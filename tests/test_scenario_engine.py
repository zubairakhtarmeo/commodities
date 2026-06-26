"""
test_scenario_engine.py
-----------------------
Tests for procurement_scenario_engine.py (PSE-5A).
Covers: all 8 validated scenarios, 4-component invariant, PULL_FORWARD reachability,
        deferral guard, WATCH status handling.
"""
from __future__ import annotations

import types
from datetime import date

import pytest

from procurement_strategy_engine import (
    DAILY_CONSUMPTION_LOCAL, DAILY_CONSUMPTION_IMPORTED,
    LOCAL_ROP_TONS, IMPORTED_ROP_TONS,
    SAFETY_STOCK_LOCAL_TONS, SAFETY_STOCK_IMPORTED_TONS,
    LOCAL_LEAD_TIME_DAYS, IMPORTED_LEAD_TIME_DAYS,
    STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE,
    _classify_supply_status,
)
from procurement_planning_engine import run_pse3d, PRICE_RISING, PRICE_FALLING, ACTION_BUY_MIXED_NOW
from procurement_calendar_engine import compute_procurement_calendar
from procurement_consolidation_engine import compute_order_consolidation
from procurement_scenario_engine import (
    _extract_base_plan, _compute_scenario_adjustment, _compute_final,
    _detect_scenario, _resolve_price_signal,
    compute_scenario_decision,
    ADJ_PULL_FORWARD_PERIODIC, ADJ_ADD_FORWARD_BUY,
    ADJ_NO_CHANGE_STOCK_DOMINATES, ADJ_DEFER_FULL,
    ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT, ACTION_DEFER, ACTION_HOLD,
)
from conftest import SCENARIO_PARAMS, EXPECTED_QTY, build_reports

TODAY     = date(2026, 6, 27)
PKR       = 278.5
CUR_PRICE = 0.78


# ===========================================================================
# Helper: run all 3 PSE-5A layers for one scenario row
# ===========================================================================

def _run_layers(source, local_inv, imp_inv, h1, h3):
    local_status = _classify_supply_status(local_inv, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS)
    imp_status   = _classify_supply_status(imp_inv,   SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS)
    total = local_inv + imp_inv

    plan = run_pse3d(
        local_inventory_tons=local_inv, imported_inventory_tons=imp_inv,
        local_status=local_status, imported_status=imp_status,
        total_inventory_tons=total,
        current_price_usd_per_lb=CUR_PRICE, forecast_h1_usd_per_lb=h1,
        forecast_h3_usd_per_lb=h3, pkr_rate=PKR, today=TODAY,
    )
    cal  = compute_procurement_calendar(
        local_inventory_tons=local_inv, imported_inventory_tons=imp_inv,
        local_status=local_status, imported_status=imp_status,
        total_inventory_tons=total, today=TODAY,
    )
    cons = compute_order_consolidation(calendar_result=cal)

    status = local_status if source == "LOCAL" else imp_status
    inv    = local_inv    if source == "LOCAL" else imp_inv
    rate   = DAILY_CONSUMPTION_LOCAL if source == "LOCAL" else DAILY_CONSUMPTION_IMPORTED
    rop    = float(LOCAL_ROP_TONS)   if source == "LOCAL" else float(IMPORTED_ROP_TONS)
    ss     = SAFETY_STOCK_LOCAL_TONS if source == "LOCAL" else SAFETY_STOCK_IMPORTED_TONS
    lt     = LOCAL_LEAD_TIME_DAYS    if source == "LOCAL" else IMPORTED_LEAD_TIME_DAYS
    deficit = plan.requirement[f"deficit_{'local' if source == 'LOCAL' else 'imported'}_tons"]
    latest_sd = plan.local_recommendation["latest_safe_order_date"] if source == "LOCAL" \
                else plan.imported_recommendation["latest_safe_order_date"]

    base   = _extract_base_plan(source, cons, plan, status, inv, rate, today=TODAY)
    fcast  = h1 if source == "LOCAL" else h3
    price_signal, delta_pct, conf = _resolve_price_signal(plan, source, CUR_PRICE, fcast, None)
    scenario = _detect_scenario(status, price_signal, inv, rop, rate)
    adj = _compute_scenario_adjustment(
        source=source, scenario=scenario, status=status,
        price_signal=price_signal, base_plan=base,
        inventory_tons=inv, total_inventory_tons=total,
        daily_rate=rate, rop_tons=rop, plan=plan, today=TODAY,
    )
    (action, fn, fl, od, arr, ds, de,
     mandatory, base_nm, opportunistic, deferred, overrides) = _compute_final(
        source=source, status=status, base_plan=base, adjustment=adj,
        latest_safe_order_date=latest_sd, today=TODAY,
        lead_time_days=lt, rop_tons=rop, safety_stock_tons=ss,
        inventory_tons=inv, deficit_from_plan=deficit,
    )
    return {
        "status": status, "action": action,
        "fn": fn, "fl": fl,
        "mandatory": mandatory, "base_nm": base_nm,
        "opportunistic": opportunistic, "deferred": deferred,
        "adj_type": adj.adjustment_type,
    }


# ===========================================================================
# CHECK A — 4-Component Invariant
# (mandatory + base_non_mandatory + opportunistic == final_now)
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_four_component_invariant(label, source, local_inv, imp_inv, h1, h3):
    """Sum of mandatory + base_nm + opportunistic must equal qty_now (final_now)."""
    r = _run_layers(source, local_inv, imp_inv, h1, h3)
    component_sum = round(r["mandatory"] + r["base_nm"] + r["opportunistic"], 2)
    assert abs(component_sum - r["fn"]) < 0.01, (
        f"{label}: mandatory({r['mandatory']:.1f}) + base_nm({r['base_nm']:.1f}) "
        f"+ opp({r['opportunistic']:.1f}) = {component_sum:.1f} ≠ fn({r['fn']:.1f})"
    )


# ===========================================================================
# CHECK B — PULL_FORWARD reachability
# ===========================================================================

def test_pull_forward_periodic_reachable():
    """At least one SAFE+RISING or WATCH+RISING scenario must reach PULL_FORWARD."""
    pull_seen = False
    for (label, src, li, ii, h1, h3) in SCENARIO_PARAMS:
        r = _run_layers(src, li, ii, h1, h3)
        if r["adj_type"] == ADJ_PULL_FORWARD_PERIODIC:
            pull_seen = True
            break
    assert pull_seen, "PULL_FORWARD_NEXT_CONSOLIDATED_ORDER is unreachable across all scenarios"


# ===========================================================================
# CHECK C — Deferral guard for CRITICAL / REORDER
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_critical_reorder_mandatory_not_deferred(label, source, local_inv, imp_inv, h1, h3):
    r = _run_layers(source, local_inv, imp_inv, h1, h3)
    if r["status"] in (STATUS_CRITICAL, STATUS_REORDER):
        assert not (r["deferred"] > 0 and r["mandatory"] > 0), (
            f"{label}: mandatory {r['mandatory']:.1f}t was deferred ({r['deferred']:.1f}t deferred)"
        )


# ===========================================================================
# CHECK D — WATCH status detected correctly
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_watch_status_correct(label, source, local_inv, imp_inv, h1, h3):
    r = _run_layers(source, local_inv, imp_inv, h1, h3)
    if "WATCH" in label:
        assert r["status"] == STATUS_WATCH, (
            f"{label}: expected WATCH status, got {r['status']}"
        )


# ===========================================================================
# Golden quantity snapshot
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_golden_quantities(label, source, local_inv, imp_inv, h1, h3):
    """Regression: procurement quantities must match the PSE-5A validated baseline."""
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    pse5a_dec = sr.local if source == "LOCAL" else sr.imported
    exp = EXPECTED_QTY[label]

    assert abs(pse5a_dec.final_qty_now_tons   - exp["qty_now"])   < 0.5, \
        f"{label}: qty_now {pse5a_dec.final_qty_now_tons:.1f} ≠ {exp['qty_now']}"
    assert abs(pse5a_dec.final_qty_later_tons - exp["qty_later"]) < 0.5, \
        f"{label}: qty_later {pse5a_dec.final_qty_later_tons:.1f} ≠ {exp['qty_later']}"
    assert abs(pse5a_dec.mandatory_component_tons - exp["mandatory"]) < 0.5, \
        f"{label}: mandatory {pse5a_dec.mandatory_component_tons:.1f} ≠ {exp['mandatory']}"


# ===========================================================================
# compute_scenario_decision — direct API test
# ===========================================================================

class TestComputeScenarioDecision:
    def _so(self, local_inv, imp_inv):
        local_status = _classify_supply_status(local_inv, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS)
        imp_status   = _classify_supply_status(imp_inv,   SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS)
        return types.SimpleNamespace(
            local_status=local_status, imported_status=imp_status,
            local_inventory_tons=local_inv, imported_inventory_tons=imp_inv,
            total_inventory_tons=local_inv + imp_inv,
            local_days_cover=round(local_inv / DAILY_CONSUMPTION_LOCAL, 1),
            imported_days_cover=round(imp_inv / DAILY_CONSUMPTION_IMPORTED, 1),
        )

    def test_returns_scenario_report(self):
        so = self._so(800.0, 8000.0)
        plan = run_pse3d(
            local_inventory_tons=800.0, imported_inventory_tons=8000.0,
            local_status=so.local_status, imported_status=so.imported_status,
            total_inventory_tons=8800.0,
            current_price_usd_per_lb=CUR_PRICE,
            forecast_h1_usd_per_lb=0.97, forecast_h3_usd_per_lb=1.06,
            pkr_rate=PKR, today=TODAY,
        )
        cal  = compute_procurement_calendar(
            local_inventory_tons=800.0, imported_inventory_tons=8000.0,
            local_status=so.local_status, imported_status=so.imported_status,
            total_inventory_tons=8800.0, today=TODAY,
        )
        cons = compute_order_consolidation(calendar_result=cal)
        sr = compute_scenario_decision(
            strategy_output=so, procurement_plan=plan,
            calendar_result=cal, consolidation_result=cons,
            current_price_usd_per_lb=CUR_PRICE,
            forecast_h1_usd_per_lb=0.97, forecast_h3_usd_per_lb=1.06,
            pkr_rate=PKR, today=TODAY,
        )
        assert sr.local  is not None
        assert sr.imported is not None
        # portfolio_action can also be BUY_MIXED_NOW when both sources act simultaneously
        valid = {ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT,
                 ACTION_DEFER, ACTION_HOLD, ACTION_BUY_MIXED_NOW}
        assert sr.portfolio_action in valid, f"Unexpected portfolio_action: {sr.portfolio_action!r}"

    def test_price_signal_in_valid_vocab(self):
        from procurement_planning_engine import PRICE_RISING, PRICE_FALLING, PRICE_NEUTRAL
        so = self._so(4000.0, 8000.0)
        plan = run_pse3d(
            local_inventory_tons=4000.0, imported_inventory_tons=8000.0,
            local_status=so.local_status, imported_status=so.imported_status,
            total_inventory_tons=12000.0,
            current_price_usd_per_lb=CUR_PRICE,
            forecast_h1_usd_per_lb=0.97, forecast_h3_usd_per_lb=1.06,
            pkr_rate=PKR, today=TODAY,
        )
        cal  = compute_procurement_calendar(
            local_inventory_tons=4000.0, imported_inventory_tons=8000.0,
            local_status=so.local_status, imported_status=so.imported_status,
            total_inventory_tons=12000.0, today=TODAY,
        )
        cons = compute_order_consolidation(calendar_result=cal)
        sr = compute_scenario_decision(
            strategy_output=so, procurement_plan=plan,
            calendar_result=cal, consolidation_result=cons,
            current_price_usd_per_lb=CUR_PRICE,
            forecast_h1_usd_per_lb=0.97, forecast_h3_usd_per_lb=1.06,
            pkr_rate=PKR, today=TODAY,
        )
        valid_signals = {PRICE_RISING, PRICE_FALLING, PRICE_NEUTRAL}
        assert sr.local.price_signal    in valid_signals
        assert sr.imported.price_signal in valid_signals
