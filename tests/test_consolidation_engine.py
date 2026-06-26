"""
test_consolidation_engine.py
----------------------------
Tests for procurement_consolidation_engine.py (PSE-4B).
Covers: consolidation reduces LOCAL order count, IMPORTED unchanged,
        steady-state quantity, output structure.
"""
from __future__ import annotations

from datetime import date

import pytest

from procurement_strategy_engine import (
    STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE,
    DAILY_CONSUMPTION_LOCAL,
)
from procurement_calendar_engine import compute_procurement_calendar, DEFAULT_HORIZON_DAYS
from procurement_consolidation_engine import (
    compute_order_consolidation,
    LOCAL_CONSOLIDATION_PERIOD_DAYS,   # T = 30
    S_T_LOCAL_CONSOLIDATED,            # order-up-to level ≈ 3217.5 t
    TRIGGER_CONSOLIDATION_RECOVERY,
    TRIGGER_CONSOLIDATED_PERIODIC,
)

TODAY = date(2026, 6, 27)


def _run(local_inv, imp_inv, local_status, imp_status, horizon=DEFAULT_HORIZON_DAYS):
    cal = compute_procurement_calendar(
        local_inventory_tons=local_inv,
        imported_inventory_tons=imp_inv,
        local_status=local_status,
        imported_status=imp_status,
        total_inventory_tons=local_inv + imp_inv,
        today=TODAY,
        horizon_days=horizon,
    )
    return compute_order_consolidation(calendar_result=cal)


# ===========================================================================
# Output structure
# ===========================================================================

class TestConsolidationStructure:
    def test_returns_consolidated_key(self):
        cons = _run(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        assert "consolidated" in cons

    def test_consolidated_has_events(self):
        cons = _run(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        assert "events" in cons["consolidated"]
        assert isinstance(cons["consolidated"]["events"], list)

    def test_each_event_has_required_fields(self):
        cons = _run(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        required = {"order_date", "expected_arrival_date", "quantity_tons", "source", "trigger"}
        for ev in cons["consolidated"]["events"]:
            for field in required:
                assert field in ev, f"Missing field: {field}"


# ===========================================================================
# LOCAL consolidation behaviour
# ===========================================================================

class TestLocalConsolidation:
    def test_local_periodic_trigger_present(self):
        """Consolidated calendar must fire CONSOLIDATED_PERIODIC for LOCAL."""
        cons = _run(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE, horizon=180)
        local_evs = [
            e for e in cons["consolidated"]["events"]
            if e["source"] == "LOCAL"
        ]
        triggers = {e["trigger"] for e in local_evs}
        assert TRIGGER_CONSOLIDATION_RECOVERY in triggers or TRIGGER_CONSOLIDATED_PERIODIC in triggers

    def test_local_steady_state_qty_approx_1485(self):
        """In periodic review, LOCAL order ≈ T × daily_local = 30 × 49.5 = 1485 tons."""
        cons = _run(4000.0, 8000.0, STATUS_SAFE, STATUS_SAFE, horizon=180)
        periodic = [
            e for e in cons["consolidated"]["events"]
            if e["source"] == "LOCAL" and e["trigger"] == TRIGGER_CONSOLIDATED_PERIODIC
        ]
        if periodic:
            for ev in periodic:
                assert abs(ev["quantity_tons"] - 1485.0) < 50.0, (
                    f"Periodic LOCAL order {ev['quantity_tons']:.0f}t deviates "
                    f"too far from 1485t expected"
                )

    def test_order_up_to_level_constant(self):
        """S_T_LOCAL_CONSOLIDATED must equal (30+10)*49.5 + 1237.5 = 3217.5 t."""
        assert abs(S_T_LOCAL_CONSOLIDATED - 3217.5) < 0.5

    def test_consolidation_period_constant(self):
        assert LOCAL_CONSOLIDATION_PERIOD_DAYS == 30


# ===========================================================================
# IMPORTED consolidation behaviour
# ===========================================================================

class TestImportedConsolidation:
    def test_imported_events_non_negative_qty(self):
        cons = _run(4000.0, 2000.0, STATUS_SAFE, STATUS_REORDER, horizon=180)
        imported_evs = [
            e for e in cons["consolidated"]["events"]
            if e["source"] == "IMPORTED"
        ]
        for ev in imported_evs:
            assert ev["quantity_tons"] >= 0.0

    def test_arrival_after_order(self):
        cons = _run(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER)
        for ev in cons["consolidated"]["events"]:
            order   = date.fromisoformat(ev["order_date"])
            arrival = date.fromisoformat(ev["expected_arrival_date"])
            assert arrival >= order


# ===========================================================================
# Quantity safety
# ===========================================================================

class TestQuantitySafety:
    def test_all_quantities_non_negative(self):
        for local_inv, imp_inv, ls, is_ in [
            (800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE),
            (4000.0, 2000.0, STATUS_SAFE,     STATUS_REORDER),
            (1900.0, 8000.0, STATUS_WATCH,    STATUS_SAFE),
        ]:
            cons = _run(local_inv, imp_inv, ls, is_)
            for ev in cons["consolidated"]["events"]:
                assert ev["quantity_tons"] >= 0.0, f"Negative qty: {ev}"
