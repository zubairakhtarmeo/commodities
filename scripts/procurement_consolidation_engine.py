"""
procurement_consolidation_engine.py
-------------------------------------
PSE-4B — Order Consolidation / Lot-Sizing Engine.

PSE-4A produces a mathematically correct procurement calendar that schedules
LOCAL cotton orders every ~10 days at ~495 tons each.  This is the minimum
possible frequency under the continuous (s, S) inventory-position policy when
starting from a CRITICAL position: the system places the smallest order that
restores the position to the order-up-to level every time the position touches
the reorder point — and with a 10-day lead time, the position touches ROP
every 10 days.

This module evaluates whether fewer, larger LOCAL orders are operationally and
financially superior — reducing administrative overhead, enabling volume
discounts, and simplifying supplier scheduling — without increasing stockout
risk or breaching the 25-day safety-stock floor.

===========================================================================
PART 1 — METHODOLOGY SELECTION
===========================================================================

Four methods were evaluated; one was selected:

    EOQ (Economic Order Quantity):
        Formula: Q* = sqrt(2 x D x S_cost / H)
        Requires: ordering cost per order (S_cost), holding cost per ton-year
        (H). NEITHER is a confirmed business rule (PSE-1 through PSE-2.7 do
        not mention fixed ordering cost or a carrying-cost percentage for
        the lot-sizing calculation). The carrying-cost rate used in PSE-3D's
        savings engine is a proxy for timing signals, not a lot-sizing input.
        Cannot apply rigorously without Finance-validated cost parameters.
        STATUS: NOT SELECTED — cost parameters unavailable.

    Fixed Order Quantity (FOQ):
        Order a confirmed lot size Q every time ROP triggers.  Requires a
        confirmed lot size (e.g. container size, bale-lot standard, or
        supplier minimum order). No such constant is confirmed anywhere in
        PSE-1 through PSE-2.7.
        STATUS: NOT SELECTED — no confirmed lot size.

    Periodic Review / Fixed Review Period (T, S):
        Review inventory every T days. Order enough to bring the position
        back to the order-up-to level S_T:
            S_T = (T + LT) x daily_rate + safety_stock
        Requires only T (a business / operational decision) — all other
        inputs (daily_rate, LT, safety_stock) are already confirmed constants
        in procurement_strategy_engine.py. The formula guarantees minimum
        on-hand = S_T - (T + LT) x daily_rate = safety_stock exactly.
        STATUS: SELECTED for LOCAL cotton, T = 30 days. Rationale below.

    Batch / Order Consolidation (post-hoc aggregation):
        Aggregate existing PSE-4A events within a time window into a single
        larger event. Rejected: post-hoc aggregation miscalculates arrival
        timing and under-sizes orders for the consolidated review period.
        Re-simulation from first principles (i.e. periodic review) is
        required instead.
        STATUS: NOT SELECTED as the algorithmic basis; the result is
        mathematically equivalent to periodic review, implemented correctly.

SELECTED METHOD — Periodic Review (T=30 days) for LOCAL cotton:

    T_local = 30 days (one monthly procurement cycle, practical for a
              supplier relationship with a 10-day delivery lead time)

    S_T_local = (T + LT_local) x daily_local + SS_local
              = (30 + 10) x 49.5 + 1,237.5
              = 40 x 49.5 + 1,237.5
              = 1,980 + 1,237.5 = 3,217.5 tons

    Minimum on-hand in steady state:
              = S_T_local - (T + LT_local) x daily_local
              = 3,217.5 - 1,980 = 1,237.5 tons = SS_local OK

    Order quantity per review (steady-state):
              = S_T_local - position ~ T x daily_local = 30 x 49.5 = 1,485 t

    Storage peak (just after a consolidated order arrives):
              ≤ S_T_local = 3,217.5 tons (well within 45,000-ton ceiling) OK

    Order frequency: ~6 per 180 days vs ~18 for PSE-4A -> 66% fewer orders

IMPORTED cotton: The natural cycle is already T ~ 90 days (the lead time
    sets the minimum review period). PSE-4A's (s, S) logic for imported cotton
    is retained unchanged in both the consolidated and original calendars.

===========================================================================
PART 2 — TWO-PHASE SIMULATION
===========================================================================

Starting from a CRITICAL position (real production: 803.2t local, 16.2-day
cover), the first periodic review cannot be placed at T=30 in the standard
way, because on-hand would fall to zero by day ~35 before the first review
order could arrive. The engine therefore uses two phases:

    Phase 1 — Recovery (LOCAL):
        Apply one modified (s, S) trigger that orders up to S_T_local
        (3,217.5t) instead of PSE-4A's smaller S_local (2,228t) when the
        position next crosses LOCAL_ROP_TONS. This single "over-sized" order
        brings the local portfolio to the consolidated order-up-to level in
        one step. In PSE-4A this same trigger fires but only orders to 2,228t,
        producing repeated 495t top-ups; here it orders 1,534t instead —
        one order that does the work of three.

    Phase 2 — Periodic Review (LOCAL):
        T days after the recovery order is placed, and every T days thereafter,
        review the position and order (S_T_local - position) tons. In steady
        state this is exactly 1,485 tons every 30 days.

    IMPORTED: (s, S) identical to PSE-4A. No phases — imported is not in
        a CRITICAL state on the real production data and is already
        operating at its natural batch size.

===========================================================================
SCOPE
===========================================================================

Does not modify: clean_inventory.py, clean_consumption.py, inventory_data.py,
consumption_data.py, update_strategy_workbook.py, procurement_engine.py,
origin_classifier.py, procurement_strategy_engine.py,
procurement_orchestrator.py, procurement_planning_engine.py,
procurement_calendar_engine.py, the Streamlit dashboard, or any UI file.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import (
    DAILY_CONSUMPTION_IMPORTED,
    DAILY_CONSUMPTION_LOCAL,
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_ROP_TONS,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_ROP_TONS,
    SAFETY_STOCK_IMPORTED_TONS,
    SAFETY_STOCK_LOCAL_TONS,
    STATUS_CRITICAL,
    STATUS_REORDER,
)
from procurement_calendar_engine import (
    DEFAULT_HORIZON_DAYS,
    TRIGGER_DEFERRED,
    TRIGGER_IMMEDIATE,
    TRIGGER_STAGGERED,
    validate_calendar,
)

# ---------------------------------------------------------------------------
# Constants — Part 1 selection outputs
# ---------------------------------------------------------------------------

# T for LOCAL periodic review (Part 1 selection: 30 days)
LOCAL_CONSOLIDATION_PERIOD_DAYS: int = 30

# Order-up-to level for consolidated local periodic review.
# S_T_local = (T + LT_local) x daily_local + SS_local
#           = (30 + 10) x 49.5 + 1,237.5 = 3,217.5 tons
S_T_LOCAL_CONSOLIDATED: float = (
    (LOCAL_CONSOLIDATION_PERIOD_DAYS + LOCAL_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_LOCAL
    + SAFETY_STOCK_LOCAL_TONS
)

# Imported order-up-to: retained from PSE-4A (natural 90-day cycle)
# S_imported = ROP_imported + LT_imported x daily_imported
#            = 6,958 + 90 x 60.5 = 12,403 tons
TARGET_STOCK_IMPORTED: float = (
    IMPORTED_ROP_TONS + IMPORTED_LEAD_TIME_DAYS * DAILY_CONSUMPTION_IMPORTED
)

# Event trigger labels
TRIGGER_CONSOLIDATION_RECOVERY = "CONSOLIDATION_RECOVERY"
TRIGGER_CONSOLIDATED_PERIODIC = "CONSOLIDATED_PERIODIC_REVIEW"
TRIGGER_PROJECTED_REORDER = "PROJECTED_REORDER"  # used for IMPORTED (unchanged)

_IMMEDIATE_TRIGGERS = frozenset({TRIGGER_IMMEDIATE, TRIGGER_DEFERRED, TRIGGER_STAGGERED})


# ===========================================================================
# PART 2 — INTERNAL SIMULATION
# ===========================================================================

def _run_consolidated_simulation(
    local_inventory_tons: float,
    imported_inventory_tons: float,
    immediate_events: list[dict],
    today: date,
    horizon_days: int,
    consolidation_period_local: int,
    s_t_local: float,
) -> tuple[list[dict], dict]:
    """Forward simulation with consolidated LOCAL periodic review + PSE-4A imported.

    Args:
        local_inventory_tons / imported_inventory_tons: starting on-hand.
        immediate_events: day-0 IMMEDIATE / DEFERRED / STAGGERED events from
            PSE-4A — preserved unchanged in the consolidated calendar.
        today: simulation start date.
        horizon_days: how many days to simulate.
        consolidation_period_local: T for local periodic review (days).
        s_t_local: order-up-to level for consolidated local review (tons).

    Returns:
        (events list, simulation stats dict)
    """
    events: list[dict] = [dict(e) for e in immediate_events]

    outstanding: dict[str, float] = {"LOCAL": 0.0, "IMPORTED": 0.0}
    for ev in events:
        outstanding[ev["source"]] += ev["quantity_tons"]

    sim_local = local_inventory_tons
    sim_imported = imported_inventory_tons
    min_local_seen = sim_local
    min_imported_seen = sim_imported

    # Phase 1 state: has the recovery order been placed?
    recovery_placed = False
    # Phase 2 state: next review day (offset from today)
    next_local_review_day: Optional[int] = None

    for day_offset in range(0, horizon_days + 1):
        current_date = today + timedelta(days=day_offset)

        if day_offset > 0:
            sim_local -= DAILY_CONSUMPTION_LOCAL
            sim_imported -= DAILY_CONSUMPTION_IMPORTED

        # Apply arrivals scheduled for today
        for ev in events:
            if ev["expected_arrival_date"] == current_date.isoformat():
                if ev["source"] == "LOCAL":
                    sim_local += ev["quantity_tons"]
                else:
                    sim_imported += ev["quantity_tons"]
                outstanding[ev["source"]] -= ev["quantity_tons"]

        min_local_seen = min(min_local_seen, sim_local)
        min_imported_seen = min(min_imported_seen, sim_imported)

        if day_offset == 0:
            continue  # day-0 already handled by immediate_events

        position_local = sim_local + outstanding["LOCAL"]
        position_imported = sim_imported + outstanding["IMPORTED"]

        # ---- LOCAL ----
        if not recovery_placed:
            # Phase 1: order up to S_T_local the first time position ≤ ROP
            if position_local <= LOCAL_ROP_TONS:
                qty = s_t_local - position_local
                if qty > 0:
                    arrival = current_date + timedelta(days=LOCAL_LEAD_TIME_DAYS)
                    events.append({
                        "source": "LOCAL",
                        "quantity_tons": qty,
                        "order_date": current_date.isoformat(),
                        "expected_arrival_date": arrival.isoformat(),
                        "reason": (
                            f"Recovery order: brings local position to the consolidated "
                            f"order-up-to level ({s_t_local:,.1f}t) in a single order, "
                            f"replacing the ~3 small top-up orders PSE-4A would place in "
                            f"the same window. After this order local cotton switches to "
                            f"{consolidation_period_local}-day periodic review."
                        ),
                        "trigger": TRIGGER_CONSOLIDATION_RECOVERY,
                        "action": "BUY_LOCAL_NOW",
                        "urgency_level": "HIGH",
                    })
                    outstanding["LOCAL"] += qty
                    recovery_placed = True
                    next_local_review_day = day_offset + consolidation_period_local

        elif next_local_review_day is not None and day_offset >= next_local_review_day:
            # Phase 2: periodic review every T days
            qty = s_t_local - position_local
            if qty > 0:
                arrival = current_date + timedelta(days=LOCAL_LEAD_TIME_DAYS)
                events.append({
                    "source": "LOCAL",
                    "quantity_tons": qty,
                    "order_date": current_date.isoformat(),
                    "expected_arrival_date": arrival.isoformat(),
                    "reason": (
                        f"Periodic review (T={consolidation_period_local} days): order to "
                        f"consolidated S-level ({s_t_local:,.1f}t), covering "
                        f"{consolidation_period_local + LOCAL_LEAD_TIME_DAYS} days of local "
                        f"supply at {DAILY_CONSUMPTION_LOCAL} t/day. "
                        f"Minimum on-hand before next arrival = {SAFETY_STOCK_LOCAL_TONS:,.1f}t "
                        f"(exactly the 25-day safety stock floor)."
                    ),
                    "trigger": TRIGGER_CONSOLIDATED_PERIODIC,
                    "action": "BUY_LOCAL_NOW",
                    "urgency_level": "PROJECTED",
                })
                outstanding["LOCAL"] += qty
            next_local_review_day += consolidation_period_local

        # ---- IMPORTED: same (s, S) as PSE-4A — no change ----
        if position_imported <= IMPORTED_ROP_TONS:
            qty = TARGET_STOCK_IMPORTED - position_imported
            if qty > 0:
                arrival = current_date + timedelta(days=IMPORTED_LEAD_TIME_DAYS)
                events.append({
                    "source": "IMPORTED",
                    "quantity_tons": qty,
                    "order_date": current_date.isoformat(),
                    "expected_arrival_date": arrival.isoformat(),
                    "reason": (
                        f"Imported position projected at reorder point ({IMPORTED_ROP_TONS:,}t). "
                        "Order now — 90-day lead time requires orders well in advance."
                    ),
                    "trigger": TRIGGER_PROJECTED_REORDER,
                    "action": "BUY_IMPORTED_NOW",
                    "urgency_level": "PROJECTED",
                })
                outstanding["IMPORTED"] += qty

    events.sort(key=lambda e: (e["order_date"], e["source"]))
    for e in events:
        e["quantity_tons"] = round(e["quantity_tons"], 2)

    return events, {
        "min_local_tons_seen": round(min_local_seen, 2),
        "min_imported_tons_seen": round(min_imported_seen, 2),
        "horizon_days_simulated": horizon_days,
        "starting_local_tons": local_inventory_tons,
        "starting_imported_tons": imported_inventory_tons,
    }


# ===========================================================================
# PART 3 — COMPARISON
# ===========================================================================

def _compare_calendars(original_events: list[dict], consolidated_events: list[dict]) -> dict:
    """Before / after metrics: order count, total tons, avg order size."""
    def _metrics(events: list[dict], source: Optional[str]) -> dict:
        subset = events if source is None else [e for e in events if e["source"] == source]
        n = len(subset)
        total = round(sum(e["quantity_tons"] for e in subset), 2)
        avg = round(total / n, 2) if n else 0.0
        return {"order_count": n, "total_tons": total, "avg_tons_per_order": avg}

    result: dict = {}
    for label, src in (("local", "LOCAL"), ("imported", "IMPORTED"), ("total", None)):
        o = _metrics(original_events, src)
        c = _metrics(consolidated_events, src)
        reduction = o["order_count"] - c["order_count"]
        reduction_pct = round(reduction / o["order_count"] * 100, 1) if o["order_count"] else 0.0
        avg_size_increase = round(c["avg_tons_per_order"] - o["avg_tons_per_order"], 2)
        result[label] = {
            "original": o,
            "consolidated": c,
            "order_count_reduction": reduction,
            "order_count_reduction_pct": reduction_pct,
            "avg_order_size_increase_tons": avg_size_increase,
        }
    return result


# ===========================================================================
# PART 4 — CONSOLIDATED CALENDAR VALIDATION
# ===========================================================================

def _validate_consolidated(consolidated_result: dict) -> dict:
    """Re-use validate_calendar() from PSE-4A — identical safety-stock logic."""
    return validate_calendar(consolidated_result)


# ===========================================================================
# PART 5 — RECOMMENDATION
# ===========================================================================

def _generate_recommendation(
    comparison: dict,
    validation_consolidated: dict,
    validation_original: dict,
    consolidation_period: int,
    s_t_local: float,
) -> dict:
    """Produce Option A / B / C recommendation with supporting rationale.

    Criteria:
        - If consolidated validation fails (recurring defect): Option A
        - If order-count reduction ≥ 60% and validation passes: Option C
        - Otherwise: Option B
    """
    passed = validation_consolidated["all_passed"]
    local_cmp = comparison["local"]
    total_cmp = comparison["total"]

    order_reduction_pct = local_cmp["order_count_reduction_pct"]
    avg_increase = local_cmp["avg_order_size_increase_tons"]

    if not passed:
        production_default = "A"
        summary = (
            "Consolidated calendar did not pass safety-stock validation "
            "(recurring defect detected). Retain PSE-4A as production default "
            "until the defect is investigated."
        )
    elif order_reduction_pct >= 60.0:
        production_default = "C"
        summary = (
            f"Consolidated calendar reduces local procurement transactions by "
            f"{order_reduction_pct:.0f}% ({local_cmp['original']['order_count']} -> "
            f"{local_cmp['consolidated']['order_count']} orders over 180 days) "
            f"without any recurring safety-stock breach. "
            f"Option C (Hybrid) is the recommended production default: use "
            f"T={consolidation_period}-day periodic review for local in normal "
            f"steady state, reverting to PSE-4A (s, S) when local status is "
            f"CRITICAL or when a PRICE_FALLING signal justifies opportunistic "
            f"close-spaced purchases."
        )
    else:
        production_default = "B"
        summary = (
            f"Consolidated calendar reduces orders by {order_reduction_pct:.0f}% "
            "with maintained security. Use as production default."
        )

    options = {
        "A": {
            "label": "Frequent-order calendar (PSE-4A)",
            "trigger_policy": "(s, S) continuous review",
            "local_order_frequency": "Every ~10 days",
            "local_order_size_tons": "~495–545t per order",
            "pros": [
                "Maximum responsiveness to price signals",
                "Smallest capital outlay per order",
                "Easiest to cancel or adjust individual orders",
                "Natural fit during CRITICAL recovery",
            ],
            "cons": [
                f"~{local_cmp['original']['order_count']} local procurement "
                "transactions per 180 days",
                "High administrative burden",
                "Fewer volume-discount opportunities",
            ],
        },
        "B": {
            "label": f"Consolidated calendar (T={consolidation_period} periodic review)",
            "trigger_policy": f"Periodic review every {consolidation_period} days",
            "local_order_frequency": f"Every {consolidation_period} days in steady state",
            "local_order_size_tons": f"~{round(consolidation_period * DAILY_CONSUMPTION_LOCAL, 1)}t per order",
            "s_t_local_tons": round(s_t_local, 2),
            "pros": [
                f"{order_reduction_pct:.0f}% fewer local procurement transactions",
                f"Average order size increases by {avg_increase:,.0f}t "
                "(volume discount potential)",
                "Simpler monthly supplier scheduling",
                f"Minimum on-hand guaranteed ≥ {SAFETY_STOCK_LOCAL_TONS:,.1f}t "
                "(25-day safety stock) in steady state",
            ],
            "cons": [
                "Larger capital commitment per order",
                "Less responsive to short-term price movements between reviews",
                "Recovery from CRITICAL requires one ramp-up order before "
                "periodic review can begin",
            ],
        },
        "C": {
            "label": "Hybrid (Recommended)",
            "trigger_policy": (
                f"T={consolidation_period} periodic review in steady state; "
                "PSE-4A (s, S) during CRITICAL or PRICE_FALLING"
            ),
            "pros": [
                "Efficiency in normal state (fewer orders)",
                "Full responsiveness when status is CRITICAL or price falls",
                "Natural two-mode structure matching MG Apparel's procurement rhythm",
                "Imported cotton unchanged — already at natural 90-day batch",
            ],
            "cons": [
                "Requires mode-switching logic in procurement execution",
                "Slightly higher planning complexity vs a single fixed rule",
            ],
        },
    }

    return {
        "production_default": production_default,
        "summary": summary,
        "options": options,
        "formulas": {
            "periodic_review_order_up_to":
                f"S_T_local = (T + LT_local) x daily_local + SS_local"
                f" = ({consolidation_period} + {LOCAL_LEAD_TIME_DAYS}) x "
                f"{DAILY_CONSUMPTION_LOCAL} + {SAFETY_STOCK_LOCAL_TONS} "
                f"= {round(s_t_local, 2)} tons",
            "min_onhand_steady_state":
                f"min_on_hand = S_T_local - (T + LT_local) x daily_local"
                f" = {round(s_t_local, 2)} - "
                f"{round((consolidation_period + LOCAL_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_LOCAL, 2)}"
                f" = {SAFETY_STOCK_LOCAL_TONS} tons (= 25-day SS floor, PASS)",
            "order_qty_steady_state":
                f"Q = T x daily_local = {consolidation_period} x "
                f"{DAILY_CONSUMPTION_LOCAL} = "
                f"{round(consolidation_period * DAILY_CONSUMPTION_LOCAL, 1)} tons",
        },
    }


# ===========================================================================
# PUBLIC API
# ===========================================================================

def compute_order_consolidation(
    calendar_result: dict,
    consolidation_period_local: int = LOCAL_CONSOLIDATION_PERIOD_DAYS,
) -> dict:
    """PSE-4B entry point — produce and compare original and consolidated calendars.

    Args:
        calendar_result: output of procurement_calendar_engine.
                         compute_procurement_calendar(). Must contain 'events',
                         'today', and 'simulation' keys.
        consolidation_period_local: T for local periodic review (days).
                                     Default: 30 (confirmed Part 1 selection).

    Returns:
        dict with keys:
            original              -- the unchanged PSE-4A calendar_result
            consolidated          -- consolidated calendar (same structure as
                                     compute_procurement_calendar() output)
            comparison            -- before/after metrics (local, imported, total)
            validation_original   -- validate_calendar() result for PSE-4A
            validation_consolidated -- validate_calendar() result for consolidated
            recommendation        -- Option A/B/C analysis with formulas
    """
    today = date.fromisoformat(calendar_result["today"])
    sim = calendar_result["simulation"]
    horizon_days = sim["horizon_days_simulated"]
    local_start = sim["starting_local_tons"]
    imported_start = sim["starting_imported_tons"]

    s_t_local = (
        (consolidation_period_local + LOCAL_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_LOCAL
        + SAFETY_STOCK_LOCAL_TONS
    )

    # Only preserve day-0 decisions from PSE-3D — projected events are regenerated
    immediate_events = [
        e for e in calendar_result["events"]
        if e.get("trigger") in _IMMEDIATE_TRIGGERS
    ]

    cons_events, cons_sim = _run_consolidated_simulation(
        local_start, imported_start, immediate_events,
        today, horizon_days, consolidation_period_local, s_t_local,
    )

    def _view(events: list[dict], days: int) -> list[dict]:
        cutoff = today + timedelta(days=days)
        return [e for e in events if date.fromisoformat(e["order_date"]) <= cutoff]

    consolidated_result = {
        "today": today.isoformat(),
        "events": cons_events,
        "view_30_days": _view(cons_events, 30),
        "view_90_days": _view(cons_events, 90),
        "view_180_days": _view(cons_events, 180),
        "simulation": cons_sim,
        "consolidation_params": {
            "period_local_days": consolidation_period_local,
            "s_t_local_tons": round(s_t_local, 2),
            "target_stock_imported_tons": round(TARGET_STOCK_IMPORTED, 2),
            "method": (
                f"LOCAL: Periodic Review T={consolidation_period_local}d (recovery "
                f"order + periodic); IMPORTED: PSE-4A (s,S) unchanged"
            ),
        },
    }

    validation_original = validate_calendar(calendar_result)
    validation_consolidated = _validate_consolidated(consolidated_result)
    comparison = _compare_calendars(calendar_result["events"], cons_events)
    recommendation = _generate_recommendation(
        comparison, validation_consolidated, validation_original,
        consolidation_period_local, s_t_local,
    )

    return {
        "original": calendar_result,
        "consolidated": consolidated_result,
        "comparison": comparison,
        "validation_original": validation_original,
        "validation_consolidated": validation_consolidated,
        "recommendation": recommendation,
    }


# ===========================================================================
# REPORT PRINTER
# ===========================================================================

def print_consolidation_report(result: dict) -> None:
    """Print a formatted PSE-4B consolidation report."""
    cmp = result["comparison"]
    vo = result["validation_original"]
    vc = result["validation_consolidated"]
    rec = result["recommendation"]
    cons_params = result["consolidated"]["consolidation_params"]
    today = result["original"]["today"]

    print("=" * 78)
    print("PSE-4B ORDER CONSOLIDATION REPORT")
    print(f"  As of : {today}")
    print(f"  Method: {cons_params['method']}")
    print("=" * 78)

    # --- Part 1: method summary ---
    print("\n-- PART 1: METHODOLOGY SELECTION --")
    print(f"  EOQ              : NOT SELECTED (ordering/holding cost parameters not confirmed)")
    print(f"  Fixed Order Qty  : NOT SELECTED (no confirmed lot size in PSE-1–2.7)")
    print(f"  Periodic Review  : SELECTED -- LOCAL T={cons_params['period_local_days']}d, "
          f"S_T={cons_params['s_t_local_tons']:,.1f}t")
    print(f"  Imported         : PSE-4A (s,S) unchanged -- already at natural 90-day cycle")

    # --- Part 1 formulas ---
    for label, formula in rec["formulas"].items():
        print(f"\n  {label}:")
        print(f"      {formula}")

    # --- Part 2: before/after comparison ---
    print("\n-- PART 4: BEFORE / AFTER COMPARISON (180-day horizon) --")
    header = f"  {'':12s} {'Orders':>8s} {'Total Tons':>12s} {'Avg Tons/Ord':>14s}"
    print(header)
    print("  " + "-" * 50)

    for label in ("local", "imported", "total"):
        src = cmp[label]
        o = src["original"]
        c = src["consolidated"]
        print(f"\n  {label.upper() + ' (before)':22s} {o['order_count']:>6}   "
              f"{o['total_tons']:>10,.1f}   {o['avg_tons_per_order']:>12,.1f}")
        print(f"  {label.upper() + ' (after)':22s} {c['order_count']:>6}   "
              f"{c['total_tons']:>10,.1f}   {c['avg_tons_per_order']:>12,.1f}")
        if src["order_count_reduction"] != 0:
            print(f"  {'  >> reduction':22s} {src['order_count_reduction']:>6}   "
                  f"{'':>10}   "
                  f"  ({src['order_count_reduction_pct']:.0f}% fewer orders)")

    # --- Part 3: validation ---
    print("\n-- PART 3: VALIDATION --")
    _v_line = lambda v, label: (
        f"  [{('PASS' if v['all_passed'] else 'FAIL')}] {label} — "
        f"min_local_after_d90={v['min_local_tons_seen_after_day_90']:,.1f}t "
        f"(SS={v['local_safety_stock_tons']:,.1f})  "
        f"min_imported_after_d90={v['min_imported_tons_seen_after_day_90']:,.1f}t "
        f"(SS={v['imported_safety_stock_tons']:,.1f})"
    )
    print(_v_line(vo, "Original (PSE-4A)   "))
    print(_v_line(vc, "Consolidated (PSE-4B)"))
    if vc["local_startup_transient"]:
        print(f"  [INFO] Local startup transient: min_local={vc['min_local_tons_seen']:,.1f}t "
              f"(within first 90 days, unavoidable — CRITICAL start position)")
    if vc["imported_startup_transient"]:
        print(f"  [INFO] Imported startup transient detected (within first 90 days)")

    # --- Part 5: recommendation ---
    print(f"\n-- PART 5: RECOMMENDATION --")
    print(f"\n  Production default: OPTION {rec['production_default']}")
    print(f"\n  {rec['summary']}")
    print()
    for opt, detail in rec["options"].items():
        marker = " <- RECOMMENDED" if opt == rec["production_default"] else ""
        print(f"  OPTION {opt}: {detail['label']}{marker}")
        print(f"    Policy : {detail.get('trigger_policy', 'see label')}")
        if "local_order_size_tons" in detail:
            print(f"    Qty    : {detail['local_order_size_tons']}")
        print(f"    Pros   : {'; '.join(detail['pros'][:2])}")
        print(f"    Cons   : {detail['cons'][0]}")
        print()

    # --- Consolidated 90-day calendar ---
    cons_view = result["consolidated"]["view_90_days"]
    orig_view = result["original"]["view_90_days"]
    print(f"-- CONSOLIDATED 90-DAY CALENDAR ({len(cons_view)} events) "
          f"vs ORIGINAL ({len(orig_view)} events) --")
    for i, e in enumerate(cons_view, 1):
        print(f"\n  [{i}] {e['source']:8s}  {e['trigger']:30s}  "
              f"{e['quantity_tons']:>8,.1f}t")
        print(f"       Order: {e['order_date']}   Arrival: {e['expected_arrival_date']}")

    print("\n" + "=" * 78)


# ===========================================================================
# CLI — end-to-end run on real production data
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "PSE-4B: Order Consolidation Engine — "
            "run on real production data and print comparison report."
        )
    )
    parser.add_argument("--workbook", default=None,
                        help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--input", default=None,
                        help="Raw Oracle export (.xlsx)")
    parser.add_argument("--period", type=int, default=LOCAL_CONSOLIDATION_PERIOD_DAYS,
                        help=f"Local consolidation period in days (default: {LOCAL_CONSOLIDATION_PERIOD_DAYS})")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS,
                        help=f"Simulation horizon in days (default: {DEFAULT_HORIZON_DAYS})")
    args = parser.parse_args()

    from procurement_orchestrator import run_orchestration
    from procurement_calendar_engine import compute_procurement_calendar

    # Step 1: Orchestrate (load -> classify -> convert -> strategy)
    orch = run_orchestration(input_path=args.input, workbook_path=args.workbook)
    so = orch["strategy_output"]

    print(f"\nLive portfolio: local={so.local_inventory_tons:,.1f}t  "
          f"imported={so.imported_inventory_tons:,.1f}t  "
          f"action={so.action}")

    # Step 2: PSE-4A calendar
    calendar = compute_procurement_calendar(
        local_inventory_tons=so.local_inventory_tons,
        imported_inventory_tons=so.imported_inventory_tons,
        local_status=so.local_status,
        imported_status=so.imported_status,
        total_inventory_tons=so.total_inventory_tons,
        horizon_days=args.horizon,
    )

    # Step 3: PSE-4B consolidation
    result = compute_order_consolidation(
        calendar_result=calendar,
        consolidation_period_local=args.period,
    )

    print_consolidation_report(result)

    vc = result["validation_consolidated"]
    sys.exit(0 if vc["all_passed"] else 1)
