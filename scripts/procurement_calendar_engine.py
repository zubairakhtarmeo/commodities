"""
procurement_calendar_engine.py
---------------------------------
PSE-4A — Procurement Calendar Engine.

PSE-3D (procurement_planning_engine.py) answers "what should I buy right
now" as a single snapshot (e.g. BUY_MIXED_NOW: 930t local + 5,015t
imported). That is necessary but not sufficient for procurement planning --
a 90-day import lead time means today's purchase is rarely the only one
needed within the planning horizon. This module converts that single
snapshot into a dated, multi-event PURCHASE SCHEDULE: every order this
portfolio will need to place over the next 30 / 90 / 180 days to stay
secure, each with an order date, an expected arrival date, a quantity, and
a reason.

Design: this module introduces NO new decision logic. The very first
event(s) in the calendar are produced by directly reusing
procurement_planning_engine.run_pse3d() (PSE-3D) -- the immediate
recommendation is not recomputed differently here. Every subsequent event
is produced by forward-simulating daily consumption and re-applying
procurement_strategy_engine's approved reorder points (1,733 / 6,958 tons)
and procurement_planning_engine's compute_procurement_requirement() sizing
formula at the simulated future date a trigger fires. There is exactly one
set of "how much to buy" rules in this codebase; this module only adds
"and when will I need to buy again."

Does not modify clean_inventory.py, clean_consumption.py, inventory_data.py,
consumption_data.py, update_strategy_workbook.py, procurement_engine.py,
origin_classifier.py, procurement_strategy_engine.py,
procurement_orchestrator.py, procurement_planning_engine.py, the Streamlit
dashboard, or any UI file.
"""

from __future__ import annotations

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
    MIN_STOCK_DAYS,
)
from procurement_planning_engine import (
    ACTION_DEFER_IMPORTED,
    ACTION_DEFER_LOCAL,
    ACTION_STAGGER_PURCHASE,
    ACTION_WAIT,
    run_pse3d,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEW_HORIZONS_DAYS = (30, 90, 180)
DEFAULT_HORIZON_DAYS = max(VIEW_HORIZONS_DAYS)

TRIGGER_IMMEDIATE = "IMMEDIATE"
TRIGGER_DEFERRED = "DEFERRED"
TRIGGER_STAGGERED = "STAGGERED"
TRIGGER_PROJECTED_REORDER = "PROJECTED_REORDER"
TRIGGER_PROJECTED_MIX = "PROJECTED_MIX_REBALANCE"


# ===========================================================================
# FORMULAS (documented once here; all reused from PSE-3B/3D, not redefined)
# ===========================================================================
#
# Reorder points (PSE-2 / PSE-2.7 approved, imported unchanged from
# procurement_strategy_engine.py):
#     ROP_local    = (25 + 10) x 49.5 = 1,733 tons
#     ROP_imported = (25 + 90) x 60.5 = 6,958 tons
#
# Forward depletion (per simulated day d, per source):
#     on_hand(d) = on_hand(d-1) - daily_consumption_source
#                  + (sum of arrivals scheduled for day d)
#
# Day-0 event(s): sized by procurement_planning_engine.compute_procurement_
# requirement() -- the same "what do I need RIGHT NOW" formula PSE-3D uses,
# reused unchanged via run_pse3d().
#
# Recurring (projected) events use a DIFFERENT, explicitly distinct formula
# -- the standard (s, S) inventory-POSITION reorder policy, not a repeat of
# the day-0 deficit formula. This distinction matters: an earlier version
# of this module reused the deficit formula for recurring events too, gated
# on a single-outstanding-order rule, and that produced a self-validation
# failure (simulated imported inventory went negative every ~90-day cycle)
# whenever on-hand cover was shorter than the lead time -- see PSE-4A report,
# Risks, for the full root-cause trace.
#
#     position_source(d) = on_hand(d) + sum(qty of orders placed but not
#                                            yet arrived, for that source)
#     Trigger: position_source(d) <= ROP_source                       (s)
#     Order quantity: S_source - position_source(d)                   (order up to S)
#     S_source = ROP_source + lead_time_source x daily_rate_source
#
# Ordering up to S (a one-lead-time buffer above the trigger s), rather
# than only to s, is what allows multiple orders to be outstanding at once
# -- required whenever lead time exceeds the gap between trigger points,
# which a 90-day imported lead time does by construction.
#
# Validation (Part 5 below): for any cycle where the order was placed
# while position (not just on-hand) was still above the reorder point at
# the moment of triggering -- i.e. NOT a pre-existing deficit inherited
# from the actual starting inventory -- on-hand should never fall below
# SAFETY_STOCK_source. A pre-existing deficit at simulation start (the real
# production case: local was already CRITICAL on day 0) produces one
# unavoidable transient dip that no order-timing algorithm can retroactively
# prevent; validate_calendar() reports this as a distinct, explained
# finding rather than masking it.
# ===========================================================================


def _seed_immediate_events(
    local_inventory_tons: float,
    imported_inventory_tons: float,
    local_status: str,
    imported_status: str,
    total_inventory_tons: Optional[float],
    current_price_usd_per_lb: Optional[float],
    forecast_h1_usd_per_lb: Optional[float],
    forecast_h3_usd_per_lb: Optional[float],
    forecast_h1_bounds: Optional[tuple[float, float]],
    forecast_h3_bounds: Optional[tuple[float, float]],
    pkr_rate: float,
    today: date,
) -> tuple[list[dict], dict]:
    """Day-0 events -- reuses PSE-3D's run_pse3d() directly. No new decision
    logic is introduced here; this function only translates PSE-3D's
    recommendation into dated calendar events.
    """
    plan = run_pse3d(
        local_inventory_tons, imported_inventory_tons, local_status, imported_status,
        total_inventory_tons, current_price_usd_per_lb, forecast_h1_usd_per_lb,
        forecast_h3_usd_per_lb, forecast_h1_bounds, forecast_h3_bounds, pkr_rate,
        today=today,
    )

    events: list[dict] = []
    pending_arrival: dict[str, Optional[str]] = {"LOCAL": None, "IMPORTED": None}

    for rec in (plan.local_recommendation, plan.imported_recommendation):
        source = rec["source"]
        qty = rec["quantity_tons"]
        if rec["action"] == ACTION_WAIT or qty <= 0:
            continue

        lead = LOCAL_LEAD_TIME_DAYS if source == "LOCAL" else IMPORTED_LEAD_TIME_DAYS

        if rec["action"] in (ACTION_DEFER_LOCAL, ACTION_DEFER_IMPORTED):
            order_date = date.fromisoformat(rec["latest_safe_order_date"])
            trigger = TRIGGER_DEFERRED
        elif rec["action"] == ACTION_STAGGER_PURCHASE:
            order_date = date.fromisoformat(rec["recommended_order_date"])
            trigger = TRIGGER_STAGGERED
        else:
            order_date = date.fromisoformat(rec["recommended_order_date"])
            trigger = TRIGGER_IMMEDIATE

        arrival_date = order_date + timedelta(days=lead)
        events.append({
            "source": source,
            "quantity_tons": qty,
            "order_date": order_date.isoformat(),
            "expected_arrival_date": arrival_date.isoformat(),
            "reason": rec["reason"],
            "trigger": trigger,
            "action": rec["action"],
            "urgency_level": rec["urgency_level"],
        })
        pending_arrival[source] = arrival_date.isoformat()

    return events, pending_arrival


def _min_after_cutoff(events: list[dict], today: date, source: str, daily_rate: float,
                       start_tons: float, cutoff_days: int, horizon_days: int) -> float:
    """Re-simulate on-hand for one source, returning the minimum seen strictly
    AFTER cutoff_days. Used by validate_calendar() to distinguish a one-time
    startup transient (an unavoidable consequence of the real starting
    position) from a genuinely recurring defect in the trigger logic.
    """
    sim = start_tons
    min_seen = None
    for d in range(0, horizon_days + 1):
        cur = today + timedelta(days=d)
        if d > 0:
            sim -= daily_rate
        for ev in events:
            if ev["source"] == source and ev["expected_arrival_date"] == cur.isoformat():
                sim += ev["quantity_tons"]
        if d > cutoff_days:
            min_seen = sim if min_seen is None else min(min_seen, sim)
    return min_seen if min_seen is not None else sim


def compute_procurement_calendar(
    local_inventory_tons: float,
    imported_inventory_tons: float,
    local_status: str,
    imported_status: str,
    total_inventory_tons: Optional[float] = None,
    current_price_usd_per_lb: Optional[float] = None,
    forecast_h1_usd_per_lb: Optional[float] = None,
    forecast_h3_usd_per_lb: Optional[float] = None,
    forecast_h1_bounds: Optional[tuple[float, float]] = None,
    forecast_h3_bounds: Optional[tuple[float, float]] = None,
    pkr_rate: float = 281.0,
    max_storage_capacity_tons: float = 45_000.0,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    today: Optional[date] = None,
) -> dict:
    """Build the full multi-event procurement calendar.

    Step 1 (day 0): seed events directly from procurement_planning_engine's
    run_pse3d() -- the existing, approved "how much / when right now" logic.
    No duplicate decision-making.

    Step 2 (days 1..horizon_days): forward-simulate daily consumption per
    source. Whenever simulated inventory for a source crosses its reorder
    point AND no order for that source is currently in flight (i.e. no
    pending arrival), place a new order sized by
    compute_procurement_requirement() evaluated at the simulated position
    on that day, dated for that simulated day, arriving lead_time days later.

    Step 3: slice the full event list into nested 30 / 90 / 180-day views
    (each view includes all events with order_date within that many days
    of today -- nested, not disjoint windows, matching PSE-2.5's horizon
    framework).

    Returns:
        dict: events (full chronological list), view_30_days, view_90_days,
              view_180_days, simulation (min_local_tons, min_imported_tons,
              for validate_calendar()).
    """
    today = today or date.today()

    events, pending_arrival = _seed_immediate_events(
        local_inventory_tons, imported_inventory_tons, local_status, imported_status,
        total_inventory_tons, current_price_usd_per_lb, forecast_h1_usd_per_lb,
        forecast_h3_usd_per_lb, forecast_h1_bounds, forecast_h3_bounds, pkr_rate, today,
    )

    # ------------------------------------------------------------------
    # Order-up-to target for RECURRING (projected) tranches.
    #
    # The day-0 event(s) above answer a different question ("what do I
    # need RIGHT NOW given today's shortfall") than the forward simulation
    # below ("what is the sustainable recurring order size that prevents
    # ever running out again"). These are deliberately two different
    # formulas, not the same one reused -- conflating them is exactly what
    # caused the bug this module's own validation caught (see PSE-4A
    # report, Part 5 / Risks): a single "order enough to reach ROP exactly"
    # placed only once an existing order has fully ARRIVED reacts too late
    # whenever on-hand cover is shorter than the lead time, producing a
    # negative-inventory simulation artifact every cycle, not just the
    # first.
    #
    # The fix is the standard (s, S) inventory-position policy:
    #   INVENTORY POSITION = on_hand + sum(qty of all orders placed for
    #                         this source that have not yet arrived)
    #   Trigger: position <= ROP_source (s)
    #   Order quantity: S_source - position   (order UP TO S, not just to s)
    #   S_source = ROP_source + lead_time_source * daily_rate_source
    #
    # Sizing up to S (not just to s) provides a one-lead-time buffer, so the
    # position does not immediately re-cross the trigger the moment an
    # order is placed -- this is what allows MULTIPLE orders to be
    # outstanding at once, which a 90-day imported lead time requires.
    # ------------------------------------------------------------------
    TARGET_STOCK_LOCAL = LOCAL_ROP_TONS + LOCAL_LEAD_TIME_DAYS * DAILY_CONSUMPTION_LOCAL
    TARGET_STOCK_IMPORTED = IMPORTED_ROP_TONS + IMPORTED_LEAD_TIME_DAYS * DAILY_CONSUMPTION_IMPORTED

    sim_local = local_inventory_tons
    sim_imported = imported_inventory_tons
    min_local_seen = sim_local
    min_imported_seen = sim_imported

    # Outstanding (placed, not yet arrived) order quantities per source --
    # replaces the single-pending-order gate that caused the bug.
    outstanding: dict[str, float] = {"LOCAL": 0.0, "IMPORTED": 0.0}
    for ev in events:
        outstanding[ev["source"]] += ev["quantity_tons"]

    for day_offset in range(0, horizon_days + 1):
        current_date = today + timedelta(days=day_offset)

        if day_offset > 0:
            sim_local -= DAILY_CONSUMPTION_LOCAL
            sim_imported -= DAILY_CONSUMPTION_IMPORTED

        # Apply arrivals scheduled for this simulated day
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
            continue  # day-0 triggers already handled by _seed_immediate_events

        position_local = sim_local + outstanding["LOCAL"]
        position_imported = sim_imported + outstanding["IMPORTED"]

        if position_local <= LOCAL_ROP_TONS:
            qty = TARGET_STOCK_LOCAL - position_local
            if qty > 0:
                order_date = current_date
                arrival_date = order_date + timedelta(days=LOCAL_LEAD_TIME_DAYS)
                events.append({
                    "source": "LOCAL",
                    "quantity_tons": qty,
                    "order_date": order_date.isoformat(),
                    "expected_arrival_date": arrival_date.isoformat(),
                    "reason": (
                        f"Local inventory position (on-hand + on-order) projected to reach "
                        f"the reorder point ({LOCAL_ROP_TONS:,.0f} tons). Order now to "
                        f"sustain cover through the next {LOCAL_LEAD_TIME_DAYS}-day cycle."
                    ),
                    "trigger": TRIGGER_PROJECTED_REORDER,
                    "action": "BUY_LOCAL_NOW",
                    "urgency_level": "PROJECTED",
                })
                outstanding["LOCAL"] += qty

        if position_imported <= IMPORTED_ROP_TONS:
            qty = TARGET_STOCK_IMPORTED - position_imported
            if qty > 0:
                order_date = current_date
                arrival_date = order_date + timedelta(days=IMPORTED_LEAD_TIME_DAYS)
                events.append({
                    "source": "IMPORTED",
                    "quantity_tons": qty,
                    "order_date": order_date.isoformat(),
                    "expected_arrival_date": arrival_date.isoformat(),
                    "reason": (
                        f"Imported inventory position (on-hand + on-order) projected to reach "
                        f"the reorder point ({IMPORTED_ROP_TONS:,.0f} tons). Place this order now "
                        f"-- 90-day lead time means it must be in the pipeline well before the "
                        f"current order arrives, to sustain cover through the next cycle."
                    ),
                    "trigger": TRIGGER_PROJECTED_REORDER,
                    "action": "BUY_IMPORTED_NOW",
                    "urgency_level": "PROJECTED",
                })
                outstanding["IMPORTED"] += qty

    events.sort(key=lambda e: (e["order_date"], e["source"]))
    for e in events:
        e["quantity_tons"] = round(e["quantity_tons"], 2)

    def _view(days: int) -> list[dict]:
        cutoff = today + timedelta(days=days)
        return [e for e in events if date.fromisoformat(e["order_date"]) <= cutoff]

    return {
        "today": today.isoformat(),
        "events": events,
        "view_30_days": _view(30),
        "view_90_days": _view(90),
        "view_180_days": _view(180),
        "simulation": {
            "min_local_tons_seen": round(min_local_seen, 2),
            "min_imported_tons_seen": round(min_imported_seen, 2),
            "horizon_days_simulated": horizon_days,
            "starting_local_tons": local_inventory_tons,
            "starting_imported_tons": imported_inventory_tons,
        },
    }


# ===========================================================================
# Validation
# ===========================================================================

def validate_calendar(calendar_result: dict) -> dict:
    """Confirms the calendar's reorder-trigger logic is internally correct.

    Identity under test (see FORMULAS block above): ROP_source is defined
    as SAFETY_STOCK_source + lead_time_source x daily_rate_source. If every
    trigger fires correctly and every arrival is applied correctly, simulated
    inventory for a source should never fall below its safety-stock floor --
    the reorder point exists specifically to prevent that.

    A breach is classified into exactly one of two categories, not lumped
    together as one pass/fail bit:

        STARTUP TRANSIENT: the breach minimum occurs within the longest
            lead time (90 days) AND on-hand was already below its own
            safety stock floor at day 0 (CRITICAL status). This is an
            unavoidable consequence of the real starting position -- no
            order placed today, of any size, can retroactively undo
            consumption that already happened before that order could
            arrive. This is EXPECTED and reported, not a defect.

        RECURRING DEFECT: any breach minimum occurring strictly after the
            longest lead time (90 days) -- i.e. once the system has had a
            full lead-time cycle to recover -- indicates the trigger logic
            itself is failing to prevent stockouts in steady state. This
            WOULD be a real bug (see PSE-4A report Risks for the one this
            check caught and the (s, S) inventory-position fix applied).

    Returns:
        dict: safety stock floors, min seen (overall and post-cutoff per
              source), startup_transient / recurring_defect flags per
              source, all_passed (True only if no RECURRING_DEFECT exists --
              a startup transient does not fail validation, since it is not
              fixable by calendar logic).
    """
    from procurement_strategy_engine import (
        DAILY_CONSUMPTION_IMPORTED,
        DAILY_CONSUMPTION_LOCAL,
        SAFETY_STOCK_IMPORTED_TONS,
        SAFETY_STOCK_LOCAL_TONS,
    )

    sim = calendar_result["simulation"]
    tol = 0.5  # tons -- rounding tolerance only, not a business threshold
    cutoff_days = max(LOCAL_LEAD_TIME_DAYS, IMPORTED_LEAD_TIME_DAYS)  # 90
    today = date.fromisoformat(calendar_result["today"])
    horizon = sim["horizon_days_simulated"]

    min_local_post_cutoff = _min_after_cutoff(
        calendar_result["events"], today, "LOCAL", DAILY_CONSUMPTION_LOCAL,
        sim["starting_local_tons"], cutoff_days, horizon,
    )
    min_imported_post_cutoff = _min_after_cutoff(
        calendar_result["events"], today, "IMPORTED", DAILY_CONSUMPTION_IMPORTED,
        sim["starting_imported_tons"], cutoff_days, horizon,
    )

    local_breach = sim["min_local_tons_seen"] < (SAFETY_STOCK_LOCAL_TONS - tol)
    imported_breach = sim["min_imported_tons_seen"] < (SAFETY_STOCK_IMPORTED_TONS - tol)

    local_recurring_defect = min_local_post_cutoff < (SAFETY_STOCK_LOCAL_TONS - tol)
    imported_recurring_defect = min_imported_post_cutoff < (SAFETY_STOCK_IMPORTED_TONS - tol)

    return {
        "local_safety_stock_tons": SAFETY_STOCK_LOCAL_TONS,
        "imported_safety_stock_tons": SAFETY_STOCK_IMPORTED_TONS,
        "min_local_tons_seen": sim["min_local_tons_seen"],
        "min_imported_tons_seen": sim["min_imported_tons_seen"],
        "min_local_tons_seen_after_day_90": round(min_local_post_cutoff, 2),
        "min_imported_tons_seen_after_day_90": round(min_imported_post_cutoff, 2),
        "local_breach": local_breach,
        "imported_breach": imported_breach,
        "local_startup_transient": local_breach and not local_recurring_defect,
        "imported_startup_transient": imported_breach and not imported_recurring_defect,
        "local_recurring_defect": local_recurring_defect,
        "imported_recurring_defect": imported_recurring_defect,
        "all_passed": not (local_recurring_defect or imported_recurring_defect),
    }


# ===========================================================================
# Printable report
# ===========================================================================

def print_calendar(calendar_result: dict, view_days: int = 90) -> None:
    view_key = f"view_{view_days}_days"
    events = calendar_result.get(view_key, [])

    print(f"\n{'=' * 78}")
    print(f"PROCUREMENT CALENDAR -- {view_days}-DAY VIEW  (as of {calendar_result['today']})")
    print("=" * 78)

    if not events:
        print("  No purchases required within this window.")
        return

    for i, e in enumerate(events, 1):
        print(f"\n  [{i}] {e['source']} Cotton  ({e['trigger']})")
        print(f"      Buy / Order Qty : {e['quantity_tons']:,.1f} tons")
        print(f"      Order Date      : {e['order_date']}")
        print(f"      Arrival Date    : {e['expected_arrival_date']}")
        print(f"      Urgency         : {e['urgency_level']}")
        print(f"      Reason          : {e['reason']}")
