"""
procurement_scenario_engine.py
--------------------------------
PSE-5A -- Scenario Decision Engine (Rework v2).

Converts PSE-3B/3D/4A/4B outputs into a 3-layer procurement decision:

    Layer A -- BASE PROCUREMENT PLAN
        What would be ordered this cycle WITHOUT any extra scenario timing
        logic? Source identified: PSE-3D deficit, PSE-4B recovery order,
        PSE-4B periodic review, PSE-4A imported reorder.

    Layer B -- SCENARIO ADJUSTMENT
        What does the price scenario CHANGE relative to the base plan?
        Explicitly answers: add forward buy / defer / pull forward / no change
        and WHY (stock dominates, base already covers window, etc.).

    Layer C -- FINAL EXECUTION
        Final = Base + Adjustment, with mandatory / opportunistic / deferred
        component split.

PRICE DATA AUDIT:
    LIVE (implemented):
        ICE Cotton No. 2 spot  : market_inputs.fetch_cotton_price()
                                  Yahoo Finance CT=F (cents/lb -> USD/lb)
                                  Fallback: FRED PCOTTINDUSDM (monthly, lagged)
        USD/PKR rate           : market_inputs.fetch_fx_rate()

    FORECAST (requires Supabase):
        H+1 (1-month)          : prediction_records, horizon_months=1
                                  Used for LOCAL (10-day lead time; price moves
                                  relevant within T=30 review period)
        H+3 (3-month)          : prediction_records, horizon_months=3
                                  Used for IMPORTED (90-day lead time)
        Bounds                 : lower_bound / upper_bound for confidence

    ASSUMPTIONS / PLACEHOLDERS:
        1. LOCAL proxy: ICE Cotton No. 2 used for LOCAL Pakistan cotton.
           No domestic Pakistan price feed confirmed. Confidence: LOW.
        2. Carrying cost: SBP policy rate proxy (not Finance-confirmed).
        3. Forward buy = physical purchase advancement. No derivatives.
        4. Forecasts unavailable without SUPABASE_URL env var.
           Engine falls back to BALANCED (calendar-driven) in that case.

SCENARIOS (per source):
    STOCK_CRITICAL  : status CRITICAL or REORDER -- security overrides price.
    PRICE_RISING    : SAFE/WATCH, forecast price > current by >= 3%.
    PRICE_FALLING   : SAFE/WATCH with deferral window > 0, price falling.
    BALANCED        : no actionable signal -- follow consolidated calendar.

ADJUSTMENT TYPES:
    NO_CHANGE_STOCK_DOMINATES
        CRITICAL/REORDER + any price: mandatory buy only, safety takes priority.
    NO_CHANGE_BASE_ALREADY_COVERS_STRATEGIC_WINDOW
        REORDER + RISING: mandatory order already covers forward price window.
    NO_CHANGE_BASE_ALREADY_COVERS_STRATEGIC_WINDOW (BALANCED)
        SAFE + BALANCED: no price action; next PSE-4B event covers needs.
    PULL_FORWARD_NEXT_CONSOLIDATED_ORDER
        SAFE/WATCH + LOCAL + RISING: advance next T=30 periodic review order.
    ADD_FORWARD_BUY
        SAFE/WATCH + IMPORTED + RISING: advance one import cycle batch.
    DEFER_PART_OF_BASE_QTY
        Any source + FALLING + has mandatory: split mandatory now / non-mandatory later.
    DEFER_FULL_BASE_QTY
        SAFE/WATCH + FALLING: defer entire cycle qty to latest safe order date.

Does not modify any prior PSE module, dashboard, or UI file.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
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
    STATUS_SAFE,
    STATUS_WATCH,
)
from procurement_planning_engine import (
    LB_PER_METRIC_TON,
    PRICE_FALLING,
    PRICE_NEUTRAL,
    PRICE_RISING,
    compute_price_signal,
    estimate_savings,
)
from procurement_calendar_engine import (
    TRIGGER_DEFERRED,
    TRIGGER_IMMEDIATE,
    TRIGGER_PROJECTED_REORDER,
    TRIGGER_STAGGERED,
)
from procurement_consolidation_engine import (
    LOCAL_CONSOLIDATION_PERIOD_DAYS,
    S_T_LOCAL_CONSOLIDATED,
    TARGET_STOCK_IMPORTED,
    TRIGGER_CONSOLIDATION_RECOVERY,
    TRIGGER_CONSOLIDATED_PERIODIC,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STORAGE_CAPACITY_TONS = 45_000.0

# Opportunistic forward-buy quanta (PRICE_RISING + SAFE/WATCH only).
# LOCAL : one consolidated review period = T x daily_local = 30 x 49.5 = 1,485t
# IMPORTED: one import lead-time cycle = LT x daily_imported = 90 x 60.5 = 5,445t
FORWARD_BUY_LOCAL_TONS    = LOCAL_CONSOLIDATION_PERIOD_DAYS * DAILY_CONSUMPTION_LOCAL
FORWARD_BUY_IMPORTED_TONS = IMPORTED_LEAD_TIME_DAYS * DAILY_CONSUMPTION_IMPORTED

# Price-signal noise floor (inherited from PSE-3D)
SIGNAL_THRESHOLD_PCT = 3.0

# PSE-3D originated events (day-0 decisions)
_PSE3D_TRIGGERS = frozenset({TRIGGER_IMMEDIATE, TRIGGER_DEFERRED, TRIGGER_STAGGERED})

# A CONSOLIDATION_RECOVERY event is "imminent" (true emergency recovery phase) if it
# fires within this many days of today. Beyond this threshold the stock is effectively
# in a plannable steady-state for scenario timing purposes, and PULL_FORWARD logic
# becomes applicable instead of ADD_FORWARD_BUY.
_RECOVERY_IMMINENT_DAYS = LOCAL_CONSOLIDATION_PERIOD_DAYS   # 30 days

# ---------------------------------------------------------------------------
# Scenario labels
# ---------------------------------------------------------------------------
SCENARIO_STOCK_CRITICAL = "STOCK_CRITICAL"
SCENARIO_PRICE_RISING   = "PRICE_RISING"
SCENARIO_PRICE_FALLING  = "PRICE_FALLING"
SCENARIO_BALANCED       = "BALANCED"

# ---------------------------------------------------------------------------
# Base plan mode labels
# ---------------------------------------------------------------------------
BASE_LOCAL_RECOVERY     = "LOCAL_RECOVERY_FROM_PSE4B"
BASE_LOCAL_PERIODIC     = "LOCAL_PERIODIC_REVIEW_FROM_PSE4B"
BASE_LOCAL_IMMEDIATE    = "LOCAL_IMMEDIATE_FROM_PSE3D"
BASE_IMPORTED_REORDER   = "IMPORTED_REORDER_FROM_PSE4A"
BASE_IMPORTED_IMMEDIATE = "IMPORTED_IMMEDIATE_FROM_PSE3D"
BASE_NO_ORDER           = "NO_BASE_ORDER_THIS_CYCLE"

# ---------------------------------------------------------------------------
# Scenario adjustment type labels
# ---------------------------------------------------------------------------
ADJ_NO_CHANGE_STOCK_DOMINATES       = "NO_CHANGE_STOCK_DOMINATES"
ADJ_NO_CHANGE_COVERS_WINDOW         = "NO_CHANGE_BASE_ALREADY_COVERS_STRATEGIC_WINDOW"
ADJ_PULL_FORWARD_PERIODIC           = "PULL_FORWARD_NEXT_CONSOLIDATED_ORDER"
ADJ_ADD_FORWARD_BUY                 = "ADD_FORWARD_BUY"
ADJ_DEFER_PART                      = "DEFER_PART_OF_BASE_QTY"
ADJ_DEFER_FULL                      = "DEFER_FULL_BASE_QTY"
ADJ_NO_CHANGE_BALANCED              = "NO_CHANGE_FOLLOW_CALENDAR"

# ---------------------------------------------------------------------------
# Final action labels
# ---------------------------------------------------------------------------
ACTION_BUY_NOW     = "BUY_NOW"
ACTION_BUY_FORWARD = "BUY_FORWARD"
ACTION_BUY_SPLIT   = "BUY_SPLIT"
ACTION_DEFER       = "DEFER"
ACTION_HOLD        = "HOLD"

_RISK_RANK = {STATUS_CRITICAL: 0, STATUS_REORDER: 1, STATUS_WATCH: 2, STATUS_SAFE: 3}


# ===========================================================================
# Output dataclasses
# ===========================================================================

@dataclass
class BasePlanLayer:
    """Layer A: what existing PSE machinery says to do this cycle."""
    mode: str
    qty_now_tons: float
    qty_later_tons: float
    source_engine: str
    next_event_trigger: Optional[str]
    next_event_qty_tons: Optional[float]
    next_event_order_date: Optional[str]
    next_event_arrival_date: Optional[str]
    reasoning: str
    in_active_recovery: bool = False  # True iff CONSOLIDATION_RECOVERY fires within _RECOVERY_IMMINENT_DAYS


@dataclass
class ScenarioAdjustmentLayer:
    """Layer B: what the price scenario changes relative to the base plan."""
    adjustment_type: str
    adjustment_now_tons: float
    adjustment_later_tons: float
    strategic_window_days: int
    days_of_cover_after_base: Optional[float]
    already_covers_window: bool
    reasoning: str


@dataclass
class SourceDecision:
    """PSE-5A three-layer scenario decision for one source."""
    source: str

    # Layer A
    base_plan: BasePlanLayer

    # Layer B
    scenario: str
    price_signal: str
    price_delta_pct: Optional[float]
    price_confidence: Optional[str]
    current_price_usd_per_lb: Optional[float]
    forecast_price_usd_per_lb: Optional[float]
    scenario_adjustment: ScenarioAdjustmentLayer

    # Layer C
    final_action: str
    final_qty_now_tons: float
    final_qty_later_tons: float
    final_order_date_now: Optional[str]
    expected_arrival_now: Optional[str]
    final_order_window_later_start: Optional[str]
    final_order_window_later_end: Optional[str]
    mandatory_component_tons: float           # stock-protection deficit (cannot be deferred)
    base_non_mandatory_component_tons: float  # structural base (mix correction / calendar) not driven by deficit
    opportunistic_component_tons: float       # price-scenario forward buy on top of base
    deferred_component_tons: float            # qty intentionally pushed to later window

    stock_risk_overrides_price: bool
    expected_savings_usd: Optional[float]
    expected_savings_pkr: Optional[float]
    expected_avoided_cost_usd: Optional[float]
    expected_avoided_cost_pkr: Optional[float]

    decision_narrative: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioReport:
    """PSE-5A full portfolio scenario report."""
    run_date: str
    local: SourceDecision
    imported: SourceDecision
    portfolio_action: str
    portfolio_risk_level: str
    portfolio_reasoning: str
    price_inputs_used: dict
    assumptions: list

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Layer A: Base Plan Extraction from PSE-4B
# ===========================================================================

def _extract_base_plan(
    source: str,
    consolidation_result: dict,
    plan,
    status: str,
    inventory_tons: float,
    daily_rate: float,
    today: Optional[date] = None,
) -> BasePlanLayer:
    """Read PSE-4B consolidated calendar events to build the base plan layer.

    For LOCAL:
        - Identifies whether the engine is in RECOVERY mode (CONSOLIDATION_RECOVERY
          trigger) or PERIODIC_REVIEW steady state.
        - base_qty_now = sum of PSE-3D immediate events for LOCAL (deficit order).
    For IMPORTED:
        - Mode = IMPORTED_REORDER_FROM_PSE4A if a projected reorder exists.
        - base_qty_now = sum of PSE-3D immediate events for IMPORTED.
    """
    today = today or date.today()
    cons_events = consolidation_result["consolidated"]["events"]
    src_events   = [e for e in cons_events if e["source"] == source]

    immediate_events = [e for e in src_events if e["trigger"] in _PSE3D_TRIGGERS]
    base_qty_now = round(sum(e["quantity_tons"] for e in immediate_events), 2)

    if source == "LOCAL":
        recovery_events = [e for e in src_events if e["trigger"] == TRIGGER_CONSOLIDATION_RECOVERY]
        periodic_events = [e for e in src_events if e["trigger"] == TRIGGER_CONSOLIDATED_PERIODIC]

        # Recovery is "imminent" only if PSE-4B fires it within one review period.
        # Beyond that the stock is in effective steady-state for scenario timing.
        recovery_near = [
            e for e in recovery_events
            if (date.fromisoformat(e["order_date"]) - today).days <= _RECOVERY_IMMINENT_DAYS
        ]
        in_active_recovery = bool(recovery_near)

        if recovery_events:
            mode = BASE_LOCAL_RECOVERY
            next_ev = recovery_events[0]
            source_engine = "PSE-3D + PSE-4B"
            if in_active_recovery:
                base_reasoning = (
                    f"LOCAL is in two-phase RECOVERY mode (PSE-4B). "
                    f"Immediate mandatory order {base_qty_now:,.1f}t (PSE-3D deficit to ROP={LOCAL_ROP_TONS:,.0f}t) "
                    f"is placed today. PSE-4B fires an imminent RECOVERY order "
                    f"({next_ev['quantity_tons']:,.1f}t, order {next_ev['order_date']}, "
                    f"arrival {next_ev['expected_arrival_date']}) "
                    f"to lift position to S_T_local={S_T_LOCAL_CONSOLIDATED:,.1f}t, "
                    f"then switches to T={LOCAL_CONSOLIDATION_PERIOD_DAYS}d periodic review."
                )
            else:
                base_reasoning = (
                    f"LOCAL has an immediate base order of {base_qty_now:,.1f}t today (PSE-3D). "
                    f"PSE-4B schedules a CONSOLIDATION_RECOVERY order "
                    f"({next_ev['quantity_tons']:,.1f}t, order {next_ev['order_date']}, "
                    f"arrival {next_ev['expected_arrival_date']}) when position next reaches ROP. "
                    f"Recovery is {(date.fromisoformat(next_ev['order_date']) - today).days}d away — "
                    f"not imminent; stock is in effective steady-state for scenario timing."
                )
        elif periodic_events:
            mode = BASE_LOCAL_PERIODIC
            next_ev = periodic_events[0]
            in_active_recovery = False
            source_engine = "PSE-4B"
            base_reasoning = (
                f"LOCAL is in PERIODIC REVIEW steady state (PSE-4B, T={LOCAL_CONSOLIDATION_PERIOD_DAYS}d). "
                f"No immediate deficit order today. "
                f"Next consolidated order: {next_ev['quantity_tons']:,.1f}t "
                f"(order {next_ev['order_date']}, arrival {next_ev['expected_arrival_date']})."
            )
        elif immediate_events:
            mode = BASE_LOCAL_IMMEDIATE
            next_ev = None
            source_engine = "PSE-3D"
            base_reasoning = (
                f"LOCAL has an immediate mandatory order of {base_qty_now:,.1f}t from PSE-3D "
                f"(no PSE-4B recovery/periodic event detected in this horizon)."
            )
            in_active_recovery = False
        else:
            mode = BASE_NO_ORDER
            next_ev = None
            in_active_recovery = False
            source_engine = "N/A"
            base_reasoning = "No LOCAL order required this cycle per PSE-3D/4B."

    else:  # IMPORTED
        in_active_recovery = False   # not applicable for IMPORTED
        projected_events = [e for e in src_events if e["trigger"] == TRIGGER_PROJECTED_REORDER]

        if immediate_events or projected_events:
            if immediate_events:
                mode = BASE_IMPORTED_REORDER
                next_ev = projected_events[0] if projected_events else None
                source_engine = "PSE-3D + PSE-4A"
                base_reasoning = (
                    f"IMPORTED has an immediate reorder of {base_qty_now:,.1f}t from PSE-3D "
                    f"(deficit to ROP={IMPORTED_ROP_TONS:,.0f}t: "
                    f"{inventory_tons:,.1f}t on-hand, {IMPORTED_ROP_TONS - inventory_tons:,.1f}t shortfall). "
                    + (
                        f"PSE-4B simulation also schedules a follow-up projected reorder "
                        f"({next_ev['quantity_tons']:,.1f}t, order {next_ev['order_date']}) "
                        f"to bring total coverage toward S_imported={TARGET_STOCK_IMPORTED:,.1f}t."
                        if next_ev else
                        "PSE-4B follow-up projected event not within this horizon."
                    )
                )
            else:
                mode = BASE_IMPORTED_REORDER
                next_ev = projected_events[0]
                base_qty_now = 0.0
                source_engine = "PSE-4A"
                base_reasoning = (
                    f"No immediate IMPORTED order from PSE-3D. "
                    f"PSE-4B/4A projects a reorder of {next_ev['quantity_tons']:,.1f}t "
                    f"(order {next_ev['order_date']})."
                )
        else:
            mode = BASE_NO_ORDER
            next_ev = None
            source_engine = "N/A"
            base_reasoning = "No IMPORTED order required this cycle per PSE-3D/4A."

    return BasePlanLayer(
        mode=mode,
        qty_now_tons=base_qty_now,
        qty_later_tons=0.0,
        source_engine=source_engine,
        next_event_trigger=next_ev["trigger"] if next_ev else None,
        next_event_qty_tons=round(next_ev["quantity_tons"], 2) if next_ev else None,
        next_event_order_date=next_ev["order_date"] if next_ev else None,
        next_event_arrival_date=next_ev["expected_arrival_date"] if next_ev else None,
        reasoning=base_reasoning,
        in_active_recovery=in_active_recovery,
    )


# ===========================================================================
# Layer B: Scenario Adjustment
# ===========================================================================

def _days_of_cover_after_base(
    source: str,
    base_plan: BasePlanLayer,
    inventory_tons: float,
    daily_rate: float,
) -> float:
    """Position coverage (in days) after the base plan immediate order is placed today.

    Always computed as (current_inventory + base_qty_now) / daily_rate so the value
    reflects the actual position management can verify, not a projected future state.
    The next PSE-4B event (recovery or periodic) is reported separately in Layer A.
    """
    pos_after = inventory_tons + base_plan.qty_now_tons
    return round(pos_after / daily_rate, 1)


def _compute_scenario_adjustment(
    source: str,
    scenario: str,
    status: str,
    price_signal: str,
    base_plan: BasePlanLayer,
    inventory_tons: float,
    total_inventory_tons: float,
    daily_rate: float,
    rop_tons: float,
    plan,
    today: Optional[date] = None,
) -> ScenarioAdjustmentLayer:
    """Determine what the price scenario changes relative to the base plan.

    Decision tree:
        STOCK_CRITICAL (CRITICAL):  security overrides all price signals.
        STOCK_CRITICAL (REORDER) + mandatory_covers_window: no extra qty needed.
        PRICE_RISING + LOCAL + SAFE/WATCH + PERIODIC:  pull forward next order.
        PRICE_RISING + IMPORTED + SAFE/WATCH + no immediate: add one batch.
        PRICE_FALLING + SAFE/WATCH: defer all non-mandatory qty.
        BALANCED: no change, follow calendar.
    """
    today = today or date.today()
    strategic_window = (
        LOCAL_CONSOLIDATION_PERIOD_DAYS if source == "LOCAL"
        else IMPORTED_LEAD_TIME_DAYS
    )
    cover_after_base = _days_of_cover_after_base(source, base_plan, inventory_tons, daily_rate)
    already_covers = cover_after_base >= strategic_window

    # Storage headroom (caps any forward buy)
    headroom = max(0.0, MAX_STORAGE_CAPACITY_TONS - total_inventory_tons)

    # -----------------------------------------------------------------------
    # STOCK_CRITICAL: status CRITICAL -- absolute security override
    # -----------------------------------------------------------------------
    if scenario == SCENARIO_STOCK_CRITICAL and status == STATUS_CRITICAL:
        # No opportunistic qty allowed; price signal is noted but overridden
        reason = (
            f"{source} status is CRITICAL ({inventory_tons:,.1f}t on-hand, "
            f"below safety floor {(SAFETY_STOCK_LOCAL_TONS if source == 'LOCAL' else SAFETY_STOCK_IMPORTED_TONS):,.1f}t). "
            f"Stock security takes absolute priority -- no price-driven quantity change. "
        )
        if price_signal == PRICE_RISING:
            if source == "LOCAL" and base_plan.mode == BASE_LOCAL_RECOVERY:
                reason += (
                    f"PSE-4B RECOVERY order ({base_plan.next_event_qty_tons:,.1f}t, "
                    f"arrival {base_plan.next_event_arrival_date}) brings position to "
                    f"S_T_local={S_T_LOCAL_CONSOLIDATED:,.1f}t ({cover_after_base:.0f}-day cover), "
                    f"already exceeding the {strategic_window}-day strategic window. "
                    f"Adding {FORWARD_BUY_LOCAL_TONS:,.0f}t extra would push beyond S_T_local -- not appropriate in RECOVERY mode."
                )
            else:
                reason += (
                    f"Rising price (+{_price_delta_str(plan, source)}%) noted but deferred "
                    f"until inventory recovers to WATCH or SAFE status."
                )
        elif price_signal == PRICE_FALLING:
            reason += (
                f"Price is falling but mandatory order cannot be deferred -- "
                f"stock is CRITICAL and must be replenished immediately regardless of price direction."
            )
        return ScenarioAdjustmentLayer(
            adjustment_type=ADJ_NO_CHANGE_STOCK_DOMINATES,
            adjustment_now_tons=0.0,
            adjustment_later_tons=0.0,
            strategic_window_days=strategic_window,
            days_of_cover_after_base=cover_after_base,
            already_covers_window=already_covers,
            reasoning=reason.strip(),
        )

    # -----------------------------------------------------------------------
    # STOCK_CRITICAL: status REORDER -- security governs but check window coverage
    # -----------------------------------------------------------------------
    if scenario == SCENARIO_STOCK_CRITICAL and status == STATUS_REORDER:
        if price_signal == PRICE_RISING and already_covers:
            reason = (
                f"{source} mandatory order {base_plan.qty_now_tons:,.1f}t brings inventory "
                f"position to {cover_after_base:.0f} days of cover "
                f"(= {inventory_tons + base_plan.qty_now_tons:,.0f}t at {daily_rate:.1f}t/day). "
                f"Strategic forward-buy window = {strategic_window} days. "
                f"Base order already covers the full forward-buy window by "
                f"{cover_after_base - strategic_window:.0f} days -- "
                f"no additional opportunistic quantity is needed or warranted this cycle."
            )
            if base_plan.next_event_qty_tons:
                reason += (
                    f" PSE-4B will automatically schedule the next "
                    f"{source} order ({base_plan.next_event_qty_tons:,.1f}t, "
                    f"trigger {base_plan.next_event_trigger}) to maintain coverage."
                )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_NO_CHANGE_COVERS_WINDOW,
                adjustment_now_tons=0.0,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=True,
                reasoning=reason,
            )
        elif price_signal == PRICE_FALLING:
            reason = (
                f"{source} status is REORDER -- mandatory {base_plan.qty_now_tons:,.1f}t "
                f"cannot be deferred even with falling prices. "
                f"Full mandatory quantity must be ordered today."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_NO_CHANGE_STOCK_DOMINATES,
                adjustment_now_tons=0.0,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )
        elif price_signal == PRICE_RISING and not already_covers:
            # Defensive branch: theoretically unreachable (already_covers always True for REORDER)
            # because mandatory deficit brings position to exactly ROP = 115d (IMPORTED) or 65d
            # (LOCAL RECOVERY). Kept explicit to avoid wrong "no signal" message if math changes.
            reason = (
                f"{source} status REORDER. Mandatory order {base_plan.qty_now_tons:,.1f}t "
                f"brings only {cover_after_base:.0f}d of cover, below the {strategic_window}d "
                f"strategic window. Stock security takes priority -- no forward buy in REORDER status. "
                f"Re-evaluate once stock recovers to WATCH/SAFE."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_NO_CHANGE_STOCK_DOMINATES,
                adjustment_now_tons=0.0,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )
        else:
            # PRICE_NEUTRAL or PRICE_UNAVAILABLE
            reason = (
                f"{source} status is REORDER. Price signal is {price_signal} -- no adjustment. "
                f"Mandatory {base_plan.qty_now_tons:,.1f}t ordered per base plan."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_NO_CHANGE_STOCK_DOMINATES,
                adjustment_now_tons=0.0,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )

    # -----------------------------------------------------------------------
    # PRICE_RISING + SAFE / WATCH
    # -----------------------------------------------------------------------
    if scenario == SCENARIO_PRICE_RISING:
        if source == "LOCAL":
            # Pull forward logic applies when recovery is NOT imminent (stock is in
            # effective steady-state) and PSE-4B has a next consolidated event scheduled.
            # When recovery IS imminent (<= _RECOVERY_IMMINENT_DAYS), adding a forward
            # buy on top of the already-firing recovery would over-order; use fixed quantum.
            if not base_plan.in_active_recovery and base_plan.next_event_qty_tons:
                # Pull forward the next PSE-4B consolidated order (recovery or periodic)
                fwd_qty = round(min(base_plan.next_event_qty_tons, headroom), 2)
                reason = (
                    f"LOCAL price forecast rising and stock is not in active recovery phase "
                    f"(next PSE-4B consolidated event is {(date.fromisoformat(base_plan.next_event_order_date) - today).days if base_plan.next_event_order_date else '?'}d away, "
                    f"> {_RECOVERY_IMMINENT_DAYS}d threshold). "
                    f"Pulling forward PSE-4B {base_plan.next_event_trigger} order "
                    f"({base_plan.next_event_qty_tons:,.1f}t, scheduled {base_plan.next_event_order_date}) "
                    f"to today to lock in current lower price. "
                    f"Stock at {inventory_tons/daily_rate:.0f}d cover ({status}) -- safe to advance. "
                    f"Quantity: min(next_event_qty={base_plan.next_event_qty_tons:,.1f}t, "
                    f"headroom={headroom:,.0f}t) = {fwd_qty:,.1f}t."
                )
                return ScenarioAdjustmentLayer(
                    adjustment_type=ADJ_PULL_FORWARD_PERIODIC,
                    adjustment_now_tons=fwd_qty,
                    adjustment_later_tons=0.0,
                    strategic_window_days=strategic_window,
                    days_of_cover_after_base=cover_after_base,
                    already_covers_window=already_covers,
                    reasoning=reason,
                )
            else:
                # Active recovery imminent OR no next event — use fixed forward-buy quantum
                fwd_qty = round(min(FORWARD_BUY_LOCAL_TONS, headroom), 2)
                reason = (
                    f"LOCAL price forecast rising. Adding forward-buy tranche of {fwd_qty:,.1f}t "
                    f"(= T={LOCAL_CONSOLIDATION_PERIOD_DAYS}d x {DAILY_CONSUMPTION_LOCAL}t/day), "
                    f"capped at storage headroom {headroom:,.0f}t. "
                    f"Stock is {inventory_tons/daily_rate:.0f}d cover -- safe to advance."
                )
                return ScenarioAdjustmentLayer(
                    adjustment_type=ADJ_ADD_FORWARD_BUY,
                    adjustment_now_tons=fwd_qty,
                    adjustment_later_tons=0.0,
                    strategic_window_days=strategic_window,
                    days_of_cover_after_base=cover_after_base,
                    already_covers_window=already_covers,
                    reasoning=reason,
                )
        else:  # IMPORTED
            fwd_qty = round(min(FORWARD_BUY_IMPORTED_TONS, headroom), 2)
            reason = (
                f"IMPORTED price forecast rising. Adding one forward-buy tranche of {fwd_qty:,.1f}t "
                f"(= LT={IMPORTED_LEAD_TIME_DAYS}d x {DAILY_CONSUMPTION_IMPORTED}t/day imported), "
                f"capped at storage headroom {headroom:,.0f}t. "
                f"This advances the next natural import batch by one lead-time cycle, "
                f"purchasing {fwd_qty:,.1f}t at today's price before forecast increase materialises. "
                f"Stock at {inventory_tons/daily_rate:.0f}d cover -- safe to advance."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_ADD_FORWARD_BUY,
                adjustment_now_tons=fwd_qty,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )

    # -----------------------------------------------------------------------
    # PRICE_FALLING + SAFE / WATCH
    # -----------------------------------------------------------------------
    if scenario == SCENARIO_PRICE_FALLING:
        if base_plan.qty_now_tons <= 0:
            # No base order, nothing to defer
            reason = (
                f"No base order this cycle and price is falling. "
                f"Defer any potential purchase to latest safe date."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_DEFER_FULL,
                adjustment_now_tons=0.0,
                adjustment_later_tons=0.0,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )
        else:
            # Defer the entire base qty (no mandatory component in SAFE/WATCH)
            defer_qty = base_plan.qty_now_tons
            reason = (
                f"{source} is SAFE/WATCH with price falling. "
                f"Base order of {defer_qty:,.1f}t deferred to the latest safe order date. "
                f"Current stock ({inventory_tons:,.1f}t, {inventory_tons/daily_rate:.0f}d cover) "
                f"provides sufficient buffer to wait for a lower price."
            )
            return ScenarioAdjustmentLayer(
                adjustment_type=ADJ_DEFER_FULL,
                adjustment_now_tons=-defer_qty,
                adjustment_later_tons=defer_qty,
                strategic_window_days=strategic_window,
                days_of_cover_after_base=cover_after_base,
                already_covers_window=already_covers,
                reasoning=reason,
            )

    # -----------------------------------------------------------------------
    # BALANCED -- follow consolidated calendar, no scenario change
    # -----------------------------------------------------------------------
    reason = (
        f"No actionable price signal (price signal: {price_signal}). "
        f"Following PSE-4B consolidated calendar. "
        + (f"Base order {base_plan.qty_now_tons:,.1f}t executed as planned."
           if base_plan.qty_now_tons > 0
           else "No order required this cycle.")
    )
    return ScenarioAdjustmentLayer(
        adjustment_type=ADJ_NO_CHANGE_BALANCED,
        adjustment_now_tons=0.0,
        adjustment_later_tons=0.0,
        strategic_window_days=strategic_window,
        days_of_cover_after_base=cover_after_base,
        already_covers_window=already_covers,
        reasoning=reason,
    )


def _price_delta_str(plan, source: str) -> str:
    """Extract a price-delta percentage string from the plan for narrative use."""
    rec = plan.local_recommendation if source == "LOCAL" else plan.imported_recommendation
    savings = rec.get("savings")
    if savings:
        delta = savings.get("price_delta_pct")
        if delta is not None:
            return f"{delta:+.1f}"
    return "N/A"


# ===========================================================================
# Layer C: Final Execution
# ===========================================================================

def _compute_final(
    source: str,
    status: str,
    base_plan: BasePlanLayer,
    adjustment: ScenarioAdjustmentLayer,
    latest_safe_order_date: str,
    today: date,
    lead_time_days: int,
    rop_tons: float,
    safety_stock_tons: float,
    inventory_tons: float,
    deficit_from_plan: float,
) -> tuple[str, float, float, Optional[str], Optional[str], Optional[str], Optional[str],
           float, float, float, float, bool]:
    """Derive Layer C from Layers A + B.

    Component model (all four must sum correctly):
        mandatory          = structural stock deficit (cannot be deferred regardless of price)
        base_non_mandatory = rest of the base plan that is structural but not deficit-driven
                             (mix-correction orders, periodic reviews, etc.)
        opportunistic      = quantity added only because of price scenario (forward buy)
        deferred           = quantity intentionally pushed to later window (falling price)

    Invariant: mandatory + base_non_mandatory + opportunistic = final_now
               (when final_now >= 0; for DEFER scenarios final_now = 0 and
                deferred = qty pushed later)

    Returns:
        (action, qty_now, qty_later, order_date_now, arrival_now,
         window_later_start, window_later_end,
         mandatory_tons, base_non_mandatory_tons, opportunistic_tons, deferred_tons,
         stock_overrides)
    """
    base_now   = base_plan.qty_now_tons
    base_later = base_plan.qty_later_tons
    adj_now    = adjustment.adjustment_now_tons
    adj_later  = adjustment.adjustment_later_tons

    final_now   = round(base_now + adj_now, 2)
    final_later = round(base_later + adj_later, 2)
    if final_now < 0:
        final_now = 0.0

    # Mandatory = pure stock-protection deficit (ROP breach + mix correction minimum)
    if status in (STATUS_CRITICAL, STATUS_REORDER):
        mandatory = round(deficit_from_plan, 2)
    else:
        mandatory = 0.0

    # Base non-mandatory = structural base order above the deficit
    # (mix correction surplus, periodic review, calendar-driven orders)
    # For DEFER scenarios: base qty is pushed to later so effective base_now contribution = 0
    effective_base_now = round(max(0.0, base_now + min(0.0, adj_now)), 2)
    base_non_mandatory = round(max(0.0, effective_base_now - mandatory), 2)

    # Opportunistic = scenario-driven addition on top of base (forward buy / pull-forward)
    opportunistic = round(max(0.0, adj_now), 2)

    # Deferred = qty pushed to later window (negative adj_now reflected in adj_later)
    deferred = round(max(0.0, adj_later), 2)

    stock_overrides = (
        status in (STATUS_CRITICAL, STATUS_REORDER) and
        adjustment.adjustment_type in (ADJ_NO_CHANGE_STOCK_DOMINATES,)
    )

    # Determine action
    # BUY_FORWARD = purely opportunistic (no mandatory or structural base component now)
    # BUY_NOW     = has any structural component (mandatory or base_non_mandatory)
    if final_now > 0 and final_later > 0:
        action = ACTION_BUY_SPLIT
    elif final_now > 0 and opportunistic > 0 and mandatory == 0 and base_non_mandatory == 0:
        action = ACTION_BUY_FORWARD    # purely price-driven, no structural need today
    elif final_now > 0:
        action = ACTION_BUY_NOW        # has structural component (mandatory or mix correction)
    elif final_later > 0:
        action = ACTION_DEFER
    else:
        action = ACTION_HOLD

    # Timing
    order_date_now = today.isoformat() if final_now > 0 else None
    arrival_now    = (today + timedelta(days=lead_time_days)).isoformat() if final_now > 0 else None
    defer_start    = (today + timedelta(days=1)).isoformat() if final_later > 0 else None
    defer_end      = latest_safe_order_date if final_later > 0 else None

    return (action, final_now, final_later, order_date_now, arrival_now,
            defer_start, defer_end, mandatory, base_non_mandatory, opportunistic, deferred,
            stock_overrides)


# ===========================================================================
# Decision narrative builder
# ===========================================================================

def _build_narrative(
    source: str,
    status: str,
    inventory_tons: float,
    days_cover: float,
    base_plan: BasePlanLayer,
    scenario: str,
    price_signal: str,
    price_delta_pct: Optional[float],
    current_price: Optional[float],
    adjustment: ScenarioAdjustmentLayer,
    final_action: str,
    final_qty_now: float,
    final_qty_later: float,
    mandatory: float,
    base_non_mandatory: float,
    opportunistic: float,
    deferred: float,
    order_date_now: Optional[str],
    arrival_now: Optional[str],
    defer_end: Optional[str],
    avoided_cost_usd: Optional[float],
    savings_usd: Optional[float],
    pkr_rate: float,
) -> str:
    """Build a decision-grade narrative paragraph for this source."""
    lines = []

    # 1. Stock position
    status_label = {
        STATUS_CRITICAL: f"CRITICAL (below {(SAFETY_STOCK_LOCAL_TONS if source == 'LOCAL' else SAFETY_STOCK_IMPORTED_TONS):,.1f}t safety floor)",
        STATUS_REORDER: f"REORDER (below {(LOCAL_ROP_TONS if source == 'LOCAL' else IMPORTED_ROP_TONS):,.0f}t ROP)",
        STATUS_WATCH: "WATCH band",
        STATUS_SAFE: "SAFE",
    }
    lines.append(
        f"Stock: {inventory_tons:,.1f}t ({days_cover:.1f}d cover) -- {status_label.get(status, status)}."
    )

    # 2. Base plan
    lines.append(f"Base plan ({base_plan.mode}): order {base_plan.qty_now_tons:,.1f}t today per {base_plan.source_engine}.")
    if base_plan.next_event_trigger:
        lines.append(
            f"Next PSE engine event: {base_plan.next_event_trigger} "
            f"{base_plan.next_event_qty_tons:,.1f}t "
            f"(order {base_plan.next_event_order_date}, arrival {base_plan.next_event_arrival_date})."
        )

    # 3. Price signal
    if price_signal not in ("UNAVAILABLE", None):
        direction = "rising" if price_signal == PRICE_RISING else ("falling" if price_signal == PRICE_FALLING else "neutral")
        delta = f" ({price_delta_pct:+.1f}%)" if price_delta_pct is not None else ""
        price_str = f"Price forecast: {direction}{delta}"
        if current_price:
            price_str += f", current {current_price:.4f} USD/lb"
        lines.append(price_str + ".")
    else:
        lines.append("Price forecast: unavailable -- following calendar only.")

    # 4. Scenario adjustment
    lines.append(f"Scenario adjustment ({adjustment.adjustment_type}): {adjustment.reasoning}")
    lines.append(
        f"Base covers {adjustment.days_of_cover_after_base:.0f}d vs "
        f"{adjustment.strategic_window_days}d strategic window -- "
        f"{'YES, window covered' if adjustment.already_covers_window else 'NO, window NOT covered'}."
    )

    # 5. Final
    if final_action == ACTION_HOLD:
        lines.append("Final: HOLD -- no purchase this cycle.")
    elif final_action == ACTION_DEFER:
        lines.append(
            f"Final: DEFER {final_qty_later:,.1f}t to latest safe date {defer_end}."
        )
    elif final_action == ACTION_BUY_SPLIT:
        lines.append(
            f"Final: BUY_SPLIT -- {final_qty_now:,.1f}t now (order {order_date_now}, "
            f"arrival {arrival_now}) + DEFER {final_qty_later:,.1f}t to {defer_end}."
        )
    else:
        label = "BUY FORWARD" if final_action == ACTION_BUY_FORWARD else "BUY NOW"
        lines.append(
            f"Final: {label} {final_qty_now:,.1f}t "
            f"(order {order_date_now}, arrival {arrival_now})."
        )

    lines.append(
        f"Components: mandatory={mandatory:,.1f}t  base_structural={base_non_mandatory:,.1f}t  "
        f"opportunistic={opportunistic:,.1f}t  deferred={deferred:,.1f}t  "
        f"[total_now={mandatory+base_non_mandatory+opportunistic:,.1f}t]."
    )

    # 6. Cost impact
    if avoided_cost_usd and avoided_cost_usd > 0:
        pkr_m = avoided_cost_usd * pkr_rate / 1_000_000
        lines.append(
            f"Buying now avoids ~USD {avoided_cost_usd:,.0f} / PKR {pkr_m:.1f}M in expected cost increase."
        )
    if savings_usd and savings_usd > 0:
        pkr_m = savings_usd * pkr_rate / 1_000_000
        lines.append(
            f"Deferring saves ~USD {savings_usd:,.0f} / PKR {pkr_m:.1f}M at forecast lower price."
        )

    return " ".join(lines)


# ===========================================================================
# Cost intelligence helper
# ===========================================================================

def _compute_cost_impact(
    qty_tons: float,
    source: str,
    scenario: str,
    adjustment_type: str,
    current_price: Optional[float],
    forecast_price: Optional[float],
    pkr_rate: float,
    lead_time_days: int,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (savings_usd, savings_pkr, avoided_usd, avoided_pkr)."""
    if not current_price or not forecast_price or qty_tons <= 0:
        return None, None, None, None
    horizon = 30 if lead_time_days == LOCAL_LEAD_TIME_DAYS else 90
    s = estimate_savings(
        qty_tons=qty_tons,
        current_price_usd_per_lb=current_price,
        forecast_price_usd_per_lb=forecast_price,
        pkr_rate=pkr_rate,
        days_until_forecast_horizon=horizon,
        source=source,
    )
    if adjustment_type in (ADJ_PULL_FORWARD_PERIODIC, ADJ_ADD_FORWARD_BUY):
        return None, None, s["expected_cost_avoided_usd"], s["expected_cost_avoided_pkr"]
    if adjustment_type in (ADJ_DEFER_FULL, ADJ_DEFER_PART):
        return s["expected_savings_if_wait_usd"], s["expected_savings_if_wait_pkr"], None, None
    return None, None, None, None


# ===========================================================================
# Price signal resolution
# ===========================================================================

def _resolve_price_signal(
    plan,
    source: str,
    current_price: Optional[float],
    forecast_price: Optional[float],
    forecast_h_bounds: Optional[tuple[float, float]],
) -> tuple[str, Optional[float], Optional[str]]:
    """Get (price_signal, price_delta_pct, price_confidence) for this source."""
    rec_key = "local_recommendation" if source == "LOCAL" else "imported_recommendation"
    ps_from_plan = getattr(plan, rec_key).get("price_signal")

    if ps_from_plan is not None:
        price_signal = ps_from_plan
        price_delta_pct: Optional[float] = None
        price_confidence: Optional[str] = None
        if current_price and forecast_price:
            ps = compute_price_signal(
                current_price, forecast_price,
                forecast_h_bounds[0] if forecast_h_bounds else None,
                forecast_h_bounds[1] if forecast_h_bounds else None,
                signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
            )
            price_delta_pct = ps["price_delta_pct"]
            price_confidence = ps["confidence"]
    elif current_price and forecast_price:
        ps = compute_price_signal(
            current_price, forecast_price,
            forecast_h_bounds[0] if forecast_h_bounds else None,
            forecast_h_bounds[1] if forecast_h_bounds else None,
            signal_threshold_pct=SIGNAL_THRESHOLD_PCT,
        )
        price_signal = ps["signal"]
        price_delta_pct = ps["price_delta_pct"]
        price_confidence = ps["confidence"]
    else:
        price_signal = "UNAVAILABLE"
        price_delta_pct = None
        price_confidence = None

    return price_signal, price_delta_pct, price_confidence


# ===========================================================================
# Scenario detection
# ===========================================================================

def _detect_scenario(
    status: str,
    price_signal: str,
    inventory_tons: float,
    rop_tons: float,
    daily_rate: float,
) -> str:
    if status in (STATUS_CRITICAL, STATUS_REORDER):
        return SCENARIO_STOCK_CRITICAL
    deferral_days = max(0.0, (inventory_tons - rop_tons) / daily_rate) if daily_rate else 0.0
    if price_signal == PRICE_RISING:
        return SCENARIO_PRICE_RISING
    if price_signal == PRICE_FALLING and deferral_days > 0:
        return SCENARIO_PRICE_FALLING
    return SCENARIO_BALANCED


# ===========================================================================
# Per-source decision builder
# ===========================================================================

def _build_source_decision(
    source: str,
    status: str,
    inventory_tons: float,
    total_inventory_tons: float,
    days_cover: float,
    rop_tons: float,
    safety_stock_tons: float,
    lead_time_days: int,
    daily_rate: float,
    deficit_from_plan: float,
    latest_safe_order_date: str,
    consolidation_result: dict,
    plan,
    today: date,
    current_price: Optional[float],
    forecast_price: Optional[float],
    forecast_h_bounds: Optional[tuple[float, float]],
    pkr_rate: float,
) -> SourceDecision:

    # ---- Layer A: Base plan from PSE-4B ----
    base_plan = _extract_base_plan(
        source, consolidation_result, plan, status, inventory_tons, daily_rate, today=today
    )

    # ---- Price signal ----
    price_signal, price_delta_pct, price_confidence = _resolve_price_signal(
        plan, source, current_price, forecast_price, forecast_h_bounds
    )

    # ---- Scenario ----
    scenario = _detect_scenario(status, price_signal, inventory_tons, rop_tons, daily_rate)

    # ---- Layer B: Scenario adjustment ----
    adjustment = _compute_scenario_adjustment(
        source=source,
        scenario=scenario,
        status=status,
        price_signal=price_signal,
        base_plan=base_plan,
        inventory_tons=inventory_tons,
        total_inventory_tons=total_inventory_tons,
        daily_rate=daily_rate,
        rop_tons=rop_tons,
        plan=plan,
        today=today,
    )

    # ---- Layer C: Final execution ----
    (action, final_now, final_later, order_date_now, arrival_now,
     defer_start, defer_end, mandatory, base_non_mandatory, opportunistic, deferred,
     stock_overrides) = _compute_final(
        source=source,
        status=status,
        base_plan=base_plan,
        adjustment=adjustment,
        latest_safe_order_date=latest_safe_order_date,
        today=today,
        lead_time_days=lead_time_days,
        rop_tons=rop_tons,
        safety_stock_tons=safety_stock_tons,
        inventory_tons=inventory_tons,
        deficit_from_plan=deficit_from_plan,
    )

    # ---- Cost intelligence ----
    cost_qty = opportunistic if opportunistic > 0 else (deferred if deferred > 0 else 0)
    savings_usd, savings_pkr, avoided_usd, avoided_pkr = _compute_cost_impact(
        qty_tons=cost_qty,
        source=source,
        scenario=scenario,
        adjustment_type=adjustment.adjustment_type,
        current_price=current_price,
        forecast_price=forecast_price,
        pkr_rate=pkr_rate,
        lead_time_days=lead_time_days,
    )

    # ---- Narrative ----
    narrative = _build_narrative(
        source=source, status=status, inventory_tons=inventory_tons,
        days_cover=days_cover, base_plan=base_plan,
        scenario=scenario, price_signal=price_signal, price_delta_pct=price_delta_pct,
        current_price=current_price, adjustment=adjustment,
        final_action=action, final_qty_now=final_now, final_qty_later=final_later,
        mandatory=mandatory, base_non_mandatory=base_non_mandatory,
        opportunistic=opportunistic, deferred=deferred,
        order_date_now=order_date_now, arrival_now=arrival_now, defer_end=defer_end,
        avoided_cost_usd=avoided_usd, savings_usd=savings_usd, pkr_rate=pkr_rate,
    )

    return SourceDecision(
        source=source,
        base_plan=base_plan,
        scenario=scenario,
        price_signal=price_signal,
        price_delta_pct=price_delta_pct,
        price_confidence=price_confidence,
        current_price_usd_per_lb=current_price,
        forecast_price_usd_per_lb=forecast_price,
        scenario_adjustment=adjustment,
        final_action=action,
        final_qty_now_tons=final_now,
        final_qty_later_tons=final_later,
        final_order_date_now=order_date_now,
        expected_arrival_now=arrival_now,
        final_order_window_later_start=defer_start,
        final_order_window_later_end=defer_end,
        mandatory_component_tons=mandatory,
        base_non_mandatory_component_tons=base_non_mandatory,
        opportunistic_component_tons=opportunistic,
        deferred_component_tons=deferred,
        stock_risk_overrides_price=stock_overrides,
        expected_savings_usd=savings_usd,
        expected_savings_pkr=savings_pkr,
        expected_avoided_cost_usd=avoided_usd,
        expected_avoided_cost_pkr=avoided_pkr,
        decision_narrative=narrative,
    )


# ===========================================================================
# PUBLIC API
# ===========================================================================

def compute_scenario_decision(
    strategy_output,
    procurement_plan,
    calendar_result: dict,
    consolidation_result: dict,
    current_price_usd_per_lb: Optional[float] = None,
    forecast_h1_usd_per_lb: Optional[float] = None,
    forecast_h3_usd_per_lb: Optional[float] = None,
    forecast_h1_bounds: Optional[tuple[float, float]] = None,
    forecast_h3_bounds: Optional[tuple[float, float]] = None,
    pkr_rate: float = 281.0,
    today: Optional[date] = None,
) -> ScenarioReport:
    """PSE-5A entry point. Produce 3-layer scenario decisions from all prior engine outputs.

    Args:
        strategy_output      : StrategyOutputV2 from PSE-3B.
        procurement_plan     : ProcurementPlan from PSE-3D.
        calendar_result      : dict from PSE-4A compute_procurement_calendar().
        consolidation_result : dict from PSE-4B compute_order_consolidation().
        current_price_usd_per_lb : ICE Cotton No. 2 live price (USD/lb).
        forecast_h1_usd_per_lb   : 1-month forecast (LOCAL horizon).
        forecast_h3_usd_per_lb   : 3-month forecast (IMPORTED horizon).
        forecast_h1/h3_bounds    : (lower, upper) forecast bounds.
        pkr_rate             : USD/PKR exchange rate.
        today                : defaults to date.today().
    """
    today = today or date.today()
    so    = strategy_output
    plan  = procurement_plan
    req   = plan.requirement

    local = _build_source_decision(
        source="LOCAL",
        status=so.local_status,
        inventory_tons=so.local_inventory_tons,
        total_inventory_tons=so.total_inventory_tons,
        days_cover=so.local_days_cover,
        rop_tons=float(LOCAL_ROP_TONS),
        safety_stock_tons=SAFETY_STOCK_LOCAL_TONS,
        lead_time_days=LOCAL_LEAD_TIME_DAYS,
        daily_rate=DAILY_CONSUMPTION_LOCAL,
        deficit_from_plan=req["deficit_local_tons"],
        latest_safe_order_date=plan.local_recommendation["latest_safe_order_date"],
        consolidation_result=consolidation_result,
        plan=plan,
        today=today,
        current_price=current_price_usd_per_lb,
        forecast_price=forecast_h1_usd_per_lb,
        forecast_h_bounds=forecast_h1_bounds,
        pkr_rate=pkr_rate,
    )

    imported = _build_source_decision(
        source="IMPORTED",
        status=so.imported_status,
        inventory_tons=so.imported_inventory_tons,
        total_inventory_tons=so.total_inventory_tons,
        days_cover=so.imported_days_cover,
        rop_tons=float(IMPORTED_ROP_TONS),
        safety_stock_tons=SAFETY_STOCK_IMPORTED_TONS,
        lead_time_days=IMPORTED_LEAD_TIME_DAYS,
        daily_rate=DAILY_CONSUMPTION_IMPORTED,
        deficit_from_plan=req["deficit_imported_tons"],
        latest_safe_order_date=plan.imported_recommendation["latest_safe_order_date"],
        consolidation_result=consolidation_result,
        plan=plan,
        today=today,
        current_price=current_price_usd_per_lb,
        forecast_price=forecast_h3_usd_per_lb,
        forecast_h_bounds=forecast_h3_bounds,
        pkr_rate=pkr_rate,
    )

    # Portfolio-level summary
    risk_label_map = {
        STATUS_CRITICAL: "CRITICAL", STATUS_REORDER: "HIGH",
        STATUS_WATCH: "MEDIUM", STATUS_SAFE: "LOW",
    }
    local_risk    = _RISK_RANK.get(so.local_status, 9)
    imported_risk = _RISK_RANK.get(so.imported_status, 9)
    worst_status  = so.local_status if local_risk <= imported_risk else so.imported_status
    portfolio_risk = risk_label_map.get(worst_status, "UNKNOWN")

    _priority = {ACTION_BUY_NOW: 1, ACTION_BUY_FORWARD: 2, ACTION_BUY_SPLIT: 3,
                 ACTION_DEFER: 4, ACTION_HOLD: 5}
    lp = _priority.get(local.final_action, 9)
    ip = _priority.get(imported.final_action, 9)
    if local.final_action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD) and \
       imported.final_action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD):
        portfolio_action = "BUY_MIXED_NOW"
    else:
        portfolio_action = local.final_action if lp <= ip else imported.final_action

    portfolio_reasoning = (
        f"LOCAL: {local.scenario} | {local.final_action} {local.final_qty_now_tons:,.1f}t now"
        + (f" + defer {local.final_qty_later_tons:,.1f}t" if local.final_qty_later_tons > 0 else "")
        + f" (mandatory={local.mandatory_component_tons:,.1f}t, "
        f"opportunistic={local.opportunistic_component_tons:,.1f}t). "
        f"IMPORTED: {imported.scenario} | {imported.final_action} {imported.final_qty_now_tons:,.1f}t now"
        + (f" + defer {imported.final_qty_later_tons:,.1f}t" if imported.final_qty_later_tons > 0 else "")
        + f" (mandatory={imported.mandatory_component_tons:,.1f}t, "
        f"opportunistic={imported.opportunistic_component_tons:,.1f}t)."
    )

    price_inputs_used = {
        "current_price_usd_per_lb": current_price_usd_per_lb,
        "forecast_h1_usd_per_lb": forecast_h1_usd_per_lb,
        "forecast_h3_usd_per_lb": forecast_h3_usd_per_lb,
        "forecast_h1_bounds": forecast_h1_bounds,
        "forecast_h3_bounds": forecast_h3_bounds,
        "pkr_rate": pkr_rate,
        "local_signal_source": (
            "PSE-3D" if plan.local_recommendation.get("price_signal") else "PSE-5A"
        ),
        "imported_signal_source": (
            "PSE-3D" if plan.imported_recommendation.get("price_signal") else "PSE-5A"
        ),
    }

    assumptions = [
        "LOCAL price proxy: ICE Cotton No. 2 (no Pakistan domestic price feed). Confidence: LOW.",
        "Carrying cost: SBP policy rate proxy (not Finance-confirmed).",
        "Forward buy = physical purchase advancement. No financial derivatives.",
        "Supabase ML forecasts require SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars.",
        "Mix 45/55 target inherited from PSE-2 confirmed business rules.",
    ]

    return ScenarioReport(
        run_date=today.isoformat(),
        local=local,
        imported=imported,
        portfolio_action=portfolio_action,
        portfolio_risk_level=portfolio_risk,
        portfolio_reasoning=portfolio_reasoning,
        price_inputs_used=price_inputs_used,
        assumptions=assumptions,
    )


# ===========================================================================
# PRICE INPUT FETCHER
# ===========================================================================

def fetch_price_inputs(timeout: int = 15) -> dict:
    """Fetch live cotton price + USD/PKR + Supabase ML forecasts."""
    from market_inputs import fetch_cotton_price, fetch_fx_rate

    result: dict = {
        "current_price_usd_per_lb": None,
        "forecast_h1_usd_per_lb": None,
        "forecast_h3_usd_per_lb": None,
        "forecast_h1_bounds": None,
        "forecast_h3_bounds": None,
        "pkr_rate": 281.0,
        "status": "UNAVAILABLE",
        "notes": [],
    }

    try:
        rec = fetch_cotton_price(timeout=timeout)
        if rec.status == "OK" and rec.metric_value:
            result["current_price_usd_per_lb"] = rec.metric_value
            result["notes"].append(f"Cotton: {rec.metric_value:.4f} USD/lb ({rec.source})")
        else:
            result["notes"].append(f"Cotton price fetch failed: {rec.error}")
    except Exception as exc:
        result["notes"].append(f"Cotton price error: {exc}")

    try:
        rec = fetch_fx_rate(timeout=timeout)
        if rec.status == "OK" and rec.metric_value:
            result["pkr_rate"] = rec.metric_value
            result["notes"].append(f"USD/PKR: {rec.metric_value:.2f} ({rec.source})")
        else:
            result["notes"].append(f"FX rate fetch failed: {rec.error}")
    except Exception as exc:
        result["notes"].append(f"FX rate error: {exc}")

    # Supabase forecasts
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    )
    if not supabase_url or not supabase_key:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        if tomllib:
            secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                try:
                    with open(secrets_path, "rb") as f:
                        secrets = tomllib.load(f)
                    supabase_url = supabase_url or secrets.get("SUPABASE_URL")
                    supabase_key = supabase_key or (
                        secrets.get("SUPABASE_SERVICE_ROLE_KEY") or secrets.get("SUPABASE_ANON_KEY")
                    )
                except Exception:
                    pass

    if supabase_url and supabase_key:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from signals.price_signals import compute_forecast_signal
            supabase_cfg = (supabase_url, supabase_key)
            current_price = result["current_price_usd_per_lb"]
            for horizon, key_p, key_b in (
                (1, "forecast_h1_usd_per_lb", "forecast_h1_bounds"),
                (3, "forecast_h3_usd_per_lb", "forecast_h3_bounds"),
            ):
                sig = compute_forecast_signal("cotton_usd", horizon, supabase_cfg,
                                              current_value=current_price)
                if sig:
                    result[key_p] = sig["forecast_value"]
                    result["notes"].append(
                        f"H+{horizon}: {sig['forecast_value']:.4f} USD/lb "
                        f"({sig['forecast_change_pct']:+.2f}%, {sig['model_name']})"
                    )
                else:
                    result["notes"].append(f"H+{horizon} forecast unavailable from Supabase")
        except Exception as exc:
            result["notes"].append(f"Supabase forecast error: {exc}")
    else:
        result["notes"].append(
            "Supabase credentials not found -- ML forecasts unavailable. "
            "Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars to enable."
        )

    has_price = bool(result["current_price_usd_per_lb"])
    has_h1    = bool(result["forecast_h1_usd_per_lb"])
    has_h3    = bool(result["forecast_h3_usd_per_lb"])
    result["status"] = ("LIVE" if (has_price and has_h1 and has_h3)
                        else ("PARTIAL" if (has_price or has_h1 or has_h3)
                              else "UNAVAILABLE"))
    return result


# ===========================================================================
# END-TO-END ORCHESTRATOR
# ===========================================================================

def run_pse5a(
    workbook_path: Optional[str] = None,
    input_path: Optional[str] = None,
    live_prices: bool = True,
    current_price_usd_per_lb: Optional[float] = None,
    forecast_h1_usd_per_lb: Optional[float] = None,
    forecast_h3_usd_per_lb: Optional[float] = None,
    forecast_h1_bounds: Optional[tuple[float, float]] = None,
    forecast_h3_bounds: Optional[tuple[float, float]] = None,
    pkr_rate: Optional[float] = None,
    horizon_days: int = 180,
    today: Optional[date] = None,
    _orch: Optional[dict] = None,
) -> ScenarioReport:
    """Full PSE-5A pipeline: inventory -> PSE-3B/3C -> PSE-3D -> PSE-4A -> PSE-4B -> PSE-5A.

    Pass _orch to reuse an already-loaded run_orchestration() result and skip
    the workbook read.  Only callers that have already called run_orchestration()
    should set this — external / CLI callers leave it None.
    """
    from procurement_planning_engine import run_pse3d
    from procurement_calendar_engine import compute_procurement_calendar
    from procurement_consolidation_engine import compute_order_consolidation

    today = today or date.today()

    if live_prices and current_price_usd_per_lb is None:
        pi = fetch_price_inputs()
        price_kwargs = {k: pi[k] for k in (
            "current_price_usd_per_lb", "forecast_h1_usd_per_lb", "forecast_h3_usd_per_lb",
            "forecast_h1_bounds", "forecast_h3_bounds", "pkr_rate",
        )}
    else:
        price_kwargs = {
            "current_price_usd_per_lb": current_price_usd_per_lb,
            "forecast_h1_usd_per_lb": forecast_h1_usd_per_lb,
            "forecast_h3_usd_per_lb": forecast_h3_usd_per_lb,
            "forecast_h1_bounds": forecast_h1_bounds,
            "forecast_h3_bounds": forecast_h3_bounds,
            "pkr_rate": pkr_rate or 281.0,
        }

    if _orch is not None:
        orch = _orch
    else:
        from procurement_orchestrator import run_orchestration
        orch = run_orchestration(input_path=input_path, workbook_path=workbook_path)
    so    = orch["strategy_output"]

    plan = run_pse3d(
        local_inventory_tons=so.local_inventory_tons,
        imported_inventory_tons=so.imported_inventory_tons,
        local_status=so.local_status,
        imported_status=so.imported_status,
        total_inventory_tons=so.total_inventory_tons,
        today=today,
        **{k: v for k, v in price_kwargs.items()
           if k in ("current_price_usd_per_lb", "forecast_h1_usd_per_lb",
                    "forecast_h3_usd_per_lb", "forecast_h1_bounds",
                    "forecast_h3_bounds", "pkr_rate")},
    )

    calendar = compute_procurement_calendar(
        local_inventory_tons=so.local_inventory_tons,
        imported_inventory_tons=so.imported_inventory_tons,
        local_status=so.local_status,
        imported_status=so.imported_status,
        total_inventory_tons=so.total_inventory_tons,
        horizon_days=horizon_days,
        today=today,
    )

    consolidation = compute_order_consolidation(calendar_result=calendar)

    return compute_scenario_decision(
        strategy_output=so,
        procurement_plan=plan,
        calendar_result=calendar,
        consolidation_result=consolidation,
        today=today,
        **price_kwargs,
    )


# ===========================================================================
# REPORT PRINTER
# ===========================================================================

def _sep(char: str = "-", width: int = 78) -> str:
    return char * width


def print_scenario_report(report: ScenarioReport) -> None:
    """Print PSE-5A 3-layer scenario decision in management action-sheet format."""
    print(_sep("="))
    print("PSE-5A SCENARIO DECISION ENGINE  --  3-LAYER PROCUREMENT ACTION SHEET")
    print(f"  Run date        : {report.run_date}")
    print(f"  Portfolio risk  : {report.portfolio_risk_level}")
    print(f"  Portfolio action: {report.portfolio_action}")
    print(_sep("="))

    for dec in (report.local, report.imported):
        print(f"\n{_sep('=')}")
        print(f"  SOURCE: {dec.source}")
        print(_sep("="))

        # ---- Layer A ----
        print(f"\n  [LAYER A -- BASE PROCUREMENT PLAN]")
        print(f"  Base plan mode    : {dec.base_plan.mode}")
        print(f"  Source engine     : {dec.base_plan.source_engine}")
        print(f"  Base qty NOW      : {dec.base_plan.qty_now_tons:>10,.1f} t")
        print(f"  Base qty LATER    : {dec.base_plan.qty_later_tons:>10,.1f} t")
        if dec.base_plan.next_event_trigger:
            print(f"  Next PSE event    : {dec.base_plan.next_event_trigger}")
            print(f"    Qty             : {dec.base_plan.next_event_qty_tons:>10,.1f} t")
            print(f"    Order date      : {dec.base_plan.next_event_order_date}")
            print(f"    Arrival date    : {dec.base_plan.next_event_arrival_date}")
        print(f"\n  Base reasoning:")
        for line in dec.base_plan.reasoning.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")

        # ---- Layer B ----
        print(f"\n  [LAYER B -- SCENARIO ADJUSTMENT]")
        print(f"  Scenario          : {dec.scenario}")
        print(f"  Price signal      : {dec.price_signal}", end="")
        if dec.price_delta_pct is not None:
            print(f"  ({dec.price_delta_pct:+.2f}%)", end="")
        print()
        if dec.current_price_usd_per_lb:
            print(f"  Current price     : {dec.current_price_usd_per_lb:.4f} USD/lb")
        if dec.forecast_price_usd_per_lb:
            print(f"  Forecast price    : {dec.forecast_price_usd_per_lb:.4f} USD/lb")
        if dec.price_confidence:
            print(f"  Confidence        : {dec.price_confidence}")
        print()
        print(f"  Adjustment type   : {dec.scenario_adjustment.adjustment_type}")
        print(f"  Adj qty NOW       : {dec.scenario_adjustment.adjustment_now_tons:>+10,.1f} t")
        print(f"  Adj qty LATER     : {dec.scenario_adjustment.adjustment_later_tons:>+10,.1f} t")
        print(f"  Strategic window  : {dec.scenario_adjustment.strategic_window_days} days")
        print(f"  Cover after base  : {dec.scenario_adjustment.days_of_cover_after_base:.0f} days  "
              f"({'COVERS' if dec.scenario_adjustment.already_covers_window else 'DOES NOT COVER'} "
              f"strategic window)")
        print(f"\n  Adjustment reasoning:")
        for line in dec.scenario_adjustment.reasoning.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")

        # ---- Layer C ----
        print(f"\n  [LAYER C -- FINAL EXECUTION]")
        print(f"  Final action      : {dec.final_action}")
        print(f"  Final qty NOW     : {dec.final_qty_now_tons:>10,.1f} t", end="")
        if dec.final_order_date_now:
            print(f"  (order {dec.final_order_date_now}, arrival {dec.expected_arrival_now})", end="")
        print()
        print(f"  Final qty LATER   : {dec.final_qty_later_tons:>10,.1f} t", end="")
        if dec.final_qty_later_tons > 0:
            print(f"  (window {dec.final_order_window_later_start} to {dec.final_order_window_later_end})", end="")
        print()
        print(f"\n  --- Component split ---")
        print(f"  Mandatory         : {dec.mandatory_component_tons:>10,.1f} t  (stock deficit, non-negotiable)")
        print(f"  Opportunistic     : {dec.opportunistic_component_tons:>10,.1f} t  (price-driven forward buy)")
        print(f"  Deferred          : {dec.deferred_component_tons:>10,.1f} t  (pushed to later window)")

        if dec.expected_avoided_cost_usd and dec.expected_avoided_cost_usd > 0:
            pkr_m = dec.expected_avoided_cost_usd * report.price_inputs_used["pkr_rate"] / 1_000_000
            print(f"\n  Cost avoided (buy now vs later) : "
                  f"USD {dec.expected_avoided_cost_usd:>12,.0f} / PKR {pkr_m:.1f}M")
        if dec.expected_savings_usd and dec.expected_savings_usd > 0:
            pkr_m = dec.expected_savings_usd * report.price_inputs_used["pkr_rate"] / 1_000_000
            print(f"\n  Savings (defer vs buy now)      : "
                  f"USD {dec.expected_savings_usd:>12,.0f} / PKR {pkr_m:.1f}M")
        if dec.stock_risk_overrides_price:
            print(f"\n  [!] STOCK RISK OVERRIDES PRICE SIGNAL")

        print(f"\n  Full decision narrative:")
        for sentence in dec.decision_narrative.split(". "):
            s = sentence.strip().rstrip(".")
            if s:
                print(f"    {s}.")

    print(f"\n{_sep('=')}")
    print("PORTFOLIO SUMMARY")
    for sentence in report.portfolio_reasoning.split(". "):
        s = sentence.strip().rstrip(".")
        if s:
            print(f"  {s}.")

    print(f"\n{_sep('=')}")
    print("PRICE INPUTS USED")
    pi = report.price_inputs_used
    print(f"  ICE Cotton (spot) : {pi.get('current_price_usd_per_lb') or 'not available'}")
    print(f"  H+1 forecast      : {pi.get('forecast_h1_usd_per_lb') or 'not available'}  (LOCAL, 1-month)")
    print(f"  H+3 forecast      : {pi.get('forecast_h3_usd_per_lb') or 'not available'}  (IMPORTED, 3-month)")
    print(f"  USD/PKR rate      : {pi.get('pkr_rate', 281.0):.2f}")
    print(f"  Signal source     : LOCAL={pi.get('local_signal_source')}  "
          f"IMPORTED={pi.get('imported_signal_source')}")

    print(f"\n{_sep('=')}")
    print("ASSUMPTIONS")
    for a in report.assumptions:
        print(f"  - {a}")
    print(_sep("="))


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PSE-5A: Scenario Decision Engine (3-layer) -- procurement action sheet"
    )
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx")
    parser.add_argument("--input",    default=None, help="Raw Oracle export (.xlsx)")
    parser.add_argument("--price",    type=float, default=None,
                        help="Current ICE Cotton price (USD/lb). Auto-fetched if omitted.")
    parser.add_argument("--h1",       type=float, default=None,
                        help="1-month forecast (USD/lb). Fetched from Supabase if omitted.")
    parser.add_argument("--h3",       type=float, default=None,
                        help="3-month forecast (USD/lb). Fetched from Supabase if omitted.")
    parser.add_argument("--pkr",      type=float, default=None, help="USD/PKR rate.")
    parser.add_argument("--no-live",  action="store_true",
                        help="Skip live price fetch.")
    parser.add_argument("--horizon",  type=int, default=180, help="Calendar horizon days.")
    args = parser.parse_args()

    report = run_pse5a(
        workbook_path=args.workbook,
        input_path=args.input,
        live_prices=not args.no_live,
        current_price_usd_per_lb=args.price,
        forecast_h1_usd_per_lb=args.h1,
        forecast_h3_usd_per_lb=args.h3,
        pkr_rate=args.pkr,
        horizon_days=args.horizon,
    )
    print_scenario_report(report)
    sys.exit(0)
