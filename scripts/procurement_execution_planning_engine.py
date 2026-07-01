"""
procurement_execution_planning_engine.py
-------------------------------------------
PSE-3.6 -- Procurement Execution Planning Layer.

This is the final intelligence layer of the Procurement Decision Engine:

    Position -> Target -> Gap -> Portfolio Optimization -+
                                                           +--> Strategic
    Market Intelligence -------------------------------- +    Assessment
                                                                   |
                                                                   v
                                          Procurement Execution Planning
                                                (this module)
                                                   |
                                                   v
                                Oracle / humans execute (NOT this module)

This module answers exactly one question -- "What procurement plan
should the organisation follow?" -- including how much, in what
sequence, within which planning window, and why. It is PLANNING, not
EXECUTION: no purchase orders, no Oracle calls, no ERP writes, no
inventory updates, no calendar dates, no executive reports.

Inputs:
    Consumes ONLY an already-built PositionSnapshot (PSE-3.0),
    PortfolioOptimizationSnapshot (PSE-3.3), MarketOpportunitySnapshot
    (PSE-3.4), and StrategicProcurementAssessment (PSE-3.5). No
    TargetSnapshot or GapSnapshot object is consumed directly -- every
    Target/Gap-derived fact this layer needs (coverage gap days, mix gap
    pct points, lead-time gap days, procurement progress gap) is already
    present inside PortfolioOptimizationSnapshot's objective
    supporting_facts (PSE-3.3 already reused GapSnapshot for exactly
    this purpose), so re-consuming Gap/Target here would be duplication,
    not reuse. The only "new" arithmetic in this module is the tons-level
    quantity derivation itself (Quantity Planning is this layer's job,
    per the brief) -- it reuses the same locked reorder-point and mix
    constants every prior phase already depends on.

Quantity derivation (per source, LOCAL / IMPORTED):
    mandatory_qty   = max(0, ROP_<source> - position.<source>_inventory_tons)
        Reuses the approved V1 reorder points (LOCAL_ROP_TONS=1,733 /
        IMPORTED_ROP_TONS=6,958, procurement_strategy_engine.py PSE-3B) --
        the same constants classify_action() already uses for REORDER
        status. Bringing inventory back up to the reorder point is the
        minimum structural requirement before the next reorder/critical
        breach.
    mix_correction  = max(0, mix_target_tons - position.<source>_inventory_tons)
        mix_target_tons = (local+imported tons) x LOCAL_MIX_TARGET/IMPORTED_MIX_TARGET
        (the same 45/55 split locked in PSE-3B), only applied when
        portfolio.mix_objective reports the portfolio outside tolerance.
    recommended_quantity_tons = max(mandatory_qty, mix_correction)
        The larger of the two structural drivers is taken (not summed) --
        satisfying the larger requirement structurally satisfies the
        smaller one, and summing would double-count the same physical
        tons. This formula is decided here, in PSE-3.6, exactly as the
        brief assigns ("Quantity Planning" is this sprint's job) -- it is
        not a recalculation of anything already published by an earlier
        layer.
    The combined LOCAL+IMPORTED total is then capped to
    position.remaining_storage_capacity_tons and (when available)
    position.remaining_procurement_tons, scaling both sources
    proportionally if either cap binds, and flagging when it does.

Expected Benefits:
    A reliable dollar cost-avoidance estimate requires a PKR/USD rate and
    a verified spot price in the correct unit -- MarketOpportunitySnapshot
    does not retain pkr_rate, and re-deriving estimate_savings() (PSE-3D)
    end to end would duplicate that module's calculation outside its own
    inputs. Per "never fabricate savings," expected_cost_avoidance_usd is
    always None/UNKNOWN here, flagged accordingly; only the already-known
    percentage signal (market.expected_price_advantage_pct, a passthrough
    fact) is surfaced as the cost-avoidance *basis*.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import (
    IMPORTED_MIX_TARGET,
    IMPORTED_ROP_TONS,
    LOCAL_MIX_TARGET,
    LOCAL_ROP_TONS,
)
from procurement_position_engine import PositionSnapshot
from procurement_optimization_engine import PortfolioOptimizationSnapshot
from procurement_market_engine import MarketOpportunitySnapshot
from procurement_strategy_assessment_engine import StrategicProcurementAssessment

# ---------------------------------------------------------------------------
# Allowed planning windows -- NOT calendar dates
# ---------------------------------------------------------------------------

EXECUTION_WINDOWS = (
    "IMMEDIATE",
    "WITHIN_LOCAL_LEAD_TIME",
    "WITHIN_IMPORTED_PLANNING_HORIZON",
    "NEXT_PLANNING_CYCLE",
    "DEFERRED_OPPORTUNITY",
    "FUTURE_OPPORTUNITY",
)

SOURCE_LOCAL = "LOCAL"
SOURCE_IMPORTED = "IMPORTED"


def _clamp_nonneg(value: float) -> float:
    return max(0.0, value)


# ===========================================================================
# OUTPUT SCHEMA
# ===========================================================================

@dataclass(frozen=True)
class ProcurementEvent:
    """One planned procurement action. Advisory only -- not an order."""

    event_id: str
    source: str                          # LOCAL / IMPORTED
    planned_quantity_tons: float
    preferred_execution_window: str       # one of EXECUTION_WINDOWS
    window_rule_fired: int
    planning_priority_score: Optional[float]
    planning_priority_level: Optional[str]
    reason: str
    supporting_facts: dict
    dependencies: tuple[str, ...]         # event_ids this event is sequenced after
    constraint_validation: dict
    confidence_score: float
    confidence_level: str
    reasoning_chain: tuple[str, ...]


@dataclass(frozen=True)
class ProcurementExecutionPlan:
    """Immutable procurement plan: how much, in what sequence, within which
    planning window, and why -- with zero execution side effects."""

    as_of: str
    generated_at: str

    plan_summary: str
    procurement_events: tuple[ProcurementEvent, ...]
    execution_sequence: tuple[str, ...]   # event_ids in recommended order

    total_planned_quantity_tons: float
    total_planned_local_tons: float
    total_planned_imported_tons: float

    constraint_validation: dict
    expected_benefits: dict

    planning_confidence_score: float
    planning_confidence_level: str

    reasoning: tuple[str, ...]

    position_data_quality_flags: tuple[str, ...]
    portfolio_data_quality_flags: tuple[str, ...]
    market_data_quality_flags: tuple[str, ...]
    strategy_data_quality_flags: tuple[str, ...]
    data_quality_flags: tuple[str, ...]

    def to_dict(self) -> dict:
        return _tuples_to_lists(asdict(self))


def _tuples_to_lists(obj):
    if isinstance(obj, tuple):
        return [_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tuples_to_lists(v) for v in obj]
    return obj


# ===========================================================================
# QUANTITY PLANNING
# ===========================================================================

def _raw_quantity_for_source(
    position: PositionSnapshot,
    portfolio: PortfolioOptimizationSnapshot,
    source: str,
) -> dict:
    classified_tons = position.local_inventory_tons + position.imported_inventory_tons

    if source == SOURCE_LOCAL:
        rop = LOCAL_ROP_TONS
        inv = position.local_inventory_tons
        mix_target_tons = classified_tons * LOCAL_MIX_TARGET
    else:
        rop = IMPORTED_ROP_TONS
        inv = position.imported_inventory_tons
        mix_target_tons = classified_tons * IMPORTED_MIX_TARGET

    mandatory_qty = round(_clamp_nonneg(rop - inv), 2)

    within_tolerance = portfolio.mix_objective.supporting_facts.get("within_tolerance", True)
    mix_correction = round(_clamp_nonneg(mix_target_tons - inv), 2) if not within_tolerance else 0.0

    recommended = round(max(mandatory_qty, mix_correction), 2)

    return {
        "reorder_point_tons": rop,
        "current_inventory_tons": inv,
        "mandatory_qty_tons": mandatory_qty,
        "mix_target_tons": round(mix_target_tons, 2),
        "mix_correction_tons": mix_correction,
        "recommended_quantity_tons": recommended,
    }


def _apply_caps(
    local_facts: dict,
    imported_facts: dict,
    position: PositionSnapshot,
) -> tuple[float, float, list[str]]:
    flags: list[str] = []
    local_qty = local_facts["recommended_quantity_tons"]
    imported_qty = imported_facts["recommended_quantity_tons"]
    total = local_qty + imported_qty

    capacity_cap = _clamp_nonneg(position.remaining_storage_capacity_tons)
    if total > capacity_cap:
        factor = (capacity_cap / total) if total > 0 else 0.0
        local_qty = round(local_qty * factor, 2)
        imported_qty = round(imported_qty * factor, 2)
        flags.append("PLANNED_QUANTITY_CAPPED_BY_STORAGE_CAPACITY")
        total = local_qty + imported_qty

    if position.remaining_procurement_tons is not None:
        annual_cap = _clamp_nonneg(position.remaining_procurement_tons)
        if total > annual_cap:
            factor = (annual_cap / total) if total > 0 else 0.0
            local_qty = round(local_qty * factor, 2)
            imported_qty = round(imported_qty * factor, 2)
            flags.append("PLANNED_QUANTITY_CAPPED_BY_ANNUAL_PROCUREMENT_TARGET")
    else:
        flags.append("ANNUAL_PROCUREMENT_PROGRESS_UNAVAILABLE_FOR_QUANTITY_CAP")

    return local_qty, imported_qty, flags


# ===========================================================================
# EXECUTION WINDOW
# ===========================================================================
#
# DECISION TREE (first match wins, per source -- deterministic, auditable):
#
#   1. IMMEDIATE   : this source's inventory is at/below its safety stock
#                     floor -> already in survival territory.
#   2. IMMEDIATE   : this source's DOH no longer covers its own lead time
#                     -> a new order today is the only way to avoid a
#                        future floor breach before delivery arrives.
#   3. IMMEDIATE   : strategy posture is PRICE_CAPTURE -> the favourable
#                     pricing window is, by definition, now.
#   4. WITHIN_<SOURCE>_LEAD_TIME/PLANNING_HORIZON :
#                     strategy posture is FORWARD_COVERAGE, or the
#                     portfolio's lead-time objective is MEDIUM priority
#                     -> exposure is elevated but not yet critical.
#   5. NEXT_PLANNING_CYCLE  : strategy posture is OPPORTUNISTIC_ACCUMULATION
#                              or BALANCED_ACCUMULATION.
#   6. DEFERRED_OPPORTUNITY : strategy posture is DEFERRED_PROCUREMENT.
#   7. FUTURE_OPPORTUNITY   : default fallback.
# ===========================================================================

def _determine_execution_window(
    source: str,
    floor_breached: bool,
    lead_time_covered: bool,
    portfolio: PortfolioOptimizationSnapshot,
    strategy: StrategicProcurementAssessment,
) -> tuple[str, int]:
    if floor_breached:
        return "IMMEDIATE", 1
    if not lead_time_covered:
        return "IMMEDIATE", 2
    if strategy.overall_procurement_posture == "PRICE_CAPTURE":
        return "IMMEDIATE", 3
    if (
        strategy.overall_procurement_posture == "FORWARD_COVERAGE"
        or portfolio.lead_time_objective.priority_level == "MEDIUM"
    ):
        window = "WITHIN_LOCAL_LEAD_TIME" if source == SOURCE_LOCAL else "WITHIN_IMPORTED_PLANNING_HORIZON"
        return window, 4
    if strategy.overall_procurement_posture in ("OPPORTUNISTIC_ACCUMULATION", "BALANCED_ACCUMULATION"):
        return "NEXT_PLANNING_CYCLE", 5
    if strategy.overall_procurement_posture == "DEFERRED_PROCUREMENT":
        return "DEFERRED_OPPORTUNITY", 6
    return "FUTURE_OPPORTUNITY", 7


# ===========================================================================
# EVENT CONSTRUCTION
# ===========================================================================

def _build_event(
    source: str,
    quantity_tons: float,
    quantity_facts: dict,
    cap_flags: list[str],
    position: PositionSnapshot,
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
    strategy: StrategicProcurementAssessment,
) -> ProcurementEvent:
    if source == SOURCE_LOCAL:
        floor_breached = position.distance_above_safety_stock_local_tons < 0
        lead_time_covered = position.local_covers_lead_time
        doh_minus_lt = position.local_doh_minus_lead_time_days
        lead_time_days = position.local_lead_time_days
    else:
        floor_breached = position.distance_above_safety_stock_imported_tons < 0
        lead_time_covered = position.imported_covers_lead_time
        doh_minus_lt = position.imported_doh_minus_lead_time_days
        lead_time_days = position.imported_lead_time_days

    window, window_rule = _determine_execution_window(source, floor_breached, lead_time_covered, portfolio, strategy)

    driver = "mandatory reorder shortfall" if quantity_facts["mandatory_qty_tons"] >= quantity_facts["mix_correction_tons"] else "portfolio mix correction"
    reason = (
        f"{source.title()} procurement of {quantity_tons:,.2f} tons driven by {driver} "
        f"(mandatory={quantity_facts['mandatory_qty_tons']:,.2f}t, "
        f"mix_correction={quantity_facts['mix_correction_tons']:,.2f}t); "
        f"window={window} (rule {window_rule}); strategy posture={strategy.overall_procurement_posture}."
    )

    priority_obj = portfolio.inventory_objective if floor_breached else (
        portfolio.lead_time_objective if not lead_time_covered else portfolio.mix_objective
    )

    constraint_validation = {
        "safety_floor": "BREACHED" if floor_breached else "SATISFIED",
        "lead_time_coverage": "AT_RISK" if not lead_time_covered else "SATISFIED",
        "storage_capacity": "CAPPED" if "PLANNED_QUANTITY_CAPPED_BY_STORAGE_CAPACITY" in cap_flags else "SATISFIED",
        "annual_procurement_target": (
            "CAPPED" if "PLANNED_QUANTITY_CAPPED_BY_ANNUAL_PROCUREMENT_TARGET" in cap_flags
            else ("UNAVAILABLE" if "ANNUAL_PROCUREMENT_PROGRESS_UNAVAILABLE_FOR_QUANTITY_CAP" in cap_flags else "SATISFIED")
        ),
    }

    reasoning_chain = (
        f"Coverage Gap: {source} inventory={quantity_facts['current_inventory_tons']:,.2f}t vs "
        f"reorder point={quantity_facts['reorder_point_tons']:,.2f}t "
        f"(mandatory shortfall={quantity_facts['mandatory_qty_tons']:,.2f}t).",
        f"Lead-Time Exposure: {source} lead time={lead_time_days} days; "
        f"DOH minus lead time={doh_minus_lt:,.2f} days; covers={lead_time_covered}.",
        f"Portfolio Objective: inventory={portfolio.inventory_objective.priority_level}, "
        f"mix={portfolio.mix_objective.priority_level}, lead_time={portfolio.lead_time_objective.priority_level}.",
        f"Market Opportunity: {market.opportunity_level} ({market.forecast_direction}, "
        f"quality={market.market_data_quality}).",
        f"Strategy: {strategy.overall_procurement_posture} -> {strategy.primary_strategic_objective}.",
        f"Planned Quantity: {quantity_tons:,.2f} tons "
        f"(= max(mandatory={quantity_facts['mandatory_qty_tons']:,.2f}, "
        f"mix_correction={quantity_facts['mix_correction_tons']:,.2f})); cap_flags={cap_flags}.",
    )

    return ProcurementEvent(
        event_id=f"{source}-1",
        source=source,
        planned_quantity_tons=quantity_tons,
        preferred_execution_window=window,
        window_rule_fired=window_rule,
        planning_priority_score=priority_obj.priority_score,
        planning_priority_level=priority_obj.priority_level,
        reason=reason,
        supporting_facts=quantity_facts,
        dependencies=tuple(),  # populated after sequencing
        constraint_validation=constraint_validation,
        confidence_score=strategy.strategy_confidence_score,
        confidence_level=strategy.strategy_confidence_level,
        reasoning_chain=reasoning_chain,
    )


# ===========================================================================
# SEQUENCING
# ===========================================================================
#
# Events are ordered by planning_priority_score descending (higher
# priority = handled first, minimising survival/feasibility risk first).
# Ties are broken LOCAL-before-IMPORTED, since the local lead time
# (10 days) is far shorter than the imported lead time (90 days) --
# placing the quicker-turnaround order first preserves optionality on the
# longer-horizon leg without changing its own urgency assessment.
# ===========================================================================

def _sequence_events(events: list[ProcurementEvent]) -> list[ProcurementEvent]:
    def _key(ev: ProcurementEvent):
        score = ev.planning_priority_score if ev.planning_priority_score is not None else -1.0
        source_rank = 0 if ev.source == SOURCE_LOCAL else 1
        return (-score, source_rank)

    ordered = sorted(events, key=_key)
    sequenced = []
    preceding_ids: list[str] = []
    for ev in ordered:
        sequenced.append(
            ProcurementEvent(
                event_id=ev.event_id, source=ev.source, planned_quantity_tons=ev.planned_quantity_tons,
                preferred_execution_window=ev.preferred_execution_window, window_rule_fired=ev.window_rule_fired,
                planning_priority_score=ev.planning_priority_score, planning_priority_level=ev.planning_priority_level,
                reason=ev.reason, supporting_facts=ev.supporting_facts,
                dependencies=tuple(preceding_ids), constraint_validation=ev.constraint_validation,
                confidence_score=ev.confidence_score, confidence_level=ev.confidence_level,
                reasoning_chain=ev.reasoning_chain,
            )
        )
        preceding_ids = preceding_ids + [ev.event_id]
    return sequenced


# ===========================================================================
# EXPECTED BENEFITS -- never fabricate savings
# ===========================================================================

def _estimate_expected_benefits(
    events: list[ProcurementEvent],
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
    strategy: StrategicProcurementAssessment,
) -> tuple[dict, list[str]]:
    flags: list[str] = []

    flags.append("EXPECTED_COST_AVOIDANCE_USD_UNAVAILABLE:no_reliable_pkr_rate_and_verified_price_basis")
    benefits = {
        "expected_cost_avoidance_usd": None,
        "expected_cost_avoidance_pct_basis": market.expected_price_advantage_pct,
        "expected_inventory_stability": (
            "IMPROVED" if any(ev.constraint_validation["safety_floor"] == "BREACHED" for ev in events)
            else ("MAINTAINED" if events else "NOT_ASSESSED")
        ),
        "mix_improvement": (
            "NOT_APPLICABLE" if portfolio.mix_objective.supporting_facts.get("within_tolerance", True)
            else ("ADDRESSED" if events else "NOT_ADDRESSED")
        ),
        "risk_reduction": (
            "HIGH" if strategy.overall_procurement_posture in ("DEFENSIVE_PROCUREMENT", "INVENTORY_PRESERVATION")
            else ("MODERATE" if events else "LOW")
        ),
    }
    return benefits, flags


# ===========================================================================
# PLANNING
# ===========================================================================

def build_execution_plan(
    position: PositionSnapshot,
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
    strategy: StrategicProcurementAssessment,
    as_of: Optional[date] = None,
) -> ProcurementExecutionPlan:
    """Build the ProcurementExecutionPlan from already-built upstream
    snapshots. No Oracle reads, no ERP writes, no inventory updates.

    Args:
        position: An already-built PositionSnapshot (PSE-3.0).
        portfolio: An already-built PortfolioOptimizationSnapshot (PSE-3.3).
        market: An already-built MarketOpportunitySnapshot (PSE-3.4).
        strategy: An already-built StrategicProcurementAssessment (PSE-3.5).
        as_of: Date this plan represents; defaults to today.

    Returns:
        ProcurementExecutionPlan -- immutable, advisory only.
    """
    as_of = as_of or date.today()
    flags: list[str] = []

    local_facts = _raw_quantity_for_source(position, portfolio, SOURCE_LOCAL)
    imported_facts = _raw_quantity_for_source(position, portfolio, SOURCE_IMPORTED)
    local_qty, imported_qty, cap_flags = _apply_caps(local_facts, imported_facts, position)
    flags.extend(cap_flags)

    events: list[ProcurementEvent] = []
    if local_qty > 0:
        events.append(_build_event(SOURCE_LOCAL, local_qty, local_facts, cap_flags, position, portfolio, market, strategy))
    if imported_qty > 0:
        events.append(_build_event(SOURCE_IMPORTED, imported_qty, imported_facts, cap_flags, position, portfolio, market, strategy))

    events = _sequence_events(events)
    execution_sequence = tuple(ev.event_id for ev in events)

    total_local = round(sum(ev.planned_quantity_tons for ev in events if ev.source == SOURCE_LOCAL), 2)
    total_imported = round(sum(ev.planned_quantity_tons for ev in events if ev.source == SOURCE_IMPORTED), 2)
    total_planned = round(total_local + total_imported, 2)

    if not events:
        flags.append("NO_PROCUREMENT_EVENTS_REQUIRED")

    constraint_validation = dict(strategy.constraint_validation)
    constraint_validation["planned_quantity_within_storage_capacity"] = (
        "SATISFIED" if total_planned <= max(0.0, position.remaining_storage_capacity_tons) + 0.01 else "BREACHED"
    )
    if position.remaining_procurement_tons is not None:
        constraint_validation["planned_quantity_within_annual_target"] = (
            "SATISFIED" if total_planned <= max(0.0, position.remaining_procurement_tons) + 0.01 else "BREACHED"
        )
    else:
        constraint_validation["planned_quantity_within_annual_target"] = "UNAVAILABLE"

    expected_benefits, benefit_flags = _estimate_expected_benefits(events, portfolio, market, strategy)
    flags.extend(benefit_flags)

    planning_confidence_score = strategy.strategy_confidence_score
    if cap_flags and any(f.startswith("PLANNED_QUANTITY_CAPPED") for f in cap_flags):
        planning_confidence_score = round(max(0.0, planning_confidence_score - 5.0), 2)
        flags.append("PLANNING_CONFIDENCE_REDUCED_BY_CAPPING")
    planning_confidence_level = strategy.strategy_confidence_level

    plan_summary = (
        f"{strategy.overall_procurement_posture}: {len(events)} procurement event(s) "
        f"totaling {total_planned:,.2f} tons (local={total_local:,.2f}t, imported={total_imported:,.2f}t)."
        if events else
        f"{strategy.overall_procurement_posture}: no procurement events required at this time."
    )

    reasoning = (
        f"Strategy posture={strategy.overall_procurement_posture} (primary objective: "
        f"{strategy.primary_strategic_objective}).",
        f"Local: mandatory={local_facts['mandatory_qty_tons']:,.2f}t, "
        f"mix_correction={local_facts['mix_correction_tons']:,.2f}t -> planned={local_qty:,.2f}t.",
        f"Imported: mandatory={imported_facts['mandatory_qty_tons']:,.2f}t, "
        f"mix_correction={imported_facts['mix_correction_tons']:,.2f}t -> planned={imported_qty:,.2f}t.",
        f"Execution sequence: {execution_sequence}.",
    )

    return ProcurementExecutionPlan(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        plan_summary=plan_summary,
        procurement_events=tuple(events),
        execution_sequence=execution_sequence,
        total_planned_quantity_tons=total_planned,
        total_planned_local_tons=total_local,
        total_planned_imported_tons=total_imported,
        constraint_validation=constraint_validation,
        expected_benefits=expected_benefits,
        planning_confidence_score=planning_confidence_score,
        planning_confidence_level=planning_confidence_level,
        reasoning=reasoning,
        position_data_quality_flags=tuple(position.data_quality_flags),
        portfolio_data_quality_flags=tuple(portfolio.data_quality_flags),
        market_data_quality_flags=tuple(market.data_quality_flags),
        strategy_data_quality_flags=(
            tuple(strategy.portfolio_data_quality_flags)
            + tuple(strategy.market_data_quality_flags)
            + tuple(strategy.data_quality_flags)
        ),
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json
    from datetime import date as _date

    parser = argparse.ArgumentParser(
        description="PSE-3.6 Procurement Execution Planning Layer -- print a "
                     "ProcurementExecutionPlan for a workbook run."
    )
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--input", default=None, help="Raw Oracle export (.xlsx)")
    parser.add_argument("--coverage-days", type=float, default=None)
    parser.add_argument("--annual-target-tons", type=float, default=None)
    parser.add_argument("--live-market", action="store_true")
    args = parser.parse_args()

    from procurement_orchestrator import run_orchestration
    from procurement_position_engine import assess_position
    from procurement_target_engine import define_strategy_target
    from procurement_gap_engine import analyze_gap
    from procurement_optimization_engine import optimize_portfolio
    from procurement_market_engine import assess_market_opportunity
    from procurement_strategy_assessment_engine import assess_strategy

    orch = run_orchestration(input_path=args.input, workbook_path=args.workbook)
    today = _date.today()
    pos = assess_position(orch["strategy_output"], as_of=today)
    tgt = define_strategy_target(
        as_of=today,
        desired_coverage_days=args.coverage_days,
        annual_target_tons=args.annual_target_tons,
    )
    gap = analyze_gap(pos, tgt)
    portfolio = optimize_portfolio(pos, tgt, gap)

    market_price_inputs = None
    if args.live_market:
        from procurement_scenario_engine import fetch_price_inputs
        market_price_inputs = fetch_price_inputs()
    market = assess_market_opportunity(market_price_inputs=market_price_inputs, as_of=today)
    strategy = assess_strategy(portfolio, market, as_of=today)

    plan = build_execution_plan(pos, portfolio, market, strategy, as_of=today)
    print(json.dumps(plan.to_dict(), indent=2))
