"""
procurement_impact_engine.py
-------------------------------
PSE-4.1 -- Decision Impact Analysis Layer.

This is NOT a decision engine. It makes no procurement decisions,
computes no new quantities, and defines no new business rules.

It is an interpretation layer: given the final ProcurementExecutionPlan
(PSE-3.6) and the five upstream snapshots that produced it, it answers
one executive question --

    "If management follows this recommendation, what is the expected
     business impact?"

-- in plain, director-facing language.

Every output field is a direct translation of an already-decided field in
one of the five upstream snapshots.  No field here is a new decision; no
arithmetic is introduced.  This is why this layer belongs AFTER PSE-3.6 in
a read-only position: it is a journalist, not a strategist.

Where "UNKNOWN" is returned:
    When an impact dimension cannot be reliably inferred from existing engine
    outputs (e.g., because market data is absent or annual progress is not
    configured), the field returns "UNKNOWN" with a data-quality flag.  The
    layer never invents language for facts it cannot verify.

Inputs (all already built by the time the dashboard renders):
    ProcurementExecutionPlan     (PSE-3.6) -- the final recommendation
    StrategicProcurementAssessment (PSE-3.5) -- strategy posture and biases
    PortfolioOptimizationSnapshot  (PSE-3.3) -- objective priority signals
    MarketOpportunitySnapshot      (PSE-3.4) -- market direction + confidence
    PositionSnapshot               (PSE-3.0) -- current inventory facts
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import MIN_STOCK_DAYS
from procurement_position_engine import PositionSnapshot
from procurement_optimization_engine import PortfolioOptimizationSnapshot
from procurement_market_engine import MarketOpportunitySnapshot
from procurement_strategy_assessment_engine import StrategicProcurementAssessment
from procurement_execution_planning_engine import ProcurementExecutionPlan


# ===========================================================================
# OUTPUT SCHEMA
# ===========================================================================

@dataclass(frozen=True)
class DecisionImpact:
    """Executive-facing interpretation of the ProcurementExecutionPlan.

    Six business-facing dimensions, each with a one-sentence statement and
    (where relevant) supporting detail.  All fields are derived directly
    from upstream engine outputs -- no decision is made here, no quantity
    is recalculated.

    "UNKNOWN" is returned (with a data-quality flag) when an impact
    dimension cannot be reliably inferred from available data.
    """

    as_of: str
    generated_at: str

    # 1. What happens to inventory if the recommendation is followed?
    inventory_outlook: str
    inventory_outlook_detail: str

    # 2. What happens to annual procurement progress?
    procurement_progress_impact: str

    # 3. What happens to the local/imported portfolio mix?
    mix_outlook: str

    # 4. What is the price-timing / market exposure implication?
    market_exposure: str

    # 5. What is the operational risk of following (or NOT following) this?
    operational_risk_level: str   # LOW / MEDIUM / HIGH / UNKNOWN
    operational_risk_reason: str

    # 6. When should the recommendation next be reviewed?
    review_guidance: str

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
# PER-DIMENSION INTERPRETATION
# Each function reads existing fields -- no new arithmetic.
# ===========================================================================

def _interpret_inventory_outlook(
    plan: ProcurementExecutionPlan,
    strategy: StrategicProcurementAssessment,
    position: PositionSnapshot,
) -> tuple[str, str]:
    """Translate the plan's inventory outcome into a director-facing statement.

    Source fields:
        strategy.overall_procurement_posture
        strategy.constraint_validation["minimum_inventory_floor_25_days"]
        plan.immediate_quantity_tons
        position.total_doh, position.total_inventory_tons,
        position.remaining_storage_capacity_tons, position.storage_utilization_pct
    """
    floor_status = strategy.constraint_validation.get("minimum_inventory_floor_25_days", "UNKNOWN")
    posture = strategy.overall_procurement_posture
    has_events = len(plan.procurement_events) > 0
    immediate_qty = plan.immediate_quantity_tons

    if posture == "INVENTORY_PRESERVATION":
        return (
            "No additional inventory accumulated — storage capacity headroom is being preserved.",
            f"Storage utilisation is {position.storage_utilization_pct:.0f}%; "
            f"{position.remaining_storage_capacity_tons:,.0f} tons of capacity remain — "
            "this is the binding constraint for this planning cycle.",
        )

    if floor_status == "BREACHED" and has_events and immediate_qty > 0:
        return (
            f"Inventory pressure relieved — immediate procurement restores coverage "
            f"above the {MIN_STOCK_DAYS}-day safety floor.",
            f"{immediate_qty:,.0f} tons procured now; current total inventory "
            f"{position.total_inventory_tons:,.0f} tons ({position.total_doh:.0f} days coverage).",
        )

    if floor_status == "BREACHED" and not has_events:
        # Edge case: floor breached but engine generated no events (e.g., fully capped).
        return (
            f"Inventory remains below the {MIN_STOCK_DAYS}-day safety floor — "
            "no procurement events are planned for this cycle.",
            f"Current coverage {position.total_doh:.0f} days; manual review is advised.",
        )

    if has_events and immediate_qty > 0:
        return (
            f"Inventory remains above the {MIN_STOCK_DAYS}-day safety floor after planned procurement.",
            f"{immediate_qty:,.0f} tons procured to maintain coverage; "
            f"current total inventory {position.total_inventory_tons:,.0f} tons "
            f"({position.total_doh:.0f} days coverage).",
        )

    return (
        f"Inventory outlook unchanged — current {position.total_doh:.0f}-day coverage is "
        "adequate and no procurement action is required.",
        f"Total inventory {position.total_inventory_tons:,.0f} tons against the "
        f"{MIN_STOCK_DAYS}-day ({position.safety_stock_total_tons:,.0f}-ton) safety floor.",
    )


def _interpret_procurement_progress(
    plan: ProcurementExecutionPlan,
    portfolio: PortfolioOptimizationSnapshot,
) -> tuple[str, list[str]]:
    """Translate annual procurement progress impact into a director-facing statement.

    Source fields:
        portfolio.procurement_progress_objective.priority_level
        plan.procurement_events (any events planned?)
    """
    flags: list[str] = []
    progress_level = portfolio.procurement_progress_objective.priority_level

    if progress_level is None:
        flags.append("PROCUREMENT_PROGRESS_IMPACT_UNKNOWN:no_annual_target_configured")
        return "UNKNOWN — no annual procurement target is configured.", flags

    has_events = len(plan.procurement_events) > 0

    if progress_level == "HIGH" and has_events:
        return "Annual procurement backlog reduced — planned restocking improves pace toward the annual target.", flags
    if progress_level == "HIGH" and not has_events:
        return "Annual procurement remains behind pace — no new volume is planned this cycle.", flags
    if progress_level == "MEDIUM" and has_events:
        return "Annual procurement pace maintained — planned volume keeps progress on track.", flags
    if progress_level == "LOW" and has_events:
        return "Annual procurement remains on schedule.", flags
    if not has_events:
        return "Procurement progress unchanged — no new volume planned this cycle.", flags

    return "Annual procurement progress maintained.", flags


def _interpret_mix_outlook(
    plan: ProcurementExecutionPlan,
    position: PositionSnapshot,
) -> str:
    """Translate portfolio mix impact into a director-facing statement.

    Source fields:
        position.mix_within_tolerance
        position.local_mix_pct, position.imported_mix_pct   (already in %)
        position.local_mix_deviation_pct_points
        plan.procurement_events[*].supporting_facts["mix_correction_tons"]
    """
    local_pct = round(position.local_mix_pct, 0)
    imported_pct = round(position.imported_mix_pct, 0)

    mix_correction_addressed = any(
        ev.supporting_facts.get("mix_correction_tons", 0) > 0
        for ev in plan.procurement_events
    )

    if position.mix_within_tolerance:
        if plan.total_planned_quantity_tons > 0:
            return (
                f"Current portfolio mix maintained near the 45/55 target "
                f"({local_pct:.0f}% local / {imported_pct:.0f}% imported)."
            )
        return (
            f"Current portfolio mix is on target "
            f"({local_pct:.0f}% local / {imported_pct:.0f}% imported)."
        )

    if mix_correction_addressed:
        direction = "underweight local" if position.local_mix_deviation_pct_points < 0 else "overweight local"
        return (
            f"Portfolio moves closer to the 45% local / 55% imported target — "
            f"current mix ({local_pct:.0f}% local) is {direction}; "
            "planned procurement corrects the imbalance."
        )

    return (
        f"Mix deviation not addressed this cycle — current mix "
        f"({local_pct:.0f}% local / {imported_pct:.0f}% imported) remains off the 45/55 target."
    )


def _interpret_market_exposure(
    plan: ProcurementExecutionPlan,
    strategy: StrategicProcurementAssessment,
    market: MarketOpportunitySnapshot,
) -> tuple[str, list[str]]:
    """Translate the price-timing implication into a director-facing statement.

    Source fields:
        market.market_data_quality, market.forecast_direction
        market.expected_price_advantage_pct
        strategy.overall_procurement_posture, strategy.execution_bias
        plan.immediate_quantity_tons, plan.deferred_quantity_tons
    """
    flags: list[str] = []

    if market.market_data_quality == "UNAVAILABLE":
        flags.append("MARKET_EXPOSURE_IMPACT_UNKNOWN:no_price_data_available")
        return "Market exposure impact UNKNOWN — no price data is available.", flags

    direction = market.forecast_direction
    posture = strategy.overall_procurement_posture
    immediate = plan.immediate_quantity_tons
    deferred = plan.deferred_quantity_tons
    advantage_pct = market.expected_price_advantage_pct

    if posture == "PRICE_CAPTURE":
        adv_str = f" (forecast advantage: {advantage_pct:+.1f}%)" if advantage_pct is not None else ""
        if immediate > 0:
            return (
                f"Captures expected price advantage — {immediate:,.0f} tons procured now "
                f"ahead of forecast price increase{adv_str}.",
                flags,
            )
        # PRICE_CAPTURE posture with no structural events (v1: no opportunistic quantity generated).
        # The market signal is still relevant; flag it even though no volume is outstanding.
        return (
            f"Favourable price opportunity identified{adv_str} — no structural procurement "
            "quantity is currently outstanding; consider opportunistic buying within storage limits.",
            flags,
        )

    if posture == "DEFERRED_PROCUREMENT" and direction == "PRICE_FALLING":
        return (
            "Preserves the opportunity to buy at lower forecasted prices — "
            "discretionary volume deferred pending market improvement.",
            flags,
        )

    if deferred > 0 and direction == "PRICE_RISING":
        return (
            f"Partial market capture — {immediate:,.0f} tons procured now, "
            f"{deferred:,.0f} tons reserved for later tranches as the price signal strengthens.",
            flags,
        )

    if deferred > 0 and direction == "PRICE_FALLING":
        return (
            f"Reduced market exposure — {immediate:,.0f} tons immediate (mandatory coverage), "
            f"{deferred:,.0f} tons held back to benefit from expected price decline.",
            flags,
        )

    if direction == "PRICE_NEUTRAL" or direction is None:
        return "Neutral market impact — price timing is not a primary driver of this recommendation.", flags

    return "Neutral market impact — market conditions are considered but not the primary driver.", flags


def _interpret_operational_risk(
    plan: ProcurementExecutionPlan,
    strategy: StrategicProcurementAssessment,
    position: PositionSnapshot,
) -> tuple[str, str]:
    """Translate the operational risk level into a director-facing rating and reason.

    Source fields:
        plan.risk_if_delayed          (already computed as HIGH/MEDIUM/LOW in PSE-3.6)
        strategy.overall_procurement_posture
        position.local_covers_lead_time, position.imported_covers_lead_time
        position.safety_stock_total_tons
    """
    risk = plan.risk_if_delayed
    posture = strategy.overall_procurement_posture

    if posture == "INVENTORY_PRESERVATION":
        return (
            "LOW",
            "No procurement action planned — current inventory is adequate and "
            "storage headroom is the binding constraint.",
        )

    if risk == "HIGH":
        if not position.imported_covers_lead_time:
            return (
                "HIGH",
                "Imported inventory no longer covers the 90-day lead time — a new order "
                "placed today is the only way to guarantee on-time delivery before a floor breach.",
            )
        if not position.local_covers_lead_time:
            return (
                "HIGH",
                "Local inventory no longer covers the 10-day lead time — "
                "immediate restocking is required to maintain uninterrupted supply.",
            )
        return (
            "HIGH",
            "Inventory is below the reorder point — delayed action risks a safety-floor breach "
            "before the next lead-time delivery arrives.",
        )

    if risk == "MEDIUM":
        return (
            "MEDIUM",
            "Lead-time coverage is reduced — further delay increases the risk of a future "
            "floor breach during the procurement planning horizon.",
        )

    return (
        "LOW",
        "Current coverage provides adequate buffer — short delays carry low operational risk.",
    )


def _interpret_review_guidance(
    plan: ProcurementExecutionPlan,
    strategy: StrategicProcurementAssessment,
) -> str:
    """Translate the review timing into a director-facing guidance sentence.

    Source fields:
        strategy.execution_bias
        plan.next_review_date
        strategy.review_trigger
    """
    bias = strategy.execution_bias
    next_review = plan.next_review_date
    trigger = strategy.review_trigger

    if bias == "IMMEDIATE":
        return (
            f"Review by {next_review} — or earlier if procurement actions cannot be confirmed "
            f"within that window. {trigger}"
        )
    if bias == "PHASED":
        return (
            f"Review by {next_review}. {trigger}"
        )
    # WAIT / DEFER_DISCRETIONARY
    return (
        f"Review by {next_review} or when this trigger occurs: {trigger}"
    )


# ===========================================================================
# PUBLIC API
# ===========================================================================

def interpret_impact(
    plan: ProcurementExecutionPlan,
    strategy: StrategicProcurementAssessment,
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
    position: PositionSnapshot,
    as_of: Optional[date] = None,
) -> DecisionImpact:
    """Build the DecisionImpact from already-built upstream snapshots.

    No I/O, no new decisions, no new arithmetic.  Every output field is a
    plain-language translation of an already-computed fact in one of the
    five upstream snapshots.

    Args:
        plan:      Already-built ProcurementExecutionPlan (PSE-3.6).
        strategy:  Already-built StrategicProcurementAssessment (PSE-3.5).
        portfolio: Already-built PortfolioOptimizationSnapshot (PSE-3.3).
        market:    Already-built MarketOpportunitySnapshot (PSE-3.4).
        position:  Already-built PositionSnapshot (PSE-3.0).
        as_of:     Date this impact represents; defaults to today.

    Returns:
        DecisionImpact -- immutable, explanation only.
    """
    as_of = as_of or date.today()
    flags: list[str] = []

    inventory_outlook, inventory_outlook_detail = _interpret_inventory_outlook(plan, strategy, position)
    procurement_progress_impact, progress_flags = _interpret_procurement_progress(plan, portfolio)
    flags.extend(progress_flags)
    mix_outlook = _interpret_mix_outlook(plan, position)
    market_exposure, market_flags = _interpret_market_exposure(plan, strategy, market)
    flags.extend(market_flags)
    operational_risk_level, operational_risk_reason = _interpret_operational_risk(plan, strategy, position)
    review_guidance = _interpret_review_guidance(plan, strategy)

    return DecisionImpact(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        inventory_outlook=inventory_outlook,
        inventory_outlook_detail=inventory_outlook_detail,
        procurement_progress_impact=procurement_progress_impact,
        mix_outlook=mix_outlook,
        market_exposure=market_exposure,
        operational_risk_level=operational_risk_level,
        operational_risk_reason=operational_risk_reason,
        review_guidance=review_guidance,
        data_quality_flags=tuple(flags),
    )
