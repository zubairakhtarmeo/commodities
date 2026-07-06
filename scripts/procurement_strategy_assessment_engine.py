"""
procurement_strategy_assessment_engine.py
--------------------------------------------
PSE-3.5 -- Strategic Procurement Assessment Layer.

Per the frozen architecture, extended through PSE-3.0 - PSE-3.4:

    Position -> Target -> Gap -> Portfolio Optimization  ("what does the
                                  business need?")
                              \\
    Market Intelligence        +--> Strategic Procurement Assessment
    ("what is the market         (this module -- "what strategy fits
     offering?")                  both?")
                                          |
                                          v
                          Execution Layer (NOT implemented here --
                          purchase quantities, timing, tranches)

This module answers exactly one question -- "Considering both
operational needs and market conditions, what procurement posture best
serves the business right now?" -- and nothing else. It defines
STRATEGY, never EXECUTION: no BUY/HOLD, no purchase quantities, no
purchase dates, no purchase orders, no tranche plans, no savings, no
executive narrative.

Inputs:
    Consumes ONLY an already-built PortfolioOptimizationSnapshot (PSE-3.3)
    and MarketOpportunitySnapshot (PSE-3.4). Performs no I/O of its own --
    no Oracle reads, no workbook reads, no forecast calls. Every
    supporting fact and constraint check here is read directly from the
    supporting_facts dicts those two snapshots already computed; nothing
    is recalculated from raw inventory/price data.

Decision philosophy:
    Neither input dominates automatically. Operational need (Portfolio
    Optimization) and market timing (Market Intelligence) are weighed
    together through a single deterministic, auditable decision tree
    (POSTURE_RULES below) -- the same first-match-wins style used by
    classify_action() in procurement_strategy_engine.py (PSE-3B) and
    _determine_posture() in procurement_optimization_engine.py (PSE-3.3).
    Survival/feasibility risk (DEFENSIVE) always outranks market
    opportunity, mirroring the Decision Hierarchy locked in PSE-2.1:
    Security > Structure > Mix > Timing.

    The rule thresholds and posture-to-objective label mappings below are
    ENGINEERING DEFAULTS, not confirmed business rules -- the same status
    as every prior phase's tunable parameters (WATCH_BUFFER_PCT,
    PRIORITY_LEVEL_LOW_MAX, OPPORTUNITY_LEVEL_LOW_MAX, etc.). They make
    the framework operable today and are explicit candidates for business
    sign-off.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_optimization_engine import PortfolioOptimizationSnapshot
from procurement_market_engine import MarketOpportunitySnapshot

# ---------------------------------------------------------------------------
# Allowed procurement postures (strategy, not execution)
# ---------------------------------------------------------------------------

ALLOWED_POSTURES = (
    "DEFENSIVE_PROCUREMENT",
    "INVENTORY_PRESERVATION",
    "FORWARD_COVERAGE",
    "PRICE_CAPTURE",
    "OPPORTUNISTIC_ACCUMULATION",
    "DEFERRED_PROCUREMENT",
    "BALANCED_ACCUMULATION",
)

_PRIMARY_OBJECTIVE_BY_POSTURE = {
    "DEFENSIVE_PROCUREMENT": "Protect inventory security and lead-time feasibility",
    "INVENTORY_PRESERVATION": "Preserve remaining storage capacity headroom",
    "FORWARD_COVERAGE": "Build coverage ahead of emerging lead-time exposure",
    "PRICE_CAPTURE": "Capture favourable market pricing ahead of structural need",
    "OPPORTUNISTIC_ACCUMULATION": "Accumulate opportunistically while conditions are favourable",
    "DEFERRED_PROCUREMENT": "Defer non-essential procurement pending better conditions",
    "BALANCED_ACCUMULATION": "Maintain steady, policy-driven accumulation toward annual plan",
}

# Engineering defaults -- NOT confirmed business rules (see module docstring)
CONFIDENCE_LEVEL_LOW_MAX = 33.0
CONFIDENCE_LEVEL_MEDIUM_MAX = 66.0


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score_to_level(score: float, low_max: float, medium_max: float) -> str:
    if score <= low_max:
        return "LOW"
    if score <= medium_max:
        return "MEDIUM"
    return "HIGH"


# ===========================================================================
# OUTPUT SCHEMA
# ===========================================================================

@dataclass(frozen=True)
class StrategyDimension:
    """One evaluated dimension: facts, the reasoning drawn from them, and
    whether the relevant business constraint is currently satisfied."""
    supporting_facts: dict
    reasoning: str
    constraint_validation: str  # SATISFIED / AT_RISK / BREACHED / NOT_APPLICABLE / UNAVAILABLE


@dataclass(frozen=True)
class StrategicProcurementAssessment:
    """Immutable snapshot of procurement STRATEGY -- never execution.

    Every field here is traceable to PortfolioOptimizationSnapshot and/or
    MarketOpportunitySnapshot; nothing is invented and no quantity, date,
    or purchase order is ever produced.
    """

    as_of: str
    generated_at: str

    overall_procurement_posture: str
    posture_rule_fired: int

    primary_strategic_objective: str
    secondary_strategic_objective: str
    secondary_objective_rule_fired: int

    # Phase 3 decision-quality contract.  These fields describe strategic
    # intent only; quantities and tranches remain the execution layer's job.
    recommended_strategy: str
    strategy_reason: str
    delay_risk: str
    execution_bias: str
    review_trigger: str
    alternative_strategy_considered: str
    alternative_rejection_reason: str

    operational_pressure_summary: dict
    market_opportunity_summary: dict

    inventory_pressure: StrategyDimension
    market_attractiveness: StrategyDimension
    portfolio_balance: StrategyDimension
    capacity_constraints: StrategyDimension
    lead_time_exposure: StrategyDimension
    procurement_progress: StrategyDimension
    flexibility_position: StrategyDimension

    constraint_validation: dict

    strategy_confidence_score: float
    strategy_confidence_level: str

    reasoning: tuple[str, ...]

    portfolio_data_quality_flags: tuple[str, ...]
    market_data_quality_flags: tuple[str, ...]
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
# PER-DIMENSION EVALUATION -- pure restatement of facts already computed
# upstream, plus a deterministic constraint check
# ===========================================================================

def _eval_inventory_pressure(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.inventory_objective
    facts = obj.supporting_facts
    status = facts.get("safety_floor_status_total")
    constraint = "BREACHED" if status == "BELOW_FLOOR" else "SATISFIED"
    reasoning = (
        f"Inventory objective priority is {obj.priority_level} (score={obj.priority_score}); "
        f"safety floor status={status}; coverage gap={facts.get('coverage_gap_days')} days."
    )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation=constraint)


def _eval_market_attractiveness(market: MarketOpportunitySnapshot) -> StrategyDimension:
    facts = {
        "opportunity_level": market.opportunity_level,
        "forecast_direction": market.forecast_direction,
        "forecast_confidence": market.forecast_confidence,
        "expected_price_advantage_pct": market.expected_price_advantage_pct,
        "market_data_quality": market.market_data_quality,
        "current_market_position": market.current_market_position,
    }
    reasoning = (
        f"Market opportunity is {market.opportunity_level or 'UNAVAILABLE'} "
        f"(data quality={market.market_data_quality}); forecast direction="
        f"{market.forecast_direction}, confidence={market.forecast_confidence}."
    )
    # Market attractiveness is an external condition, not a business
    # constraint -- there is nothing to validate it against.
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation="NOT_APPLICABLE")


def _eval_portfolio_balance(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.mix_objective
    facts = obj.supporting_facts
    within_tolerance = facts.get("within_tolerance")
    constraint = "SATISFIED" if within_tolerance else "AT_RISK"
    reasoning = (
        f"Mix objective priority is {obj.priority_level} (score={obj.priority_score}); "
        f"local gap={facts.get('local_mix_gap_pct_points')} pct points, "
        f"within tolerance={within_tolerance}."
    )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation=constraint)


def _eval_capacity_constraints(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.capacity_objective
    facts = obj.supporting_facts
    remaining = facts.get("remaining_storage_capacity_tons")
    if remaining is not None and remaining < 0:
        constraint = "BREACHED"
    elif obj.priority_level == "HIGH":
        constraint = "AT_RISK"
    else:
        constraint = "SATISFIED"
    reasoning = (
        f"Capacity objective priority is {obj.priority_level} (score={obj.priority_score}); "
        f"storage utilisation={facts.get('storage_utilization_pct')}%, "
        f"remaining capacity={remaining} tons."
    )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation=constraint)


def _eval_lead_time_exposure(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.lead_time_objective
    facts = obj.supporting_facts
    local_gap = facts.get("local_lead_time_gap_days")
    imported_gap = facts.get("imported_lead_time_gap_days")
    at_risk = (local_gap is not None and local_gap < 0) or (imported_gap is not None and imported_gap < 0)
    constraint = "AT_RISK" if at_risk else "SATISFIED"
    reasoning = (
        f"Lead-time objective priority is {obj.priority_level} (score={obj.priority_score}); "
        f"local gap={local_gap} days, imported gap={imported_gap} days."
    )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation=constraint)


def _eval_procurement_progress(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.procurement_progress_objective
    facts = obj.supporting_facts
    if obj.priority_level is None:
        constraint = "UNAVAILABLE"
        reasoning = "Procurement progress objective is unavailable -- no annual target data was configured."
    else:
        constraint = "AT_RISK" if obj.priority_level == "HIGH" else "SATISFIED"
        reasoning = (
            f"Procurement progress objective priority is {obj.priority_level} "
            f"(score={obj.priority_score}); annual progress gap="
            f"{facts.get('annual_progress_gap_pct_points')} pct points."
        )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation=constraint)


def _eval_flexibility_position(portfolio: PortfolioOptimizationSnapshot) -> StrategyDimension:
    obj = portfolio.flexibility_objective
    facts = obj.supporting_facts
    reasoning = (
        "Flexibility objective is unavailable -- PositionSnapshot tracks no current "
        "flexibility fact (see PSE-3.0/PSE-3.2/PSE-3.3 docstrings)."
    )
    return StrategyDimension(supporting_facts=facts, reasoning=reasoning, constraint_validation="UNAVAILABLE")


# ===========================================================================
# OVERALL PROCUREMENT POSTURE
# ===========================================================================
#
# DECISION TREE (evaluated top to bottom, first match wins -- deterministic,
# auditable, no ML/LLM):
#
#   1. DEFENSIVE_PROCUREMENT     : portfolio.overall_portfolio_posture ==
#                                   DEFENSIVE -> survival/feasibility risk
#                                   overrides market timing entirely
#                                   (Security > Timing, PSE-2.1).
#   2. INVENTORY_PRESERVATION    : capacity objective priority is HIGH ->
#                                   storage headroom is the binding
#                                   constraint; do not accumulate further
#                                   regardless of market attractiveness.
#   3. FORWARD_COVERAGE          : lead-time objective priority is MEDIUM ->
#                                   exposure is elevated but not yet
#                                   critical; build coverage ahead of the
#                                   horizon before it becomes a Rule-1
#                                   situation.
#   4. PRICE_CAPTURE             : market opportunity is HIGH, forecast
#                                   direction is PRICE_RISING, and market
#                                   data is usable -> capture today's price
#                                   ahead of structural need.
#   5. OPPORTUNISTIC_ACCUMULATION: portfolio posture is OPPORTUNISTIC and
#                                   market opportunity is MEDIUM or HIGH ->
#                                   no urgent need, market at least
#                                   moderately favourable.
#   6. DEFERRED_PROCUREMENT      : portfolio posture is OPPORTUNISTIC and
#                                   market opportunity is LOW -> no urgent
#                                   need and market is unfavourable.
#   7. DEFERRED_PROCUREMENT      : portfolio posture is BALANCED and market
#                                   opportunity is LOW with a PRICE_FALLING
#                                   direction confirmed by FULL-quality data
#                                   -> no structural urgency, prices clearly
#                                   heading lower; defer discretionary volume
#                                   as an experienced director would.
#   8. BALANCED_ACCUMULATION     : default -- steady, policy-driven
#                                   accumulation; neither urgency nor
#                                   opportunism dominates.
# ===========================================================================

def _determine_posture(
    portfolio: PortfolioOptimizationSnapshot, market: MarketOpportunitySnapshot
) -> tuple[str, int]:
    if portfolio.overall_portfolio_posture == "DEFENSIVE":
        return "DEFENSIVE_PROCUREMENT", 1

    if portfolio.capacity_objective.priority_level == "HIGH":
        return "INVENTORY_PRESERVATION", 2

    if portfolio.lead_time_objective.priority_level == "MEDIUM":
        return "FORWARD_COVERAGE", 3

    if (
        market.opportunity_level == "HIGH"
        and market.forecast_direction == "PRICE_RISING"
        and market.market_data_quality != "UNAVAILABLE"
    ):
        return "PRICE_CAPTURE", 4

    if portfolio.overall_portfolio_posture == "OPPORTUNISTIC" and market.opportunity_level in ("MEDIUM", "HIGH"):
        return "OPPORTUNISTIC_ACCUMULATION", 5

    if portfolio.overall_portfolio_posture == "OPPORTUNISTIC" and market.opportunity_level == "LOW":
        return "DEFERRED_PROCUREMENT", 6

    if (
        portfolio.overall_portfolio_posture == "BALANCED"
        and market.opportunity_level == "LOW"
        and market.forecast_direction == "PRICE_FALLING"
        and market.market_data_quality == "FULL"
    ):
        return "DEFERRED_PROCUREMENT", 7

    return "BALANCED_ACCUMULATION", 8


# ===========================================================================
# SECONDARY STRATEGIC OBJECTIVE
# ===========================================================================
#
# DECISION TREE (first match wins), evaluated independently of the primary
# posture -- surfaces whichever secondary pressure is currently most
# binding among Mix and Procurement Progress (Flexibility is always
# unavailable today; see _eval_flexibility_position):
#
#   A. Restore portfolio mix         : mix objective priority is HIGH
#   B. Accelerate procurement progress: procurement progress priority is HIGH
#   C. No secondary objective identified: default
# ===========================================================================

def _determine_secondary_objective(portfolio: PortfolioOptimizationSnapshot) -> tuple[str, int]:
    if portfolio.mix_objective.priority_level == "HIGH":
        return "Restore portfolio mix toward the 45/55 local-imported target", 1
    if portfolio.procurement_progress_objective.priority_level == "HIGH":
        return "Accelerate annual procurement progress back onto pace", 2
    return "No secondary objective identified", 3


# ===========================================================================
# STRATEGY CONFIDENCE
# ===========================================================================

def _score_strategy_confidence(
    portfolio: PortfolioOptimizationSnapshot, market: MarketOpportunitySnapshot
) -> float:
    score = 100.0
    if market.market_data_quality == "UNAVAILABLE":
        score -= 40.0
    elif market.market_data_quality == "PARTIAL":
        score -= 15.0
    if market.forecast_confidence == "LOW":
        score -= 15.0
    if portfolio.procurement_progress_objective.priority_level is None:
        score -= 10.0
    if portfolio.flexibility_objective.priority_level is None:
        score -= 10.0
    return round(_clamp(score), 2)


def _build_strategy_intelligence(
    posture: str,
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
) -> dict:
    """Translate the selected posture into director-level strategic intent.

    This is deliberately qualitative.  The execution layer remains solely
    responsible for turning the intent into quantities and timing.
    """
    mapping = {
        "DEFENSIVE_PROCUREMENT": ("SURVIVAL_COVERAGE", "IMMEDIATE", "HIGH"),
        "INVENTORY_PRESERVATION": ("WAIT", "WAIT", "LOW"),
        "FORWARD_COVERAGE": ("STRATEGIC_FORWARD_PROCUREMENT", "PHASED", "MEDIUM"),
        "PRICE_CAPTURE": ("OPPORTUNITY_CAPTURE", "IMMEDIATE", "MEDIUM"),
        "OPPORTUNISTIC_ACCUMULATION": ("GRADUAL_ACCUMULATION", "PHASED", "LOW"),
        "DEFERRED_PROCUREMENT": ("WAIT", "DEFER_DISCRETIONARY", "LOW"),
        "BALANCED_ACCUMULATION": ("GRADUAL_ACCUMULATION", "PHASED", "MEDIUM"),
    }
    strategy, bias, risk = mapping[posture]

    opportunity = market.opportunity_level or "UNAVAILABLE"
    direction = market.forecast_direction or "UNAVAILABLE"
    reason = (
        f"{strategy.replace('_', ' ').title()} fits portfolio posture "
        f"{portfolio.overall_portfolio_posture} with market opportunity "
        f"{opportunity} and forecast direction {direction}."
    )

    if bias == "IMMEDIATE":
        trigger = "Review after the immediate coverage action or if market conditions reverse."
        alternative = "WAIT"
        rejection = "Waiting was rejected because inventory security, lead-time exposure, or a time-sensitive price opportunity dominates."
    elif bias == "PHASED":
        trigger = "Review at the next planning cycle or when forecast direction, confidence, or inventory safety changes."
        alternative = "FULL_IMMEDIATE_BUY"
        rejection = "A full immediate buy was rejected to preserve flexibility while the position remains manageable."
    else:
        trigger = "Review when price opportunity improves, forecast direction changes, or safety/lead-time coverage deteriorates."
        alternative = "BUY_NOW"
        rejection = "Buying discretionary volume now was rejected because current coverage permits waiting and the market does not justify early commitment."

    return {
        "recommended_strategy": strategy,
        "strategy_reason": reason,
        "delay_risk": risk,
        "execution_bias": bias,
        "review_trigger": trigger,
        "alternative_strategy_considered": alternative,
        "alternative_rejection_reason": rejection,
    }


# ===========================================================================
# ASSESSMENT
# ===========================================================================

def assess_strategy(
    portfolio: PortfolioOptimizationSnapshot,
    market: MarketOpportunitySnapshot,
    as_of: Optional[date] = None,
) -> StrategicProcurementAssessment:
    """Build the StrategicProcurementAssessment from an already-built
    PortfolioOptimizationSnapshot and MarketOpportunitySnapshot.

    Args:
        portfolio: An already-built PortfolioOptimizationSnapshot (PSE-3.3).
        market: An already-built MarketOpportunitySnapshot (PSE-3.4).
        as_of: Date this assessment represents; defaults to today.

    Returns:
        StrategicProcurementAssessment -- immutable, strategy only.
    """
    as_of = as_of or date.today()
    flags: list[str] = []

    inventory_pressure = _eval_inventory_pressure(portfolio)
    market_attractiveness = _eval_market_attractiveness(market)
    portfolio_balance = _eval_portfolio_balance(portfolio)
    capacity_constraints = _eval_capacity_constraints(portfolio)
    lead_time_exposure = _eval_lead_time_exposure(portfolio)
    procurement_progress = _eval_procurement_progress(portfolio)
    flexibility_position = _eval_flexibility_position(portfolio)

    if procurement_progress.constraint_validation == "UNAVAILABLE":
        flags.append("PROCUREMENT_PROGRESS_DATA_UNAVAILABLE")
    flags.append("FLEXIBILITY_DATA_UNAVAILABLE")
    if market.market_data_quality == "UNAVAILABLE":
        flags.append("MARKET_DATA_UNAVAILABLE_AFFECTING_STRATEGY")

    posture, posture_rule = _determine_posture(portfolio, market)
    secondary_objective, secondary_rule = _determine_secondary_objective(portfolio)
    primary_objective = _PRIMARY_OBJECTIVE_BY_POSTURE[posture]
    intelligence = _build_strategy_intelligence(posture, portfolio, market)

    confidence_score = _score_strategy_confidence(portfolio, market)
    confidence_level = _score_to_level(confidence_score, CONFIDENCE_LEVEL_LOW_MAX, CONFIDENCE_LEVEL_MEDIUM_MAX)

    local_gap = lead_time_exposure.supporting_facts.get("local_lead_time_gap_days")
    imported_gap = lead_time_exposure.supporting_facts.get("imported_lead_time_gap_days")
    constraint_validation = {
        "minimum_inventory_floor_25_days": inventory_pressure.constraint_validation,
        "maximum_storage_45000_tons": capacity_constraints.constraint_validation,
        "local_imported_mix_45_55": portfolio_balance.constraint_validation,
        "local_lead_time_10_days": "SATISFIED" if (local_gap is None or local_gap >= 0) else "AT_RISK",
        "imported_lead_time_90_days": "SATISFIED" if (imported_gap is None or imported_gap >= 0) else "AT_RISK",
    }

    operational_pressure_summary = {
        "portfolio_posture": portfolio.overall_portfolio_posture,
        "inventory_priority": portfolio.inventory_objective.priority_level,
        "mix_priority": portfolio.mix_objective.priority_level,
        "procurement_progress_priority": portfolio.procurement_progress_objective.priority_level,
        "lead_time_priority": portfolio.lead_time_objective.priority_level,
        "capacity_priority": portfolio.capacity_objective.priority_level,
        "flexibility_priority": portfolio.flexibility_objective.priority_level,
    }
    market_opportunity_summary = {
        "opportunity_level": market.opportunity_level,
        "forecast_direction": market.forecast_direction,
        "forecast_confidence": market.forecast_confidence,
        "market_data_quality": market.market_data_quality,
    }

    reasoning = (
        f"Rule {posture_rule} fired -> {posture}.",
        f"Portfolio posture={portfolio.overall_portfolio_posture}; "
        f"inventory={portfolio.inventory_objective.priority_level}, "
        f"lead_time={portfolio.lead_time_objective.priority_level}, "
        f"capacity={portfolio.capacity_objective.priority_level}, "
        f"mix={portfolio.mix_objective.priority_level}.",
        f"Market opportunity={market.opportunity_level} "
        f"(quality={market.market_data_quality}); forecast={market.forecast_direction}.",
        f"Secondary objective rule {secondary_rule} fired -> {secondary_objective}.",
    )

    return StrategicProcurementAssessment(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        overall_procurement_posture=posture,
        posture_rule_fired=posture_rule,
        primary_strategic_objective=primary_objective,
        secondary_strategic_objective=secondary_objective,
        secondary_objective_rule_fired=secondary_rule,
        recommended_strategy=intelligence["recommended_strategy"],
        strategy_reason=intelligence["strategy_reason"],
        delay_risk=intelligence["delay_risk"],
        execution_bias=intelligence["execution_bias"],
        review_trigger=intelligence["review_trigger"],
        alternative_strategy_considered=intelligence["alternative_strategy_considered"],
        alternative_rejection_reason=intelligence["alternative_rejection_reason"],
        operational_pressure_summary=operational_pressure_summary,
        market_opportunity_summary=market_opportunity_summary,
        inventory_pressure=inventory_pressure,
        market_attractiveness=market_attractiveness,
        portfolio_balance=portfolio_balance,
        capacity_constraints=capacity_constraints,
        lead_time_exposure=lead_time_exposure,
        procurement_progress=procurement_progress,
        flexibility_position=flexibility_position,
        constraint_validation=constraint_validation,
        strategy_confidence_score=confidence_score,
        strategy_confidence_level=confidence_level,
        reasoning=reasoning,
        portfolio_data_quality_flags=tuple(portfolio.data_quality_flags),
        market_data_quality_flags=tuple(market.data_quality_flags),
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json
    from datetime import date as _date

    parser = argparse.ArgumentParser(
        description="PSE-3.5 Strategic Procurement Assessment Layer -- print a "
                     "StrategicProcurementAssessment for a workbook run."
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
    print(json.dumps(strategy.to_dict(), indent=2))
