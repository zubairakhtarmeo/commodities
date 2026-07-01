"""
procurement_optimization_engine.py
-------------------------------------
PSE-3.3 -- Portfolio Optimization Layer.

Per the frozen architecture (PSE-2.1 -> PSE-2.4, extended in PSE-3.0/3.1/3.2):

    Procurement Position Assessment -> Strategy Target Definition ->
    Gap Analysis -> Portfolio Optimization -> Opportunity Window Classifier
    -> Tranche Allocation -> Tactical Execution -> Executive Narrative

This module implements ONLY the fourth box. It answers a single question --
"Given the current gaps and business constraints, what procurement
objectives should be prioritised?" -- and nothing else. It produces
PRIORITIES, not actions:

    "Increase Imported Coverage" (valid output of this layer)
    "Buy 4,958 tons of imported cotton by 2026-07-15" (NOT this layer --
        that requires Opportunity Windows + Tranche Allocation + Tactical
        Execution, none of which are implemented here)

It contains no BUY/HOLD logic, no purchase quantities, no purchase
timing, no savings math, no opportunity-window classification, no
tranche plans, and no executive narrative.

Inputs:
    Consumes ONLY an already-built PositionSnapshot, TargetSnapshot, and
    GapSnapshot. Performs no I/O of its own and recomputes nothing those
    three objects already produced -- every supporting fact reported here
    is read directly off one of them.

Optimisation methodology (deterministic, auditable, no ML/LLM):
    Each dimension below produces a continuous priority_score (0-100) and
    a discrete priority_level (LOW/MEDIUM/HIGH) from a fixed, documented
    formula over GapSnapshot/PositionSnapshot facts. The score thresholds
    (PRIORITY_LEVEL_LOW_MAX / PRIORITY_LEVEL_MEDIUM_MAX) and the per-point
    weightings below them are ENGINEERING DEFAULTS, not confirmed business
    rules -- the same status as WATCH_BUFFER_PCT / MIX_TOLERANCE_PCT_POINTS
    in procurement_strategy_engine.py (PSE-3B) and the unresolved business
    parameters flagged in PSE-2.2/PSE-2.4. They make the framework operable
    today and are explicit candidates for business sign-off; they do not
    change the structural separation of "measure" (Sprints 1-3) from
    "prioritise" (this sprint) from "decide" (future sprints).

Flexibility Objective:
    PositionSnapshot tracks no current-flexibility fact (see PSE-3.0 /
    PSE-3.2 docstrings). Per "never fabricate data," this objective is
    always priority_score=None / priority_level=None with a data-quality
    flag, exactly mirroring FlexibilityGap in PSE-3.2.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_position_engine import PositionSnapshot
from procurement_target_engine import TargetSnapshot
from procurement_gap_engine import GapSnapshot

# ---------------------------------------------------------------------------
# Engineering defaults -- NOT confirmed business rules (see module docstring)
# ---------------------------------------------------------------------------

PRIORITY_LEVEL_LOW_MAX = 33.0
PRIORITY_LEVEL_MEDIUM_MAX = 66.0

VALID_POSTURES = ("DEFENSIVE", "BALANCED", "OPPORTUNISTIC")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score_to_level(score: float) -> str:
    if score <= PRIORITY_LEVEL_LOW_MAX:
        return "LOW"
    if score <= PRIORITY_LEVEL_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


# ===========================================================================
# OUTPUT SCHEMA
# ===========================================================================

@dataclass(frozen=True)
class ObjectivePriority:
    """One optimisation dimension's result. Priority only -- no action."""
    priority_score: Optional[float]   # 0-100, continuous; None if not computable
    priority_level: Optional[str]      # LOW / MEDIUM / HIGH; None if not computable
    supporting_facts: dict


@dataclass(frozen=True)
class PortfolioOptimizationSnapshot:
    """Immutable snapshot of procurement-objective priorities.

    Answers "what should be prioritised right now?" -- never "what should
    we buy, how much, or when." Those questions belong to the Opportunity
    Window, Tranche Allocation, and Tactical Execution layers (not yet
    implemented).
    """

    as_of: str
    generated_at: str

    inventory_objective: ObjectivePriority
    mix_objective: ObjectivePriority
    procurement_progress_objective: ObjectivePriority
    lead_time_objective: ObjectivePriority
    capacity_objective: ObjectivePriority
    flexibility_objective: ObjectivePriority

    overall_portfolio_posture: str    # DEFENSIVE / BALANCED / OPPORTUNISTIC
    posture_rule_fired: int           # which numbered rule below produced the posture

    position_data_quality_flags: tuple[str, ...]
    target_data_quality_flags: tuple[str, ...]
    gap_data_quality_flags: tuple[str, ...]
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
# DIMENSION SCORING -- each is an independent, pure function of facts
# already present on position/target/gap
# ===========================================================================

def _score_inventory_objective(position: PositionSnapshot, gap: GapSnapshot) -> ObjectivePriority:
    """Coverage gap + safety floor + capacity pressure."""
    statuses = (
        gap.inventory_gap.safety_floor_status_total,
        gap.inventory_gap.safety_floor_status_local,
        gap.inventory_gap.safety_floor_status_imported,
    )
    breach_count = sum(1 for s in statuses if s == "BELOW_FLOOR")
    safety_points = breach_count / 3.0 * 50.0

    coverage_gap_days = gap.inventory_gap.coverage_gap_days
    coverage_points = min(40.0, abs(coverage_gap_days) * 2.0) if coverage_gap_days < 0 else 0.0

    util_pct = position.storage_utilization_pct
    if util_pct >= 90.0:
        capacity_points = 10.0
    elif util_pct >= 75.0:
        capacity_points = 5.0
    else:
        capacity_points = 0.0

    score = _clamp(safety_points + coverage_points + capacity_points)
    return ObjectivePriority(
        priority_score=round(score, 2),
        priority_level=_score_to_level(score),
        supporting_facts={
            "coverage_gap_days": coverage_gap_days,
            "safety_floor_breach_count": breach_count,
            "safety_floor_status_total": gap.inventory_gap.safety_floor_status_total,
            "storage_utilization_pct": util_pct,
        },
    )


def _score_mix_objective(gap: GapSnapshot) -> ObjectivePriority:
    """Mix deviation magnitude + tolerance breach."""
    local_dev = gap.mix_gap.local_mix_gap_pct_points
    imported_dev = gap.mix_gap.imported_mix_gap_pct_points
    worst_dev = max(abs(local_dev), abs(imported_dev))

    base = min(60.0, worst_dev * 3.0)
    if not gap.mix_gap.within_tolerance:
        base += 20.0
    score = _clamp(base)

    return ObjectivePriority(
        priority_score=round(score, 2),
        priority_level=_score_to_level(score),
        supporting_facts={
            "local_mix_gap_pct_points": local_dev,
            "imported_mix_gap_pct_points": imported_dev,
            "within_tolerance": gap.mix_gap.within_tolerance,
        },
    )


def _score_procurement_progress_objective(gap: GapSnapshot) -> ObjectivePriority:
    """How far behind the time-based annual pace actual purchasing is."""
    annual_gap = gap.procurement_gap.annual_progress_gap_pct_points
    if annual_gap is None:
        return ObjectivePriority(
            priority_score=None,
            priority_level=None,
            supporting_facts={
                "annual_progress_gap_pct_points": None,
                "reason": "PROCUREMENT_PROGRESS_DATA_UNAVAILABLE",
            },
        )

    behind_pace = max(0.0, -annual_gap)
    score = _clamp(behind_pace * 2.5)
    return ObjectivePriority(
        priority_score=round(score, 2),
        priority_level=_score_to_level(score),
        supporting_facts={"annual_progress_gap_pct_points": annual_gap},
    )


def _score_lead_time_objective(gap: GapSnapshot) -> ObjectivePriority:
    """Worst-case deficit (in days) against either source's planning horizon."""
    deficit_local = max(0.0, -gap.lead_time_gap.local_lead_time_gap_days)
    deficit_imported = max(0.0, -gap.lead_time_gap.imported_lead_time_gap_days)
    worst_deficit_days = max(deficit_local, deficit_imported)

    score = _clamp(worst_deficit_days * 1.0)
    return ObjectivePriority(
        priority_score=round(score, 2),
        priority_level=_score_to_level(score),
        supporting_facts={
            "local_lead_time_gap_days": gap.lead_time_gap.local_lead_time_gap_days,
            "imported_lead_time_gap_days": gap.lead_time_gap.imported_lead_time_gap_days,
            "worst_deficit_days": worst_deficit_days,
        },
    )


def _score_capacity_objective(position: PositionSnapshot, gap: GapSnapshot) -> ObjectivePriority:
    """Storage utilisation pressure, independent of whether a target buffer
    was configured (always computable from PositionSnapshot facts alone)."""
    util_pct = position.storage_utilization_pct
    score = _clamp(max(0.0, util_pct - 50.0) * 2.0)
    return ObjectivePriority(
        priority_score=round(score, 2),
        priority_level=_score_to_level(score),
        supporting_facts={
            "storage_utilization_pct": util_pct,
            "remaining_storage_capacity_tons": position.remaining_storage_capacity_tons,
            "storage_buffer_gap_tons": gap.capacity_gap.storage_buffer_gap_tons,
        },
    )


def _score_flexibility_objective(gap: GapSnapshot) -> ObjectivePriority:
    """Always unavailable today -- PositionSnapshot tracks no current
    flexibility fact. Never fabricated."""
    return ObjectivePriority(
        priority_score=None,
        priority_level=None,
        supporting_facts={
            "current_flexibility": gap.flexibility_gap.current_flexibility,
            "desired_flexibility": gap.flexibility_gap.desired_flexibility,
            "reason": "FLEXIBILITY_DATA_UNAVAILABLE",
        },
    )


# ===========================================================================
# OVERALL PORTFOLIO POSTURE
# ===========================================================================
#
# DECISION TREE for posture (evaluated top to bottom, first match wins --
# same deterministic, auditable style as classify_action() in
# procurement_strategy_engine.py PSE-3B):
#
#   1. DEFENSIVE    : inventory or lead-time objective is HIGH priority
#                      -> survival/feasibility risk dominates (mirrors the
#                         Decision Hierarchy Level 0/1 precedence from
#                         PSE-2.1: security always overrides optimisation).
#   2. OPPORTUNISTIC: every computable objective is LOW priority
#                      -> comfortable margins everywhere; no constraint is
#                         binding. Objectives with no data (None level) are
#                         excluded from this check, not treated as LOW.
#   3. BALANCED      : default -- everything else (some MEDIUM pressure,
#                      or mixed LOW/MEDIUM, with no HIGH-priority survival
#                      risk).
# ===========================================================================

def _determine_posture(objectives: list[ObjectivePriority]) -> tuple[str, int]:
    levels = [o.priority_level for o in objectives if o.priority_level is not None]

    inventory, mix, progress, lead_time, capacity, flexibility = objectives
    if inventory.priority_level == "HIGH" or lead_time.priority_level == "HIGH":
        return "DEFENSIVE", 1

    if levels and all(level == "LOW" for level in levels):
        return "OPPORTUNISTIC", 2

    return "BALANCED", 3


# ===========================================================================
# OPTIMIZATION
# ===========================================================================

def optimize_portfolio(
    position: PositionSnapshot,
    target: TargetSnapshot,
    gap: GapSnapshot,
) -> PortfolioOptimizationSnapshot:
    """Build the PortfolioOptimizationSnapshot: which objectives matter most
    right now, given the current gaps. No actions, quantities, or timing.

    Args:
        position: An already-built PositionSnapshot (PSE-3.0).
        target: An already-built TargetSnapshot (PSE-3.1).
        gap: An already-built GapSnapshot (PSE-3.2), built from the same
            position/target pair.

    Returns:
        PortfolioOptimizationSnapshot -- immutable, priorities only.
    """
    flags: list[str] = []

    inventory_obj = _score_inventory_objective(position, gap)
    mix_obj = _score_mix_objective(gap)
    progress_obj = _score_procurement_progress_objective(gap)
    lead_time_obj = _score_lead_time_objective(gap)
    capacity_obj = _score_capacity_objective(position, gap)
    flexibility_obj = _score_flexibility_objective(gap)

    if progress_obj.priority_level is None:
        flags.append("PROCUREMENT_PROGRESS_OBJECTIVE_UNAVAILABLE")
    flags.append("FLEXIBILITY_OBJECTIVE_UNAVAILABLE:position_has_no_current_flexibility_field")

    posture, rule_fired = _determine_posture(
        [inventory_obj, mix_obj, progress_obj, lead_time_obj, capacity_obj, flexibility_obj]
    )

    return PortfolioOptimizationSnapshot(
        as_of=gap.as_of,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        inventory_objective=inventory_obj,
        mix_objective=mix_obj,
        procurement_progress_objective=progress_obj,
        lead_time_objective=lead_time_obj,
        capacity_objective=capacity_obj,
        flexibility_objective=flexibility_obj,
        overall_portfolio_posture=posture,
        posture_rule_fired=rule_fired,
        position_data_quality_flags=tuple(position.data_quality_flags),
        target_data_quality_flags=tuple(target.data_quality_flags),
        gap_data_quality_flags=tuple(gap.data_quality_flags),
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json
    from datetime import date

    parser = argparse.ArgumentParser(
        description="PSE-3.3 Portfolio Optimization Layer -- print a PortfolioOptimizationSnapshot"
    )
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--input", default=None, help="Raw Oracle export (.xlsx)")
    parser.add_argument("--coverage-days", type=float, default=None)
    parser.add_argument("--annual-target-tons", type=float, default=None)
    args = parser.parse_args()

    from procurement_orchestrator import run_orchestration
    from procurement_position_engine import assess_position
    from procurement_target_engine import define_strategy_target
    from procurement_gap_engine import analyze_gap

    orch = run_orchestration(input_path=args.input, workbook_path=args.workbook)
    today = date.today()
    pos = assess_position(orch["strategy_output"], as_of=today)
    tgt = define_strategy_target(
        as_of=today,
        desired_coverage_days=args.coverage_days,
        annual_target_tons=args.annual_target_tons,
    )
    gap = analyze_gap(pos, tgt)
    opt = optimize_portfolio(pos, tgt, gap)
    print(json.dumps(opt.to_dict(), indent=2))
