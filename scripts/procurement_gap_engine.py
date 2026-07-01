"""
procurement_gap_engine.py
---------------------------
PSE-3.2 -- Gap Analysis Layer.

Per the frozen architecture (PSE-2.1 -> PSE-2.4):

    Procurement Position Assessment -> Strategy Target Definition ->
    Gap Analysis -> Portfolio Optimization -> Opportunity Window Classifier
    -> Tranche Allocation -> Tactical Execution -> Executive Narrative

This module implements ONLY the third box. It answers a single question --
"How far are we from where we should be?" -- as pure arithmetic:

    PositionSnapshot (reality)  -  TargetSnapshot (objective)  =  GapSnapshot

It contains no interpretation, no urgency classification, no BUY/HOLD
logic, no purchase quantities, no savings, no opportunity windows, no
tranche plans, and no executive narrative. Gap Analysis measures;
Portfolio Optimization (Sprint 4, not implemented here) decides.

Inputs:
    Consumes ONLY an already-built PositionSnapshot and TargetSnapshot.
    Performs no I/O of its own -- no workbook reads, no Oracle access, no
    forecast calls, no re-derivation of any value either snapshot already
    computed. The one constant this module imports directly,
    DAILY_CONSUMPTION_TOTAL (from procurement_strategy_engine, the same
    single source of truth both PositionSnapshot and TargetSnapshot
    already depend on transitively), is used only to convert
    TargetSnapshot's desired_inventory_coverage_days into a comparable
    tons figure for stock_gap_tons -- it does not introduce a new
    calculation, only a unit conversion of an existing locked constant.

Fields with no comparable counterpart on both sides (Flexibility,
Market Context) are left None with a data-quality flag explaining why,
per "never fabricate data" -- this mirrors the same discipline already
applied in PSE-3.0 and PSE-3.1.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import DAILY_CONSUMPTION_TOTAL
from procurement_position_engine import PositionSnapshot
from procurement_target_engine import TargetSnapshot


# ===========================================================================
# OUTPUT SCHEMA -- GapSnapshot (and its nested sections)
# ===========================================================================

@dataclass(frozen=True)
class InventoryGap:
    coverage_gap_days: float                       # position.total_doh - target.desired_inventory_coverage_days
    stock_gap_tons: float                           # position stock - (desired coverage days x daily consumption)
    storage_headroom_gap_tons: Optional[float]       # position remaining capacity - target desired buffer
    safety_floor_status_total: str                  # ABOVE_FLOOR / BELOW_FLOOR (restates position fact)
    safety_floor_status_local: str
    safety_floor_status_imported: str


@dataclass(frozen=True)
class ProcurementGap:
    annual_progress_gap_pct_points: Optional[float]  # position completion% - target progress%
    monthly_progress_gap_tons: Optional[float]        # position month purchased - target monthly plan
    remaining_procurement_gap_tons: Optional[float]   # position remaining - target remaining planned


@dataclass(frozen=True)
class MixGap:
    local_mix_gap_pct_points: float                  # position local% - target desired local%
    imported_mix_gap_pct_points: float
    within_tolerance: bool                            # reused directly from PositionSnapshot, not recomputed


@dataclass(frozen=True)
class LeadTimeGap:
    local_lead_time_gap_days: float                  # position local DOH - target local planning horizon
    imported_lead_time_gap_days: float
    local_covers_planning_horizon: bool
    imported_covers_planning_horizon: bool


@dataclass(frozen=True)
class CapacityGap:
    utilization_gap_pct_points: Optional[float]       # position utilisation% - target desired utilisation%
    storage_buffer_gap_tons: Optional[float]          # position remaining capacity - target desired buffer


@dataclass(frozen=True)
class FlexibilityGap:
    current_flexibility: Optional[float]              # always None -- PositionSnapshot tracks no such field
    desired_flexibility: Optional[float]               # passthrough from TargetSnapshot
    flexibility_gap: Optional[float]                   # always None until a Position-side measure exists


@dataclass(frozen=True)
class MarketGap:
    current_market_price_usd_per_lb: Optional[float]   # passthrough from PositionSnapshot
    target_market_price_usd_per_lb: Optional[float]     # always None -- TargetSnapshot defines no price objective
    price_gap_usd_per_lb: Optional[float]               # always None -- nothing comparable to diff against


@dataclass(frozen=True)
class GapSnapshot:
    """Immutable, comparison-only snapshot: PositionSnapshot - TargetSnapshot.

    Pure arithmetic over two already-built snapshots. No interpretation,
    no urgency, no decisions -- every downstream optimization layer reads
    these gaps rather than re-deriving them from Position/Target directly.
    """

    as_of: str
    generated_at: str
    position_as_of: str
    target_as_of: str

    inventory_gap: InventoryGap
    procurement_gap: ProcurementGap
    mix_gap: MixGap
    lead_time_gap: LeadTimeGap
    capacity_gap: CapacityGap
    flexibility_gap: FlexibilityGap
    market_gap: MarketGap

    position_data_quality_flags: tuple[str, ...]
    target_data_quality_flags: tuple[str, ...]
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
# ANALYSIS
# ===========================================================================

def analyze_gap(position: PositionSnapshot, target: TargetSnapshot) -> GapSnapshot:
    """Compute the GapSnapshot: position (reality) minus target (objective).

    Args:
        position: An already-built PositionSnapshot (PSE-3.0). Not
            recomputed, reloaded, or re-derived here.
        target: An already-built TargetSnapshot (PSE-3.1). Not recomputed,
            reloaded, or re-derived here.

    Returns:
        GapSnapshot -- immutable, comparison-only.
    """
    flags: list[str] = []

    if position.as_of != target.as_of:
        flags.append(
            f"AS_OF_DATE_MISMATCH:position={position.as_of}_target={target.as_of}"
        )

    # --- 1. Inventory Gap ---
    coverage_gap_days = round(position.total_doh - target.desired_inventory_coverage_days, 2)
    desired_stock_tons = target.desired_inventory_coverage_days * DAILY_CONSUMPTION_TOTAL
    stock_gap_tons = round(position.total_inventory_tons - desired_stock_tons, 2)

    if target.desired_storage_buffer_tons is not None:
        storage_headroom_gap_tons = round(
            position.remaining_storage_capacity_tons - target.desired_storage_buffer_tons, 2
        )
    else:
        storage_headroom_gap_tons = None
        flags.append("STORAGE_HEADROOM_GAP_UNAVAILABLE:target_buffer_not_configured")

    def _floor_status(distance_above_floor: float) -> str:
        return "ABOVE_FLOOR" if distance_above_floor >= 0 else "BELOW_FLOOR"

    inventory_gap = InventoryGap(
        coverage_gap_days=coverage_gap_days,
        stock_gap_tons=stock_gap_tons,
        storage_headroom_gap_tons=storage_headroom_gap_tons,
        safety_floor_status_total=_floor_status(position.distance_above_safety_stock_total_tons),
        safety_floor_status_local=_floor_status(position.distance_above_safety_stock_local_tons),
        safety_floor_status_imported=_floor_status(position.distance_above_safety_stock_imported_tons),
    )

    # --- 2. Procurement Progress Gap ---
    if position.procurement_completion_pct is not None and target.target_procurement_progress_pct is not None:
        annual_progress_gap = round(
            position.procurement_completion_pct - target.target_procurement_progress_pct, 2
        )
    else:
        annual_progress_gap = None
        flags.append("ANNUAL_PROGRESS_GAP_UNAVAILABLE:procurement_progress_data_incomplete")

    if position.current_month_purchased_tons is not None and target.monthly_target_procurement_tons is not None:
        monthly_progress_gap = round(
            position.current_month_purchased_tons - target.monthly_target_procurement_tons, 2
        )
    else:
        monthly_progress_gap = None
        flags.append("MONTHLY_PROGRESS_GAP_UNAVAILABLE:procurement_progress_data_incomplete")

    if position.remaining_procurement_tons is not None and target.remaining_planned_procurement_tons is not None:
        remaining_procurement_gap = round(
            position.remaining_procurement_tons - target.remaining_planned_procurement_tons, 2
        )
    else:
        remaining_procurement_gap = None
        flags.append("REMAINING_PROCUREMENT_GAP_UNAVAILABLE:procurement_progress_data_incomplete")

    procurement_gap = ProcurementGap(
        annual_progress_gap_pct_points=annual_progress_gap,
        monthly_progress_gap_tons=monthly_progress_gap,
        remaining_procurement_gap_tons=remaining_procurement_gap,
    )

    # --- 3. Mix Gap ---
    mix_gap = MixGap(
        local_mix_gap_pct_points=round(position.local_mix_pct - target.desired_local_pct, 2),
        imported_mix_gap_pct_points=round(position.imported_mix_pct - target.desired_imported_pct, 2),
        within_tolerance=position.mix_within_tolerance,
    )

    # --- 4. Lead-Time Gap ---
    local_lt_gap = round(position.local_doh - target.local_planning_horizon_days, 2)
    imported_lt_gap = round(position.imported_doh - target.imported_planning_horizon_days, 2)
    lead_time_gap = LeadTimeGap(
        local_lead_time_gap_days=local_lt_gap,
        imported_lead_time_gap_days=imported_lt_gap,
        local_covers_planning_horizon=local_lt_gap >= 0,
        imported_covers_planning_horizon=imported_lt_gap >= 0,
    )

    # --- 5. Capacity Gap ---
    if target.desired_capacity_utilization_pct is not None:
        utilization_gap = round(
            position.storage_utilization_pct - target.desired_capacity_utilization_pct, 2
        )
    else:
        utilization_gap = None
        flags.append("CAPACITY_UTILIZATION_GAP_UNAVAILABLE:target_utilization_not_configured")

    if target.desired_storage_buffer_tons is not None:
        storage_buffer_gap = round(
            position.remaining_storage_capacity_tons - target.desired_storage_buffer_tons, 2
        )
    else:
        storage_buffer_gap = None
        flags.append("CAPACITY_STORAGE_BUFFER_GAP_UNAVAILABLE:target_buffer_not_configured")

    capacity_gap = CapacityGap(
        utilization_gap_pct_points=utilization_gap,
        storage_buffer_gap_tons=storage_buffer_gap,
    )

    # --- 6. Flexibility Gap ---
    # PositionSnapshot tracks no current-flexibility measure (no such fact
    # source exists yet -- see PSE-3.0 module docstring). Per "never
    # fabricate data," the gap is left None rather than guessed.
    flags.append("FLEXIBILITY_GAP_UNAVAILABLE:position_has_no_current_flexibility_field")
    flexibility_gap = FlexibilityGap(
        current_flexibility=None,
        desired_flexibility=target.desired_flexibility,
        flexibility_gap=None,
    )

    # --- 7. Market Context Gap ---
    # TargetSnapshot defines no market-price objective (price is a fact of
    # reality, not a business policy target) -- there is nothing compatible
    # on the Target side to diff PositionSnapshot's market fields against.
    flags.append("MARKET_CONTEXT_GAP_NOT_APPLICABLE:target_defines_no_market_objective")
    market_gap = MarketGap(
        current_market_price_usd_per_lb=position.current_market_price_usd_per_lb,
        target_market_price_usd_per_lb=None,
        price_gap_usd_per_lb=None,
    )

    return GapSnapshot(
        as_of=position.as_of,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        position_as_of=position.as_of,
        target_as_of=target.as_of,
        inventory_gap=inventory_gap,
        procurement_gap=procurement_gap,
        mix_gap=mix_gap,
        lead_time_gap=lead_time_gap,
        capacity_gap=capacity_gap,
        flexibility_gap=flexibility_gap,
        market_gap=market_gap,
        position_data_quality_flags=tuple(position.data_quality_flags),
        target_data_quality_flags=tuple(target.data_quality_flags),
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json
    from datetime import date

    parser = argparse.ArgumentParser(
        description="PSE-3.2 Gap Analysis Layer -- print a GapSnapshot for a workbook run"
    )
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--input", default=None, help="Raw Oracle export (.xlsx)")
    parser.add_argument("--coverage-days", type=float, default=None)
    parser.add_argument("--annual-target-tons", type=float, default=None)
    args = parser.parse_args()

    from procurement_orchestrator import run_orchestration
    from procurement_position_engine import assess_position
    from procurement_target_engine import define_strategy_target

    orch = run_orchestration(input_path=args.input, workbook_path=args.workbook)
    today = date.today()
    pos = assess_position(orch["strategy_output"], as_of=today)
    tgt = define_strategy_target(
        as_of=today,
        desired_coverage_days=args.coverage_days,
        annual_target_tons=args.annual_target_tons,
    )
    gap = analyze_gap(pos, tgt)
    print(json.dumps(gap.to_dict(), indent=2))
