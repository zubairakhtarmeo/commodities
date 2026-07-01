"""
procurement_position_engine.py
--------------------------------
PSE-3.0 -- Procurement Position Assessment Layer (Layer 0).

Per the frozen architecture (PSE-2.1 -> PSE-2.4):

    Procurement Position Assessment -> Strategy Target Definition ->
    Gap Analysis -> Portfolio Optimization -> Opportunity Window Classifier
    -> Tranche Allocation -> Tactical Execution -> Executive Narrative

This module implements ONLY the first box. It answers a single question --
"Where are we right now?" -- and answers it with facts alone. It contains
no optimization, no recommendation, no BUY/HOLD/MONITOR logic, no
opportunity-window classification, and no purchase-quantity or savings
math. Every downstream layer (not yet built) is expected to consume the
PositionSnapshot produced here rather than recompute any of these facts
itself.

Reuse, not duplication:
    - Inventory totals, DOH, mix %, and reorder status all come from
      StrategyOutputV2 (procurement_strategy_engine.build_strategy_output_v2,
      PSE-3B), produced by procurement_orchestrator.run_orchestration()
      (PSE-3C). This module does not re-read the workbook, re-classify
      origin, or re-run the Kg->Tons conversion.
    - Locked business constants (safety stock, lead times, mix targets,
      minimum stock days) are imported from procurement_strategy_engine.py,
      the single source of truth established in PSE-3B.
    - MAX_STORAGE_CAPACITY_TONS is imported from procurement_orchestrator.py,
      where it is already defined as the single source of truth for that
      constant.
    - Forecast confidence is derived via procurement_planning_engine's
      compute_price_signal(), the existing PSE-3D function for that
      calculation -- not reimplemented here.

What this module does NOT do:
    - It does not fetch live prices itself. market_price_inputs is an
      optional caller-supplied dict (e.g. the output of
      procurement_scenario_engine.fetch_price_inputs()). If omitted, all
      Market Context fields are None and a data-quality flag is set.
    - It does not track annual procurement targets / purchases-to-date --
      no such data source exists anywhere in this codebase today.
      procurement_progress is an optional caller-supplied dict for future
      wiring; if omitted, all Procurement Progress fields are None and a
      data-quality flag is set. This module invents no new tracking logic
      for that data, per the reuse-only mandate for this phase.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_orchestrator import MAX_STORAGE_CAPACITY_TONS
from procurement_planning_engine import compute_price_signal
from procurement_strategy_engine import (
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_MIX_TARGET,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_MIX_TARGET,
    MIN_STOCK_DAYS,
    SAFETY_STOCK_IMPORTED_TONS,
    SAFETY_STOCK_LOCAL_TONS,
    SAFETY_STOCK_TOTAL_TONS,
    StrategyOutputV2,
)


# ===========================================================================
# OUTPUT SCHEMA -- PositionSnapshot
# ===========================================================================

@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable, fact-only snapshot of the current procurement position.

    Every future optimization / recommendation layer consumes this object
    rather than recalculating DOH, mix, storage utilisation, or procurement
    progress independently (frozen architecture requirement, PSE-2.4).
    """

    as_of: str
    generated_at: str

    # --- Inventory Position ---
    total_inventory_tons: float
    local_inventory_tons: float
    imported_inventory_tons: float
    total_doh: float
    local_doh: float
    imported_doh: float
    min_stock_days: int
    safety_stock_total_tons: float
    safety_stock_local_tons: float
    safety_stock_imported_tons: float
    distance_above_safety_stock_total_tons: float
    distance_above_safety_stock_local_tons: float
    distance_above_safety_stock_imported_tons: float
    max_storage_capacity_tons: float
    remaining_storage_capacity_tons: float
    storage_utilization_pct: float

    # --- Procurement Progress Position (facts only when supplied) ---
    annual_target_tons: Optional[float]
    purchased_to_date_tons: Optional[float]
    remaining_procurement_tons: Optional[float]
    procurement_completion_pct: Optional[float]
    current_month_purchased_tons: Optional[float]

    # --- Mix Position ---
    local_mix_pct: float
    imported_mix_pct: float
    local_mix_target_pct: float
    imported_mix_target_pct: float
    local_mix_deviation_pct_points: float
    imported_mix_deviation_pct_points: float
    mix_within_tolerance: bool

    # --- Lead-Time Position (facts only -- no recommendations) ---
    local_lead_time_days: int
    imported_lead_time_days: int
    local_doh_minus_lead_time_days: float
    imported_doh_minus_lead_time_days: float
    local_covers_lead_time: bool
    imported_covers_lead_time: bool

    # --- Market Context (facts only -- no interpretation) ---
    current_market_price_usd_per_lb: Optional[float]
    forecast_h1_usd_per_lb: Optional[float]
    forecast_h3_usd_per_lb: Optional[float]
    forecast_confidence: Optional[str]

    # --- Metadata ---
    data_freshness: str
    data_quality_flags: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_quality_flags"] = list(self.data_quality_flags)
        return d


# ===========================================================================
# ASSESSMENT
# ===========================================================================

def assess_position(
    strategy_output: StrategyOutputV2,
    market_price_inputs: Optional[dict] = None,
    procurement_progress: Optional[dict] = None,
    as_of: Optional[date] = None,
) -> PositionSnapshot:
    """Build the single-source-of-truth PositionSnapshot for the current run.

    Args:
        strategy_output: StrategyOutputV2 -- the output of
            procurement_orchestrator.run_orchestration()["strategy_output"]
            (or procurement_strategy_engine.build_strategy_output_v2()
            directly). Reused as-is; nothing here recomputes inventory,
            mix, or reorder facts.
        market_price_inputs: Optional dict matching the shape of
            procurement_scenario_engine.fetch_price_inputs() --
            current_price_usd_per_lb, forecast_h1_usd_per_lb,
            forecast_h3_usd_per_lb, forecast_h1_bounds. This layer does not
            fetch prices itself; the caller decides the source (live fetch,
            cached value, or test fixture). If omitted, Market Context
            fields are None and a data-quality flag is set.
        procurement_progress: Optional dict with annual_target_tons,
            purchased_to_date_tons, current_month_purchased_tons. No
            existing module in this codebase tracks this data yet -- if
            omitted, Procurement Progress fields are None and a
            data-quality flag is set, rather than inventing new tracking
            logic in this phase.
        as_of: Date this snapshot represents; defaults to today.

    Returns:
        PositionSnapshot -- immutable, facts only.
    """
    so = strategy_output
    as_of = as_of or date.today()
    flags: list[str] = []

    # --- Inventory Position ---
    remaining_capacity_tons = MAX_STORAGE_CAPACITY_TONS - so.total_inventory_tons
    storage_utilization_pct = (
        so.total_inventory_tons / MAX_STORAGE_CAPACITY_TONS * 100.0
        if MAX_STORAGE_CAPACITY_TONS
        else 0.0
    )
    distance_total = so.total_inventory_tons - SAFETY_STOCK_TOTAL_TONS
    distance_local = so.local_inventory_tons - SAFETY_STOCK_LOCAL_TONS
    distance_imported = so.imported_inventory_tons - SAFETY_STOCK_IMPORTED_TONS

    if so.unknown_inventory_tons > 0:
        flags.append(
            f"UNCLASSIFIED_INVENTORY_PRESENT:{so.unknown_inventory_tons:.1f}_tons"
        )

    # --- Procurement Progress Position ---
    if procurement_progress:
        annual_target = procurement_progress.get("annual_target_tons")
        purchased = procurement_progress.get("purchased_to_date_tons")
        current_month = procurement_progress.get("current_month_purchased_tons")
        if annual_target is not None and purchased is not None:
            remaining = annual_target - purchased
            completion_pct = (purchased / annual_target * 100.0) if annual_target else None
        else:
            remaining = None
            completion_pct = None
            flags.append("PROCUREMENT_PROGRESS_INCOMPLETE")
    else:
        annual_target = purchased = remaining = completion_pct = current_month = None
        flags.append("PROCUREMENT_PROGRESS_UNAVAILABLE")

    # --- Mix Position ---
    local_mix_target_pct = LOCAL_MIX_TARGET * 100.0
    imported_mix_target_pct = IMPORTED_MIX_TARGET * 100.0
    local_mix_deviation = so.local_mix_pct - local_mix_target_pct
    imported_mix_deviation = so.imported_mix_pct - imported_mix_target_pct

    # --- Lead-Time Position ---
    local_doh_minus_lt = so.local_days_cover - LOCAL_LEAD_TIME_DAYS
    imported_doh_minus_lt = so.imported_days_cover - IMPORTED_LEAD_TIME_DAYS

    # --- Market Context ---
    if market_price_inputs:
        current_price = market_price_inputs.get("current_price_usd_per_lb")
        forecast_h1 = market_price_inputs.get("forecast_h1_usd_per_lb")
        forecast_h3 = market_price_inputs.get("forecast_h3_usd_per_lb")
        h1_bounds = market_price_inputs.get("forecast_h1_bounds")
        if current_price is not None and forecast_h1 is not None:
            lb, ub = h1_bounds if h1_bounds else (None, None)
            signal = compute_price_signal(current_price, forecast_h1, lb, ub)
            forecast_confidence = signal["confidence"]
        else:
            forecast_confidence = None
            flags.append("MARKET_CONTEXT_INCOMPLETE")
    else:
        current_price = forecast_h1 = forecast_h3 = forecast_confidence = None
        flags.append("MARKET_CONTEXT_UNAVAILABLE")

    return PositionSnapshot(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        total_inventory_tons=so.total_inventory_tons,
        local_inventory_tons=so.local_inventory_tons,
        imported_inventory_tons=so.imported_inventory_tons,
        total_doh=so.total_days_cover,
        local_doh=so.local_days_cover,
        imported_doh=so.imported_days_cover,
        min_stock_days=MIN_STOCK_DAYS,
        safety_stock_total_tons=SAFETY_STOCK_TOTAL_TONS,
        safety_stock_local_tons=SAFETY_STOCK_LOCAL_TONS,
        safety_stock_imported_tons=SAFETY_STOCK_IMPORTED_TONS,
        distance_above_safety_stock_total_tons=round(distance_total, 2),
        distance_above_safety_stock_local_tons=round(distance_local, 2),
        distance_above_safety_stock_imported_tons=round(distance_imported, 2),
        max_storage_capacity_tons=MAX_STORAGE_CAPACITY_TONS,
        remaining_storage_capacity_tons=round(remaining_capacity_tons, 2),
        storage_utilization_pct=round(storage_utilization_pct, 2),
        annual_target_tons=annual_target,
        purchased_to_date_tons=purchased,
        remaining_procurement_tons=round(remaining, 2) if remaining is not None else None,
        procurement_completion_pct=round(completion_pct, 2) if completion_pct is not None else None,
        current_month_purchased_tons=current_month,
        local_mix_pct=so.local_mix_pct,
        imported_mix_pct=so.imported_mix_pct,
        local_mix_target_pct=round(local_mix_target_pct, 2),
        imported_mix_target_pct=round(imported_mix_target_pct, 2),
        local_mix_deviation_pct_points=round(local_mix_deviation, 2),
        imported_mix_deviation_pct_points=round(imported_mix_deviation, 2),
        mix_within_tolerance=not so.rebalance_required,
        local_lead_time_days=LOCAL_LEAD_TIME_DAYS,
        imported_lead_time_days=IMPORTED_LEAD_TIME_DAYS,
        local_doh_minus_lead_time_days=round(local_doh_minus_lt, 2),
        imported_doh_minus_lead_time_days=round(imported_doh_minus_lt, 2),
        local_covers_lead_time=local_doh_minus_lt >= 0,
        imported_covers_lead_time=imported_doh_minus_lt >= 0,
        current_market_price_usd_per_lb=current_price,
        forecast_h1_usd_per_lb=forecast_h1,
        forecast_h3_usd_per_lb=forecast_h3,
        forecast_confidence=forecast_confidence,
        data_freshness=f"strategy_output.run_date={so.run_date}",
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PSE-3.0 Position Assessment Layer -- print a PositionSnapshot"
    )
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--input", default=None, help="Raw Oracle export (.xlsx)")
    args = parser.parse_args()

    from procurement_orchestrator import run_orchestration

    orch = run_orchestration(input_path=args.input, workbook_path=args.workbook)
    snapshot = assess_position(orch["strategy_output"])

    import json
    print(json.dumps(snapshot.to_dict(), indent=2))
