"""
procurement_target_engine.py
------------------------------
PSE-3.1 -- Strategy Target Definition Layer.

Per the frozen architecture (PSE-2.1 -> PSE-2.4):

    Procurement Position Assessment -> Strategy Target Definition ->
    Gap Analysis -> Portfolio Optimization -> Opportunity Window Classifier
    -> Tranche Allocation -> Tactical Execution -> Executive Narrative

This module implements ONLY the second box. It answers a single question --
"Where SHOULD we be?" -- purely from business policy. It is the mirror
image of procurement_position_engine.py (PSE-3.0):

    PositionSnapshot = reality   (depends on current inventory/market data)
    TargetSnapshot    = objective (depends on business policy only)
    Gap Analysis       = the difference (not implemented here -- Sprint 3)

This module therefore:
    - never imports or reads PositionSnapshot / procurement_position_engine
    - never reads live inventory, mix, or market data
    - never computes a shortage, surplus, gap, or procurement quantity
    - never calls BUY/HOLD logic, optimization, forecasting, or
      recommendation engines

Reuse, not duplication:
    - Locked business constants (minimum stock days, mix targets, lead
      times) are imported from procurement_strategy_engine.py -- the same
      single source of truth used by PSE-3.0.
    - MAX_STORAGE_CAPACITY_TONS is imported from procurement_orchestrator.py.

No production data source exists today for an annual procurement target,
a monthly procurement schedule, a desired flexibility/dry-powder measure,
desired strategic posture, desired capacity utilisation, or a desired
storage buffer (confirmed during PSE-3.0 research and unchanged since).
Per "never fabricate data," every one of these is accepted as an optional
caller-supplied input. When omitted, the corresponding TargetSnapshot
field is None and a data-quality flag records the gap.

Frozen constraint from PSE-2.4: the Target must never be set below the
hard survival Floor (MIN_STOCK_DAYS). define_strategy_target() enforces
this by raising ValueError rather than silently clamping or flagging --
this is a structural invariant of the frozen architecture, not a
runtime data-quality concern.
"""

from __future__ import annotations

import calendar
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_orchestrator import MAX_STORAGE_CAPACITY_TONS
from procurement_strategy_engine import (
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_MIX_TARGET,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_MIX_TARGET,
    MIN_STOCK_DAYS,
)

# ---------------------------------------------------------------------------
# Traceability labels (not business rules -- version tags only)
# ---------------------------------------------------------------------------

BUSINESS_RULE_VERSION = "PSE-2.7"      # vintage of the locked constants reused above
CONFIGURATION_VERSION = "PSE-3.1"      # this layer's own implementation version

VALID_STRATEGIC_POSTURES = ("DEFENSIVE", "BALANCED", "OPPORTUNISTIC")


# ===========================================================================
# OUTPUT SCHEMA -- TargetSnapshot
# ===========================================================================

@dataclass(frozen=True)
class TargetSnapshot:
    """Immutable, policy-only snapshot of the desired procurement state.

    Contains no reference to current inventory, market prices, or actual
    purchases. Two TargetSnapshots built for the same as_of date with the
    same inputs are always identical -- it is a pure function of business
    policy and the calendar.
    """

    as_of: str
    generated_at: str

    # --- Inventory Targets ---
    desired_inventory_coverage_days: float
    inventory_floor_days: int
    maximum_inventory_tons: float
    desired_local_inventory_pct: float
    desired_imported_inventory_pct: float

    # --- Procurement Targets (facts only when annual_target_tons supplied) ---
    annual_procurement_target_tons: Optional[float]
    target_procurement_progress_pct: Optional[float]
    monthly_target_procurement_tons: Optional[float]
    remaining_planned_procurement_tons: Optional[float]

    # --- Portfolio Targets ---
    desired_local_pct: float
    desired_imported_pct: float
    desired_flexibility: Optional[float]
    desired_strategic_posture: Optional[str]

    # --- Lead Time Targets ---
    local_planning_horizon_days: int
    imported_planning_horizon_days: int

    # --- Operational Targets ---
    desired_capacity_utilization_pct: Optional[float]
    desired_storage_buffer_tons: Optional[float]

    # --- Metadata ---
    business_rule_version: str
    configuration_version: str
    data_quality_flags: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_quality_flags"] = list(self.data_quality_flags)
        return d


# ===========================================================================
# DEFINITION
# ===========================================================================

def define_strategy_target(
    as_of: Optional[date] = None,
    desired_coverage_days: Optional[float] = None,
    annual_target_tons: Optional[float] = None,
    desired_flexibility: Optional[float] = None,
    desired_strategic_posture: Optional[str] = None,
    desired_capacity_utilization_pct: Optional[float] = None,
    desired_storage_buffer_tons: Optional[float] = None,
) -> TargetSnapshot:
    """Build the single-source-of-truth TargetSnapshot for the current run.

    Args:
        as_of: Date this target represents; defaults to today. Used only
            for calendar-based pacing of the (optional) annual procurement
            plan -- never to look up actual inventory or purchases.
        desired_coverage_days: Desired inventory coverage, in days. Must be
            >= MIN_STOCK_DAYS (the confirmed survival floor) -- raises
            ValueError otherwise, per the PSE-2.4 frozen constraint that
            the Target may never fall below the Floor. Defaults to the
            floor itself when omitted (flagged, since "floor" and "desired
            operating level" are not confirmed to be the same business
            parameter).
        annual_target_tons: Optional annual procurement plan total. No
            production data source for this exists yet (see module
            docstring) -- caller-supplied or omitted entirely.
        desired_flexibility: Optional dry-powder / optionality measure
            (0..1). Not a confirmed business rule -- caller-supplied.
        desired_strategic_posture: One of DEFENSIVE / BALANCED /
            OPPORTUNISTIC, or None. Not a confirmed business rule --
            caller-supplied.
        desired_capacity_utilization_pct: Optional target storage
            utilisation. Not a confirmed business rule -- caller-supplied.
        desired_storage_buffer_tons: Optional target headroom below
            MAX_STORAGE_CAPACITY_TONS. Not a confirmed business rule --
            caller-supplied.

    Returns:
        TargetSnapshot -- immutable, policy-only.

    Raises:
        ValueError: if desired_coverage_days < MIN_STOCK_DAYS, or
            desired_strategic_posture is not a recognised value.
    """
    as_of = as_of or date.today()
    flags: list[str] = []

    # --- Inventory Targets ---
    if desired_coverage_days is None:
        coverage_days = float(MIN_STOCK_DAYS)
        flags.append("DESIRED_COVERAGE_DEFAULTED_TO_FLOOR")
    else:
        if desired_coverage_days < MIN_STOCK_DAYS:
            raise ValueError(
                f"desired_coverage_days ({desired_coverage_days}) cannot be below "
                f"the confirmed survival floor of MIN_STOCK_DAYS ({MIN_STOCK_DAYS}). "
                "Per PSE-2.4, the Target must never be set below the Floor."
            )
        coverage_days = float(desired_coverage_days)

    local_mix_pct = round(LOCAL_MIX_TARGET * 100.0, 2)
    imported_mix_pct = round(IMPORTED_MIX_TARGET * 100.0, 2)

    # --- Procurement Targets ---
    if annual_target_tons is not None:
        days_in_year = 366 if calendar.isleap(as_of.year) else 365
        elapsed_fraction = as_of.timetuple().tm_yday / days_in_year
        target_progress_pct = round(elapsed_fraction * 100.0, 2)
        monthly_target_tons = round(annual_target_tons / 12.0, 2)
        remaining_planned_tons = round(annual_target_tons * (1.0 - elapsed_fraction), 2)
        flags.append("CALENDAR_YEAR_PACING_ASSUMED")
        flags.append("UNIFORM_MONTHLY_PACING_ASSUMED")
    else:
        target_progress_pct = None
        monthly_target_tons = None
        remaining_planned_tons = None
        flags.append("ANNUAL_PROCUREMENT_TARGET_NOT_CONFIGURED")

    # --- Portfolio Targets ---
    if desired_strategic_posture is not None:
        posture = desired_strategic_posture.upper()
        if posture not in VALID_STRATEGIC_POSTURES:
            raise ValueError(
                f"desired_strategic_posture must be one of {VALID_STRATEGIC_POSTURES}, "
                f"got {desired_strategic_posture!r}."
            )
    else:
        posture = None
        flags.append("STRATEGIC_POSTURE_NOT_CONFIGURED")

    if desired_flexibility is None:
        flags.append("DESIRED_FLEXIBILITY_NOT_CONFIGURED")

    # --- Operational Targets ---
    if desired_capacity_utilization_pct is None:
        flags.append("DESIRED_CAPACITY_UTILIZATION_NOT_CONFIGURED")
    if desired_storage_buffer_tons is None:
        flags.append("DESIRED_STORAGE_BUFFER_NOT_CONFIGURED")

    return TargetSnapshot(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        desired_inventory_coverage_days=coverage_days,
        inventory_floor_days=MIN_STOCK_DAYS,
        maximum_inventory_tons=MAX_STORAGE_CAPACITY_TONS,
        desired_local_inventory_pct=local_mix_pct,
        desired_imported_inventory_pct=imported_mix_pct,
        annual_procurement_target_tons=annual_target_tons,
        target_procurement_progress_pct=target_progress_pct,
        monthly_target_procurement_tons=monthly_target_tons,
        remaining_planned_procurement_tons=remaining_planned_tons,
        desired_local_pct=local_mix_pct,
        desired_imported_pct=imported_mix_pct,
        desired_flexibility=desired_flexibility,
        desired_strategic_posture=posture,
        local_planning_horizon_days=LOCAL_LEAD_TIME_DAYS,
        imported_planning_horizon_days=IMPORTED_LEAD_TIME_DAYS,
        desired_capacity_utilization_pct=desired_capacity_utilization_pct,
        desired_storage_buffer_tons=desired_storage_buffer_tons,
        business_rule_version=BUSINESS_RULE_VERSION,
        configuration_version=CONFIGURATION_VERSION,
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="PSE-3.1 Strategy Target Definition Layer -- print a TargetSnapshot"
    )
    parser.add_argument("--coverage-days", type=float, default=None)
    parser.add_argument("--annual-target-tons", type=float, default=None)
    parser.add_argument("--posture", default=None, choices=[None, *VALID_STRATEGIC_POSTURES])
    args = parser.parse_args()

    snapshot = define_strategy_target(
        desired_coverage_days=args.coverage_days,
        annual_target_tons=args.annual_target_tons,
        desired_strategic_posture=args.posture,
    )
    print(json.dumps(snapshot.to_dict(), indent=2))
