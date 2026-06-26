"""
procurement_strategy_engine.py
--------------------------------
PSE-3B — Core Procurement Strategy Engine.

Transforms the existing org x commodity BUY/HOLD/MONITOR logic
(procurement_engine.py, unmodified) into a portfolio-level LOCAL/IMPORTED
procurement decision engine, per the PSE-2 / PSE-2.7 approved design.

This module is purely a calculation layer -- no I/O, no Oracle access, no
forecast calls. It consumes:
    - origin_classifier.aggregate_by_origin() output (PSE-3A)
for inventory position, and exposes pure functions that a future
orchestration script can wire to clean_consumption.py / run_forecasts.py
outputs (PSE-3C onward). clean_inventory.py, clean_consumption.py, and
run_forecasts.py are not imported or modified by this module.

Scope (confirmed PSE-2 business rules, approved in PSE-2.7):
    Daily consumption (total)    = 110 tons/day
    Local daily consumption      =  49.5 tons/day (45% of total)
    Imported daily consumption   =  60.5 tons/day (55% of total)
    Minimum stock policy         = 25 days
    Local lead time              = 10 days
    Imported lead time           = 90 days
    Local mix target             = 45%
    Imported mix target          = 55%
    Local reorder point  (ROP_L) = (25 + 10) x 49.5 = 1,732.5 -> 1,733 tons
    Imported reorder point(ROP_I)= (25 + 90) x 60.5 = 6,957.5 -> 6,958 tons

Two parameters used below are engineering defaults, not confirmed business
rules (flagged the same way OWT/MOQ were flagged in PSE-2.5/2.7 -- they do
not change the structural math, only the WATCH-band sensitivity):
    WATCH_BUFFER_PCT          = 0.15  (15% above ROP triggers WATCH, not SAFE)
    MIX_TOLERANCE_PCT_POINTS  = 10.0  (mix must drift >10 points to rebalance)

Decision logic is 100% deterministic threshold comparison. No ML, no LLM,
no fuzzy heuristics -- every branch in classify_action() is a documented,
auditable rule (see DECISION TREE in the module docstring of Part 4 below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional

import pandas as pd


def _round_half_up(value: float) -> int:
    """Round-half-up (1732.5 -> 1733), unlike Python's built-in round() which
    uses banker's rounding (round-half-to-even) and would give 1732.
    """
    return math.floor(value + 0.5)

# ---------------------------------------------------------------------------
# Constants -- confirmed business rules (PSE-1 / PSE-2 / PSE-2.7 approved)
# ---------------------------------------------------------------------------

DAILY_CONSUMPTION_TOTAL = 110.0
LOCAL_MIX_TARGET = 0.45
IMPORTED_MIX_TARGET = 0.55

DAILY_CONSUMPTION_LOCAL = DAILY_CONSUMPTION_TOTAL * LOCAL_MIX_TARGET      # 49.5
DAILY_CONSUMPTION_IMPORTED = DAILY_CONSUMPTION_TOTAL * IMPORTED_MIX_TARGET  # 60.5

MIN_STOCK_DAYS = 25
LOCAL_LEAD_TIME_DAYS = 10
IMPORTED_LEAD_TIME_DAYS = 90

SAFETY_STOCK_TOTAL_TONS = MIN_STOCK_DAYS * DAILY_CONSUMPTION_TOTAL  # 2,750

# Safety stock allocated proportionally per source (PSE-2 Part 2.2,
# "conservative" interpretation -- formally ratified in PSE-2.7 Condition 3).
SAFETY_STOCK_LOCAL_TONS = MIN_STOCK_DAYS * DAILY_CONSUMPTION_LOCAL       # 1,237.5
SAFETY_STOCK_IMPORTED_TONS = MIN_STOCK_DAYS * DAILY_CONSUMPTION_IMPORTED  # 1,512.5

# Reorder points -- approved V1 values (PSE-2.7). Computed here from the
# formula and asserted against the approved rounded constants so any future
# drift between formula and approved value fails loudly rather than silently.
LOCAL_ROP_TONS = _round_half_up((MIN_STOCK_DAYS + LOCAL_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_LOCAL)        # 1,733
IMPORTED_ROP_TONS = _round_half_up((MIN_STOCK_DAYS + IMPORTED_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_IMPORTED)  # 6,958
assert LOCAL_ROP_TONS == 1733, f"Local ROP drifted from approved value: {LOCAL_ROP_TONS}"
assert IMPORTED_ROP_TONS == 6958, f"Imported ROP drifted from approved value: {IMPORTED_ROP_TONS}"

# Engineering defaults -- NOT confirmed business rules. Tunable without
# changing structural reorder-point math. See module docstring.
WATCH_BUFFER_PCT = 0.15
MIX_TOLERANCE_PCT_POINTS = 10.0

# ---------------------------------------------------------------------------
# Status / action vocabularies
# ---------------------------------------------------------------------------

STATUS_SAFE = "SAFE"
STATUS_WATCH = "WATCH"
STATUS_REORDER = "REORDER"
STATUS_CRITICAL = "CRITICAL"

ACTION_BUY_LOCAL = "BUY_LOCAL"
ACTION_BUY_IMPORTED = "BUY_IMPORTED"
ACTION_BUY_MIXED = "BUY_MIXED"
ACTION_HOLD = "HOLD"
ACTION_WATCH = "WATCH"
ACTION_CRITICAL = "CRITICAL"


# ===========================================================================
# PART 1 -- INVENTORY POSITION ENGINE
# ===========================================================================

def compute_inventory_position(
    origin_summary_df: pd.DataFrame,
    daily_consumption_total: float = DAILY_CONSUMPTION_TOTAL,
    daily_consumption_local: float = DAILY_CONSUMPTION_LOCAL,
    daily_consumption_imported: float = DAILY_CONSUMPTION_IMPORTED,
) -> dict:
    """Compute the portfolio-level inventory position from classified inventory.

    Args:
        origin_summary_df: Output of origin_classifier.aggregate_by_origin()
                            (PSE-3A) -- columns: org_name, origin, tons.
                            Rows with origin == "UNKNOWN" are included in the
                            physical total (they are real stock sitting in
                            the warehouse) but excluded from local/imported
                            mix and reorder calculations, since their source
                            cannot be determined -- per the "do not guess"
                            rule carried over from PSE-3A.
        daily_consumption_total/local/imported: Planning rates (tons/day).

    Returns:
        dict with keys:
            total_inventory_tons      -- physical total (LOCAL+IMPORTED+UNKNOWN)
            local_inventory_tons
            imported_inventory_tons
            unknown_inventory_tons    -- excluded from mix/cover-by-source
            classified_inventory_tons -- LOCAL+IMPORTED (denominator for mix %)
            local_mix_pct             -- % of CLASSIFIED inventory that is local
            imported_mix_pct          -- % of CLASSIFIED inventory that is imported
            local_days_cover          -- local_tons / 49.5
            imported_days_cover       -- imported_tons / 60.5
            total_days_cover          -- total physical tons / 110 (true stockout horizon)
    """
    if origin_summary_df is None or origin_summary_df.empty:
        local_tons = imported_tons = unknown_tons = 0.0
    else:
        grouped = origin_summary_df.groupby("origin")["tons"].sum()
        local_tons = float(grouped.get("LOCAL", 0.0))
        imported_tons = float(grouped.get("IMPORTED", 0.0))
        unknown_tons = float(grouped.get("UNKNOWN", 0.0))

    classified_tons = local_tons + imported_tons
    total_tons = classified_tons + unknown_tons

    local_mix_pct = (local_tons / classified_tons * 100.0) if classified_tons else 0.0
    imported_mix_pct = (imported_tons / classified_tons * 100.0) if classified_tons else 0.0

    local_days_cover = (local_tons / daily_consumption_local) if daily_consumption_local else 0.0
    imported_days_cover = (
        imported_tons / daily_consumption_imported if daily_consumption_imported else 0.0
    )
    total_days_cover = (total_tons / daily_consumption_total) if daily_consumption_total else 0.0

    return {
        "total_inventory_tons": round(total_tons, 2),
        "local_inventory_tons": round(local_tons, 2),
        "imported_inventory_tons": round(imported_tons, 2),
        "unknown_inventory_tons": round(unknown_tons, 2),
        "classified_inventory_tons": round(classified_tons, 2),
        "local_mix_pct": round(local_mix_pct, 2),
        "imported_mix_pct": round(imported_mix_pct, 2),
        "local_days_cover": round(local_days_cover, 2),
        "imported_days_cover": round(imported_days_cover, 2),
        "total_days_cover": round(total_days_cover, 2),
    }


# ===========================================================================
# PART 2 -- REORDER ENGINE
# ===========================================================================

def _classify_supply_status(
    inventory_tons: float,
    safety_stock_tons: float,
    reorder_point_tons: float,
    watch_buffer_pct: float = WATCH_BUFFER_PCT,
) -> str:
    """Deterministic four-band status classifier for a single source.

    Bands (in order, first match wins):
        CRITICAL : inventory_tons <= safety_stock_tons
                   (below the pure 25-day safety floor for this source --
                   emergency; lead time alone cannot save you)
        REORDER  : safety_stock_tons < inventory_tons <= reorder_point_tons
                   (above the floor, but at/below the point where a new
                   order must be placed today to avoid breaching the floor
                   before the next delivery arrives)
        WATCH    : reorder_point_tons < inventory_tons
                   <= reorder_point_tons * (1 + watch_buffer_pct)
                   (above the reorder point but within the buffer band --
                   getting close, monitor closely)
        SAFE     : inventory_tons > reorder_point_tons * (1 + watch_buffer_pct)
    """
    if inventory_tons <= safety_stock_tons:
        return STATUS_CRITICAL
    if inventory_tons <= reorder_point_tons:
        return STATUS_REORDER
    if inventory_tons <= reorder_point_tons * (1.0 + watch_buffer_pct):
        return STATUS_WATCH
    return STATUS_SAFE


def compute_reorder_triggers(
    local_inventory_tons: float,
    imported_inventory_tons: float,
) -> dict:
    """Compute reorder status for local and imported cotton.

    Uses the approved V1 reorder points (PSE-2.7):
        Local ROP    = 1,733 tons
        Imported ROP = 6,958 tons

    Returns:
        dict with keys:
            local_reorder_trigger     -- 1733 (constant, returned for traceability)
            imported_reorder_trigger  -- 6958 (constant, returned for traceability)
            local_safety_stock_tons
            imported_safety_stock_tons
            local_status     -- SAFE / WATCH / REORDER / CRITICAL
            imported_status  -- SAFE / WATCH / REORDER / CRITICAL
    """
    local_status = _classify_supply_status(
        local_inventory_tons, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS
    )
    imported_status = _classify_supply_status(
        imported_inventory_tons, SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS
    )

    return {
        "local_reorder_trigger": LOCAL_ROP_TONS,
        "imported_reorder_trigger": IMPORTED_ROP_TONS,
        "local_safety_stock_tons": SAFETY_STOCK_LOCAL_TONS,
        "imported_safety_stock_tons": SAFETY_STOCK_IMPORTED_TONS,
        "local_status": local_status,
        "imported_status": imported_status,
    }


# ===========================================================================
# PART 3 -- MIX ANALYSIS
# ===========================================================================

def compute_mix_gap(
    local_inventory_tons: float,
    imported_inventory_tons: float,
    local_mix_target: float = LOCAL_MIX_TARGET,
    imported_mix_target: float = IMPORTED_MIX_TARGET,
    tolerance_pct_points: float = MIX_TOLERANCE_PCT_POINTS,
) -> dict:
    """Compare current local/imported holdings against the 45/55 mix target.

    local_gap_tons / imported_gap_tons are signed:
        positive -> under-weight (need to buy more of this source to reach target)
        negative -> over-weight  (currently holding more than the target share)

    rebalance_required fires only when the deviation exceeds
    +/- tolerance_pct_points (default 10 percentage points) -- this avoids
    reacting to normal day-to-day fluctuation from lumpy import arrivals
    (see PSE-2.5 Weakness 3).

    Returns:
        dict with keys:
            classified_inventory_tons
            local_mix_pct / imported_mix_pct
            target_local_tons / target_imported_tons
            local_gap_tons / imported_gap_tons
            overweight_local / overweight_imported   (bool)
            rebalance_required                        (bool)
    """
    classified_tons = local_inventory_tons + imported_inventory_tons

    if classified_tons <= 0:
        return {
            "classified_inventory_tons": 0.0,
            "local_mix_pct": 0.0,
            "imported_mix_pct": 0.0,
            "target_local_tons": 0.0,
            "target_imported_tons": 0.0,
            "local_gap_tons": 0.0,
            "imported_gap_tons": 0.0,
            "overweight_local": False,
            "overweight_imported": False,
            "rebalance_required": False,
        }

    local_mix_pct = local_inventory_tons / classified_tons * 100.0
    imported_mix_pct = imported_inventory_tons / classified_tons * 100.0

    target_local_tons = classified_tons * local_mix_target
    target_imported_tons = classified_tons * imported_mix_target

    local_gap_tons = target_local_tons - local_inventory_tons
    imported_gap_tons = target_imported_tons - imported_inventory_tons

    overweight_local = local_mix_pct > (local_mix_target * 100.0 + tolerance_pct_points)
    overweight_imported = imported_mix_pct > (imported_mix_target * 100.0 + tolerance_pct_points)

    return {
        "classified_inventory_tons": round(classified_tons, 2),
        "local_mix_pct": round(local_mix_pct, 2),
        "imported_mix_pct": round(imported_mix_pct, 2),
        "target_local_tons": round(target_local_tons, 2),
        "target_imported_tons": round(target_imported_tons, 2),
        "local_gap_tons": round(local_gap_tons, 2),
        "imported_gap_tons": round(imported_gap_tons, 2),
        "overweight_local": overweight_local,
        "overweight_imported": overweight_imported,
        "rebalance_required": overweight_local or overweight_imported,
    }


# ===========================================================================
# PART 4 -- PROCUREMENT ACTION ENGINE
# ===========================================================================
#
# DECISION TREE (evaluated top to bottom -- first match wins; deterministic
# threshold logic only, no ML / LLM / fuzzy heuristics):
#
#   1. CRITICAL  : local_status == CRITICAL OR imported_status == CRITICAL
#                  -> security overrides everything else, including mix.
#
#   2. BUY_MIXED : local_status == REORDER AND imported_status == REORDER
#                  -> both sources need ordering simultaneously.
#
#   3. BUY_LOCAL : local_status == REORDER (imported_status != REORDER)
#                  -> only local needs ordering.
#
#   4. BUY_IMPORTED : imported_status == REORDER (local_status != REORDER)
#                  -> only imported needs ordering.
#
#   5. WATCH     : local_status == WATCH OR imported_status == WATCH
#                  -> neither source requires an order yet, but at least one
#                     is inside its buffer band; flag for closer monitoring.
#                     (Evaluated only after REORDER/CRITICAL are ruled out.)
#
#   6. Mix correction (both sources SAFE, no urgency):
#        BUY_LOCAL    if rebalance_required AND overweight_imported
#        BUY_IMPORTED if rebalance_required AND overweight_local
#                  -> structural/security requirements (1-5) always take
#                     priority over mix optimization, per the PSE-2.5/2.6
#                     constraint hierarchy (Security > Structure > Mix).
#
#   7. HOLD      : both sources SAFE and mix within tolerance.
#                  -> default state when nothing above fires.
# ===========================================================================

def classify_action(
    reorder_result: dict,
    mix_result: dict,
) -> dict:
    """Deterministic procurement action classification.

    Args:
        reorder_result: Output of compute_reorder_triggers().
        mix_result:      Output of compute_mix_gap().

    Returns:
        dict with keys:
            action      -- one of BUY_LOCAL / BUY_IMPORTED / BUY_MIXED /
                            HOLD / WATCH / CRITICAL
            reason      -- short human-readable justification (audit trail)
            rule_fired  -- which numbered rule (1-7) in the decision tree fired
    """
    local_status = reorder_result["local_status"]
    imported_status = reorder_result["imported_status"]
    rebalance_required = mix_result["rebalance_required"]
    overweight_local = mix_result["overweight_local"]
    overweight_imported = mix_result["overweight_imported"]

    # Rule 1 -- CRITICAL overrides everything
    if local_status == STATUS_CRITICAL or imported_status == STATUS_CRITICAL:
        breached = []
        if local_status == STATUS_CRITICAL:
            breached.append("local")
        if imported_status == STATUS_CRITICAL:
            breached.append("imported")
        return {
            "action": ACTION_CRITICAL,
            "reason": f"Inventory below safety stock floor for: {', '.join(breached)}.",
            "rule_fired": 1,
        }

    # Rule 2 -- both at reorder point
    if local_status == STATUS_REORDER and imported_status == STATUS_REORDER:
        return {
            "action": ACTION_BUY_MIXED,
            "reason": "Both local and imported inventory at or below their reorder points.",
            "rule_fired": 2,
        }

    # Rule 3 -- local only at reorder point
    if local_status == STATUS_REORDER:
        return {
            "action": ACTION_BUY_LOCAL,
            "reason": "Local inventory at or below reorder point (1,733 tons).",
            "rule_fired": 3,
        }

    # Rule 4 -- imported only at reorder point
    if imported_status == STATUS_REORDER:
        return {
            "action": ACTION_BUY_IMPORTED,
            "reason": "Imported inventory at or below reorder point (6,958 tons).",
            "rule_fired": 4,
        }

    # Rule 5 -- watch band
    if local_status == STATUS_WATCH or imported_status == STATUS_WATCH:
        watching = []
        if local_status == STATUS_WATCH:
            watching.append("local")
        if imported_status == STATUS_WATCH:
            watching.append("imported")
        return {
            "action": ACTION_WATCH,
            "reason": f"Inventory approaching reorder point for: {', '.join(watching)}.",
            "rule_fired": 5,
        }

    # Rule 6 -- mix correction (only reached when both sources are SAFE)
    if rebalance_required and overweight_imported:
        return {
            "action": ACTION_BUY_LOCAL,
            "reason": "Both sources secure; portfolio mix overweight imported "
                      "(> tolerance). Next purchase should be local to restore 45/55 mix.",
            "rule_fired": 6,
        }
    if rebalance_required and overweight_local:
        return {
            "action": ACTION_BUY_IMPORTED,
            "reason": "Both sources secure; portfolio mix overweight local "
                      "(> tolerance). Next purchase should be imported to restore 45/55 mix.",
            "rule_fired": 6,
        }

    # Rule 7 -- default
    return {
        "action": ACTION_HOLD,
        "reason": "Both sources above reorder points and mix within tolerance. "
                  "No procurement action required.",
        "rule_fired": 7,
    }


# ===========================================================================
# PART 5 -- OUTPUT SCHEMA (strategy_output_v2)
# ===========================================================================

@dataclass
class StrategyOutputV2:
    """PSE-3B output schema -- one row per engine run (portfolio level).

    Deliberately excludes anything from PSE-2.5 (staging/horizons) or
    PSE-2.6 (scenarios/timing) -- those are out of scope for this phase.
    """
    run_date: str

    # Inventory metrics (Part 1)
    total_inventory_tons: float
    local_inventory_tons: float
    imported_inventory_tons: float
    unknown_inventory_tons: float
    classified_inventory_tons: float
    local_mix_pct: float
    imported_mix_pct: float
    local_days_cover: float
    imported_days_cover: float
    total_days_cover: float

    # Reorder metrics (Part 2)
    local_reorder_trigger: float
    imported_reorder_trigger: float
    local_safety_stock_tons: float
    imported_safety_stock_tons: float
    local_status: str
    imported_status: str

    # Mix metrics (Part 3)
    target_local_tons: float
    target_imported_tons: float
    local_gap_tons: float
    imported_gap_tons: float
    overweight_local: bool
    overweight_imported: bool
    rebalance_required: bool

    # Action classification (Part 4)
    action: str
    reason: str
    rule_fired: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_strategy_output_v2(
    origin_summary_df: pd.DataFrame,
    run_date: Optional[str] = None,
) -> StrategyOutputV2:
    """Orchestrator: run Parts 1-4 and assemble the strategy_output_v2 row.

    Args:
        origin_summary_df: Output of origin_classifier.aggregate_by_origin().
        run_date:           ISO date string; defaults to today.

    Returns:
        StrategyOutputV2 instance (call .to_dict() for a flat dict, or wrap
        in pd.DataFrame([...]) for tabular output across multiple runs).
    """
    position = compute_inventory_position(origin_summary_df)
    reorder = compute_reorder_triggers(
        position["local_inventory_tons"], position["imported_inventory_tons"]
    )
    mix = compute_mix_gap(
        position["local_inventory_tons"], position["imported_inventory_tons"]
    )
    action = classify_action(reorder, mix)

    return StrategyOutputV2(
        run_date=run_date or date.today().isoformat(),
        total_inventory_tons=position["total_inventory_tons"],
        local_inventory_tons=position["local_inventory_tons"],
        imported_inventory_tons=position["imported_inventory_tons"],
        unknown_inventory_tons=position["unknown_inventory_tons"],
        classified_inventory_tons=position["classified_inventory_tons"],
        local_mix_pct=position["local_mix_pct"],
        imported_mix_pct=position["imported_mix_pct"],
        local_days_cover=position["local_days_cover"],
        imported_days_cover=position["imported_days_cover"],
        total_days_cover=position["total_days_cover"],
        local_reorder_trigger=reorder["local_reorder_trigger"],
        imported_reorder_trigger=reorder["imported_reorder_trigger"],
        local_safety_stock_tons=reorder["local_safety_stock_tons"],
        imported_safety_stock_tons=reorder["imported_safety_stock_tons"],
        local_status=reorder["local_status"],
        imported_status=reorder["imported_status"],
        target_local_tons=mix["target_local_tons"],
        target_imported_tons=mix["target_imported_tons"],
        local_gap_tons=mix["local_gap_tons"],
        imported_gap_tons=mix["imported_gap_tons"],
        overweight_local=mix["overweight_local"],
        overweight_imported=mix["overweight_imported"],
        rebalance_required=mix["rebalance_required"],
        action=action["action"],
        reason=action["reason"],
        rule_fired=action["rule_fired"],
    )
