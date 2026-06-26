"""
procurement_planning_engine.py
---------------------------------
PSE-3D — Procurement Planning Engine.

Transforms the inventory signals produced by procurement_strategy_engine.py
(PSE-3B) and procurement_orchestrator.py (PSE-3C) into executable
procurement plans: how much to buy, when to buy it, local vs imported,
and the expected cost impact of buying now vs waiting.

Objective (explicit, not trading): minimize procurement cost while
maintaining inventory security. Every formula below treats the purchase
quantity as a structural requirement (driven by inventory position, lead
time, and the 45/55 mix policy) -- price only ever influences WHEN within
an already-safe window a required purchase happens, never WHETHER it
happens, and never pushes a purchase later than the point inventory
security allows. This mirrors the constraint hierarchy approved across
PSE-2 / PSE-2.5 / PSE-2.6 / PSE-2.7: Security > Structure > Mix > Timing.

This module creates NO new forecasting model. Part 3 (compute_price_signal)
consumes the existing output of run_forecasts.py only -- the
commodity_prices / prediction_records schema documented in that file:
    commodity, model_name, as_of_date, horizon_months (1, 3, or 6),
    target_date, predicted_value, lower_bound, upper_bound, unit
(see run_forecasts.py lines 708-741 for the authoritative schema).

Does not modify clean_inventory.py, clean_consumption.py, inventory_data.py,
consumption_data.py, update_strategy_workbook.py, procurement_engine.py,
origin_classifier.py, procurement_strategy_engine.py, or
procurement_orchestrator.py. Imports the public functions/constants of
procurement_strategy_engine.py only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import (
    DAILY_CONSUMPTION_IMPORTED,
    DAILY_CONSUMPTION_LOCAL,
    DAILY_CONSUMPTION_TOTAL,
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_MIX_TARGET,
    IMPORTED_ROP_TONS,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_MIX_TARGET,
    LOCAL_ROP_TONS,
    MIN_STOCK_DAYS,
    SAFETY_STOCK_TOTAL_TONS,
    STATUS_CRITICAL,
    STATUS_REORDER,
    STATUS_SAFE,
    STATUS_WATCH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 1 metric ton = 1,000 kg = 2,204.622 lb. ICE Cotton No. 2 is quoted in
# USD/lb (and c/lb); cotton inventory in this system is tracked in metric
# tons. This is the single unit-conversion constant for the savings engine.
LB_PER_METRIC_TON = 2204.622

# Price-signal classification threshold (carried forward unchanged from the
# PSE-2.6 design, Part 4.3 -- "3% minimum to act on a signal; below this,
# noise > signal"). Engineering default, not a confirmed business rule.
PRICE_SIGNAL_THRESHOLD_PCT = 3.0

# Confidence-spread filter (PSE-2.6 Part 4.3): forecast bound spread wider
# than this fraction of the predicted value downgrades signal confidence.
FORECAST_SPREAD_LOW_CONFIDENCE_PCT = 15.0

# Default carrying-cost rate when none is supplied -- SBP policy rate is the
# live-fetched proxy already used elsewhere in this codebase (market_inputs.py).
# Flagged, as in every prior phase, as an engineering default standing in for
# the company's true cost of capital until confirmed by Finance.
DEFAULT_CARRYING_COST_ANNUAL_RATE = 0.105

PRICE_RISING = "PRICE_RISING"
PRICE_FALLING = "PRICE_FALLING"
PRICE_NEUTRAL = "PRICE_NEUTRAL"

URGENCY_CRITICAL = "CRITICAL"
URGENCY_HIGH = "HIGH"
URGENCY_MEDIUM = "MEDIUM"
URGENCY_LOW = "LOW"

ACTION_BUY_NOW = "BUY_NOW"
ACTION_BUY_LOCAL_NOW = "BUY_LOCAL_NOW"
ACTION_BUY_IMPORTED_NOW = "BUY_IMPORTED_NOW"
ACTION_BUY_MIXED_NOW = "BUY_MIXED_NOW"
ACTION_WAIT = "WAIT"
ACTION_DEFER_LOCAL = "DEFER_LOCAL"
ACTION_DEFER_IMPORTED = "DEFER_IMPORTED"
ACTION_STAGGER_PURCHASE = "STAGGER_PURCHASE"
ACTION_ACCELERATE_IMPORT_ORDER = "ACCELERATE_IMPORT_ORDER"

_STATUS_TO_URGENCY = {
    STATUS_CRITICAL: URGENCY_CRITICAL,
    STATUS_REORDER: URGENCY_HIGH,
    STATUS_WATCH: URGENCY_MEDIUM,
    STATUS_SAFE: URGENCY_LOW,
}

# Action priority for selecting the single "primary" headline action when
# local and imported recommendations differ -- lower number = more urgent,
# wins. Used only to pick a top-line summary action; the full per-source
# detail is always returned alongside it.
_ACTION_PRIORITY = {
    ACTION_ACCELERATE_IMPORT_ORDER: 1,
    ACTION_BUY_MIXED_NOW: 2,
    ACTION_BUY_LOCAL_NOW: 3,
    ACTION_BUY_IMPORTED_NOW: 3,
    ACTION_DEFER_LOCAL: 4,
    ACTION_DEFER_IMPORTED: 4,
    ACTION_STAGGER_PURCHASE: 5,
    ACTION_WAIT: 6,
}


# ===========================================================================
# PART 1 -- PROCUREMENT REQUIREMENT ENGINE
# ===========================================================================

def compute_procurement_requirement(
    local_inventory_tons: float,
    imported_inventory_tons: float,
    total_inventory_tons: Optional[float] = None,
    max_storage_capacity_tons: float = 45_000.0,
    daily_consumption_total: float = DAILY_CONSUMPTION_TOTAL,
    local_mix_target: float = LOCAL_MIX_TARGET,
    imported_mix_target: float = IMPORTED_MIX_TARGET,
    future_horizon_days: int = 90,
) -> dict:
    """How much should I buy, right now and across the planning horizon?

    Three components, computed in order (documented formulas):

    1. STRUCTURAL DEFICIT (reorder-point gap) -- the approved V1 reorder
       points from procurement_strategy_engine.py:
           ROP_L = 1,733 tons   ROP_I = 6,958 tons
           deficit_L = max(0, ROP_L - local_inventory_tons)
           deficit_I = max(0, ROP_I - imported_inventory_tons)

    2. MIX CORRECTION (PSE-2 Part 2.3, Step 2) -- bring the portfolio back
       toward the 45/55 target:
           classified_total   = local + imported
           target_L_tons      = classified_total * 0.45
           target_I_tons      = classified_total * 0.55
           mix_correction_L   = max(0, target_L_tons - local_inventory_tons)
           mix_correction_I   = max(0, target_I_tons - imported_inventory_tons)

       The structural deficit always takes priority over mix correction
       (PSE-2 Part 2.3, Step 3: "Deficit quantity takes priority over mix
       correction because security always overrides optimization"):
           immediate_L = max(deficit_L, mix_correction_L)
           immediate_I = max(deficit_I, mix_correction_I)

    3. CAPACITY CONSTRAINT (PSE-2 Part 2.3, Step 3) -- never recommend a
       purchase that would breach the 45,000-ton ceiling:
           headroom = max_storage_capacity_tons - total_inventory_tons
           if (immediate_L + immediate_I) > headroom:
               scale both down proportionally to fit exactly within headroom

    4. FUTURE REQUIREMENT (PSE-2.5 Part 1.3, single horizon) -- how much
       MORE will be needed within future_horizon_days (default 90, matched
       to the imported lead time) beyond what is being bought right now:
           Net_Req(H) = (H * daily_consumption_total) - total_inventory_tons
                        + SAFETY_STOCK_TOTAL_TONS
           future_requirement = max(0, Net_Req(H) - immediate_requirement)

    Args:
        local_inventory_tons / imported_inventory_tons: current position (tons).
        total_inventory_tons: physical total (local+imported+unknown). If
                               None, defaults to local+imported (i.e. assumes
                               no unclassified inventory).
        max_storage_capacity_tons: confirmed PSE-1 business rule (45,000 t).
        daily_consumption_total: confirmed business rule (110 t/day).
        future_horizon_days: horizon for the "future_requirement" figure.
                              Default 90 days = imported lead time, so this
                              answers "how much total procurement activity
                              (immediate + future) is needed to stay secure
                              through one full import cycle?"

    Returns:
        dict: local_qty_required, imported_qty_required, total_qty_required,
              immediate_requirement, future_requirement,
              deficit_local_tons, deficit_imported_tons,
              mix_correction_local_tons, mix_correction_imported_tons,
              capacity_constrained (bool)
    """
    if total_inventory_tons is None:
        total_inventory_tons = local_inventory_tons + imported_inventory_tons

    # --- Step 1: structural deficit ---
    deficit_local = max(0.0, LOCAL_ROP_TONS - local_inventory_tons)
    deficit_imported = max(0.0, IMPORTED_ROP_TONS - imported_inventory_tons)

    # --- Step 2: mix correction ---
    classified_total = local_inventory_tons + imported_inventory_tons
    target_local_tons = classified_total * local_mix_target
    target_imported_tons = classified_total * imported_mix_target
    mix_correction_local = max(0.0, target_local_tons - local_inventory_tons)
    mix_correction_imported = max(0.0, target_imported_tons - imported_inventory_tons)

    immediate_local = max(deficit_local, mix_correction_local)
    immediate_imported = max(deficit_imported, mix_correction_imported)

    # --- Step 3: capacity constraint ---
    headroom = max(0.0, max_storage_capacity_tons - total_inventory_tons)
    proposed_total = immediate_local + immediate_imported
    capacity_constrained = proposed_total > headroom and proposed_total > 0
    if capacity_constrained:
        scale = headroom / proposed_total
        immediate_local *= scale
        immediate_imported *= scale

    total_qty_required = immediate_local + immediate_imported
    immediate_requirement = total_qty_required

    # --- Step 4: future requirement ---
    net_req_horizon = (
        future_horizon_days * daily_consumption_total
        - total_inventory_tons
        + SAFETY_STOCK_TOTAL_TONS
    )
    future_requirement = max(0.0, net_req_horizon - immediate_requirement)

    return {
        "local_qty_required": round(immediate_local, 2),
        "imported_qty_required": round(immediate_imported, 2),
        "total_qty_required": round(total_qty_required, 2),
        "immediate_requirement": round(immediate_requirement, 2),
        "future_requirement": round(future_requirement, 2),
        "deficit_local_tons": round(deficit_local, 2),
        "deficit_imported_tons": round(deficit_imported, 2),
        "mix_correction_local_tons": round(mix_correction_local, 2),
        "mix_correction_imported_tons": round(mix_correction_imported, 2),
        "capacity_constrained": capacity_constrained,
    }


# ===========================================================================
# PART 2 -- PROCUREMENT TIMING ENGINE
# ===========================================================================

def compute_purchase_timing(
    inventory_tons: float,
    status: str,
    reorder_point_tons: float,
    daily_consumption_rate: float,
    price_signal: Optional[str] = None,
    today: Optional[date] = None,
) -> dict:
    """When should I buy, for a single source (local or imported)?

    DEFERRAL WINDOW (days until the reorder point is breached at current
    consumption rate):
        deferral_window_days = max(0, (inventory_tons - reorder_point_tons)
                                       / daily_consumption_rate)

    LATEST SAFE ORDER DATE -- the absolute deadline; ordering later than
    this breaches the reorder point before the next delivery could arrive:
        latest_safe_order_date = today + deferral_window_days

    RECOMMENDED ORDER DATE -- price can only ever pull this EARLIER than
    latest_safe_order_date, never later (Security > Timing in the approved
    constraint hierarchy -- PSE-2.5/2.6/2.7):
        if status in (CRITICAL, REORDER):
            recommended_order_date = today                  (no latitude)
        elif price_signal == PRICE_RISING:
            recommended_order_date = today                  (capture price before it rises)
        else:
            recommended_order_date = latest_safe_order_date  (standard reorder-point timing)

    URGENCY_LEVEL -- mapped directly from the existing PSE-3B status
    classification (no new thresholds introduced):
        CRITICAL status -> urgency CRITICAL
        REORDER  status -> urgency HIGH
        WATCH    status -> urgency MEDIUM
        SAFE     status -> urgency LOW

    Args:
        inventory_tons: current inventory for this source (tons).
        status: SAFE / WATCH / REORDER / CRITICAL (from compute_reorder_triggers).
        reorder_point_tons: ROP for this source (1,733 local / 6,958 imported).
        daily_consumption_rate: 49.5 (local) or 60.5 (imported).
        price_signal: optional PRICE_RISING / PRICE_FALLING / PRICE_NEUTRAL
                       from compute_price_signal(). None = no price latitude applied.
        today: defaults to date.today().

    Returns:
        dict: deferral_window_days, latest_safe_order_date (ISO str),
              recommended_order_date (ISO str), urgency_level
    """
    today = today or date.today()

    deferral_window_days = max(0.0, (inventory_tons - reorder_point_tons) / daily_consumption_rate) \
        if daily_consumption_rate else 0.0

    latest_safe_order_date = today + timedelta(days=deferral_window_days)

    if status in (STATUS_CRITICAL, STATUS_REORDER):
        recommended_order_date = today
    elif price_signal == PRICE_RISING:
        recommended_order_date = today
    else:
        recommended_order_date = latest_safe_order_date

    urgency_level = _STATUS_TO_URGENCY.get(status, URGENCY_LOW)

    return {
        "deferral_window_days": round(deferral_window_days, 1),
        "latest_safe_order_date": latest_safe_order_date.isoformat(),
        "recommended_order_date": recommended_order_date.isoformat(),
        "urgency_level": urgency_level,
    }


# ===========================================================================
# PART 3 -- FORECAST INTEGRATION (no new models -- consumes run_forecasts.py output)
# ===========================================================================
#
# run_forecasts.py's confirmed output schema (commodity_prices /
# prediction_records table, see run_forecasts.py lines 708-741):
#     commodity, model_name, as_of_date, horizon_months (1, 3, or 6),
#     target_date, predicted_value, lower_bound, upper_bound, unit
#
# Available horizons: h1 (1 month), h3 (3 months), h6 (6 months).
# Available confidence proxy: (upper_bound - lower_bound) spread around
# predicted_value -- there is no separate "confidence" column in the
# schema, so spread-as-percentage-of-predicted-value is used, exactly as
# designed in PSE-2.6 Part 9.3 ("Forecast Probability Derivation").
#
# Per PSE-2.6 Part 4.4 (asymmetric timing rule), the horizon used differs
# by source: LOCAL uses h1 (10-day lead time -- near-term signal is
# reliable), IMPORTED uses h3 (90-day lead time -- matched horizon, even
# though confidence is lower at that range). This module does not decide
# which horizon to use internally -- the caller passes whichever horizon's
# predicted_value/bounds are relevant; see run_pse3d() for the h1/h3 split.
# ===========================================================================

def compute_price_signal(
    current_price: float,
    predicted_value: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    signal_threshold_pct: float = PRICE_SIGNAL_THRESHOLD_PCT,
) -> dict:
    """Classify a forecast into PRICE_RISING / PRICE_FALLING / PRICE_NEUTRAL.

    price_delta_pct = (predicted_value - current_price) / current_price * 100

        > +signal_threshold_pct  -> PRICE_RISING
        < -signal_threshold_pct  -> PRICE_FALLING
        otherwise                -> PRICE_NEUTRAL  (within noise band)

    Confidence (only computed when bounds are supplied):
        spread_pct = (upper_bound - lower_bound) / predicted_value * 100
        spread_pct > FORECAST_SPREAD_LOW_CONFIDENCE_PCT (15%) -> confidence LOW
        otherwise                                              -> confidence HIGH
    When bounds are not supplied, confidence is reported as "UNKNOWN" --
    this function never silently assumes HIGH confidence with no evidence.

    Args:
        current_price: latest known spot price (any unit -- USD/lb, PKR, etc.,
                        as long as it matches predicted_value's unit).
        predicted_value: forecast price at the chosen horizon.
        lower_bound / upper_bound: optional forecast bounds, same unit.
        signal_threshold_pct: noise floor (default 3%, PSE-2.6 Part 4.3).

    Returns:
        dict: signal, price_delta_pct, confidence, spread_pct (or None)
    """
    if current_price == 0:
        return {"signal": PRICE_NEUTRAL, "price_delta_pct": 0.0,
                "confidence": "UNKNOWN", "spread_pct": None}

    price_delta_pct = (predicted_value - current_price) / current_price * 100.0

    if price_delta_pct > signal_threshold_pct:
        signal = PRICE_RISING
    elif price_delta_pct < -signal_threshold_pct:
        signal = PRICE_FALLING
    else:
        signal = PRICE_NEUTRAL

    confidence = "UNKNOWN"
    spread_pct = None
    if lower_bound is not None and upper_bound is not None and predicted_value:
        spread_pct = (upper_bound - lower_bound) / predicted_value * 100.0
        confidence = "LOW" if spread_pct > FORECAST_SPREAD_LOW_CONFIDENCE_PCT else "HIGH"

    return {
        "signal": signal,
        "price_delta_pct": round(price_delta_pct, 2),
        "confidence": confidence,
        "spread_pct": round(spread_pct, 2) if spread_pct is not None else None,
    }


# ===========================================================================
# PART 4 -- SAVINGS ENGINE
# ===========================================================================

def estimate_savings(
    qty_tons: float,
    current_price_usd_per_lb: float,
    forecast_price_usd_per_lb: float,
    pkr_rate: float,
    days_until_forecast_horizon: float,
    source: str = "IMPORTED",
    carrying_cost_annual_rate: float = DEFAULT_CARRYING_COST_ANNUAL_RATE,
) -> dict:
    """Buy now vs wait -- expected cost impact in USD and PKR.

    ASSUMPTIONS (must be read before trusting the output):
        1. The quantity (qty_tons) is a STRUCTURAL requirement from
           compute_procurement_requirement() -- this function does not
           decide whether to buy, only what buying now vs later is worth.
           This is a procurement cost-minimization calculation, not a
           speculative trading signal.
        2. current_price_usd_per_lb is ICE Cotton No. 2 (USD/lb), the only
           live price source confirmed in this system (market_inputs.py).
           For source="IMPORTED" this is the correct reference price.
           For source="LOCAL" this is a PROXY, not a confirmed local
           Pakistan cotton price -- this gap was identified in PSE-1
           (finding M2) and remains UNRESOLVED. Results for LOCAL are
           tagged price_basis_confidence="LOW" for this reason; results
           for IMPORTED are tagged "HIGH".
        3. carrying_cost_annual_rate defaults to the SBP policy rate (the
           only live-fetched rate in this system) as a proxy for the
           company's true cost of capital -- flagged, as in every prior
           phase, as an engineering default pending Finance confirmation.
        4. 1 metric ton = 2,204.622 lb (LB_PER_METRIC_TON).

    FORMULAS:
        cost_now_usd    = qty_tons * current_price_usd_per_lb  * LB_PER_METRIC_TON
        cost_future_usd = qty_tons * forecast_price_usd_per_lb * LB_PER_METRIC_TON
        price_delta_usd = cost_future_usd - cost_now_usd
            > 0  -> price expected to RISE -> buying now avoids this cost
            < 0  -> price expected to FALL -> waiting captures this saving

        carrying_cost_usd = cost_now_usd * carrying_cost_annual_rate
                             * (days_until_forecast_horizon / 365)
            (cost of buying now and holding stock until the forecast date;
             reported separately -- see note below on when it applies)

        If price_delta_usd > 0 (RISING):
            expected_cost_avoided_usd       = price_delta_usd
            expected_extra_cost_if_wait_usd = price_delta_usd
            expected_savings_if_wait_usd    = 0.0
        If price_delta_usd <= 0 (FALLING or flat):
            expected_savings_if_wait_usd    = -price_delta_usd
            expected_cost_avoided_usd       = 0.0
            expected_extra_cost_if_wait_usd = 0.0

    NOTE on carrying_cost_usd: this only reduces the case for buying early
    when the alternative being evaluated is a genuinely optional forward
    purchase (FORWARD_BUY / STAGGER scenarios). For a structurally required
    purchase (REORDER/CRITICAL), the stock must be held regardless of which
    day it is bought on, so carrying cost is reported for transparency but
    is NOT netted against expected_cost_avoided_usd by this function -- the
    caller decides whether it is relevant to the specific recommendation.

    Returns:
        dict: cost_now_usd, cost_future_usd, price_delta_usd,
              expected_cost_avoided_usd, expected_extra_cost_if_wait_usd,
              expected_savings_if_wait_usd, carrying_cost_usd,
              (...and _pkr equivalents for the cost/savings figures),
              price_basis_confidence
    """
    cost_now_usd = qty_tons * current_price_usd_per_lb * LB_PER_METRIC_TON
    cost_future_usd = qty_tons * forecast_price_usd_per_lb * LB_PER_METRIC_TON
    price_delta_usd = cost_future_usd - cost_now_usd

    carrying_cost_usd = (
        cost_now_usd * carrying_cost_annual_rate * (days_until_forecast_horizon / 365.0)
    )

    if price_delta_usd > 0:
        expected_cost_avoided_usd = price_delta_usd
        expected_extra_cost_if_wait_usd = price_delta_usd
        expected_savings_if_wait_usd = 0.0
    else:
        expected_savings_if_wait_usd = -price_delta_usd
        expected_cost_avoided_usd = 0.0
        expected_extra_cost_if_wait_usd = 0.0

    price_basis_confidence = "HIGH" if source.upper() == "IMPORTED" else "LOW"

    return {
        "cost_now_usd": round(cost_now_usd, 2),
        "cost_future_usd": round(cost_future_usd, 2),
        "price_delta_usd": round(price_delta_usd, 2),
        "expected_cost_avoided_usd": round(expected_cost_avoided_usd, 2),
        "expected_extra_cost_if_wait_usd": round(expected_extra_cost_if_wait_usd, 2),
        "expected_savings_if_wait_usd": round(expected_savings_if_wait_usd, 2),
        "carrying_cost_usd": round(carrying_cost_usd, 2),
        "cost_now_pkr": round(cost_now_usd * pkr_rate, 2),
        "expected_cost_avoided_pkr": round(expected_cost_avoided_usd * pkr_rate, 2),
        "expected_savings_if_wait_pkr": round(expected_savings_if_wait_usd * pkr_rate, 2),
        "price_basis_confidence": price_basis_confidence,
    }


# ===========================================================================
# PART 5 -- PROCUREMENT RECOMMENDATIONS
# ===========================================================================

def _recommend_for_source(
    source_name: str,
    status: str,
    qty_required: float,
    timing: dict,
    price_signal_result: Optional[dict],
    savings_result: Optional[dict],
) -> dict:
    """Single-source recommendation -- action, quantity, timing, reason.

    DECISION LOGIC (deterministic; evaluated in this order):

        1. status == CRITICAL:
             IMPORTED -> ACCELERATE_IMPORT_ORDER (the normal 90-day pipeline
                         has already failed to protect the safety floor --
                         this needs expediting, not a routine order)
             LOCAL    -> BUY_LOCAL_NOW (10-day lead time is already fast;
                         no "accelerate" framing needed)
             Timing forced to today. Price signal ignored.

        2. status == REORDER:
             BUY_LOCAL_NOW / BUY_IMPORTED_NOW. Timing forced to today.
             Price signal ignored (structural requirement, not optional).

        3. status in (WATCH, SAFE) AND qty_required > 0 (mix correction
           triggered even though the reorder point itself is not breached):
             price_signal == PRICE_FALLING -> DEFER_LOCAL / DEFER_IMPORTED
                 (wait for the lower price; timing = latest_safe_order_date,
                  never later than the structural deadline)
             price_signal == PRICE_RISING  -> BUY_LOCAL_NOW / BUY_IMPORTED_NOW
                 (buy now to avoid the forecast cost increase)
             price_signal == PRICE_NEUTRAL or unavailable -> STAGGER_PURCHASE
                 (no price information justifies full acceleration or full
                  deferral -- split the mix-correction quantity across the
                  normal procurement cycle instead of one lump purchase)

        4. status in (WATCH, SAFE) AND qty_required == 0:
             WAIT -- inventory secure, no mix correction needed.

    Returns:
        dict: source, action, quantity_tons, timing (recommended_order_date),
              urgency_level, reason, savings (or None)
    """
    signal = price_signal_result["signal"] if price_signal_result else None

    if status == STATUS_CRITICAL:
        if source_name == "IMPORTED":
            action = ACTION_ACCELERATE_IMPORT_ORDER
            reason = (
                "Imported inventory below the safety stock floor -- the normal "
                "90-day lead time can no longer be relied on to protect supply. "
                "Expedite this order immediately (alternate freight, supplier "
                "escalation); do not treat as a routine purchase."
            )
        else:
            action = ACTION_BUY_LOCAL_NOW
            reason = (
                "Local inventory below the safety stock floor. Order immediately "
                "-- 10-day lead time, but every day of delay extends the exposure."
            )

    elif status == STATUS_REORDER:
        action = ACTION_BUY_LOCAL_NOW if source_name == "LOCAL" else ACTION_BUY_IMPORTED_NOW
        rop = LOCAL_ROP_TONS if source_name == "LOCAL" else IMPORTED_ROP_TONS
        reason = (
            f"{source_name.title()} inventory at or below its reorder point "
            f"({rop:,} tons). Order now -- this is a structural requirement, "
            "not a price-timing decision."
        )

    elif qty_required > 0:
        if signal == PRICE_FALLING:
            action = ACTION_DEFER_LOCAL if source_name == "LOCAL" else ACTION_DEFER_IMPORTED
            reason = (
                f"{source_name.title()} inventory secure; portfolio mix needs "
                f"rebalancing, but the forecast shows falling prices "
                f"({price_signal_result['price_delta_pct']:.1f}%). Defer to "
                f"{timing['latest_safe_order_date']} -- the latest date this "
                "remains safe without breaching the reorder point."
            )
        elif signal == PRICE_RISING:
            action = ACTION_BUY_LOCAL_NOW if source_name == "LOCAL" else ACTION_BUY_IMPORTED_NOW
            reason = (
                f"{source_name.title()} inventory secure, but mix rebalance is "
                f"needed and the forecast shows rising prices "
                f"(+{price_signal_result['price_delta_pct']:.1f}%). Buy now to "
                "avoid the higher forecast cost."
            )
        else:
            action = ACTION_STAGGER_PURCHASE
            reason = (
                f"{source_name.title()} inventory secure; mix rebalance needed but "
                "no actionable price signal. Stagger the purchase across the "
                "normal procurement cycle rather than buying the full amount at once."
            )
    else:
        action = ACTION_WAIT
        reason = f"{source_name.title()} inventory secure. No procurement action required."

    return {
        "source": source_name,
        "action": action,
        "quantity_tons": round(qty_required, 2),
        "recommended_order_date": timing["recommended_order_date"],
        "latest_safe_order_date": timing["latest_safe_order_date"],
        "urgency_level": timing["urgency_level"],
        "reason": reason,
        "price_signal": signal,
        "savings": savings_result,
    }


@dataclass
class ProcurementPlan:
    """PSE-3D output -- the full executable procurement plan for one run."""
    run_date: str
    should_buy_now: bool
    primary_action: str
    local_recommendation: dict
    imported_recommendation: dict
    requirement: dict
    risk_level: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_recommendations(
    requirement: dict,
    local_status: str,
    imported_status: str,
    local_timing: dict,
    imported_timing: dict,
    local_price_signal: Optional[dict] = None,
    imported_price_signal: Optional[dict] = None,
    local_savings: Optional[dict] = None,
    imported_savings: Optional[dict] = None,
    today: Optional[date] = None,
) -> ProcurementPlan:
    """Part 5 orchestrator -- combine source-level recommendations into one plan.

    should_buy_now answers Q3 ("buy now or wait?") directly: True if either
    source's recommended_order_date == today.

    primary_action is the single highest-priority action across both
    sources (see _ACTION_PRIORITY), EXCEPT when both sources independently
    recommend a BUY_*_NOW action simultaneously -- in that case the headline
    is BUY_MIXED_NOW, since both purchases should be placed together.

    risk_level mirrors the more severe of the two urgency levels
    (CRITICAL > HIGH > MEDIUM > LOW).
    """
    today = today or date.today()

    local_rec = _recommend_for_source(
        "LOCAL", local_status, requirement["local_qty_required"],
        local_timing, local_price_signal, local_savings,
    )
    imported_rec = _recommend_for_source(
        "IMPORTED", imported_status, requirement["imported_qty_required"],
        imported_timing, imported_price_signal, imported_savings,
    )

    should_buy_now = (
        local_rec["recommended_order_date"] == today.isoformat()
        or imported_rec["recommended_order_date"] == today.isoformat()
    )

    buy_now_actions = {ACTION_BUY_LOCAL_NOW, ACTION_BUY_IMPORTED_NOW}
    if local_rec["action"] in buy_now_actions and imported_rec["action"] in buy_now_actions:
        primary_action = ACTION_BUY_MIXED_NOW
    else:
        candidates = [local_rec["action"], imported_rec["action"]]
        primary_action = min(candidates, key=lambda a: _ACTION_PRIORITY.get(a, 99))

    urgency_rank = {URGENCY_CRITICAL: 0, URGENCY_HIGH: 1, URGENCY_MEDIUM: 2, URGENCY_LOW: 3}
    risk_level = min(
        [local_rec["urgency_level"], imported_rec["urgency_level"]],
        key=lambda u: urgency_rank.get(u, 9),
    )

    return ProcurementPlan(
        run_date=today.isoformat(),
        should_buy_now=should_buy_now,
        primary_action=primary_action,
        local_recommendation=local_rec,
        imported_recommendation=imported_rec,
        requirement=requirement,
        risk_level=risk_level,
    )


# ===========================================================================
# Orchestration helper -- ties Parts 1-5 together for a single engine run
# ===========================================================================

def run_pse3d(
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
    carrying_cost_annual_rate: float = DEFAULT_CARRYING_COST_ANNUAL_RATE,
    today: Optional[date] = None,
) -> ProcurementPlan:
    """Convenience entry point: runs Parts 1-5 in sequence for one position.

    Horizon convention (PSE-2.6 Part 4.4, carried forward unchanged):
        LOCAL price signal/timing    uses the H+1 (1-month) forecast --
            matches the 10-day lead time; near-term signal is reliable.
        IMPORTED price signal/timing uses the H+3 (3-month) forecast --
            matches the 90-day lead time, even though confidence is lower
            at that range.

    If current_price_usd_per_lb / forecast_* are not supplied, price signal
    and savings are skipped (None) and timing/requirement still run on pure
    inventory-position logic -- the engine never requires a forecast to
    answer Q1/Q2 (how much / structural timing), only Q5/Q6 (savings) and
    the optimization-zone portion of Q3/Q4.
    """
    today = today or date.today()

    requirement = compute_procurement_requirement(
        local_inventory_tons, imported_inventory_tons, total_inventory_tons,
    )

    local_price_signal = imported_price_signal = None
    local_savings = imported_savings = None

    if current_price_usd_per_lb is not None and forecast_h1_usd_per_lb is not None:
        lb1 = forecast_h1_bounds[0] if forecast_h1_bounds else None
        ub1 = forecast_h1_bounds[1] if forecast_h1_bounds else None
        local_price_signal = compute_price_signal(
            current_price_usd_per_lb, forecast_h1_usd_per_lb, lb1, ub1
        )
        if requirement["local_qty_required"] > 0:
            local_savings = estimate_savings(
                requirement["local_qty_required"], current_price_usd_per_lb,
                forecast_h1_usd_per_lb, pkr_rate, days_until_forecast_horizon=30,
                source="LOCAL", carrying_cost_annual_rate=carrying_cost_annual_rate,
            )

    if current_price_usd_per_lb is not None and forecast_h3_usd_per_lb is not None:
        lb3 = forecast_h3_bounds[0] if forecast_h3_bounds else None
        ub3 = forecast_h3_bounds[1] if forecast_h3_bounds else None
        imported_price_signal = compute_price_signal(
            current_price_usd_per_lb, forecast_h3_usd_per_lb, lb3, ub3
        )
        if requirement["imported_qty_required"] > 0:
            imported_savings = estimate_savings(
                requirement["imported_qty_required"], current_price_usd_per_lb,
                forecast_h3_usd_per_lb, pkr_rate, days_until_forecast_horizon=90,
                source="IMPORTED", carrying_cost_annual_rate=carrying_cost_annual_rate,
            )

    local_timing = compute_purchase_timing(
        local_inventory_tons, local_status, LOCAL_ROP_TONS, DAILY_CONSUMPTION_LOCAL,
        price_signal=local_price_signal["signal"] if local_price_signal else None,
        today=today,
    )
    imported_timing = compute_purchase_timing(
        imported_inventory_tons, imported_status, IMPORTED_ROP_TONS, DAILY_CONSUMPTION_IMPORTED,
        price_signal=imported_price_signal["signal"] if imported_price_signal else None,
        today=today,
    )

    return generate_recommendations(
        requirement, local_status, imported_status,
        local_timing, imported_timing,
        local_price_signal, imported_price_signal,
        local_savings, imported_savings,
        today=today,
    )
