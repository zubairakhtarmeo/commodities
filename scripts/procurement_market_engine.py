"""
procurement_market_engine.py
-------------------------------
PSE-3.4 -- Market Intelligence Layer.

Engine only -- no Streamlit, no dashboard, no UI. Per the frozen
architecture, this is a sibling input to Portfolio Optimization (PSE-3.3),
not a step in the Position -> Target -> Gap -> Optimization chain:

    PortfolioOptimizationSnapshot  (procurement priorities -- "what matters")
            +
    MarketOpportunitySnapshot       (market attractiveness -- "is today good")
            |
            v
    Procurement Strategy Layer (NOT implemented here -- future sprint)
        decides: whether to buy, how much, when.

This module answers exactly one question -- "How attractive is today's
market for procurement?" -- and nothing else. It contains no BUY/HOLD
logic, no purchase quantities, no purchase dates, no tranche plans, no
savings math, and no executive narrative.

Reuse, not duplication:
    - compute_price_signal() (procurement_planning_engine.py, PSE-3D) is
      reused as-is for forecast direction/confidence classification, for
      both the 1-horizon (h1) and 3-horizon (h3) forecasts. This module
      defines no new price-signal logic.
    - This module does not fetch live prices itself. market_price_inputs
      is an optional caller-supplied dict matching the shape of
      procurement_scenario_engine.fetch_price_inputs() -- the caller
      decides the source (live fetch, cached value, or test fixture),
      exactly the same decoupling already used by
      procurement_position_engine.assess_position() (PSE-3.0).

No production data source exists today for a 52-week/seasonal historical
price range or a seasonality calendar (confirmed by inspection -- no
module in this codebase computes either). Per "never fabricate data,"
both are optional caller-supplied inputs; when omitted, the corresponding
output is reported as unavailable with a data-quality flag rather than
invented.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_planning_engine import compute_price_signal

# ---------------------------------------------------------------------------
# Engineering defaults -- NOT confirmed business rules (same status as
# WATCH_BUFFER_PCT in procurement_strategy_engine.py and the priority-score
# thresholds in procurement_optimization_engine.py).
# ---------------------------------------------------------------------------

OPPORTUNITY_LEVEL_LOW_MAX = 33.0
OPPORTUNITY_LEVEL_MEDIUM_MAX = 66.0

HISTORICAL_POSITION_LOW_MAX_PCTILE = 33.0
HISTORICAL_POSITION_HIGH_MIN_PCTILE = 67.0

LOW_CONFIDENCE_DAMPENING_FACTOR = 0.5  # how strongly LOW confidence pulls the score toward neutral (50)
HISTORICAL_POSITION_SCORE_ADJUSTMENT = 10.0  # +/- points applied for NEAR_LOW / NEAR_HIGH


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _score_to_level(score: float) -> str:
    if score <= OPPORTUNITY_LEVEL_LOW_MAX:
        return "LOW"
    if score <= OPPORTUNITY_LEVEL_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


# ===========================================================================
# OUTPUT SCHEMA
# ===========================================================================

@dataclass(frozen=True)
class MarketOpportunitySnapshot:
    """Immutable snapshot of how attractive today's market is for
    procurement. Facts and a single derived opportunity score -- no
    BUY/HOLD, quantities, dates, tranches, or savings.
    """

    as_of: str
    generated_at: str

    current_market_position: str            # NEAR_LOW / MID_RANGE / NEAR_HIGH / UNKNOWN
    forecast_direction: Optional[str]         # PRICE_RISING / PRICE_FALLING / PRICE_NEUTRAL / None
    forecast_confidence: Optional[str]         # HIGH / LOW / UNKNOWN / None
    expected_price_advantage_pct: Optional[float]

    market_opportunity_score: Optional[float]  # 0-100, continuous; None if no usable price data
    opportunity_level: Optional[str]            # LOW / MEDIUM / HIGH; None if no usable price data

    market_data_quality: str                    # FULL / PARTIAL / UNAVAILABLE

    supporting_facts: dict
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
# ASSESSMENT
# ===========================================================================

def assess_market_opportunity(
    market_price_inputs: Optional[dict] = None,
    historical_price_context: Optional[dict] = None,
    seasonality_context: Optional[dict] = None,
    as_of: Optional[date] = None,
) -> MarketOpportunitySnapshot:
    """Build the MarketOpportunitySnapshot for the current run.

    Args:
        market_price_inputs: Optional dict matching the shape of
            procurement_scenario_engine.fetch_price_inputs() --
            current_price_usd_per_lb, forecast_h1_usd_per_lb,
            forecast_h3_usd_per_lb, forecast_h1_bounds, forecast_h3_bounds.
            This layer does not fetch prices itself. If omitted (or
            missing current/h1 price), the snapshot reports
            market_data_quality="UNAVAILABLE" and all derived fields are
            None.
        historical_price_context: Optional dict with
            period_low_usd_per_lb / period_high_usd_per_lb. No production
            source for this exists yet -- caller-supplied. If omitted,
            current_market_position is "UNKNOWN" and a flag is set.
        seasonality_context: Optional dict, passed through verbatim into
            supporting_facts (e.g. {"typical_direction": ..., "note": ...}).
            No production source for this exists yet -- caller-supplied.
            If omitted, a flag is set.
        as_of: Date this snapshot represents; defaults to today.

    Returns:
        MarketOpportunitySnapshot -- immutable, market-attractiveness only.
    """
    as_of = as_of or date.today()
    flags: list[str] = []
    facts: dict = {}

    current_price = (market_price_inputs or {}).get("current_price_usd_per_lb")
    forecast_h1 = (market_price_inputs or {}).get("forecast_h1_usd_per_lb")
    forecast_h3 = (market_price_inputs or {}).get("forecast_h3_usd_per_lb")
    h1_bounds = (market_price_inputs or {}).get("forecast_h1_bounds")
    h3_bounds = (market_price_inputs or {}).get("forecast_h3_bounds")

    facts["current_price_usd_per_lb"] = current_price
    facts["forecast_h1_usd_per_lb"] = forecast_h1
    facts["forecast_h3_usd_per_lb"] = forecast_h3

    if current_price is None or forecast_h1 is None:
        flags.append("MARKET_DATA_UNAVAILABLE:current_price_or_h1_forecast_missing")
        current_market_position = "UNKNOWN"
        forecast_direction = None
        forecast_confidence = None
        expected_price_advantage_pct = None
        market_opportunity_score = None
        opportunity_level = None
        market_data_quality = "UNAVAILABLE"
    else:
        h1_lb, h1_ub = h1_bounds if h1_bounds else (None, None)
        signal_h1 = compute_price_signal(current_price, forecast_h1, h1_lb, h1_ub)
        forecast_direction = signal_h1["signal"]
        forecast_confidence = signal_h1["confidence"]
        expected_price_advantage_pct = signal_h1["price_delta_pct"]
        facts["h1_spread_pct"] = signal_h1["spread_pct"]

        if forecast_h3 is not None:
            h3_lb, h3_ub = h3_bounds if h3_bounds else (None, None)
            signal_h3 = compute_price_signal(current_price, forecast_h3, h3_lb, h3_ub)
            facts["h3_signal"] = signal_h3
        else:
            facts["h3_signal"] = None
            flags.append("H3_FORECAST_UNAVAILABLE")

        if forecast_confidence == "UNKNOWN":
            market_data_quality = "PARTIAL"
            flags.append("FORECAST_CONFIDENCE_UNKNOWN:no_forecast_bounds_supplied")
        else:
            market_data_quality = "FULL"

        # --- Market Opportunity Score ---
        # 50 = neutral. Rising forecast -> higher score (favorable to buy
        # now, ahead of the increase). Falling forecast -> lower score
        # (favorable to wait). LOW/UNKNOWN confidence dampens the signal
        # toward neutral rather than acting on an uncertain forecast.
        score = 50.0 + _clamp(expected_price_advantage_pct * 2.0, -40.0, 40.0)
        if forecast_confidence == "LOW":
            score = 50.0 + (score - 50.0) * LOW_CONFIDENCE_DAMPENING_FACTOR
        elif forecast_confidence == "UNKNOWN":
            score = 50.0

        # --- Historical Price Position (if available) ---
        period_low = (historical_price_context or {}).get("period_low_usd_per_lb")
        period_high = (historical_price_context or {}).get("period_high_usd_per_lb")
        if period_low is not None and period_high is not None and period_high > period_low:
            percentile = _clamp(
                (current_price - period_low) / (period_high - period_low) * 100.0
            )
            facts["historical_price_percentile"] = round(percentile, 2)
            if percentile <= HISTORICAL_POSITION_LOW_MAX_PCTILE:
                current_market_position = "NEAR_LOW"
                score += HISTORICAL_POSITION_SCORE_ADJUSTMENT
            elif percentile >= HISTORICAL_POSITION_HIGH_MIN_PCTILE:
                current_market_position = "NEAR_HIGH"
                score -= HISTORICAL_POSITION_SCORE_ADJUSTMENT
            else:
                current_market_position = "MID_RANGE"
        else:
            current_market_position = "UNKNOWN"
            facts["historical_price_percentile"] = None
            flags.append("HISTORICAL_PRICE_CONTEXT_UNAVAILABLE")

        market_opportunity_score = round(_clamp(score), 2)
        opportunity_level = _score_to_level(market_opportunity_score)

    # --- Seasonality (if available) -- passthrough fact only, never fabricated ---
    if seasonality_context:
        facts["seasonality"] = seasonality_context
    else:
        facts["seasonality"] = None
        flags.append("SEASONALITY_CONTEXT_UNAVAILABLE")

    return MarketOpportunitySnapshot(
        as_of=as_of.isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        current_market_position=current_market_position,
        forecast_direction=forecast_direction,
        forecast_confidence=forecast_confidence,
        expected_price_advantage_pct=expected_price_advantage_pct,
        market_opportunity_score=market_opportunity_score,
        opportunity_level=opportunity_level,
        market_data_quality=market_data_quality,
        supporting_facts=facts,
        data_quality_flags=tuple(flags),
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="PSE-3.4 Market Intelligence Layer -- print a MarketOpportunitySnapshot "
                     "from live-fetched price/forecast data."
    )
    parser.add_argument("--live", action="store_true", help="Fetch live price/forecast via fetch_price_inputs()")
    args = parser.parse_args()

    market_price_inputs = None
    if args.live:
        from procurement_scenario_engine import fetch_price_inputs
        market_price_inputs = fetch_price_inputs()

    snapshot = assess_market_opportunity(market_price_inputs=market_price_inputs)
    print(json.dumps(snapshot.to_dict(), indent=2))
