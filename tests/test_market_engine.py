"""
test_market_engine.py
-----------------------
Tests for procurement_market_engine.py (PSE-3.4 -- Market Intelligence
Layer).

This layer answers only "how attractive is today's market?" -- these
tests assert the scoring formula matches the documented methodology,
that optional inputs (historical/seasonality) are never fabricated, and
that no BUY/HOLD/quantity/date/tranche/savings concept leaks in.
"""
import inspect

import pytest

import procurement_market_engine as market_mod
from procurement_planning_engine import compute_price_signal
from procurement_market_engine import (
    OPPORTUNITY_LEVEL_LOW_MAX,
    OPPORTUNITY_LEVEL_MEDIUM_MAX,
    MarketOpportunitySnapshot,
    assess_market_opportunity,
)


# ===========================================================================
# Architectural isolation
# ===========================================================================

class TestArchitecturalIsolation:
    def test_function_body_has_no_io_or_fetch_calls(self):
        import procurement_market_engine
        code_consts = assess_market_opportunity.__code__.co_names
        for forbidden in ("fetch_price_inputs", "fetch_cotton_price", "run_orchestration", "load_inventory"):
            assert forbidden not in code_consts

    def test_does_not_require_any_inventory_object(self):
        sig = inspect.signature(assess_market_opportunity)
        for name in sig.parameters:
            assert "position" not in name.lower()
            assert "inventory" not in name.lower()

    def test_identical_inputs_produce_identical_snapshot(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.85}
        a = assess_market_opportunity(market_price_inputs=prices)
        b = assess_market_opportunity(market_price_inputs=prices)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db


# ===========================================================================
# Market data unavailable -- never fabricated
# ===========================================================================

class TestMarketDataUnavailable:
    def test_no_inputs_at_all(self):
        snap = assess_market_opportunity()
        assert snap.market_data_quality == "UNAVAILABLE"
        assert snap.forecast_direction is None
        assert snap.forecast_confidence is None
        assert snap.expected_price_advantage_pct is None
        assert snap.market_opportunity_score is None
        assert snap.opportunity_level is None
        assert snap.current_market_position == "UNKNOWN"
        assert any("MARKET_DATA_UNAVAILABLE" in f for f in snap.data_quality_flags)

    def test_missing_h1_forecast_only(self):
        snap = assess_market_opportunity(market_price_inputs={"current_price_usd_per_lb": 0.78})
        assert snap.market_data_quality == "UNAVAILABLE"
        assert snap.market_opportunity_score is None


# ===========================================================================
# Forecast direction / confidence -- reused from compute_price_signal
# ===========================================================================

class TestForecastReuse:
    def test_direction_and_confidence_match_compute_price_signal(self):
        prices = {
            "current_price_usd_per_lb": 0.78,
            "forecast_h1_usd_per_lb": 0.85,
            "forecast_h1_bounds": (0.80, 0.90),
        }
        snap = assess_market_opportunity(market_price_inputs=prices)
        expected = compute_price_signal(0.78, 0.85, 0.80, 0.90)
        assert snap.forecast_direction == expected["signal"]
        assert snap.forecast_confidence == expected["confidence"]
        assert snap.expected_price_advantage_pct == expected["price_delta_pct"]

    def test_rising_forecast_is_price_rising(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.90}
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.forecast_direction == "PRICE_RISING"

    def test_falling_forecast_is_price_falling(self):
        prices = {"current_price_usd_per_lb": 0.90, "forecast_h1_usd_per_lb": 0.78}
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.forecast_direction == "PRICE_FALLING"

    def test_no_bounds_gives_unknown_confidence_and_partial_quality(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.85}
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.forecast_confidence == "UNKNOWN"
        assert snap.market_data_quality == "PARTIAL"

    def test_h3_missing_flagged(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.85}
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert "H3_FORECAST_UNAVAILABLE" in snap.data_quality_flags
        assert snap.supporting_facts["h3_signal"] is None

    def test_h3_present_surfaced_in_supporting_facts(self):
        prices = {
            "current_price_usd_per_lb": 0.78,
            "forecast_h1_usd_per_lb": 0.85,
            "forecast_h3_usd_per_lb": 0.92,
        }
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.supporting_facts["h3_signal"] is not None
        assert snap.supporting_facts["h3_signal"]["signal"] in ("PRICE_RISING", "PRICE_FALLING", "PRICE_NEUTRAL")


# ===========================================================================
# Market Opportunity Score -- documented formula
# ===========================================================================

class TestOpportunityScore:
    def test_neutral_forecast_scores_near_50(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.785}  # < 3% threshold
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.forecast_direction == "PRICE_NEUTRAL"
        assert 40.0 <= snap.market_opportunity_score <= 60.0

    def test_strong_rising_forecast_scores_high(self):
        prices = {
            "current_price_usd_per_lb": 0.70,
            "forecast_h1_usd_per_lb": 0.90,
            "forecast_h1_bounds": (0.88, 0.92),
        }
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.market_opportunity_score > 60.0
        assert snap.opportunity_level == "HIGH"

    def test_strong_falling_forecast_scores_low(self):
        prices = {
            "current_price_usd_per_lb": 0.90,
            "forecast_h1_usd_per_lb": 0.70,
            "forecast_h1_bounds": (0.68, 0.72),
        }
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.market_opportunity_score < 40.0
        assert snap.opportunity_level == "LOW"

    def test_low_confidence_dampens_score_toward_neutral(self):
        tight_bounds = {
            "current_price_usd_per_lb": 0.70,
            "forecast_h1_usd_per_lb": 0.90,
            "forecast_h1_bounds": (0.89, 0.91),  # narrow -> HIGH confidence
        }
        wide_bounds = {
            "current_price_usd_per_lb": 0.70,
            "forecast_h1_usd_per_lb": 0.90,
            "forecast_h1_bounds": (0.50, 1.30),  # very wide -> LOW confidence
        }
        snap_tight = assess_market_opportunity(market_price_inputs=tight_bounds)
        snap_wide = assess_market_opportunity(market_price_inputs=wide_bounds)
        assert snap_tight.forecast_confidence == "HIGH"
        assert snap_wide.forecast_confidence == "LOW"
        # Same price_delta_pct, but the LOW-confidence score should sit
        # closer to the neutral midpoint (50) than the HIGH-confidence one.
        assert abs(snap_wide.market_opportunity_score - 50.0) < abs(snap_tight.market_opportunity_score - 50.0)

    def test_score_always_within_0_100(self):
        prices = {
            "current_price_usd_per_lb": 0.10,
            "forecast_h1_usd_per_lb": 5.00,
            "forecast_h1_bounds": (4.90, 5.10),
        }
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert 0.0 <= snap.market_opportunity_score <= 100.0


# ===========================================================================
# Historical Price Position -- optional, never fabricated
# ===========================================================================

class TestHistoricalPricePosition:
    def test_unknown_when_not_supplied(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.85}
        snap = assess_market_opportunity(market_price_inputs=prices)
        assert snap.current_market_position == "UNKNOWN"
        assert any("HISTORICAL_PRICE_CONTEXT_UNAVAILABLE" in f for f in snap.data_quality_flags)

    def test_near_low_when_current_price_near_period_low(self):
        prices = {"current_price_usd_per_lb": 0.61, "forecast_h1_usd_per_lb": 0.65}
        hist = {"period_low_usd_per_lb": 0.60, "period_high_usd_per_lb": 1.00}
        snap = assess_market_opportunity(market_price_inputs=prices, historical_price_context=hist)
        assert snap.current_market_position == "NEAR_LOW"

    def test_near_high_when_current_price_near_period_high(self):
        prices = {"current_price_usd_per_lb": 0.98, "forecast_h1_usd_per_lb": 0.85}
        hist = {"period_low_usd_per_lb": 0.60, "period_high_usd_per_lb": 1.00}
        snap = assess_market_opportunity(market_price_inputs=prices, historical_price_context=hist)
        assert snap.current_market_position == "NEAR_HIGH"

    def test_mid_range_in_the_middle(self):
        prices = {"current_price_usd_per_lb": 0.80, "forecast_h1_usd_per_lb": 0.85}
        hist = {"period_low_usd_per_lb": 0.60, "period_high_usd_per_lb": 1.00}
        snap = assess_market_opportunity(market_price_inputs=prices, historical_price_context=hist)
        assert snap.current_market_position == "MID_RANGE"

    def test_near_low_position_increases_score_vs_unknown(self):
        prices = {
            "current_price_usd_per_lb": 0.61,
            "forecast_h1_usd_per_lb": 0.61,  # neutral forecast in isolation
        }
        hist = {"period_low_usd_per_lb": 0.60, "period_high_usd_per_lb": 1.00}
        snap_without = assess_market_opportunity(market_price_inputs=prices)
        snap_with = assess_market_opportunity(market_price_inputs=prices, historical_price_context=hist)
        assert snap_with.market_opportunity_score > snap_without.market_opportunity_score


# ===========================================================================
# Seasonality -- optional passthrough only, never fabricated
# ===========================================================================

class TestSeasonality:
    def test_unavailable_when_not_supplied(self):
        snap = assess_market_opportunity()
        assert snap.supporting_facts["seasonality"] is None
        assert any("SEASONALITY_CONTEXT_UNAVAILABLE" in f for f in snap.data_quality_flags)

    def test_passthrough_when_supplied(self):
        season = {"typical_direction": "RISING", "note": "Pre-harvest tightness"}
        snap = assess_market_opportunity(seasonality_context=season)
        assert snap.supporting_facts["seasonality"] == season
        assert not any("SEASONALITY_CONTEXT_UNAVAILABLE" in f for f in snap.data_quality_flags)


# ===========================================================================
# Constraints -- no BUY/HOLD/quantities/dates/tranches/savings
# ===========================================================================

class TestNoDecisionLogic:
    def test_no_forbidden_fields_in_output(self):
        prices = {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.85}
        snap = assess_market_opportunity(market_price_inputs=prices)
        d = snap.to_dict()
        forbidden = {
            "action", "buy", "hold", "qty_now_tons", "qty_later_tons",
            "recommended_order_date", "savings", "tranche", "recommendation",
        }
        assert forbidden.isdisjoint(d.keys())

    def test_module_defines_no_decision_jargon(self):
        names = dir(market_mod)
        for name in names:
            for forbidden in ("Tranche", "Recommend", "PurchaseQty", "BuyNow", "Savings"):
                assert forbidden not in name


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self):
        snap = assess_market_opportunity()
        with pytest.raises(Exception):
            snap.market_data_quality = "FULL"

    def test_to_dict_converts_tuples_to_lists(self):
        snap = assess_market_opportunity()
        d = snap.to_dict()
        assert isinstance(d["data_quality_flags"], list)

    def test_priority_level_thresholds_consistent(self):
        from procurement_market_engine import _score_to_level
        assert _score_to_level(0.0) == "LOW"
        assert _score_to_level(OPPORTUNITY_LEVEL_LOW_MAX) == "LOW"
        assert _score_to_level(OPPORTUNITY_LEVEL_LOW_MAX + 0.01) == "MEDIUM"
        assert _score_to_level(OPPORTUNITY_LEVEL_MEDIUM_MAX) == "MEDIUM"
        assert _score_to_level(OPPORTUNITY_LEVEL_MEDIUM_MAX + 0.01) == "HIGH"
        assert _score_to_level(100.0) == "HIGH"
