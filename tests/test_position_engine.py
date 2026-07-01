"""
test_position_engine.py
------------------------
Tests for procurement_position_engine.py (PSE-3.0 -- Position Assessment
Layer / Layer 0).

Builds StrategyOutputV2 from synthetic origin_summary data (no workbook
I/O) and asserts the PositionSnapshot reflects it as pure facts, with no
optimization/recommendation logic and no silent recalculation of values
already produced upstream.
"""
from datetime import date

import pandas as pd
import pytest

from procurement_strategy_engine import (
    DAILY_CONSUMPTION_LOCAL,
    DAILY_CONSUMPTION_IMPORTED,
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_MIX_TARGET,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_MIX_TARGET,
    MIN_STOCK_DAYS,
    SAFETY_STOCK_IMPORTED_TONS,
    SAFETY_STOCK_LOCAL_TONS,
    SAFETY_STOCK_TOTAL_TONS,
    build_strategy_output_v2,
)
from procurement_orchestrator import MAX_STORAGE_CAPACITY_TONS
from procurement_position_engine import PositionSnapshot, assess_position


def _make_origin_summary(local_tons: float, imported_tons: float, unknown_tons: float = 0.0) -> pd.DataFrame:
    rows = []
    if local_tons:
        rows.append({"org_name": "MTM", "origin": "LOCAL", "tons": local_tons})
    if imported_tons:
        rows.append({"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons})
    if unknown_tons:
        rows.append({"org_name": "MTM", "origin": "UNKNOWN", "tons": unknown_tons})
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


@pytest.fixture
def safe_strategy_output():
    """Both sources comfortably SAFE and roughly on the 45/55 mix target."""
    df = _make_origin_summary(local_tons=4000.0, imported_tons=8000.0)
    return build_strategy_output_v2(df, run_date="2026-06-27")


@pytest.fixture
def critical_strategy_output():
    """Both sources below their safety stock floor."""
    df = _make_origin_summary(local_tons=800.0, imported_tons=1000.0)
    return build_strategy_output_v2(df, run_date="2026-06-27")


# ===========================================================================
# Pure reuse -- snapshot must match StrategyOutputV2 facts exactly
# ===========================================================================

class TestInventoryPositionReuse:
    def test_total_local_imported_match_strategy_output(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.total_inventory_tons == safe_strategy_output.total_inventory_tons
        assert snap.local_inventory_tons == safe_strategy_output.local_inventory_tons
        assert snap.imported_inventory_tons == safe_strategy_output.imported_inventory_tons

    def test_doh_matches_strategy_output_days_cover(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.total_doh == safe_strategy_output.total_days_cover
        assert snap.local_doh == safe_strategy_output.local_days_cover
        assert snap.imported_doh == safe_strategy_output.imported_days_cover

    def test_safety_stock_constants_match_locked_values(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.safety_stock_total_tons == SAFETY_STOCK_TOTAL_TONS
        assert snap.safety_stock_local_tons == SAFETY_STOCK_LOCAL_TONS
        assert snap.safety_stock_imported_tons == SAFETY_STOCK_IMPORTED_TONS
        assert snap.min_stock_days == MIN_STOCK_DAYS

    def test_distance_above_safety_stock(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        expected_total = safe_strategy_output.total_inventory_tons - SAFETY_STOCK_TOTAL_TONS
        expected_local = safe_strategy_output.local_inventory_tons - SAFETY_STOCK_LOCAL_TONS
        expected_imported = safe_strategy_output.imported_inventory_tons - SAFETY_STOCK_IMPORTED_TONS
        assert snap.distance_above_safety_stock_total_tons == round(expected_total, 2)
        assert snap.distance_above_safety_stock_local_tons == round(expected_local, 2)
        assert snap.distance_above_safety_stock_imported_tons == round(expected_imported, 2)

    def test_distance_negative_when_critical(self, critical_strategy_output):
        snap = assess_position(critical_strategy_output)
        assert snap.distance_above_safety_stock_local_tons < 0
        assert snap.distance_above_safety_stock_imported_tons < 0

    def test_storage_capacity_and_utilization(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.max_storage_capacity_tons == MAX_STORAGE_CAPACITY_TONS
        expected_remaining = MAX_STORAGE_CAPACITY_TONS - safe_strategy_output.total_inventory_tons
        assert snap.remaining_storage_capacity_tons == round(expected_remaining, 2)
        expected_util = safe_strategy_output.total_inventory_tons / MAX_STORAGE_CAPACITY_TONS * 100.0
        assert snap.storage_utilization_pct == round(expected_util, 2)

    def test_unclassified_inventory_flagged(self):
        df = _make_origin_summary(local_tons=4000.0, imported_tons=8000.0, unknown_tons=500.0)
        so = build_strategy_output_v2(df, run_date="2026-06-27")
        snap = assess_position(so)
        assert any("UNCLASSIFIED_INVENTORY_PRESENT" in f for f in snap.data_quality_flags)


class TestMixPositionReuse:
    def test_mix_pct_matches_strategy_output(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.local_mix_pct == safe_strategy_output.local_mix_pct
        assert snap.imported_mix_pct == safe_strategy_output.imported_mix_pct

    def test_mix_targets_match_locked_constants(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.local_mix_target_pct == round(LOCAL_MIX_TARGET * 100.0, 2)
        assert snap.imported_mix_target_pct == round(IMPORTED_MIX_TARGET * 100.0, 2)

    def test_within_tolerance_matches_rebalance_flag(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.mix_within_tolerance == (not safe_strategy_output.rebalance_required)

    def test_overweight_imported_flagged_out_of_tolerance(self):
        # Local far below target share -> imported overweight -> deviation should be negative for local
        df = _make_origin_summary(local_tons=500.0, imported_tons=9500.0)
        so = build_strategy_output_v2(df, run_date="2026-06-27")
        snap = assess_position(so)
        assert snap.local_mix_deviation_pct_points < 0
        assert snap.mix_within_tolerance == (not so.rebalance_required)


class TestLeadTimePosition:
    def test_lead_time_constants(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.local_lead_time_days == LOCAL_LEAD_TIME_DAYS
        assert snap.imported_lead_time_days == IMPORTED_LEAD_TIME_DAYS

    def test_covers_lead_time_true_when_safe(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.local_covers_lead_time is True
        assert snap.imported_covers_lead_time is True

    def test_covers_lead_time_false_when_critical(self, critical_strategy_output):
        # Local lead time (10 days) is short enough that even CRITICAL local
        # inventory can still exceed it -- this is exactly why this layer
        # reports facts only and leaves interpretation to later layers.
        # Imported lead time (90 days) is long enough that CRITICAL imported
        # inventory cannot cover it.
        snap = assess_position(critical_strategy_output)
        assert snap.imported_covers_lead_time is False

    def test_doh_minus_lead_time_is_facts_only(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        expected_local = safe_strategy_output.local_days_cover - LOCAL_LEAD_TIME_DAYS
        expected_imported = safe_strategy_output.imported_days_cover - IMPORTED_LEAD_TIME_DAYS
        assert snap.local_doh_minus_lead_time_days == round(expected_local, 2)
        assert snap.imported_doh_minus_lead_time_days == round(expected_imported, 2)


# ===========================================================================
# Optional inputs -- Procurement Progress / Market Context
# ===========================================================================

class TestProcurementProgressOptional:
    def test_unavailable_by_default(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.annual_target_tons is None
        assert snap.purchased_to_date_tons is None
        assert snap.remaining_procurement_tons is None
        assert snap.procurement_completion_pct is None
        assert "PROCUREMENT_PROGRESS_UNAVAILABLE" in snap.data_quality_flags

    def test_computed_when_supplied(self, safe_strategy_output):
        progress = {"annual_target_tons": 40000.0, "purchased_to_date_tons": 10000.0}
        snap = assess_position(safe_strategy_output, procurement_progress=progress)
        assert snap.annual_target_tons == 40000.0
        assert snap.purchased_to_date_tons == 10000.0
        assert snap.remaining_procurement_tons == 30000.0
        assert snap.procurement_completion_pct == 25.0
        assert "PROCUREMENT_PROGRESS_UNAVAILABLE" not in snap.data_quality_flags

    def test_incomplete_flagged(self, safe_strategy_output):
        progress = {"annual_target_tons": 40000.0}  # missing purchased_to_date_tons
        snap = assess_position(safe_strategy_output, procurement_progress=progress)
        assert snap.remaining_procurement_tons is None
        assert "PROCUREMENT_PROGRESS_INCOMPLETE" in snap.data_quality_flags


class TestMarketContextOptional:
    def test_unavailable_by_default(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.current_market_price_usd_per_lb is None
        assert snap.forecast_h1_usd_per_lb is None
        assert snap.forecast_confidence is None
        assert "MARKET_CONTEXT_UNAVAILABLE" in snap.data_quality_flags

    def test_populated_when_supplied(self, safe_strategy_output):
        prices = {
            "current_price_usd_per_lb": 0.78,
            "forecast_h1_usd_per_lb": 0.85,
            "forecast_h3_usd_per_lb": 0.90,
            "forecast_h1_bounds": (0.80, 0.90),
        }
        snap = assess_position(safe_strategy_output, market_price_inputs=prices)
        assert snap.current_market_price_usd_per_lb == 0.78
        assert snap.forecast_h1_usd_per_lb == 0.85
        assert snap.forecast_h3_usd_per_lb == 0.90
        assert snap.forecast_confidence in ("HIGH", "LOW")
        assert "MARKET_CONTEXT_UNAVAILABLE" not in snap.data_quality_flags

    def test_incomplete_flagged_when_no_forecast(self, safe_strategy_output):
        prices = {"current_price_usd_per_lb": 0.78}
        snap = assess_position(safe_strategy_output, market_price_inputs=prices)
        assert snap.forecast_confidence is None
        assert "MARKET_CONTEXT_INCOMPLETE" in snap.data_quality_flags


# ===========================================================================
# Immutability / metadata / no recommendation logic leakage
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        with pytest.raises(Exception):
            snap.total_inventory_tons = 0.0

    def test_to_dict_roundtrip(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        d = snap.to_dict()
        assert d["total_inventory_tons"] == snap.total_inventory_tons
        assert isinstance(d["data_quality_flags"], list)

    def test_as_of_defaults_to_today(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert snap.as_of == date.today().isoformat()

    def test_as_of_overridable(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output, as_of=date(2026, 1, 1))
        assert snap.as_of == "2026-01-01"

    def test_no_action_or_buy_hold_fields_present(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        d = snap.to_dict()
        forbidden = {"action", "reason", "rule_fired", "recommended_qty_tons", "buy", "hold"}
        assert forbidden.isdisjoint(d.keys())

    def test_data_freshness_references_strategy_output_run_date(self, safe_strategy_output):
        snap = assess_position(safe_strategy_output)
        assert "2026-06-27" in snap.data_freshness
