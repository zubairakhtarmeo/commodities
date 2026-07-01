"""
test_gap_engine.py
--------------------
Tests for procurement_gap_engine.py (PSE-3.2 -- Gap Analysis Layer).

GapSnapshot = PositionSnapshot - TargetSnapshot, pure arithmetic. These
tests build a PositionSnapshot from synthetic inventory data and a
TargetSnapshot from explicit policy inputs, then assert the gap is
exactly the arithmetic difference -- no interpretation, no I/O reloads.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import procurement_gap_engine
from procurement_strategy_engine import DAILY_CONSUMPTION_TOTAL, build_strategy_output_v2
from procurement_position_engine import assess_position
from procurement_target_engine import define_strategy_target
from procurement_gap_engine import GapSnapshot, analyze_gap


def _make_origin_summary(local_tons: float, imported_tons: float) -> pd.DataFrame:
    rows = [
        {"org_name": "MTM", "origin": "LOCAL", "tons": local_tons},
        {"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons},
    ]
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


@pytest.fixture
def position():
    df = _make_origin_summary(local_tons=4000.0, imported_tons=8000.0)
    so = build_strategy_output_v2(df, run_date="2026-06-30")
    return assess_position(so, as_of=date(2026, 6, 30))


@pytest.fixture
def target():
    return define_strategy_target(as_of=date(2026, 6, 30), desired_coverage_days=40.0)


@pytest.fixture
def target_full():
    return define_strategy_target(
        as_of=date(2026, 6, 30),
        desired_coverage_days=40.0,
        annual_target_tons=40000.0,
        desired_flexibility=0.3,
        desired_strategic_posture="BALANCED",
        desired_capacity_utilization_pct=60.0,
        desired_storage_buffer_tons=5000.0,
    )


@pytest.fixture
def position_full():
    df = _make_origin_summary(local_tons=4000.0, imported_tons=8000.0)
    so = build_strategy_output_v2(df, run_date="2026-06-30")
    return assess_position(
        so,
        as_of=date(2026, 6, 30),
        market_price_inputs={"current_price_usd_per_lb": 0.80, "forecast_h1_usd_per_lb": 0.85},
        procurement_progress={
            "annual_target_tons": 40000.0,
            "purchased_to_date_tons": 18000.0,
            "current_month_purchased_tons": 1200.0,
        },
    )


# ===========================================================================
# Architectural isolation -- pure comparison, no I/O reloads
# ===========================================================================

class TestArchitecturalIsolation:
    def test_module_does_not_reload_oracle_or_workbook_data(self):
        source = inspect.getsource(procurement_gap_engine)
        # The CLI __main__ block legitimately reuses run_orchestration for a
        # standalone demo; analyze_gap() itself must not.
        analyze_gap_source = inspect.getsource(procurement_gap_engine.analyze_gap)
        assert "run_orchestration" not in analyze_gap_source
        assert "load_inventory" not in analyze_gap_source
        assert "fetch_price_inputs" not in analyze_gap_source

    def test_analyze_gap_signature_is_two_snapshots_only(self):
        sig = inspect.signature(analyze_gap)
        assert list(sig.parameters.keys()) == ["position", "target"]

    def test_identical_inputs_produce_identical_gap(self, position, target):
        a = analyze_gap(position, target)
        b = analyze_gap(position, target)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db


# ===========================================================================
# 1. Inventory Gap
# ===========================================================================

class TestInventoryGap:
    def test_coverage_gap_is_position_minus_target(self, position, target):
        gap = analyze_gap(position, target)
        expected = round(position.total_doh - target.desired_inventory_coverage_days, 2)
        assert gap.inventory_gap.coverage_gap_days == expected

    def test_stock_gap_uses_locked_daily_consumption_constant(self, position, target):
        gap = analyze_gap(position, target)
        desired_stock = target.desired_inventory_coverage_days * DAILY_CONSUMPTION_TOTAL
        expected = round(position.total_inventory_tons - desired_stock, 2)
        assert gap.inventory_gap.stock_gap_tons == expected

    def test_storage_headroom_gap_none_when_target_buffer_not_configured(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.inventory_gap.storage_headroom_gap_tons is None
        assert any("STORAGE_HEADROOM_GAP_UNAVAILABLE" in f for f in gap.data_quality_flags)

    def test_storage_headroom_gap_computed_when_configured(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        expected = round(
            position_full.remaining_storage_capacity_tons - target_full.desired_storage_buffer_tons, 2
        )
        assert gap.inventory_gap.storage_headroom_gap_tons == expected

    def test_safety_floor_status_above_when_safe(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.inventory_gap.safety_floor_status_total == "ABOVE_FLOOR"
        assert gap.inventory_gap.safety_floor_status_local == "ABOVE_FLOOR"
        assert gap.inventory_gap.safety_floor_status_imported == "ABOVE_FLOOR"

    def test_safety_floor_status_below_when_critical(self, target):
        df = _make_origin_summary(local_tons=500.0, imported_tons=900.0)
        so = build_strategy_output_v2(df, run_date="2026-06-30")
        pos = assess_position(so, as_of=date(2026, 6, 30))
        gap = analyze_gap(pos, target)
        assert gap.inventory_gap.safety_floor_status_local == "BELOW_FLOOR"
        assert gap.inventory_gap.safety_floor_status_imported == "BELOW_FLOOR"


# ===========================================================================
# 2. Procurement Progress Gap
# ===========================================================================

class TestProcurementGap:
    def test_unavailable_when_position_progress_missing(self, position, target_full):
        gap = analyze_gap(position, target_full)
        assert gap.procurement_gap.annual_progress_gap_pct_points is None
        assert gap.procurement_gap.monthly_progress_gap_tons is None
        assert gap.procurement_gap.remaining_procurement_gap_tons is None

    def test_computed_when_both_sides_available(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        expected_annual = round(
            position_full.procurement_completion_pct - target_full.target_procurement_progress_pct, 2
        )
        assert gap.procurement_gap.annual_progress_gap_pct_points == expected_annual

        expected_monthly = round(
            position_full.current_month_purchased_tons - target_full.monthly_target_procurement_tons, 2
        )
        assert gap.procurement_gap.monthly_progress_gap_tons == expected_monthly

        expected_remaining = round(
            position_full.remaining_procurement_tons - target_full.remaining_planned_procurement_tons, 2
        )
        assert gap.procurement_gap.remaining_procurement_gap_tons == expected_remaining


# ===========================================================================
# 3. Mix Gap
# ===========================================================================

class TestMixGap:
    def test_mix_gap_is_position_minus_target(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.mix_gap.local_mix_gap_pct_points == round(
            position.local_mix_pct - target.desired_local_pct, 2
        )
        assert gap.mix_gap.imported_mix_gap_pct_points == round(
            position.imported_mix_pct - target.desired_imported_pct, 2
        )

    def test_within_tolerance_reused_not_recomputed(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.mix_gap.within_tolerance == position.mix_within_tolerance


# ===========================================================================
# 4. Lead-Time Gap
# ===========================================================================

class TestLeadTimeGap:
    def test_lead_time_gap_is_doh_minus_horizon(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.lead_time_gap.local_lead_time_gap_days == round(
            position.local_doh - target.local_planning_horizon_days, 2
        )
        assert gap.lead_time_gap.imported_lead_time_gap_days == round(
            position.imported_doh - target.imported_planning_horizon_days, 2
        )

    def test_covers_planning_horizon_flags(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.lead_time_gap.local_covers_planning_horizon == (
            gap.lead_time_gap.local_lead_time_gap_days >= 0
        )
        assert gap.lead_time_gap.imported_covers_planning_horizon == (
            gap.lead_time_gap.imported_lead_time_gap_days >= 0
        )

    def test_no_urgency_classification_present(self, position, target):
        gap = analyze_gap(position, target)
        d = gap.to_dict()
        assert "urgency" not in str(d["lead_time_gap"]).lower()


# ===========================================================================
# 5. Capacity Gap
# ===========================================================================

class TestCapacityGap:
    def test_unavailable_when_target_not_configured(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.capacity_gap.utilization_gap_pct_points is None
        assert gap.capacity_gap.storage_buffer_gap_tons is None

    def test_computed_when_configured(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        expected_util = round(
            position_full.storage_utilization_pct - target_full.desired_capacity_utilization_pct, 2
        )
        expected_buffer = round(
            position_full.remaining_storage_capacity_tons - target_full.desired_storage_buffer_tons, 2
        )
        assert gap.capacity_gap.utilization_gap_pct_points == expected_util
        assert gap.capacity_gap.storage_buffer_gap_tons == expected_buffer


# ===========================================================================
# 6. Flexibility Gap -- never fabricated
# ===========================================================================

class TestFlexibilityGap:
    def test_current_flexibility_always_none(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        assert gap.flexibility_gap.current_flexibility is None
        assert gap.flexibility_gap.flexibility_gap is None

    def test_desired_flexibility_passthrough(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        assert gap.flexibility_gap.desired_flexibility == target_full.desired_flexibility

    def test_flagged_unavailable(self, position, target):
        gap = analyze_gap(position, target)
        assert any("FLEXIBILITY_GAP_UNAVAILABLE" in f for f in gap.data_quality_flags)


# ===========================================================================
# 7. Market Context Gap -- never fabricated
# ===========================================================================

class TestMarketGap:
    def test_target_side_always_none(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        assert gap.market_gap.target_market_price_usd_per_lb is None
        assert gap.market_gap.price_gap_usd_per_lb is None

    def test_position_price_passthrough(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        assert gap.market_gap.current_market_price_usd_per_lb == position_full.current_market_price_usd_per_lb

    def test_flagged_not_applicable(self, position, target):
        gap = analyze_gap(position, target)
        assert any("MARKET_CONTEXT_GAP_NOT_APPLICABLE" in f for f in gap.data_quality_flags)


# ===========================================================================
# Constraints -- no decisions, no recommendations
# ===========================================================================

class TestNoDecisionLogic:
    def test_no_buy_hold_or_action_fields(self, position_full, target_full):
        gap = analyze_gap(position_full, target_full)
        d = gap.to_dict()
        forbidden = {"action", "reason", "rule_fired", "buy", "hold", "savings",
                     "opportunity_window", "tranche", "recommendation"}
        assert forbidden.isdisjoint(d.keys())
        assert forbidden.isdisjoint(d["inventory_gap"].keys())
        assert forbidden.isdisjoint(d["procurement_gap"].keys())

    def test_module_does_not_define_optimization_concepts(self):
        names = dir(procurement_gap_engine)
        forbidden_substrings = ("OpportunityWindow", "Tranche", "Optimiz", "Recommend")
        for name in names:
            for forbidden in forbidden_substrings:
                assert forbidden not in name, f"{name} suggests optimization logic leaked into Gap Analysis"


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self, position, target):
        gap = analyze_gap(position, target)
        with pytest.raises(Exception):
            gap.as_of = "2000-01-01"

    def test_nested_sections_are_frozen(self, position, target):
        gap = analyze_gap(position, target)
        with pytest.raises(Exception):
            gap.inventory_gap.coverage_gap_days = 0.0

    def test_to_dict_converts_tuples_to_lists(self, position, target):
        gap = analyze_gap(position, target)
        d = gap.to_dict()
        assert isinstance(d["data_quality_flags"], list)
        assert isinstance(d["position_data_quality_flags"], list)

    def test_position_and_target_flags_passthrough(self, position, target):
        gap = analyze_gap(position, target)
        assert tuple(gap.position_data_quality_flags) == tuple(position.data_quality_flags)
        assert tuple(gap.target_data_quality_flags) == tuple(target.data_quality_flags)

    def test_as_of_mismatch_flagged(self, position):
        mismatched_target = define_strategy_target(as_of=date(2026, 1, 1), desired_coverage_days=40.0)
        gap = analyze_gap(position, mismatched_target)
        assert any("AS_OF_DATE_MISMATCH" in f for f in gap.data_quality_flags)

    def test_as_of_matches_position_when_aligned(self, position, target):
        gap = analyze_gap(position, target)
        assert gap.as_of == position.as_of == target.as_of
        assert not any("AS_OF_DATE_MISMATCH" in f for f in gap.data_quality_flags)
