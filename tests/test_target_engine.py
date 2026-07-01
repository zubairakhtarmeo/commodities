"""
test_target_engine.py
-----------------------
Tests for procurement_target_engine.py (PSE-3.1 -- Strategy Target
Definition Layer).

TargetSnapshot is a pure function of business policy and the calendar --
these tests never construct a PositionSnapshot or any inventory/market
data, and explicitly assert this module has no dependency on either.
"""
import inspect
from datetime import date

import pytest

import procurement_target_engine
from procurement_orchestrator import MAX_STORAGE_CAPACITY_TONS
from procurement_strategy_engine import (
    IMPORTED_LEAD_TIME_DAYS,
    IMPORTED_MIX_TARGET,
    LOCAL_LEAD_TIME_DAYS,
    LOCAL_MIX_TARGET,
    MIN_STOCK_DAYS,
)
from procurement_target_engine import (
    BUSINESS_RULE_VERSION,
    CONFIGURATION_VERSION,
    TargetSnapshot,
    VALID_STRATEGIC_POSTURES,
    define_strategy_target,
)


# ===========================================================================
# Architectural isolation -- never depends on PositionSnapshot / reality
# ===========================================================================

class TestArchitecturalIsolation:
    def test_module_does_not_import_position_engine(self):
        import_lines = [
            line for line in inspect.getsource(procurement_target_engine).splitlines()
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        joined = "\n".join(import_lines)
        assert "procurement_position_engine" not in joined
        assert "PositionSnapshot" not in joined

    def test_define_strategy_target_signature_has_no_position_param(self):
        sig = inspect.signature(define_strategy_target)
        for name in sig.parameters:
            assert "position" not in name.lower()
            assert "inventory_tons" not in name.lower()
            assert "current" not in name.lower()

    def test_identical_inputs_produce_identical_snapshot(self):
        a = define_strategy_target(as_of=date(2026, 6, 30), annual_target_tons=40000.0)
        b = define_strategy_target(as_of=date(2026, 6, 30), annual_target_tons=40000.0)
        assert a.to_dict() == {**b.to_dict(), "generated_at": a.generated_at}


# ===========================================================================
# Inventory Targets -- reused, not duplicated constants
# ===========================================================================

class TestInventoryTargets:
    def test_inventory_floor_matches_locked_constant(self):
        snap = define_strategy_target()
        assert snap.inventory_floor_days == MIN_STOCK_DAYS

    def test_maximum_inventory_matches_orchestrator_constant(self):
        snap = define_strategy_target()
        assert snap.maximum_inventory_tons == MAX_STORAGE_CAPACITY_TONS

    def test_desired_mix_matches_locked_constants(self):
        snap = define_strategy_target()
        assert snap.desired_local_inventory_pct == round(LOCAL_MIX_TARGET * 100.0, 2)
        assert snap.desired_imported_inventory_pct == round(IMPORTED_MIX_TARGET * 100.0, 2)

    def test_coverage_defaults_to_floor_and_is_flagged(self):
        snap = define_strategy_target()
        assert snap.desired_inventory_coverage_days == float(MIN_STOCK_DAYS)
        assert "DESIRED_COVERAGE_DEFAULTED_TO_FLOOR" in snap.data_quality_flags

    def test_coverage_above_floor_accepted(self):
        snap = define_strategy_target(desired_coverage_days=40.0)
        assert snap.desired_inventory_coverage_days == 40.0
        assert "DESIRED_COVERAGE_DEFAULTED_TO_FLOOR" not in snap.data_quality_flags

    def test_coverage_at_floor_exactly_accepted(self):
        snap = define_strategy_target(desired_coverage_days=float(MIN_STOCK_DAYS))
        assert snap.desired_inventory_coverage_days == float(MIN_STOCK_DAYS)

    def test_coverage_below_floor_raises(self):
        with pytest.raises(ValueError, match="survival floor"):
            define_strategy_target(desired_coverage_days=10.0)


# ===========================================================================
# Procurement Targets -- optional, calendar-based pacing only
# ===========================================================================

class TestProcurementTargets:
    def test_unconfigured_by_default(self):
        snap = define_strategy_target()
        assert snap.annual_procurement_target_tons is None
        assert snap.target_procurement_progress_pct is None
        assert snap.monthly_target_procurement_tons is None
        assert snap.remaining_planned_procurement_tons is None
        assert "ANNUAL_PROCUREMENT_TARGET_NOT_CONFIGURED" in snap.data_quality_flags

    def test_monthly_target_is_annual_over_twelve(self):
        snap = define_strategy_target(annual_target_tons=120000.0)
        assert snap.monthly_target_procurement_tons == 10000.0

    def test_progress_pct_at_year_start(self):
        snap = define_strategy_target(as_of=date(2026, 1, 1), annual_target_tons=36500.0)
        assert snap.target_procurement_progress_pct == pytest.approx(0.27, abs=0.05)

    def test_progress_pct_at_year_end(self):
        snap = define_strategy_target(as_of=date(2026, 12, 31), annual_target_tons=36500.0)
        assert snap.target_procurement_progress_pct == pytest.approx(100.0, abs=0.5)

    def test_remaining_planned_plus_progress_share_equals_annual_target(self):
        snap = define_strategy_target(as_of=date(2026, 7, 1), annual_target_tons=40000.0)
        elapsed_share = snap.annual_procurement_target_tons * snap.target_procurement_progress_pct / 100.0
        assert elapsed_share + snap.remaining_planned_procurement_tons == pytest.approx(40000.0, abs=5.0)

    def test_pacing_assumptions_flagged_when_configured(self):
        snap = define_strategy_target(annual_target_tons=40000.0)
        assert "CALENDAR_YEAR_PACING_ASSUMED" in snap.data_quality_flags
        assert "UNIFORM_MONTHLY_PACING_ASSUMED" in snap.data_quality_flags

    def test_no_procurement_quantity_or_gap_fields_present(self):
        snap = define_strategy_target(annual_target_tons=40000.0)
        d = snap.to_dict()
        forbidden = {"shortage", "surplus", "gap", "qty_now_tons", "qty_later_tons", "action"}
        assert forbidden.isdisjoint(d.keys())


# ===========================================================================
# Portfolio Targets
# ===========================================================================

class TestPortfolioTargets:
    def test_unconfigured_flexibility_and_posture_by_default(self):
        snap = define_strategy_target()
        assert snap.desired_flexibility is None
        assert snap.desired_strategic_posture is None
        assert "DESIRED_FLEXIBILITY_NOT_CONFIGURED" in snap.data_quality_flags
        assert "STRATEGIC_POSTURE_NOT_CONFIGURED" in snap.data_quality_flags

    def test_desired_local_imported_pct_match_mix_targets(self):
        snap = define_strategy_target()
        assert snap.desired_local_pct == snap.desired_local_inventory_pct
        assert snap.desired_imported_pct == snap.desired_imported_inventory_pct

    @pytest.mark.parametrize("posture", VALID_STRATEGIC_POSTURES)
    def test_valid_postures_accepted(self, posture):
        snap = define_strategy_target(desired_strategic_posture=posture)
        assert snap.desired_strategic_posture == posture

    def test_posture_case_insensitive(self):
        snap = define_strategy_target(desired_strategic_posture="balanced")
        assert snap.desired_strategic_posture == "BALANCED"

    def test_invalid_posture_raises(self):
        with pytest.raises(ValueError):
            define_strategy_target(desired_strategic_posture="AGGRESSIVE")

    def test_desired_flexibility_passthrough(self):
        snap = define_strategy_target(desired_flexibility=0.3)
        assert snap.desired_flexibility == 0.3
        assert "DESIRED_FLEXIBILITY_NOT_CONFIGURED" not in snap.data_quality_flags


# ===========================================================================
# Lead Time Targets -- reused, not duplicated
# ===========================================================================

class TestLeadTimeTargets:
    def test_planning_horizons_match_locked_lead_times(self):
        snap = define_strategy_target()
        assert snap.local_planning_horizon_days == LOCAL_LEAD_TIME_DAYS
        assert snap.imported_planning_horizon_days == IMPORTED_LEAD_TIME_DAYS


# ===========================================================================
# Operational Targets -- optional
# ===========================================================================

class TestOperationalTargets:
    def test_unconfigured_by_default(self):
        snap = define_strategy_target()
        assert snap.desired_capacity_utilization_pct is None
        assert snap.desired_storage_buffer_tons is None
        assert "DESIRED_CAPACITY_UTILIZATION_NOT_CONFIGURED" in snap.data_quality_flags
        assert "DESIRED_STORAGE_BUFFER_NOT_CONFIGURED" in snap.data_quality_flags

    def test_configured_values_passthrough(self):
        snap = define_strategy_target(
            desired_capacity_utilization_pct=70.0,
            desired_storage_buffer_tons=5000.0,
        )
        assert snap.desired_capacity_utilization_pct == 70.0
        assert snap.desired_storage_buffer_tons == 5000.0


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self):
        snap = define_strategy_target()
        with pytest.raises(Exception):
            snap.inventory_floor_days = 0

    def test_version_metadata_present(self):
        snap = define_strategy_target()
        assert snap.business_rule_version == BUSINESS_RULE_VERSION
        assert snap.configuration_version == CONFIGURATION_VERSION

    def test_to_dict_roundtrip(self):
        snap = define_strategy_target()
        d = snap.to_dict()
        assert d["inventory_floor_days"] == snap.inventory_floor_days
        assert isinstance(d["data_quality_flags"], list)

    def test_as_of_defaults_to_today(self):
        snap = define_strategy_target()
        assert snap.as_of == date.today().isoformat()

    def test_as_of_overridable(self):
        snap = define_strategy_target(as_of=date(2026, 1, 1))
        assert snap.as_of == "2026-01-01"
