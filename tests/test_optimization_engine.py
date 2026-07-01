"""
test_optimization_engine.py
-----------------------------
Tests for procurement_optimization_engine.py (PSE-3.3 -- Portfolio
Optimization Layer).

Builds Position -> Target -> Gap from synthetic data, then asserts the
PortfolioOptimizationSnapshot scores match the documented formulas exactly
and that no BUY/HOLD/quantity/timing/savings concept leaks into this layer.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import procurement_optimization_engine as opt_mod
from procurement_strategy_engine import build_strategy_output_v2
from procurement_position_engine import assess_position
from procurement_target_engine import define_strategy_target
from procurement_gap_engine import analyze_gap
from procurement_optimization_engine import (
    PRIORITY_LEVEL_LOW_MAX,
    PRIORITY_LEVEL_MEDIUM_MAX,
    VALID_POSTURES,
    optimize_portfolio,
)


def _make_origin_summary(local_tons: float, imported_tons: float) -> pd.DataFrame:
    rows = [
        {"org_name": "MTM", "origin": "LOCAL", "tons": local_tons},
        {"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons},
    ]
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


def _build(local_tons, imported_tons, as_of=date(2026, 6, 30), desired_coverage_days=45.0, **target_kwargs):
    so = build_strategy_output_v2(_make_origin_summary(local_tons, imported_tons), run_date=as_of.isoformat())
    pos = assess_position(so, as_of=as_of)
    tgt = define_strategy_target(as_of=as_of, desired_coverage_days=desired_coverage_days, **target_kwargs)
    gap = analyze_gap(pos, tgt)
    return pos, tgt, gap


@pytest.fixture
def comfortable():
    """Both sources SAFE, mix on target, plenty of storage headroom."""
    return _build(local_tons=4000.0, imported_tons=8000.0, desired_coverage_days=45.0)


@pytest.fixture
def critical():
    """Both sources below safety floor -- should drive a HIGH/DEFENSIVE result."""
    return _build(local_tons=500.0, imported_tons=900.0)


# ===========================================================================
# Architectural isolation
# ===========================================================================

class TestArchitecturalIsolation:
    def test_optimize_portfolio_signature_is_three_snapshots_only(self):
        sig = inspect.signature(optimize_portfolio)
        assert list(sig.parameters.keys()) == ["position", "target", "gap"]

    def test_function_body_has_no_io_calls(self):
        source = inspect.getsource(optimize_portfolio)
        for forbidden in ("run_orchestration", "load_inventory", "fetch_price_inputs", "fetch_cotton_price"):
            assert forbidden not in source

    def test_identical_inputs_produce_identical_snapshot(self, comfortable):
        pos, tgt, gap = comfortable
        a = optimize_portfolio(pos, tgt, gap)
        b = optimize_portfolio(pos, tgt, gap)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db


# ===========================================================================
# Inventory Objective
# ===========================================================================

class TestInventoryObjective:
    def test_comfortable_position_is_low_or_medium(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.inventory_objective.priority_level in ("LOW", "MEDIUM")

    def test_critical_position_breaches_safety_floor(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.inventory_objective.supporting_facts["safety_floor_breach_count"] >= 1
        assert snap.inventory_objective.priority_score > 0

    def test_score_formula_matches_documented_weights(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        statuses = (
            gap.inventory_gap.safety_floor_status_total,
            gap.inventory_gap.safety_floor_status_local,
            gap.inventory_gap.safety_floor_status_imported,
        )
        breach_count = sum(1 for s in statuses if s == "BELOW_FLOOR")
        safety_points = breach_count / 3.0 * 50.0
        cov = gap.inventory_gap.coverage_gap_days
        coverage_points = min(40.0, abs(cov) * 2.0) if cov < 0 else 0.0
        util = pos.storage_utilization_pct
        capacity_points = 10.0 if util >= 90 else (5.0 if util >= 75 else 0.0)
        expected = round(min(100.0, safety_points + coverage_points + capacity_points), 2)
        assert snap.inventory_objective.priority_score == expected

    def test_score_always_within_0_100(self, comfortable, critical):
        for pos, tgt, gap in (comfortable, critical):
            snap = optimize_portfolio(pos, tgt, gap)
            assert 0.0 <= snap.inventory_objective.priority_score <= 100.0


# ===========================================================================
# Mix Objective
# ===========================================================================

class TestMixObjective:
    def test_within_tolerance_scores_lower_than_breach(self):
        pos1, tgt1, gap1 = _build(local_tons=4500.0, imported_tons=5500.0)  # exactly on 45/55
        pos2, tgt2, gap2 = _build(local_tons=500.0, imported_tons=9500.0)   # far overweight imported
        snap1 = optimize_portfolio(pos1, tgt1, gap1)
        snap2 = optimize_portfolio(pos2, tgt2, gap2)
        assert snap1.mix_objective.priority_score <= snap2.mix_objective.priority_score

    def test_supporting_facts_include_deviation_and_tolerance(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        facts = snap.mix_objective.supporting_facts
        assert "local_mix_gap_pct_points" in facts
        assert "within_tolerance" in facts


# ===========================================================================
# Procurement Progress Objective
# ===========================================================================

class TestProcurementProgressObjective:
    def test_unavailable_when_no_progress_data(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.procurement_progress_objective.priority_score is None
        assert snap.procurement_progress_objective.priority_level is None
        assert "PROCUREMENT_PROGRESS_OBJECTIVE_UNAVAILABLE" in snap.data_quality_flags

    def test_behind_pace_scores_positive(self):
        as_of = date(2026, 7, 1)
        so = build_strategy_output_v2(_make_origin_summary(4000.0, 8000.0), run_date=as_of.isoformat())
        pos = assess_position(
            so, as_of=as_of,
            procurement_progress={"annual_target_tons": 40000.0, "purchased_to_date_tons": 5000.0},
        )
        tgt = define_strategy_target(as_of=as_of, desired_coverage_days=25.0, annual_target_tons=40000.0)
        gap = analyze_gap(pos, tgt)
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.procurement_progress_objective.priority_score is not None
        assert snap.procurement_progress_objective.priority_score > 0

    def test_ahead_of_pace_scores_zero(self):
        as_of = date(2026, 7, 1)
        so = build_strategy_output_v2(_make_origin_summary(4000.0, 8000.0), run_date=as_of.isoformat())
        pos = assess_position(
            so, as_of=as_of,
            procurement_progress={"annual_target_tons": 40000.0, "purchased_to_date_tons": 39000.0},
        )
        tgt = define_strategy_target(as_of=as_of, desired_coverage_days=25.0, annual_target_tons=40000.0)
        gap = analyze_gap(pos, tgt)
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.procurement_progress_objective.priority_score == 0.0
        assert snap.procurement_progress_objective.priority_level == "LOW"


# ===========================================================================
# Lead-Time Objective
# ===========================================================================

class TestLeadTimeObjective:
    def test_critical_inventory_drives_high_lead_time_priority(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        # Imported lead time (90 days) cannot be covered by low imported DOH.
        assert snap.lead_time_objective.priority_score > 0

    def test_no_urgency_wording_in_supporting_facts(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        facts_str = str(snap.lead_time_objective.supporting_facts).lower()
        assert "urgency" not in facts_str
        assert "critical" not in facts_str


# ===========================================================================
# Capacity Objective
# ===========================================================================

class TestCapacityObjective:
    def test_low_utilization_is_low_priority(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert pos.storage_utilization_pct < 50.0
        assert snap.capacity_objective.priority_score == 0.0
        assert snap.capacity_objective.priority_level == "LOW"

    def test_high_utilization_is_high_priority(self):
        # Push total inventory close to MAX_STORAGE_CAPACITY_TONS (45,000).
        pos, tgt, gap = _build(local_tons=20000.0, imported_tons=22000.0)
        snap = optimize_portfolio(pos, tgt, gap)
        assert pos.storage_utilization_pct > 90.0
        assert snap.capacity_objective.priority_level == "HIGH"

    def test_score_computable_without_target_buffer_configured(self, comfortable):
        pos, tgt, gap = comfortable
        assert tgt.desired_storage_buffer_tons is None  # not configured
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.capacity_objective.priority_score is not None


# ===========================================================================
# Flexibility Objective -- never fabricated
# ===========================================================================

class TestFlexibilityObjective:
    def test_always_unavailable(self, comfortable, critical):
        for pos, tgt, gap in (comfortable, critical):
            snap = optimize_portfolio(pos, tgt, gap)
            assert snap.flexibility_objective.priority_score is None
            assert snap.flexibility_objective.priority_level is None
            assert "FLEXIBILITY_OBJECTIVE_UNAVAILABLE" in " ".join(snap.data_quality_flags)

    def test_desired_flexibility_surfaced_in_supporting_facts_when_configured(self):
        pos, tgt, gap = _build(4000.0, 8000.0, desired_flexibility=0.3)
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.flexibility_objective.supporting_facts["desired_flexibility"] == 0.3


# ===========================================================================
# Overall Portfolio Posture
# ===========================================================================

class TestOverallPosture:
    def test_posture_is_one_of_allowed_values(self, comfortable, critical):
        for pos, tgt, gap in (comfortable, critical):
            snap = optimize_portfolio(pos, tgt, gap)
            assert snap.overall_portfolio_posture in VALID_POSTURES

    def test_critical_inventory_yields_defensive(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.overall_portfolio_posture == "DEFENSIVE"
        assert snap.posture_rule_fired == 1

    def test_comfortable_position_is_not_defensive(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.overall_portfolio_posture != "DEFENSIVE"


# ===========================================================================
# Constraints -- no BUY/HOLD/quantities/timing/savings/recommendations
# ===========================================================================

class TestNoDecisionLogic:
    def test_no_action_quantity_timing_savings_fields(self, critical):
        pos, tgt, gap = critical
        snap = optimize_portfolio(pos, tgt, gap)
        d = snap.to_dict()
        forbidden = {
            "action", "buy", "hold", "qty_now_tons", "qty_later_tons",
            "recommended_order_date", "savings", "opportunity_window", "tranche",
        }
        assert forbidden.isdisjoint(d.keys())
        assert forbidden.isdisjoint(d["inventory_objective"].keys())

    def test_module_defines_no_optimization_jargon_beyond_priority(self):
        names = dir(opt_mod)
        for name in names:
            for forbidden in ("OpportunityWindow", "Tranche", "Recommend", "PurchaseQty", "BuyNow"):
                assert forbidden not in name


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        with pytest.raises(Exception):
            snap.overall_portfolio_posture = "DEFENSIVE"

    def test_nested_objective_is_frozen(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        with pytest.raises(Exception):
            snap.inventory_objective.priority_score = 0.0

    def test_to_dict_converts_tuples_to_lists(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        d = snap.to_dict()
        assert isinstance(d["data_quality_flags"], list)
        assert isinstance(d["position_data_quality_flags"], list)

    def test_upstream_flags_passthrough(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert tuple(snap.position_data_quality_flags) == tuple(pos.data_quality_flags)
        assert tuple(snap.target_data_quality_flags) == tuple(tgt.data_quality_flags)
        assert tuple(snap.gap_data_quality_flags) == tuple(gap.data_quality_flags)

    def test_as_of_matches_gap(self, comfortable):
        pos, tgt, gap = comfortable
        snap = optimize_portfolio(pos, tgt, gap)
        assert snap.as_of == gap.as_of

    def test_priority_level_thresholds_consistent(self):
        from procurement_optimization_engine import _score_to_level
        assert _score_to_level(0.0) == "LOW"
        assert _score_to_level(PRIORITY_LEVEL_LOW_MAX) == "LOW"
        assert _score_to_level(PRIORITY_LEVEL_LOW_MAX + 0.01) == "MEDIUM"
        assert _score_to_level(PRIORITY_LEVEL_MEDIUM_MAX) == "MEDIUM"
        assert _score_to_level(PRIORITY_LEVEL_MEDIUM_MAX + 0.01) == "HIGH"
        assert _score_to_level(100.0) == "HIGH"
