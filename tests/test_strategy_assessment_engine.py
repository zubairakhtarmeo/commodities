"""
test_strategy_assessment_engine.py
-------------------------------------
Tests for procurement_strategy_assessment_engine.py (PSE-3.5 -- Strategic
Procurement Assessment Layer).

Builds Position -> Target -> Gap -> PortfolioOptimization (PSE-3.0/1/2/3)
from synthetic inventory data, builds MarketOpportunitySnapshot (PSE-3.4)
from explicit price inputs, then asserts assess_strategy() combines them
via the documented decision tree -- no I/O, no BUY/HOLD/quantity/date
concepts, full traceability.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import procurement_strategy_assessment_engine as strat_mod
from procurement_strategy_engine import build_strategy_output_v2
from procurement_position_engine import assess_position
from procurement_target_engine import define_strategy_target
from procurement_gap_engine import analyze_gap
from procurement_optimization_engine import optimize_portfolio
from procurement_market_engine import assess_market_opportunity
from procurement_strategy_assessment_engine import (
    ALLOWED_POSTURES,
    assess_strategy,
)


def _make_origin_summary(local_tons: float, imported_tons: float) -> pd.DataFrame:
    rows = [
        {"org_name": "MTM", "origin": "LOCAL", "tons": local_tons},
        {"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons},
    ]
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


def _build_portfolio(local_tons, imported_tons, as_of=date(2026, 6, 30), desired_coverage_days=45.0, **target_kwargs):
    so = build_strategy_output_v2(_make_origin_summary(local_tons, imported_tons), run_date=as_of.isoformat())
    pos = assess_position(so, as_of=as_of)
    tgt = define_strategy_target(as_of=as_of, desired_coverage_days=desired_coverage_days, **target_kwargs)
    gap = analyze_gap(pos, tgt)
    return optimize_portfolio(pos, tgt, gap)


@pytest.fixture
def comfortable_portfolio():
    """Both sources SAFE, mix on target -- low pressure everywhere."""
    return _build_portfolio(local_tons=4000.0, imported_tons=8000.0)


@pytest.fixture
def critical_portfolio():
    """Both sources below safety floor -- drives DEFENSIVE posture."""
    return _build_portfolio(local_tons=500.0, imported_tons=900.0)


@pytest.fixture
def tight_capacity_portfolio():
    """High storage utilisation, otherwise comfortable -- drives capacity pressure."""
    return _build_portfolio(local_tons=20000.0, imported_tons=22000.0, desired_coverage_days=45.0)


def _market(opportunity="neutral", **overrides):
    presets = {
        "neutral": {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.785},
        "high_rising": {
            "current_price_usd_per_lb": 0.70,
            "forecast_h1_usd_per_lb": 0.90,
            "forecast_h1_bounds": (0.88, 0.92),
        },
        "low_falling": {
            "current_price_usd_per_lb": 0.90,
            "forecast_h1_usd_per_lb": 0.70,
            "forecast_h1_bounds": (0.68, 0.72),
        },
        "unavailable": None,
    }
    prices = presets[opportunity]
    if overrides and prices is not None:
        prices = {**prices, **overrides}
    return assess_market_opportunity(market_price_inputs=prices, as_of=date(2026, 6, 30))


# ===========================================================================
# Architectural isolation
# ===========================================================================

class TestArchitecturalIsolation:
    def test_assess_strategy_signature(self):
        sig = inspect.signature(assess_strategy)
        names = list(sig.parameters.keys())
        assert names[:2] == ["portfolio", "market"]

    def test_function_body_has_no_io_calls(self):
        code_names = assess_strategy.__code__.co_names
        for forbidden in ("run_orchestration", "load_inventory", "fetch_price_inputs", "fetch_cotton_price"):
            assert forbidden not in code_names

    def test_identical_inputs_produce_identical_assessment(self, comfortable_portfolio):
        market = _market("neutral")
        a = assess_strategy(comfortable_portfolio, market)
        b = assess_strategy(comfortable_portfolio, market)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db


# ===========================================================================
# Overall Procurement Posture -- decision tree
# ===========================================================================

class TestOverallPosture:
    def test_posture_is_one_of_allowed_values(self, comfortable_portfolio, critical_portfolio):
        market = _market("neutral")
        for portfolio in (comfortable_portfolio, critical_portfolio):
            snap = assess_strategy(portfolio, market)
            assert snap.overall_procurement_posture in ALLOWED_POSTURES

    def test_critical_portfolio_yields_defensive_regardless_of_market(self, critical_portfolio):
        for opp in ("neutral", "high_rising", "low_falling"):
            snap = assess_strategy(critical_portfolio, _market(opp))
            assert snap.overall_procurement_posture == "DEFENSIVE_PROCUREMENT"
            assert snap.posture_rule_fired == 1

    def test_tight_capacity_yields_inventory_preservation(self, tight_capacity_portfolio):
        snap = assess_strategy(tight_capacity_portfolio, _market("high_rising"))
        assert snap.overall_procurement_posture == "INVENTORY_PRESERVATION"
        assert snap.posture_rule_fired == 2

    def test_strong_rising_market_yields_price_capture_when_no_overriding_pressure(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("high_rising"))
        assert snap.overall_procurement_posture in ("PRICE_CAPTURE", "OPPORTUNISTIC_ACCUMULATION")

    def test_unfavorable_market_with_comfortable_portfolio_defers_or_balances(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("low_falling"))
        assert snap.overall_procurement_posture in ("DEFERRED_PROCUREMENT", "BALANCED_ACCUMULATION")

    def test_unavailable_market_falls_back_gracefully(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("unavailable"))
        assert snap.overall_procurement_posture in ALLOWED_POSTURES
        assert "MARKET_DATA_UNAVAILABLE_AFFECTING_STRATEGY" in snap.data_quality_flags


# ===========================================================================
# Primary / Secondary Strategic Objective
# ===========================================================================

class TestObjectives:
    def test_primary_objective_matches_posture(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        assert snap.primary_strategic_objective == strat_mod._PRIMARY_OBJECTIVE_BY_POSTURE[
            snap.overall_procurement_posture
        ]

    def test_secondary_objective_restore_mix_when_mix_high(self):
        # Far overweight imported -> mix objective should be HIGH priority.
        portfolio = _build_portfolio(local_tons=200.0, imported_tons=9800.0)
        snap = assess_strategy(portfolio, _market("neutral"))
        if portfolio.mix_objective.priority_level == "HIGH":
            assert "mix" in snap.secondary_strategic_objective.lower()
            assert snap.secondary_objective_rule_fired == 1

    def test_secondary_objective_default_when_no_pressure(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        if (
            comfortable_portfolio.mix_objective.priority_level != "HIGH"
            and comfortable_portfolio.procurement_progress_objective.priority_level != "HIGH"
        ):
            assert snap.secondary_objective_rule_fired == 3


# ===========================================================================
# Per-dimension evaluation -- traceable to upstream facts
# ===========================================================================

class TestDimensionEvaluation:
    def test_inventory_pressure_breached_when_critical(self, critical_portfolio):
        snap = assess_strategy(critical_portfolio, _market("neutral"))
        assert snap.inventory_pressure.constraint_validation == "BREACHED"

    def test_inventory_pressure_satisfied_when_comfortable(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        assert snap.inventory_pressure.constraint_validation == "SATISFIED"

    def test_market_attractiveness_not_applicable_as_constraint(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        assert snap.market_attractiveness.constraint_validation == "NOT_APPLICABLE"

    def test_capacity_constraints_at_risk_when_tight(self, tight_capacity_portfolio):
        snap = assess_strategy(tight_capacity_portfolio, _market("neutral"))
        assert snap.capacity_constraints.constraint_validation in ("AT_RISK", "BREACHED")

    def test_flexibility_position_always_unavailable(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        assert snap.flexibility_position.constraint_validation == "UNAVAILABLE"
        assert "FLEXIBILITY_DATA_UNAVAILABLE" in snap.data_quality_flags

    def test_procurement_progress_unavailable_without_annual_target(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        assert snap.procurement_progress.constraint_validation == "UNAVAILABLE"
        assert "PROCUREMENT_PROGRESS_DATA_UNAVAILABLE" in snap.data_quality_flags

    def test_each_dimension_reasoning_is_nonempty_string(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        for dim in (
            snap.inventory_pressure, snap.market_attractiveness, snap.portfolio_balance,
            snap.capacity_constraints, snap.lead_time_exposure, snap.procurement_progress,
            snap.flexibility_position,
        ):
            assert isinstance(dim.reasoning, str) and len(dim.reasoning) > 0


# ===========================================================================
# Constraint validation summary -- locked business rules
# ===========================================================================

class TestConstraintValidationSummary:
    def test_all_five_business_constraints_present(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        expected_keys = {
            "minimum_inventory_floor_25_days",
            "maximum_storage_45000_tons",
            "local_imported_mix_45_55",
            "local_lead_time_10_days",
            "imported_lead_time_90_days",
        }
        assert expected_keys == set(snap.constraint_validation.keys())

    def test_floor_breached_propagates_to_summary(self, critical_portfolio):
        snap = assess_strategy(critical_portfolio, _market("neutral"))
        assert snap.constraint_validation["minimum_inventory_floor_25_days"] == "BREACHED"


# ===========================================================================
# Strategy Confidence
# ===========================================================================

class TestStrategyConfidence:
    def test_confidence_lower_when_market_unavailable(self, comfortable_portfolio):
        snap_avail = assess_strategy(comfortable_portfolio, _market("neutral"))
        snap_unavail = assess_strategy(comfortable_portfolio, _market("unavailable"))
        assert snap_unavail.strategy_confidence_score < snap_avail.strategy_confidence_score

    def test_confidence_within_bounds(self, comfortable_portfolio, critical_portfolio):
        for portfolio in (comfortable_portfolio, critical_portfolio):
            for opp in ("neutral", "high_rising", "low_falling", "unavailable"):
                snap = assess_strategy(portfolio, _market(opp))
                assert 0.0 <= snap.strategy_confidence_score <= 100.0
                assert snap.strategy_confidence_level in ("LOW", "MEDIUM", "HIGH")


# ===========================================================================
# Constraints -- no BUY/HOLD/quantities/dates/tranches/savings
# ===========================================================================

class TestNoExecutionLogic:
    def test_no_forbidden_fields_anywhere(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("high_rising"))
        d = snap.to_dict()
        forbidden = {
            "action", "buy", "hold", "qty_now_tons", "qty_later_tons",
            "recommended_order_date", "purchase_order", "savings", "tranche",
        }
        assert forbidden.isdisjoint(d.keys())

    def test_module_defines_no_execution_jargon(self):
        names = dir(strat_mod)
        for name in names:
            for forbidden in ("Tranche", "PurchaseOrder", "BuyNow", "Savings", "ExecutiveDashboard"):
                assert forbidden not in name


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestSnapshotContract:
    def test_snapshot_is_frozen(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        with pytest.raises(Exception):
            snap.overall_procurement_posture = "DEFENSIVE_PROCUREMENT"

    def test_dimension_is_frozen(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        with pytest.raises(Exception):
            snap.inventory_pressure.constraint_validation = "SATISFIED"

    def test_to_dict_converts_tuples_to_lists(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"))
        d = snap.to_dict()
        assert isinstance(d["reasoning"], list)
        assert isinstance(d["data_quality_flags"], list)

    def test_upstream_flags_passthrough(self, comfortable_portfolio):
        market = _market("neutral")
        snap = assess_strategy(comfortable_portfolio, market)
        assert tuple(snap.portfolio_data_quality_flags) == tuple(comfortable_portfolio.data_quality_flags)
        assert tuple(snap.market_data_quality_flags) == tuple(market.data_quality_flags)

    def test_as_of_overridable(self, comfortable_portfolio):
        snap = assess_strategy(comfortable_portfolio, _market("neutral"), as_of=date(2026, 1, 1))
        assert snap.as_of == "2026-01-01"
