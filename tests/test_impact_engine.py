"""
test_impact_engine.py
-----------------------
Tests for procurement_impact_engine.py (PSE-4.1 -- Decision Impact Analysis).

Builds the full upstream pipeline (PSE-3.0 through PSE-3.6) from synthetic
inventory data and price inputs, then asserts that interpret_impact() returns
correct business-facing language for six impact dimensions -- without
recalculating any procurement decisions.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import procurement_impact_engine as impact_mod
from procurement_strategy_engine import build_strategy_output_v2
from procurement_position_engine import assess_position
from procurement_target_engine import define_strategy_target
from procurement_gap_engine import analyze_gap
from procurement_optimization_engine import optimize_portfolio
from procurement_market_engine import assess_market_opportunity
from procurement_strategy_assessment_engine import assess_strategy
from procurement_execution_planning_engine import build_execution_plan
from procurement_impact_engine import DecisionImpact, interpret_impact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_origin_summary(local_tons: float, imported_tons: float) -> pd.DataFrame:
    rows = [
        {"org_name": "MTM", "origin": "LOCAL",    "tons": local_tons},
        {"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons},
    ]
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


def _market(opportunity="neutral"):
    presets = {
        "neutral":    {"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.785},
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
    return assess_market_opportunity(
        market_price_inputs=presets[opportunity], as_of=date(2026, 6, 30)
    )


def _build_all(local_tons, imported_tons, market_opportunity="neutral",
               desired_coverage_days=45.0, as_of=date(2026, 6, 30)):
    so       = build_strategy_output_v2(_make_origin_summary(local_tons, imported_tons),
                                        run_date=as_of.isoformat())
    pos      = assess_position(so, as_of=as_of)
    tgt      = define_strategy_target(as_of=as_of, desired_coverage_days=desired_coverage_days)
    gap      = analyze_gap(pos, tgt)
    portfolio = optimize_portfolio(pos, tgt, gap)
    market   = _market(market_opportunity)
    strategy = assess_strategy(portfolio, market, as_of=as_of)
    plan     = build_execution_plan(pos, portfolio, market, strategy, as_of=as_of)
    return pos, portfolio, market, strategy, plan


def _impact(*args, **kwargs):
    pos, portfolio, market, strategy, plan = _build_all(*args, **kwargs)
    return interpret_impact(plan, strategy, portfolio, market, pos)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def critical_impact():
    """Both sources below safety floor -- highest urgency."""
    return _impact(local_tons=500.0, imported_tons=900.0)


@pytest.fixture
def comfortable_impact():
    """Both sources above ROP, mix on target -- no procurement needed."""
    return _impact(local_tons=9000.0, imported_tons=11000.0)


@pytest.fixture
def near_full_impact():
    """Storage nearly full -- INVENTORY_PRESERVATION posture."""
    return _impact(local_tons=20000.0, imported_tons=22000.0)


@pytest.fixture
def rising_market_impact():
    """Healthy inventory + clearly rising market -- PRICE_CAPTURE territory."""
    return _impact(local_tons=9000.0, imported_tons=11000.0, market_opportunity="high_rising")


@pytest.fixture
def falling_market_impact():
    """Healthy inventory + clearly falling market -- defer discretionary."""
    return _impact(local_tons=9000.0, imported_tons=11000.0, market_opportunity="low_falling")


@pytest.fixture
def no_market_impact():
    """No price data available -- market fields should return UNKNOWN."""
    return _impact(local_tons=500.0, imported_tons=900.0, market_opportunity="unavailable")


# ===========================================================================
# Architectural isolation
# ===========================================================================

class TestArchitecture:
    def test_interpret_impact_signature(self):
        sig   = inspect.signature(interpret_impact)
        names = list(sig.parameters.keys())
        assert names[:5] == ["plan", "strategy", "portfolio", "market", "position"]

    def test_function_body_makes_no_erp_or_oracle_calls(self):
        code = interpret_impact.__code__.co_names
        for forbidden in ("run_orchestration", "load_inventory", "fetch_price_inputs",
                          "to_excel", "write_to_oracle"):
            assert forbidden not in code

    def test_no_engine_layer_imported(self):
        # Impact engine must NOT import target or gap engines (no new decisions).
        import procurement_impact_engine as m
        assert not hasattr(m, "define_strategy_target")
        assert not hasattr(m, "analyze_gap")
        assert not hasattr(m, "optimize_portfolio")

    def test_identical_inputs_produce_identical_impact(self, critical_impact):
        pos, portfolio, market, strategy, plan = _build_all(500.0, 900.0)
        a = interpret_impact(plan, strategy, portfolio, market, pos)
        b = interpret_impact(plan, strategy, portfolio, market, pos)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db

    def test_output_is_frozen(self, critical_impact):
        with pytest.raises(Exception):
            critical_impact.inventory_outlook = "modified"

    def test_to_dict_converts_tuples_to_lists(self, critical_impact):
        d = critical_impact.to_dict()
        assert isinstance(d["data_quality_flags"], list)

    def test_as_of_overridable(self):
        pos, portfolio, market, strategy, plan = _build_all(500.0, 900.0)
        impact = interpret_impact(plan, strategy, portfolio, market, pos, as_of=date(2026, 1, 1))
        assert impact.as_of == "2026-01-01"


# ===========================================================================
# Inventory Outlook
# ===========================================================================

class TestInventoryOutlook:
    def test_critical_relieves_pressure(self, critical_impact):
        assert "relieved" in critical_impact.inventory_outlook.lower() or \
               "safety floor" in critical_impact.inventory_outlook.lower()

    def test_comfortable_outlook_is_adequate(self, comfortable_impact):
        text = comfortable_impact.inventory_outlook.lower()
        assert "adequate" in text or "above" in text or "unchanged" in text

    def test_inventory_preservation_mentions_storage(self, near_full_impact):
        text = near_full_impact.inventory_outlook.lower()
        assert "storage" in text or "capacity" in text or "preserved" in text

    def test_detail_is_nonempty_string(self, critical_impact, comfortable_impact):
        for impact in (critical_impact, comfortable_impact):
            assert isinstance(impact.inventory_outlook_detail, str)
            assert len(impact.inventory_outlook_detail) > 0

    def test_detail_mentions_inventory_tons(self, critical_impact):
        assert "tons" in critical_impact.inventory_outlook_detail.lower()


# ===========================================================================
# Procurement Progress Impact
# ===========================================================================

class TestProcurementProgress:
    def test_returns_unknown_when_no_annual_target(self, critical_impact):
        # Default build has no annual_target_tons configured.
        assert "UNKNOWN" in critical_impact.procurement_progress_impact
        assert any("PROCUREMENT_PROGRESS_IMPACT_UNKNOWN" in f
                   for f in critical_impact.data_quality_flags)

    def test_flag_set_for_unknown_progress(self, critical_impact):
        assert "PROCUREMENT_PROGRESS_IMPACT_UNKNOWN:no_annual_target_configured" \
               in critical_impact.data_quality_flags

    def test_progress_is_string(self, comfortable_impact):
        assert isinstance(comfortable_impact.procurement_progress_impact, str)
        assert len(comfortable_impact.procurement_progress_impact) > 0


# ===========================================================================
# Mix Outlook
# ===========================================================================

class TestMixOutlook:
    def test_comfortable_mix_on_target(self, comfortable_impact):
        text = comfortable_impact.mix_outlook.lower()
        assert "target" in text or "maintained" in text or "on target" in text

    def test_mix_correction_addressed_when_local_short(self):
        # local=3000, imported=17000 -> local very short, mix correction event expected
        impact = _impact(local_tons=3000.0, imported_tons=17000.0)
        text = impact.mix_outlook.lower()
        # Either mix is being corrected or deviation is noted
        assert "target" in text or "closer" in text or "mix" in text

    def test_mix_outlook_is_nonempty_string(self, critical_impact, comfortable_impact):
        for impact in (critical_impact, comfortable_impact):
            assert isinstance(impact.mix_outlook, str)
            assert len(impact.mix_outlook) > 0

    def test_mix_mentions_percentages(self, comfortable_impact):
        # Should contain some % reference
        assert "%" in comfortable_impact.mix_outlook


# ===========================================================================
# Market Exposure
# ===========================================================================

class TestMarketExposure:
    def test_no_market_data_returns_unknown(self, no_market_impact):
        assert "UNKNOWN" in no_market_impact.market_exposure
        assert "MARKET_EXPOSURE_IMPACT_UNKNOWN:no_price_data_available" \
               in no_market_impact.data_quality_flags

    def test_rising_market_mentions_advantage_or_capture(self, rising_market_impact):
        text = rising_market_impact.market_exposure.lower()
        # Either actively capturing (quantity > 0) or signalling a favourable opportunity.
        assert any(w in text for w in ("advantage", "capture", "favourable", "price", "ahead", "opportun"))

    def test_falling_market_mentions_defer_or_preserve(self, falling_market_impact):
        text = falling_market_impact.market_exposure.lower()
        assert any(w in text for w in ("defer", "preserves", "decline", "falling", "lower"))

    def test_neutral_market_is_neutral(self, comfortable_impact):
        text = comfortable_impact.market_exposure.lower()
        assert "neutral" in text or "not a primary" in text

    def test_market_exposure_is_nonempty_string(self, critical_impact):
        assert isinstance(critical_impact.market_exposure, str)
        assert len(critical_impact.market_exposure) > 0


# ===========================================================================
# Operational Risk
# ===========================================================================

class TestOperationalRisk:
    def test_critical_scenario_yields_high_risk(self, critical_impact):
        assert critical_impact.operational_risk_level == "HIGH"

    def test_comfortable_scenario_yields_low_risk(self, comfortable_impact):
        assert comfortable_impact.operational_risk_level == "LOW"

    def test_storage_full_scenario_yields_low_risk(self, near_full_impact):
        # INVENTORY_PRESERVATION -- no procurement needed, low operational risk
        assert near_full_impact.operational_risk_level == "LOW"

    def test_risk_level_is_allowed_value(self, critical_impact, comfortable_impact):
        allowed = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}
        for impact in (critical_impact, comfortable_impact):
            assert impact.operational_risk_level in allowed

    def test_risk_reason_is_nonempty_string(self, critical_impact, comfortable_impact):
        for impact in (critical_impact, comfortable_impact):
            assert isinstance(impact.operational_risk_reason, str)
            assert len(impact.operational_risk_reason) > 0

    def test_high_risk_reason_explains_consequence(self, critical_impact):
        text = critical_impact.operational_risk_reason.lower()
        assert any(w in text for w in ("reorder", "lead time", "floor", "breach", "delivery"))


# ===========================================================================
# Review Guidance
# ===========================================================================

class TestReviewGuidance:
    def test_review_guidance_is_nonempty_string(self, critical_impact, comfortable_impact):
        for impact in (critical_impact, comfortable_impact):
            assert isinstance(impact.review_guidance, str)
            assert len(impact.review_guidance) > 0

    def test_review_guidance_contains_a_date(self, critical_impact):
        # next_review_date from PSE-3.6 is an ISO date string like 2026-07-09
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2}", critical_impact.review_guidance)

    def test_critical_review_mentions_urgency(self, critical_impact):
        text = critical_impact.review_guidance.lower()
        assert any(w in text for w in ("earlier", "window", "confirm", "review"))

    def test_comfortable_review_mentions_planning(self, comfortable_impact):
        text = comfortable_impact.review_guidance.lower()
        assert "review" in text


# ===========================================================================
# No execution side effects (engine frozen validation)
# ===========================================================================

class TestNoSideEffects:
    def test_module_defines_no_decision_jargon(self):
        # Only check names actually defined in this module (not legitimately imported types).
        own_names = [
            name for name in dir(impact_mod)
            if getattr(getattr(impact_mod, name, None), "__module__", "") == "procurement_impact_engine"
        ]
        for name in own_names:
            for forbidden in ("PurchaseOrder", "OracleWrite", "ERPUpdate", "StrategicPosture"):
                assert forbidden not in name, f"Forbidden jargon '{forbidden}' found in '{name}'"

    def test_impact_has_no_quantity_fields(self, critical_impact):
        d = critical_impact.to_dict()
        qty_keys = {k for k in d if "quantity" in k or "tons" in k or "rop" in k}
        assert qty_keys == set(), f"Unexpected quantity fields: {qty_keys}"

    def test_upstream_snapshots_unchanged_after_interpret(self):
        pos, portfolio, market, strategy, plan = _build_all(500.0, 900.0)
        original_posture = strategy.overall_procurement_posture
        original_qty     = plan.total_planned_quantity_tons
        interpret_impact(plan, strategy, portfolio, market, pos)
        assert strategy.overall_procurement_posture == original_posture
        assert plan.total_planned_quantity_tons == original_qty
