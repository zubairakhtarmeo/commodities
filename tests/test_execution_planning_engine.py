"""
test_execution_planning_engine.py
------------------------------------
Tests for procurement_execution_planning_engine.py (PSE-3.6 -- Procurement
Execution Planning Layer).

Builds Position -> Target -> Gap -> PortfolioOptimization -> Strategy
(PSE-3.0/1/2/3/5) plus MarketOpportunitySnapshot (PSE-3.4) from synthetic
data, then asserts build_execution_plan() derives quantities, windows,
and sequencing deterministically -- no Oracle/ERP calls, no fabricated
savings, full traceability.
"""
import inspect
from datetime import date

import pandas as pd
import pytest

import procurement_execution_planning_engine as plan_mod
from procurement_strategy_engine import (
    IMPORTED_ROP_TONS,
    LOCAL_ROP_TONS,
    build_strategy_output_v2,
)
from procurement_position_engine import assess_position
from procurement_target_engine import define_strategy_target
from procurement_gap_engine import analyze_gap
from procurement_optimization_engine import optimize_portfolio
from procurement_market_engine import assess_market_opportunity
from procurement_strategy_assessment_engine import assess_strategy
from procurement_execution_planning_engine import (
    EXECUTION_WINDOWS,
    build_execution_plan,
)


def _make_origin_summary(local_tons: float, imported_tons: float) -> pd.DataFrame:
    rows = [
        {"org_name": "MTM", "origin": "LOCAL", "tons": local_tons},
        {"org_name": "MTM", "origin": "IMPORTED", "tons": imported_tons},
    ]
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


def _market(opportunity="neutral"):
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
    return assess_market_opportunity(market_price_inputs=presets[opportunity], as_of=date(2026, 6, 30))


def _build_all(local_tons, imported_tons, as_of=date(2026, 6, 30), desired_coverage_days=45.0,
                market_opportunity="neutral", **target_kwargs):
    so = build_strategy_output_v2(_make_origin_summary(local_tons, imported_tons), run_date=as_of.isoformat())
    pos = assess_position(so, as_of=as_of)
    tgt = define_strategy_target(as_of=as_of, desired_coverage_days=desired_coverage_days, **target_kwargs)
    gap = analyze_gap(pos, tgt)
    portfolio = optimize_portfolio(pos, tgt, gap)
    market = _market(market_opportunity)
    strategy = assess_strategy(portfolio, market, as_of=as_of)
    return pos, portfolio, market, strategy


@pytest.fixture
def comfortable():
    """Both sources SAFE, mix exactly on the 45/55 target -- no procurement events expected."""
    return _build_all(local_tons=9000.0, imported_tons=11000.0)


@pytest.fixture
def critical():
    """Both sources below safety floor -- mandatory events on both sources."""
    return _build_all(local_tons=500.0, imported_tons=900.0)


@pytest.fixture
def local_only_short():
    """Only local below its reorder point."""
    return _build_all(local_tons=500.0, imported_tons=8000.0)


# ===========================================================================
# Architectural isolation
# ===========================================================================

class TestArchitecturalIsolation:
    def test_build_execution_plan_signature(self):
        sig = inspect.signature(build_execution_plan)
        names = list(sig.parameters.keys())
        assert names[:4] == ["position", "portfolio", "market", "strategy"]

    def test_function_body_has_no_io_or_erp_calls(self):
        code_names = build_execution_plan.__code__.co_names
        for forbidden in (
            "run_orchestration", "load_inventory", "fetch_price_inputs",
            "fetch_cotton_price", "to_excel", "write_to_oracle",
        ):
            assert forbidden not in code_names

    def test_identical_inputs_produce_identical_plan(self, critical):
        pos, portfolio, market, strategy = critical
        a = build_execution_plan(pos, portfolio, market, strategy)
        b = build_execution_plan(pos, portfolio, market, strategy)
        da, db = a.to_dict(), b.to_dict()
        da.pop("generated_at"); db.pop("generated_at")
        assert da == db


# ===========================================================================
# Quantity Planning
# ===========================================================================

class TestQuantityPlanning:
    def test_no_events_when_comfortable(self, comfortable):
        pos, portfolio, market, strategy = comfortable
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.procurement_events == ()
        assert plan.total_planned_quantity_tons == 0.0
        assert "NO_PROCUREMENT_EVENTS_REQUIRED" in plan.data_quality_flags

    def test_mandatory_quantity_matches_reorder_point_shortfall(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        local_event = next(ev for ev in plan.procurement_events if ev.source == "LOCAL")
        imported_event = next(ev for ev in plan.procurement_events if ev.source == "IMPORTED")
        assert local_event.planned_quantity_tons == pytest.approx(LOCAL_ROP_TONS - pos.local_inventory_tons, abs=0.5)
        assert imported_event.planned_quantity_tons == pytest.approx(
            IMPORTED_ROP_TONS - pos.imported_inventory_tons, abs=0.5
        )

    def test_only_short_source_gets_an_event(self, local_only_short):
        pos, portfolio, market, strategy = local_only_short
        plan = build_execution_plan(pos, portfolio, market, strategy)
        sources = {ev.source for ev in plan.procurement_events}
        assert "LOCAL" in sources
        assert "IMPORTED" not in sources

    def test_quantity_never_invented_beyond_max_of_mandatory_and_mix(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            facts = ev.supporting_facts
            assert ev.planned_quantity_tons <= max(facts["mandatory_qty_tons"], facts["mix_correction_tons"]) + 0.01

    def test_total_planned_quantity_equals_sum_of_events(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        expected_total = round(sum(ev.planned_quantity_tons for ev in plan.procurement_events), 2)
        assert plan.total_planned_quantity_tons == expected_total

    def test_quantity_capped_by_storage_capacity(self):
        # Push inventory near full capacity but still below both reorder points
        # is not feasible structurally; instead verify the cap activates when
        # remaining capacity is artificially tiny via a near-full position.
        pos, portfolio, market, strategy = _build_all(local_tons=500.0, imported_tons=44000.0)
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.total_planned_quantity_tons <= pos.remaining_storage_capacity_tons + 0.5


# ===========================================================================
# Execution Windows -- not calendar dates
# ===========================================================================

class TestExecutionWindows:
    def test_all_windows_are_allowed_values(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            assert ev.preferred_execution_window in EXECUTION_WINDOWS

    def test_floor_breach_yields_immediate(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            assert ev.preferred_execution_window == "IMMEDIATE"
            assert ev.window_rule_fired in (1, 2)

    def test_no_window_is_a_calendar_date(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            assert "-" not in ev.preferred_execution_window or ev.preferred_execution_window.count("_") > 0
            # crude but effective: no ISO date pattern like YYYY-MM-DD
            import re
            assert not re.match(r"^\d{4}-\d{2}-\d{2}$", ev.preferred_execution_window)


# ===========================================================================
# Purchase Sequencing
# ===========================================================================

class TestSequencing:
    def test_execution_sequence_matches_event_order(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.execution_sequence == tuple(ev.event_id for ev in plan.procurement_events)

    def test_higher_priority_event_sequenced_first(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        scores = [ev.planning_priority_score for ev in plan.procurement_events]
        assert scores == sorted(scores, reverse=True)

    def test_dependencies_reflect_preceding_events_in_sequence(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        if len(plan.procurement_events) >= 2:
            first, second = plan.procurement_events[0], plan.procurement_events[1]
            assert first.dependencies == ()
            assert second.dependencies == (first.event_id,)

    def test_local_before_imported_when_priority_tied(self):
        # Symmetric critical shortfall -> equal priority scores -> tie-break LOCAL first.
        pos, portfolio, market, strategy = _build_all(local_tons=500.0, imported_tons=900.0)
        plan = build_execution_plan(pos, portfolio, market, strategy)
        if len(plan.procurement_events) == 2 and (
            plan.procurement_events[0].planning_priority_score == plan.procurement_events[1].planning_priority_score
        ):
            assert plan.procurement_events[0].source == "LOCAL"


# ===========================================================================
# Traceability -- reasoning chain
# ===========================================================================

class TestTraceability:
    def test_reasoning_chain_has_six_stages(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            assert len(ev.reasoning_chain) == 6

    def test_reasoning_chain_mentions_documented_stages(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            joined = " ".join(ev.reasoning_chain)
            for stage in ("Coverage Gap", "Lead-Time Exposure", "Portfolio Objective", "Market Opportunity", "Strategy", "Planned Quantity"):
                assert stage in joined


# ===========================================================================
# Expected Benefits -- never fabricate savings
# ===========================================================================

class TestExpectedBenefits:
    def test_cost_avoidance_usd_always_none(self, critical, comfortable):
        for pos, portfolio, market, strategy in (critical, comfortable):
            plan = build_execution_plan(pos, portfolio, market, strategy)
            assert plan.expected_benefits["expected_cost_avoidance_usd"] is None
            assert any("EXPECTED_COST_AVOIDANCE_USD_UNAVAILABLE" in f for f in plan.data_quality_flags)

    def test_cost_avoidance_pct_basis_passthrough(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.expected_benefits["expected_cost_avoidance_pct_basis"] == market.expected_price_advantage_pct

    def test_inventory_stability_improved_when_floor_breach_addressed(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.expected_benefits["expected_inventory_stability"] == "IMPROVED"

    def test_inventory_stability_not_assessed_when_no_events(self, comfortable):
        pos, portfolio, market, strategy = comfortable
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.expected_benefits["expected_inventory_stability"] == "NOT_ASSESSED"


# ===========================================================================
# Constraint Validation
# ===========================================================================

class TestConstraintValidation:
    def test_includes_strategy_level_constraints(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for key in strategy.constraint_validation:
            assert key in plan.constraint_validation

    def test_includes_new_quantity_constraints(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert "planned_quantity_within_storage_capacity" in plan.constraint_validation
        assert "planned_quantity_within_annual_target" in plan.constraint_validation

    def test_storage_capacity_constraint_satisfied_after_capping(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert plan.constraint_validation["planned_quantity_within_storage_capacity"] == "SATISFIED"


# ===========================================================================
# Constraints -- no execution side effects
# ===========================================================================

class TestNoExecutionSideEffects:
    def test_no_forbidden_fields_anywhere(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        d = plan.to_dict()
        forbidden = {"purchase_order", "po_number", "erp_status", "oracle_transaction_id", "executed_at"}
        assert forbidden.isdisjoint(d.keys())

    def test_module_defines_no_execution_side_effect_jargon(self):
        names = dir(plan_mod)
        for name in names:
            for forbidden in ("PurchaseOrder", "OracleWrite", "ERPUpdate", "ExecutiveReport"):
                assert forbidden not in name

    def test_no_calendar_date_fields_in_event(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        for ev in plan.procurement_events:
            d_ev = ev.__dict__ if hasattr(ev, "__dict__") else {}
        # ProcurementEvent fields themselves -- assert none is literally named "date"
        from dataclasses import fields
        field_names = {f.name for f in fields(plan_mod.ProcurementEvent)}
        assert "execution_date" not in field_names
        assert "order_date" not in field_names


# ===========================================================================
# Metadata / contract
# ===========================================================================

class TestPlanContract:
    def test_plan_is_frozen(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        with pytest.raises(Exception):
            plan.total_planned_quantity_tons = 0.0

    def test_event_is_frozen(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        with pytest.raises(Exception):
            plan.procurement_events[0].planned_quantity_tons = 0.0

    def test_to_dict_converts_tuples_to_lists(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        d = plan.to_dict()
        assert isinstance(d["procurement_events"], list)
        assert isinstance(d["execution_sequence"], list)
        assert isinstance(d["data_quality_flags"], list)

    def test_upstream_flags_passthrough(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy)
        assert tuple(plan.position_data_quality_flags) == tuple(pos.data_quality_flags)
        assert tuple(plan.portfolio_data_quality_flags) == tuple(portfolio.data_quality_flags)
        assert tuple(plan.market_data_quality_flags) == tuple(market.data_quality_flags)

    def test_as_of_overridable(self, critical):
        pos, portfolio, market, strategy = critical
        plan = build_execution_plan(pos, portfolio, market, strategy, as_of=date(2026, 1, 1))
        assert plan.as_of == "2026-01-01"
