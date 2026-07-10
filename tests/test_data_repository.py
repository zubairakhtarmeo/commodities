"""
test_data_repository.py
------------------------
PSE-4.3B.1 -- Tests for procurement_data_repository.py.

Coverage:
    TestDomainModels        -- OrgInventoryEntry / InventorySnapshot immutability
                               and to_origin_summary_df() contract
    TestRepositoryHealth    -- RepositoryHealth immutability and field invariants
    TestRepositoryContract  -- Abstract interface, WorkbookInventoryRepository
                               behaviour when workbook is absent
    TestEngineCompatibility -- Synthetic snapshots fed into the full PSE chain
    TestIntegration         -- Real workbook (guarded with @pytest.mark.integration)
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors conftest.py)
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).parent.parent
SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from procurement_data_repository import (
    OrgInventoryEntry,
    InventorySnapshot,
    RepositoryHealth,
    RepositoryUnavailableError,
    ProcurementInventoryRepository,
    WorkbookInventoryRepository,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_AS_OF = date(2026, 6, 30)
_NOW   = datetime(2026, 6, 30, 12, 0, 0)

_SAMPLE_ENTRIES = (
    OrgInventoryEntry(org_name="MTM", origin="LOCAL",    tons=9000.0),
    OrgInventoryEntry(org_name="MTM", origin="IMPORTED", tons=11000.0),
)

_SAMPLE_DQ: tuple[tuple[str, str], ...] = (
    ("source",               "workbook"),
    ("workbook",             "Strategies.xlsx"),
    ("row_count",            "2"),
    ("conversion_validated", "true"),
    ("sanity_warning_count", "0"),
    ("read_latency_ms",      "123.4"),
)


def _make_snapshot(
    entries=_SAMPLE_ENTRIES,
    dq=_SAMPLE_DQ,
    as_of=_AS_OF,
    ingested_at=_NOW,
) -> InventorySnapshot:
    return InventorySnapshot(
        as_of_date=as_of,
        ingested_at=ingested_at,
        pipeline_run=None,
        org_breakdowns=entries,
        data_quality=dq,
    )


# ===========================================================================
# Domain model tests
# ===========================================================================

class TestDomainModels:
    # ------------------------------------------------------------------
    # OrgInventoryEntry
    # ------------------------------------------------------------------

    def test_org_entry_is_frozen(self):
        entry = OrgInventoryEntry(org_name="MTM", origin="LOCAL", tons=500.0)
        with pytest.raises((AttributeError, TypeError)):
            entry.tons = 999.0  # type: ignore[misc]

    def test_org_entry_fields(self):
        entry = OrgInventoryEntry(org_name="GBF", origin="IMPORTED", tons=3000.5)
        assert entry.org_name == "GBF"
        assert entry.origin   == "IMPORTED"
        assert entry.tons     == 3000.5

    def test_org_entry_is_hashable(self):
        entry = OrgInventoryEntry(org_name="MTM", origin="LOCAL", tons=100.0)
        s = {entry}
        assert entry in s

    # ------------------------------------------------------------------
    # InventorySnapshot
    # ------------------------------------------------------------------

    def test_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.as_of_date = date(2025, 1, 1)  # type: ignore[misc]

    def test_snapshot_is_hashable(self):
        snap = _make_snapshot()
        s = {snap}
        assert snap in s

    def test_data_quality_is_tuple_of_tuples(self):
        snap = _make_snapshot()
        assert isinstance(snap.data_quality, tuple)
        for item in snap.data_quality:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_data_quality_as_dict_round_trips(self):
        snap = _make_snapshot()
        d = snap.data_quality_as_dict()
        assert d["source"]   == "workbook"
        assert d["workbook"] == "Strategies.xlsx"
        assert d["row_count"] == "2"

    # ------------------------------------------------------------------
    # to_origin_summary_df — columns, order, dtypes
    # ------------------------------------------------------------------

    def test_to_origin_summary_df_has_three_columns(self):
        df = _make_snapshot().to_origin_summary_df()
        assert list(df.columns) == ["org_name", "origin", "tons"]

    def test_to_origin_summary_df_column_dtypes(self):
        df = _make_snapshot().to_origin_summary_df()
        # is_string_dtype accepts both legacy object and pandas 2.x StringDtype
        assert pd.api.types.is_string_dtype(df["org_name"])
        assert pd.api.types.is_string_dtype(df["origin"])
        assert df["tons"].dtype == "float64"

    def test_to_origin_summary_df_row_count(self):
        df = _make_snapshot().to_origin_summary_df()
        assert len(df) == 2

    def test_to_origin_summary_df_values(self):
        df = _make_snapshot().to_origin_summary_df()
        local_row = df[df["origin"] == "LOCAL"].iloc[0]
        assert local_row["org_name"] == "MTM"
        assert local_row["tons"]     == pytest.approx(9000.0)

        imported_row = df[df["origin"] == "IMPORTED"].iloc[0]
        assert imported_row["tons"] == pytest.approx(11000.0)

    def test_to_origin_summary_df_empty_breakdowns(self):
        snap = InventorySnapshot(
            as_of_date=_AS_OF,
            ingested_at=_NOW,
            pipeline_run=None,
            org_breakdowns=(),
            data_quality=(),
        )
        df = snap.to_origin_summary_df()
        assert list(df.columns) == ["org_name", "origin", "tons"]
        assert len(df) == 0
        assert df["tons"].dtype == "float64"

    def test_to_origin_summary_df_is_new_object_each_call(self):
        snap = _make_snapshot()
        df1 = snap.to_origin_summary_df()
        df2 = snap.to_origin_summary_df()
        assert df1 is not df2

    def test_to_origin_summary_df_mutation_does_not_affect_snapshot(self):
        snap = _make_snapshot()
        df = snap.to_origin_summary_df()
        df.loc[0, "tons"] = 0.0
        df2 = snap.to_origin_summary_df()
        assert df2["tons"].sum() == pytest.approx(20000.0)

    def test_to_origin_summary_df_tons_are_float(self):
        entries = (
            OrgInventoryEntry(org_name="X", origin="LOCAL", tons=1),  # int-like
        )
        snap = InventorySnapshot(
            as_of_date=_AS_OF, ingested_at=_NOW, pipeline_run=None,
            org_breakdowns=entries, data_quality=(),
        )
        df = snap.to_origin_summary_df()
        assert df["tons"].dtype == "float64"


# ===========================================================================
# RepositoryHealth tests
# ===========================================================================

class TestRepositoryHealth:
    def _healthy(self) -> RepositoryHealth:
        return RepositoryHealth(
            backend="workbook",
            status="healthy",
            is_available=True,
            last_refresh=_NOW,
            latency_ms=45.2,
            record_count=6,
            freshness_hours=2.1,
            message="Workbook available — last modified 2026-06-30 12:00",
        )

    def test_health_is_frozen(self):
        h = self._healthy()
        with pytest.raises((AttributeError, TypeError)):
            h.status = "degraded"  # type: ignore[misc]

    def test_health_is_hashable(self):
        h = self._healthy()
        s = {h}
        assert h in s

    def test_unavailable_health(self):
        h = RepositoryHealth(
            backend="workbook", status="unavailable", is_available=False,
            last_refresh=None, latency_ms=None, record_count=None,
            freshness_hours=None, message="Workbook not found: Strategies.xlsx",
        )
        assert not h.is_available
        assert h.status == "unavailable"
        assert h.last_refresh is None

    def test_allowed_statuses(self):
        allowed = {"healthy", "degraded", "unavailable"}
        for s in allowed:
            h = RepositoryHealth(
                backend="workbook", status=s, is_available=(s != "unavailable"),
                last_refresh=None, latency_ms=None, record_count=None,
                freshness_hours=None, message="",
            )
            assert h.status == s


# ===========================================================================
# Repository contract tests
# ===========================================================================

class TestRepositoryContract:
    def test_abstract_class_not_instantiable(self):
        with pytest.raises(TypeError):
            ProcurementInventoryRepository()  # type: ignore[abstract]

    def test_workbook_repo_is_subclass(self):
        assert issubclass(WorkbookInventoryRepository, ProcurementInventoryRepository)

    def test_workbook_repo_implements_interface(self):
        import inspect
        repo = WorkbookInventoryRepository("/nonexistent/path/Strategies.xlsx")
        assert callable(repo.get_snapshot)
        assert callable(repo.health)
        assert not inspect.isabstract(repo)

    def test_health_returns_unavailable_when_file_missing(self, tmp_path):
        repo = WorkbookInventoryRepository(tmp_path / "missing.xlsx")
        h = repo.health()
        assert h.is_available is False
        assert h.status == "unavailable"
        assert "missing.xlsx" in h.message

    def test_health_never_raises(self, tmp_path):
        repo = WorkbookInventoryRepository(tmp_path / "nowhere" / "missing.xlsx")
        h = repo.health()
        assert isinstance(h, RepositoryHealth)

    def test_get_snapshot_raises_unavailable_when_file_missing(self, tmp_path):
        repo = WorkbookInventoryRepository(tmp_path / "missing.xlsx")
        with pytest.raises(RepositoryUnavailableError):
            repo.get_snapshot()

    def test_repository_unavailable_error_is_runtime_error(self):
        assert issubclass(RepositoryUnavailableError, RuntimeError)

    def test_health_returns_healthy_when_file_present(self, tmp_path):
        wb = tmp_path / "Strategies.xlsx"
        wb.write_bytes(b"fake")
        repo = WorkbookInventoryRepository(wb)
        h = repo.health()
        assert h.is_available is True
        assert h.backend == "workbook"
        assert h.freshness_hours is not None
        assert h.freshness_hours >= 0.0

    def test_health_reports_correct_backend(self, tmp_path):
        repo = WorkbookInventoryRepository(tmp_path / "missing.xlsx")
        h = repo.health()
        assert h.backend == "workbook"


# ===========================================================================
# Engine compatibility tests (synthetic data, no workbook)
# ===========================================================================

class TestEngineCompatibility:
    """Verify that InventorySnapshot.to_origin_summary_df() can drive the
    full PSE chain (PSE-3.0 through PSE-3.6) identically to direct
    run_orchestration() output."""

    def _run_pse_chain(self, snap: InventorySnapshot):
        from procurement_strategy_engine import build_strategy_output_v2
        from procurement_position_engine import assess_position
        from procurement_target_engine import define_strategy_target
        from procurement_gap_engine import analyze_gap
        from procurement_optimization_engine import optimize_portfolio
        from procurement_market_engine import assess_market_opportunity
        from procurement_strategy_assessment_engine import assess_strategy
        from procurement_execution_planning_engine import build_execution_plan

        df       = snap.to_origin_summary_df()
        so       = build_strategy_output_v2(df, run_date=_AS_OF.isoformat())
        pos      = assess_position(so, as_of=_AS_OF)
        tgt      = define_strategy_target(as_of=_AS_OF, desired_coverage_days=45.0)
        gap      = analyze_gap(pos, tgt)
        portfolio = optimize_portfolio(pos, tgt, gap)
        market   = assess_market_opportunity(
            market_price_inputs={"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.785},
            as_of=_AS_OF,
        )
        strategy = assess_strategy(portfolio, market, as_of=_AS_OF)
        plan     = build_execution_plan(pos, portfolio, market, strategy, as_of=_AS_OF)
        return so, pos, tgt, gap, portfolio, market, strategy, plan

    def test_full_pse_chain_runs_without_error(self):
        snap = _make_snapshot()
        result = self._run_pse_chain(snap)
        assert len(result) == 8

    def test_pse_chain_strategy_output_total_matches_snapshot(self):
        snap = _make_snapshot()
        so, pos, *_ = self._run_pse_chain(snap)
        assert so.total_inventory_tons == pytest.approx(20000.0, rel=1e-3)

    def test_pse_chain_position_snapshot_not_none(self):
        snap = _make_snapshot()
        _, pos, *_ = self._run_pse_chain(snap)
        assert pos is not None
        assert pos.total_inventory_tons > 0

    def test_pse_chain_execution_plan_not_none(self):
        snap = _make_snapshot()
        *_, plan = self._run_pse_chain(snap)
        assert plan is not None

    def test_critical_inventory_triggers_procurement(self):
        entries = (
            OrgInventoryEntry(org_name="MTM", origin="LOCAL",    tons=500.0),
            OrgInventoryEntry(org_name="MTM", origin="IMPORTED", tons=900.0),
        )
        snap = InventorySnapshot(
            as_of_date=_AS_OF, ingested_at=_NOW, pipeline_run=None,
            org_breakdowns=entries, data_quality=(),
        )
        *_, plan = self._run_pse_chain(snap)
        assert plan.total_planned_quantity_tons > 0

    def test_near_full_inventory_triggers_preservation(self):
        entries = (
            OrgInventoryEntry(org_name="MTM", origin="LOCAL",    tons=20000.0),
            OrgInventoryEntry(org_name="MTM", origin="IMPORTED", tons=22000.0),
        )
        snap = InventorySnapshot(
            as_of_date=_AS_OF, ingested_at=_NOW, pipeline_run=None,
            org_breakdowns=entries, data_quality=(),
        )
        *_, strategy, plan = self._run_pse_chain(snap)
        assert strategy.overall_procurement_posture == "INVENTORY_PRESERVATION"

    def test_df_from_snapshot_has_correct_shape_for_engine(self):
        snap = _make_snapshot()
        df = snap.to_origin_summary_df()
        assert set(df.columns) == {"org_name", "origin", "tons"}
        assert len(df) == len(_SAMPLE_ENTRIES)

    def test_multiorg_snapshot_runs_correctly(self):
        entries = (
            OrgInventoryEntry(org_name="MTM", origin="LOCAL",    tons=5000.0),
            OrgInventoryEntry(org_name="GBF", origin="LOCAL",    tons=2000.0),
            OrgInventoryEntry(org_name="MTM", origin="IMPORTED", tons=8000.0),
            OrgInventoryEntry(org_name="GBF", origin="IMPORTED", tons=3000.0),
        )
        snap = InventorySnapshot(
            as_of_date=_AS_OF, ingested_at=_NOW, pipeline_run=None,
            org_breakdowns=entries, data_quality=(),
        )
        so, pos, *_ = self._run_pse_chain(snap)
        assert so.total_inventory_tons == pytest.approx(18000.0, rel=1e-3)

    def test_impact_engine_accepts_snapshot_derived_plan(self):
        from procurement_impact_engine import interpret_impact
        snap = _make_snapshot()
        _, pos, _, _, portfolio, market, strategy, plan = self._run_pse_chain(snap)
        impact = interpret_impact(plan, strategy, portfolio, market, pos)
        assert impact is not None
        assert isinstance(impact.inventory_outlook, str)
        assert len(impact.inventory_outlook) > 0


# ===========================================================================
# Integration tests (real workbook)
# ===========================================================================

class TestIntegration:
    """Tests that require Strategies.xlsx on disk.

    Deselect in CI with:  pytest -m "not integration"
    """

    @pytest.mark.integration
    def test_get_snapshot_returns_inventory_snapshot(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert isinstance(snap, InventorySnapshot)

    @pytest.mark.integration
    def test_get_snapshot_has_entries(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) > 0

    @pytest.mark.integration
    def test_get_snapshot_entries_have_valid_origin(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        valid_origins = {"LOCAL", "IMPORTED", "UNKNOWN"}
        for entry in snap.org_breakdowns:
            assert entry.origin in valid_origins, (
                f"Unexpected origin '{entry.origin}' for org '{entry.org_name}'"
            )

    @pytest.mark.integration
    def test_get_snapshot_tons_are_positive(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        for entry in snap.org_breakdowns:
            assert entry.tons >= 0.0, (
                f"Negative tons for {entry.org_name}/{entry.origin}: {entry.tons}"
            )

    @pytest.mark.integration
    def test_get_snapshot_data_quality_has_source_key(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert dq.get("source") == "workbook"

    @pytest.mark.integration
    def test_get_snapshot_data_quality_records_latency(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert "read_latency_ms" in dq
        latency = float(dq["read_latency_ms"])
        assert latency > 0.0

    @pytest.mark.integration
    def test_health_returns_available_with_real_workbook(self, workbook_path):
        repo = WorkbookInventoryRepository(workbook_path)
        h = repo.health()
        assert h.is_available is True
        assert h.status in {"healthy", "degraded"}
        assert h.freshness_hours is not None

    @pytest.mark.integration
    def test_to_origin_summary_df_matches_run_orchestration(self, workbook_path):
        """InventorySnapshot.to_origin_summary_df() must produce the same
        DataFrame slice that run_orchestration() feeds to build_strategy_output_v2().
        """
        from procurement_orchestrator import run_orchestration

        # Direct path (current behaviour in dashboard)
        result  = run_orchestration(workbook_path=str(workbook_path))
        direct_df = (
            result["origin_summary_converted"][["org_name", "origin", "tons"]]
            .reset_index(drop=True)
        )

        # Repository path
        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        repo_df = snap.to_origin_summary_df().reset_index(drop=True)

        # Shape
        assert list(repo_df.columns) == list(direct_df.columns)
        assert len(repo_df) == len(direct_df)

        # Values (sort both by org_name+origin to eliminate row-order sensitivity)
        sort_keys = ["org_name", "origin"]
        direct_sorted = direct_df.sort_values(sort_keys).reset_index(drop=True)
        repo_sorted   = repo_df.sort_values(sort_keys).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            repo_sorted, direct_sorted,
            check_dtype=True, check_like=False, rtol=1e-6,
        )

    @pytest.mark.integration
    def test_full_pse_chain_via_repository_produces_valid_plan(self, workbook_path):
        """Snapshot from WorkbookInventoryRepository drives the full PSE chain."""
        from procurement_strategy_engine import build_strategy_output_v2
        from procurement_position_engine import assess_position
        from procurement_target_engine import define_strategy_target
        from procurement_gap_engine import analyze_gap
        from procurement_optimization_engine import optimize_portfolio
        from procurement_market_engine import assess_market_opportunity
        from procurement_strategy_assessment_engine import assess_strategy
        from procurement_execution_planning_engine import build_execution_plan

        repo = WorkbookInventoryRepository(workbook_path)
        snap = repo.get_snapshot(as_of=_AS_OF)
        df   = snap.to_origin_summary_df()

        so       = build_strategy_output_v2(df, run_date=_AS_OF.isoformat())
        pos      = assess_position(so, as_of=_AS_OF)
        tgt      = define_strategy_target(as_of=_AS_OF, desired_coverage_days=45.0)
        gap      = analyze_gap(pos, tgt)
        portfolio = optimize_portfolio(pos, tgt, gap)
        market   = assess_market_opportunity(
            market_price_inputs={"current_price_usd_per_lb": 0.78, "forecast_h1_usd_per_lb": 0.785},
            as_of=_AS_OF,
        )
        strategy = assess_strategy(portfolio, market, as_of=_AS_OF)
        plan     = build_execution_plan(pos, portfolio, market, strategy, as_of=_AS_OF)

        assert plan is not None
        assert plan.total_planned_quantity_tons >= 0.0
        assert plan.next_review_date is not None
        assert strategy.overall_procurement_posture in {
            "EMERGENCY_PROCUREMENT", "URGENT_PROCUREMENT", "BALANCED_ACCUMULATION",
            "DEFERRED_PROCUREMENT", "PRICE_CAPTURE", "INVENTORY_PRESERVATION",
            "FORWARD_COVERAGE",
        }
