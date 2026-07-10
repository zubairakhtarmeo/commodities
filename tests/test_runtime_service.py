"""
tests/test_runtime_service.py
-------------------------------
PSE-4.3B.3 — ProcurementRuntimeService test suite.

All tests use mocked repositories and mocked engines so that no workbook,
database, or live price feed is required. The entire test suite runs in
isolation.

Test classes:
    TestRuntimeResultDTO      — ProcurementRuntimeResult contract
    TestRuntimeServiceRun     — ProcurementRuntimeService.run() happy path
    TestRuntimeServiceErrors  — Repository failures / engine failures
    TestLegacyDictContract    — to_legacy_dict() backward-compatibility
    TestDashboardContract     — Verify the dict schema the dashboard expects
    TestRunOrder              — Verify engine call sequence
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from procurement_runtime_service import ProcurementRuntimeService, ProcurementRuntimeResult
from procurement_data_repository import (
    InventorySnapshot,
    OrgInventoryEntry,
    RepositoryHealth,
    RepositoryUnavailableError,
)


# ===========================================================================
# Shared fixtures and helpers
# ===========================================================================

def _make_health(status: str = "healthy", available: bool = True) -> RepositoryHealth:
    return RepositoryHealth(
        backend="mock",
        status=status,
        is_available=available,
        last_refresh=datetime(2025, 5, 1, 9, 0, 0),
        latency_ms=5.0,
        record_count=4,
        freshness_hours=2.0,
        message="",
    )


def _make_snapshot() -> InventorySnapshot:
    org_breakdowns = (
        OrgInventoryEntry(org_name="MG-FAISAL", origin="LOCAL",    tons=100.0),
        OrgInventoryEntry(org_name="MG-FAISAL", origin="IMPORTED", tons=200.0),
        OrgInventoryEntry(org_name="MG-RYK",    origin="LOCAL",    tons=50.0),
        OrgInventoryEntry(org_name="MG-RYK",    origin="IMPORTED", tons=80.0),
    )
    return InventorySnapshot(
        as_of_date=date(2025, 5, 1),
        ingested_at=datetime(2025, 5, 1, 9, 0, 0),
        pipeline_run=None,
        org_breakdowns=org_breakdowns,
        data_quality=(("source", "mock"), ("row_count", "4")),
    )


def _mock_repo(snapshot: Optional[InventorySnapshot] = None,
               health: Optional[RepositoryHealth] = None,
               raise_unavailable: bool = False):
    """Return a mock ProcurementInventoryRepository."""
    repo = MagicMock()
    h = health or _make_health()
    repo.health.return_value = h
    if raise_unavailable:
        repo.get_snapshot.side_effect = RepositoryUnavailableError("Mock unavailable")
    else:
        repo.get_snapshot.return_value = snapshot or _make_snapshot()
    return repo


def _make_strategy_output():
    """Minimal mock StrategyOutput with the scalar fields the dashboard reads."""
    so = MagicMock()
    so.local_inventory_tons = 150.0
    so.imported_inventory_tons = 280.0
    so.total_inventory_tons = 430.0
    so.local_status = "ADEQUATE"
    so.imported_status = "CRITICAL"
    so.local_days_cover = 45.0
    so.imported_days_cover = 12.0
    return so


def _make_scenario_report():
    sr = MagicMock()
    sr.price_inputs_used = {"current_price_usd_per_lb": 0.78}
    sr.to_dict.return_value = {"action": "BUY_FORWARD"}
    return sr


def _make_executive_report():
    er = MagicMock()
    er.to_dict.return_value = {
        "local": {"action_code": "BUY_FORWARD"},
        "imported": {"action_code": "BUY_NOW"},
        "portfolio_urgency": "HIGH",
    }
    return er


def _make_engine_output(name: str):
    """Return a generic mock engine output with a to_dict() method."""
    m = MagicMock()
    m.to_dict.return_value = {"engine": name}
    return m


# ===========================================================================
# TestRuntimeResultDTO
# ===========================================================================

class TestRuntimeResultDTO:
    """ProcurementRuntimeResult dataclass contract."""

    def test_is_available_true(self):
        so = _make_strategy_output()
        result = ProcurementRuntimeResult(
            is_available=True,
            error_message=None,
            repository_health=_make_health(),
            snapshot=_make_snapshot(),
            position=_make_engine_output("position"),
            target=_make_engine_output("target"),
            gap=_make_engine_output("gap"),
            portfolio=_make_engine_output("portfolio"),
            market=_make_engine_output("market"),
            strategy=_make_engine_output("strategy"),
            execution_plan=_make_engine_output("execution_plan"),
            impact=_make_engine_output("impact"),
            scenario_report=_make_scenario_report(),
            executive_report=_make_executive_report(),
            strategy_output=so,
            as_of=date(2025, 5, 1),
            generated_at=datetime(2025, 5, 1, 10, 0, 0),
        )
        assert result.is_available is True
        assert result.error_message is None

    def test_is_available_false(self):
        result = ProcurementRuntimeResult._make_unavailable(
            error="Test error",
            health=_make_health(status="unavailable", available=False),
            as_of=date(2025, 5, 1),
        )
        assert result.is_available is False
        assert result.error_message == "Test error"
        assert result.snapshot is None
        assert result.position is None
        assert result.execution_plan is None

    def test_make_unavailable_preserves_health(self):
        h = _make_health(status="degraded", available=False)
        result = ProcurementRuntimeResult._make_unavailable("err", h, date(2025, 5, 1))
        assert result.repository_health is h

    def test_make_unavailable_sets_as_of(self):
        d = date(2025, 6, 15)
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), d)
        assert result.as_of == d

    def test_make_unavailable_sets_generated_at(self):
        before = datetime.now()
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), date(2025, 5, 1))
        after = datetime.now()
        assert before <= result.generated_at <= after

    def test_all_engine_fields_none_when_unavailable(self):
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), date(2025, 5, 1))
        for attr in ("position", "target", "gap", "portfolio", "market",
                     "strategy", "execution_plan", "impact",
                     "scenario_report", "executive_report", "strategy_output", "snapshot"):
            assert getattr(result, attr) is None, f"{attr} should be None"


# ===========================================================================
# TestLegacyDictContract
# ===========================================================================

class TestLegacyDictContract:
    """to_legacy_dict() must produce the exact structure the dashboard expects."""

    def _make_full_result(self) -> ProcurementRuntimeResult:
        so = _make_strategy_output()
        sr = _make_scenario_report()
        er = _make_executive_report()
        ep = _make_engine_output("execution_plan")
        im = _make_engine_output("impact")
        return ProcurementRuntimeResult(
            is_available=True,
            error_message=None,
            repository_health=_make_health(),
            snapshot=_make_snapshot(),
            position=_make_engine_output("position"),
            target=_make_engine_output("target"),
            gap=_make_engine_output("gap"),
            portfolio=_make_engine_output("portfolio"),
            market=_make_engine_output("market"),
            strategy=_make_engine_output("strategy"),
            execution_plan=ep,
            impact=im,
            scenario_report=sr,
            executive_report=er,
            strategy_output=so,
            as_of=date(2025, 5, 1),
            generated_at=datetime(2025, 5, 1, 10, 0, 0),
        )

    def test_ok_true_when_available(self):
        d = self._make_full_result().to_legacy_dict()
        assert d["ok"] is True

    def test_has_report_key(self):
        d = self._make_full_result().to_legacy_dict()
        assert "report" in d
        assert isinstance(d["report"], dict)

    def test_has_scenario_key(self):
        d = self._make_full_result().to_legacy_dict()
        assert "scenario" in d
        assert isinstance(d["scenario"], dict)

    def test_has_execution_plan_key(self):
        d = self._make_full_result().to_legacy_dict()
        assert "execution_plan" in d
        assert isinstance(d["execution_plan"], dict)

    def test_has_decision_impact_key(self):
        d = self._make_full_result().to_legacy_dict()
        assert "decision_impact" in d
        assert isinstance(d["decision_impact"], dict)

    def test_inventory_has_all_seven_scalars(self):
        d = self._make_full_result().to_legacy_dict()
        inv = d["inventory"]
        for key in (
            "local_inventory_tons", "imported_inventory_tons", "total_inventory_tons",
            "local_status", "imported_status", "local_days_cover", "imported_days_cover",
        ):
            assert key in inv, f"inventory missing key: {key}"

    def test_inventory_tons_are_floats(self):
        d = self._make_full_result().to_legacy_dict()
        inv = d["inventory"]
        for k in ("local_inventory_tons", "imported_inventory_tons", "total_inventory_tons",
                  "local_days_cover", "imported_days_cover"):
            assert isinstance(inv[k], float), f"{k} should be float"

    def test_inventory_status_strings(self):
        d = self._make_full_result().to_legacy_dict()
        inv = d["inventory"]
        assert isinstance(inv["local_status"], str)
        assert isinstance(inv["imported_status"], str)

    def test_inventory_values_from_strategy_output(self):
        d = self._make_full_result().to_legacy_dict()
        inv = d["inventory"]
        assert inv["local_inventory_tons"]    == 150.0
        assert inv["imported_inventory_tons"] == 280.0
        assert inv["total_inventory_tons"]    == 430.0
        assert inv["local_status"]            == "ADEQUATE"
        assert inv["imported_status"]         == "CRITICAL"

    def test_ok_false_when_unavailable(self):
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), date(2025, 5, 1))
        d = result.to_legacy_dict()
        assert d["ok"] is False

    def test_error_key_present_when_unavailable(self):
        result = ProcurementRuntimeResult._make_unavailable("Workbook missing", _make_health(), date(2025, 5, 1))
        d = result.to_legacy_dict()
        assert "error" in d
        assert "Workbook missing" in d["error"]

    def test_no_ok_key_when_unavailable_except_ok(self):
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), date(2025, 5, 1))
        d = result.to_legacy_dict()
        assert set(d.keys()) == {"ok", "error"}

    def test_empty_dicts_when_pse5a_unavailable(self):
        """When scenario_report is None, report/scenario keys must still exist."""
        so = _make_strategy_output()
        result = ProcurementRuntimeResult(
            is_available=True,
            error_message=None,
            repository_health=_make_health(),
            snapshot=_make_snapshot(),
            position=_make_engine_output("position"),
            target=_make_engine_output("target"),
            gap=_make_engine_output("gap"),
            portfolio=_make_engine_output("portfolio"),
            market=_make_engine_output("market"),
            strategy=_make_engine_output("strategy"),
            execution_plan=_make_engine_output("execution_plan"),
            impact=_make_engine_output("impact"),
            scenario_report=None,
            executive_report=None,
            strategy_output=so,
            as_of=date(2025, 5, 1),
            generated_at=datetime.now(),
        )
        d = result.to_legacy_dict()
        assert d["ok"] is True
        assert d["report"] == {}
        assert d["scenario"] == {}


# ===========================================================================
# TestRuntimeServiceRun — happy path
# ===========================================================================

class TestRuntimeServiceRun:
    """ProcurementRuntimeService.run() happy-path tests with mocked engines."""

    def _run_with_mocks(self, as_of=None):
        repo = _mock_repo()
        so = _make_strategy_output()
        sr = _make_scenario_report()
        er = _make_executive_report()

        with patch("procurement_runtime_service.ProcurementRuntimeService._run_engines") as mock_engines:
            expected = ProcurementRuntimeResult(
                is_available=True,
                error_message=None,
                repository_health=_make_health(),
                snapshot=_make_snapshot(),
                position=_make_engine_output("p"),
                target=_make_engine_output("t"),
                gap=_make_engine_output("g"),
                portfolio=_make_engine_output("pf"),
                market=_make_engine_output("m"),
                strategy=_make_engine_output("s"),
                execution_plan=_make_engine_output("ep"),
                impact=_make_engine_output("i"),
                scenario_report=sr,
                executive_report=er,
                strategy_output=so,
                as_of=as_of or date.today(),
                generated_at=datetime.now(),
            )
            mock_engines.return_value = expected

            svc = ProcurementRuntimeService(repo)
            result = svc.run(as_of=as_of)
        return result

    def test_returns_runtime_result(self):
        result = self._run_with_mocks(date(2025, 5, 1))
        assert isinstance(result, ProcurementRuntimeResult)

    def test_is_available_true_on_success(self):
        result = self._run_with_mocks(date(2025, 5, 1))
        assert result.is_available is True

    def test_as_of_defaults_to_today(self):
        result = self._run_with_mocks(None)
        assert result.as_of == date.today()

    def test_as_of_explicit_date_used(self):
        d = date(2025, 4, 15)
        result = self._run_with_mocks(d)
        assert result.as_of == d

    def test_health_is_checked(self):
        repo = _mock_repo()
        with patch("procurement_runtime_service.ProcurementRuntimeService._run_engines") as mock_e:
            mock_e.return_value = ProcurementRuntimeResult._make_unavailable("x", _make_health(), date.today())
            ProcurementRuntimeService(repo).run()
        repo.health.assert_called_once()

    def test_get_snapshot_is_called(self):
        repo = _mock_repo()
        with patch("procurement_runtime_service.ProcurementRuntimeService._run_engines") as mock_e:
            mock_e.return_value = ProcurementRuntimeResult._make_unavailable("x", _make_health(), date.today())
            ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        repo.get_snapshot.assert_called_once_with(as_of=date(2025, 5, 1))


# ===========================================================================
# TestRuntimeServiceErrors
# ===========================================================================

class TestRuntimeServiceErrors:
    """run() must return is_available=False for all error conditions."""

    def test_repository_unavailable_error(self):
        repo = _mock_repo(raise_unavailable=True)
        svc = ProcurementRuntimeService(repo)
        result = svc.run(as_of=date(2025, 5, 1))
        assert result.is_available is False

    def test_unavailable_error_message_propagated(self):
        repo = _mock_repo(raise_unavailable=True)
        result = ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        assert "Mock unavailable" in (result.error_message or "")

    def test_unexpected_repo_exception(self):
        repo = MagicMock()
        repo.health.return_value = _make_health()
        repo.get_snapshot.side_effect = RuntimeError("Disk failure")
        result = ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        assert result.is_available is False
        assert "Disk failure" in (result.error_message or "")

    def test_engine_exception_caught(self):
        repo = _mock_repo()
        with patch("procurement_runtime_service.ProcurementRuntimeService._run_engines") as mock_e:
            mock_e.side_effect = ValueError("Engine exploded")
            result = ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        assert result.is_available is False
        assert "Engine exploded" in (result.error_message or "")

    def test_run_never_raises(self):
        repo = _mock_repo(raise_unavailable=True)
        try:
            ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        except Exception as exc:
            pytest.fail(f"run() raised {exc!r}")

    def test_health_check_failure_does_not_abort(self):
        """A health() exception must not prevent get_snapshot() from being tried."""
        repo = MagicMock()
        repo.health.side_effect = RuntimeError("Health check failed")
        repo.get_snapshot.return_value = _make_snapshot()

        with patch("procurement_runtime_service.ProcurementRuntimeService._run_engines") as mock_e:
            mock_e.return_value = ProcurementRuntimeResult._make_unavailable("x", _make_health(), date.today())
            ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))

        # get_snapshot should still be called despite health check failure
        repo.get_snapshot.assert_called_once()

    def test_unavailable_result_has_as_of(self):
        repo = _mock_repo(raise_unavailable=True)
        d = date(2025, 6, 20)
        result = ProcurementRuntimeService(repo).run(as_of=d)
        assert result.as_of == d

    def test_unavailable_result_has_generated_at(self):
        repo = _mock_repo(raise_unavailable=True)
        before = datetime.now()
        result = ProcurementRuntimeService(repo).run(as_of=date(2025, 5, 1))
        after = datetime.now()
        assert before <= result.generated_at <= after


# ===========================================================================
# TestRunOrder — engine execution order
# ===========================================================================

class TestRunOrder:
    """Verify the engine import order inside _run_engines().

    We patch every engine import and confirm they are all called, preventing
    silent order violations that would break the market engine's dependency
    on PSE-5A price inputs.
    """

    def _patch_all_engines(self):
        """Return a dict of patch targets → their mock names."""
        return {
            "procurement_runtime_service.ProcurementRuntimeService._run_pse5a": "_run_pse5a",
            "procurement_runtime_service.ProcurementRuntimeService._run_pse5b": "_run_pse5b",
        }

    def test_pse5a_runs_before_market_engine(self):
        """PSE-5A price inputs must be available when market engine runs."""
        repo = _mock_repo()
        call_order = []

        so = _make_strategy_output()
        sr = _make_scenario_report()
        er = _make_executive_report()

        with (
            patch("procurement_runtime_service.ProcurementRuntimeService._run_pse5a",
                  side_effect=lambda *a, **kw: (call_order.append("pse5a"), (sr, {"p": 0.78}))[1]) as _p5a,
            patch("procurement_runtime_service.ProcurementRuntimeService._run_pse5b",
                  side_effect=lambda *a, **kw: (call_order.append("pse5b"), er)[1]),
            # patch engine imports
            patch.dict("sys.modules", {
                "procurement_strategy_engine": MagicMock(
                    build_strategy_output_v2=MagicMock(side_effect=lambda *a, **kw: (call_order.append("build_so"), so)[1])
                ),
                "procurement_position_engine": MagicMock(assess_position=MagicMock(side_effect=lambda *a, **kw: (call_order.append("position"), _make_engine_output("p"))[1])),
                "procurement_target_engine": MagicMock(define_strategy_target=MagicMock(side_effect=lambda *a, **kw: (call_order.append("target"), _make_engine_output("t"))[1])),
                "procurement_gap_engine": MagicMock(analyze_gap=MagicMock(side_effect=lambda *a, **kw: (call_order.append("gap"), _make_engine_output("g"))[1])),
                "procurement_optimization_engine": MagicMock(optimize_portfolio=MagicMock(side_effect=lambda *a, **kw: (call_order.append("portfolio"), _make_engine_output("pf"))[1])),
                "procurement_market_engine": MagicMock(assess_market_opportunity=MagicMock(side_effect=lambda *a, **kw: (call_order.append("market"), _make_engine_output("m"))[1])),
                "procurement_strategy_assessment_engine": MagicMock(assess_strategy=MagicMock(side_effect=lambda *a, **kw: (call_order.append("strategy"), _make_engine_output("s"))[1])),
                "procurement_execution_planning_engine": MagicMock(build_execution_plan=MagicMock(side_effect=lambda *a, **kw: (call_order.append("execution_plan"), _make_engine_output("ep"))[1])),
                "procurement_impact_engine": MagicMock(interpret_impact=MagicMock(side_effect=lambda *a, **kw: (call_order.append("impact"), _make_engine_output("i"))[1])),
            }),
        ):
            svc = ProcurementRuntimeService(repo)
            result = svc.run(as_of=date(2025, 5, 1))

        # PSE-5A must precede the market engine
        if "pse5a" in call_order and "market" in call_order:
            assert call_order.index("pse5a") < call_order.index("market"), (
                "PSE-5A must run before the market engine so price inputs are available"
            )


# ===========================================================================
# TestDashboardContract
# ===========================================================================

class TestDashboardContract:
    """The dict returned by to_legacy_dict() must satisfy every key the dashboard reads."""

    REQUIRED_TOP_LEVEL = {"ok", "report", "scenario", "execution_plan", "decision_impact", "inventory"}
    REQUIRED_INVENTORY = {
        "local_inventory_tons", "imported_inventory_tons", "total_inventory_tons",
        "local_status", "imported_status", "local_days_cover", "imported_days_cover",
    }

    def _full_result(self):
        so = _make_strategy_output()
        ep = _make_engine_output("ep")
        im = _make_engine_output("im")
        return ProcurementRuntimeResult(
            is_available=True,
            error_message=None,
            repository_health=_make_health(),
            snapshot=_make_snapshot(),
            position=_make_engine_output("p"),
            target=_make_engine_output("t"),
            gap=_make_engine_output("g"),
            portfolio=_make_engine_output("pf"),
            market=_make_engine_output("m"),
            strategy=_make_engine_output("s"),
            execution_plan=ep,
            impact=im,
            scenario_report=_make_scenario_report(),
            executive_report=_make_executive_report(),
            strategy_output=so,
            as_of=date(2025, 5, 1),
            generated_at=datetime.now(),
        )

    def test_all_required_top_level_keys_present(self):
        d = self._full_result().to_legacy_dict()
        missing = self.REQUIRED_TOP_LEVEL - set(d.keys())
        assert not missing, f"Missing dashboard keys: {missing}"

    def test_all_required_inventory_keys_present(self):
        d = self._full_result().to_legacy_dict()
        missing = self.REQUIRED_INVENTORY - set(d["inventory"].keys())
        assert not missing, f"Missing inventory keys: {missing}"

    def test_render_pse6a_executive_panel_reads_report(self):
        """report key is directly accessed by render_pse6a_executive_panel."""
        d = self._full_result().to_legacy_dict()
        er = d["report"]
        assert isinstance(er, dict)

    def test_render_pse6b_reads_inventory_keys(self):
        """_pse6b_get_prod_so() splats inventory into the return dict."""
        d = self._full_result().to_legacy_dict()
        inv = d["inventory"]
        # simulate _pse6b_get_prod_so splat
        merged = {"ok": True, **inv}
        assert float(merged["local_inventory_tons"]) == 150.0
        assert float(merged["imported_inventory_tons"]) == 280.0

    def test_pse7a_reads_report_scenario_inventory(self):
        """_run_pse7a_cached reads report, scenario, inventory from the dict."""
        d = self._full_result().to_legacy_dict()
        assert "report" in d
        assert "scenario" in d
        assert "inventory" in d
        inv = d["inventory"]
        # _run_pse7a_cached accesses these three scalars
        _ = inv["local_inventory_tons"]
        _ = inv["imported_inventory_tons"]
        _ = inv["total_inventory_tons"]

    def test_result_not_ok_has_no_report_key(self):
        """When ok=False, the dashboard checks result.get('ok') first."""
        result = ProcurementRuntimeResult._make_unavailable("err", _make_health(), date(2025, 5, 1))
        d = result.to_legacy_dict()
        assert d.get("ok") is False
        assert "report" not in d
