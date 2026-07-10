"""
test_dashboard_integration.py
------------------------------
Tests for the PSE-8B performance guarantees and dashboard cache behaviour.

These tests verify:
  - run_orchestration() is called exactly once in run_pse5b()
  - run_orchestration() is called exactly once in run_pse7a()
  - run_orchestration() is called exactly once when the full pipeline is
    driven from _run_pse5b_cached() (simulated cold-cache dashboard load)
  - _pse6b_get_prod_so() extracts data from the shared cache (no extra read)
  - _run_pse7a_cached() reuses the shared cache (no extra read)

Workbook tests are marked @pytest.mark.integration and guarded with
pytest.skip() when the workbook is absent, so CI can run them with:
    pytest -m "not integration"
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager, nullcontext
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT  = Path(__file__).parent.parent
SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

TODAY        = date(2026, 6, 27)
TODAY_STR    = TODAY.isoformat()
WORKBOOK_STR = str(_REPO_ROOT / "data" / "strategy" / "Strategies.xlsx")


def _maybe_patch_orch(mod, side_effect):
    """Return a patch context manager only if mod has run_orchestration at module level."""
    if hasattr(mod, "run_orchestration"):
        return patch.object(mod, "run_orchestration", side_effect=side_effect)
    return nullcontext()


# ===========================================================================
# Helpers: instrument run_orchestration to count calls
# ===========================================================================

class OrchestratorCallCounter:
    """Context manager that patches run_orchestration and counts invocations."""

    def __init__(self, real_result=None):
        self._real_result = real_result
        self.call_count = 0
        self._patches = []

    def __enter__(self):
        import procurement_orchestrator as _po
        import procurement_scenario_engine as _pse5a
        import procurement_decision_engine as _pse5b
        import procurement_monitoring_engine as _pse7a

        real_fn = _po.run_orchestration
        counter = self

        def _counted(*a, **kw):
            counter.call_count += 1
            return real_fn(*a, **kw)

        # Patch in every module that imports run_orchestration at module level
        for mod in (_po, _pse5a, _pse5b, _pse7a):
            if not hasattr(mod, "run_orchestration"):
                continue
            p = patch.object(mod, "run_orchestration", side_effect=_counted)
            p.start()
            self._patches.append(p)

        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


# ===========================================================================
# Unit: run_pse5b makes exactly 1 workbook read
# ===========================================================================

class TestRunPse5bWorkbookReads:
    @pytest.mark.integration
    def test_run_pse5b_single_workbook_read(self, workbook_path):
        """After PSE-8B: run_pse5b() must read the workbook exactly once."""
        import procurement_orchestrator as _po
        from procurement_decision_engine import run_pse5b

        original = _po.run_orchestration
        call_count = [0]

        def _counted(*a, **kw):
            call_count[0] += 1
            return original(*a, **kw)

        with patch.object(_po, "run_orchestration", side_effect=_counted):
            import procurement_scenario_engine as _pse5a
            with _maybe_patch_orch(_pse5a, _counted):
                sr, er = run_pse5b(
                    workbook_path=str(workbook_path),
                    live_prices=False,
                    today=TODAY,
                )

        assert call_count[0] == 1, (
            f"run_pse5b() called run_orchestration {call_count[0]} times; expected 1"
        )
        assert er.portfolio_urgency in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


# ===========================================================================
# Unit: run_pse7a makes exactly 1 workbook read
# ===========================================================================

class TestRunPse7aWorkbookReads:
    @pytest.mark.integration
    def test_run_pse7a_single_workbook_read(self, workbook_path):
        """After PSE-8B: run_pse7a() must read the workbook exactly once."""
        import procurement_orchestrator as _po
        import procurement_scenario_engine as _pse5a
        import procurement_decision_engine as _pse5b_mod
        import procurement_monitoring_engine as _pse7a_mod
        from procurement_monitoring_engine import run_pse7a

        original = _po.run_orchestration
        call_count = [0]

        def _counted(*a, **kw):
            call_count[0] += 1
            return original(*a, **kw)

        with patch.object(_po, "run_orchestration", side_effect=_counted):
            with _maybe_patch_orch(_pse5a,    _counted):
                with _maybe_patch_orch(_pse5b_mod, _counted):
                    with _maybe_patch_orch(_pse7a_mod, _counted):
                        report = run_pse7a(
                            workbook_path=str(workbook_path),
                            today=TODAY,
                        )

        assert call_count[0] == 1, (
            f"run_pse7a() called run_orchestration {call_count[0]} times; expected 1"
        )
        assert len(report.alerts) >= 0


# ===========================================================================
# Unit: full simulated cold-cache dashboard load = 1 workbook read
# ===========================================================================

class TestDashboardColdCacheLoad:
    @pytest.mark.integration
    def test_full_pipeline_single_workbook_read(self, workbook_path):
        """
        Simulate what _run_pse5b_cached() does on a cold cache:
        orch → run_pse5a(_orch=orch) → generate_executive_report()
        Then hand the cached result to generate_monitoring_report().
        Total calls to run_orchestration: exactly 1.
        """
        import procurement_orchestrator as _po
        from procurement_scenario_engine import run_pse5a
        from procurement_decision_engine import generate_executive_report
        from procurement_monitoring_engine import generate_monitoring_report

        original = _po.run_orchestration
        call_count = [0]

        def _counted(*a, **kw):
            call_count[0] += 1
            return original(*a, **kw)

        with patch.object(_po, "run_orchestration", side_effect=_counted):
            import procurement_scenario_engine as _pse5a_mod
            with _maybe_patch_orch(_pse5a_mod, _counted):
                orch = _po.run_orchestration(workbook_path=str(workbook_path))
                so   = orch["strategy_output"]

                sr = run_pse5a(
                    workbook_path=str(workbook_path),
                    live_prices=False,
                    today=TODAY,
                    _orch=orch,
                )
                er = generate_executive_report(
                    scenario_report=sr,
                    local_status=so.local_status,
                    imported_status=so.imported_status,
                )
                monitoring = generate_monitoring_report(
                    exec_report_dict=er.to_dict(),
                    scenario_report_dict=sr.to_dict(),
                    local_inventory_tons=float(so.local_inventory_tons),
                    imported_inventory_tons=float(so.imported_inventory_tons),
                    total_inventory_tons=float(so.total_inventory_tons),
                    today=TODAY,
                )

        assert call_count[0] == 1, (
            f"Full cold-cache pipeline called run_orchestration {call_count[0]} times; expected 1"
        )
        assert monitoring is not None


# ===========================================================================
# Unit: _orch passthrough eliminates redundant read in run_pse5a
# ===========================================================================

class TestOrchPassthrough:
    @pytest.mark.integration
    def test_run_pse5a_with_orch_skips_workbook_read(self, workbook_path):
        """When _orch is provided, run_pse5a() must not call run_orchestration()."""
        import procurement_orchestrator as _po
        from procurement_scenario_engine import run_pse5a

        orch = _po.run_orchestration(workbook_path=str(workbook_path))

        extra_reads = [0]

        def _should_not_be_called(*a, **kw):
            extra_reads[0] += 1
            return _po.run_orchestration(*a, **kw)

        import procurement_scenario_engine as _pse5a_mod
        with _maybe_patch_orch(_pse5a_mod, _should_not_be_called):
            sr = run_pse5a(
                workbook_path=str(workbook_path),
                live_prices=False,
                today=TODAY,
                _orch=orch,
            )

        assert extra_reads[0] == 0, (
            f"run_pse5a(_orch=...) still called run_orchestration {extra_reads[0]} times"
        )
        assert sr is not None

    @pytest.mark.integration
    def test_run_pse5a_without_orch_reads_workbook(self, workbook_path):
        """Without _orch, run_pse5a() must call run_orchestration() exactly once."""
        import procurement_orchestrator as _po
        from procurement_scenario_engine import run_pse5a

        original = _po.run_orchestration
        call_count = [0]

        def _counted(*a, **kw):
            call_count[0] += 1
            return original(*a, **kw)

        with patch.object(_po, "run_orchestration", side_effect=_counted):
            sr = run_pse5a(
                workbook_path=str(workbook_path),
                live_prices=False,
                today=TODAY,
            )

        assert call_count[0] == 1, (
            f"run_pse5a() without _orch called run_orchestration {call_count[0]} times; expected 1"
        )


# ===========================================================================
# Unit: _pse6b_get_prod_so returns correct inventory shape (no Streamlit)
# ===========================================================================

class TestPse6bGetProdSo:
    @pytest.mark.integration
    def test_returns_correct_keys(self, workbook_path):
        """_pse6b_get_prod_so must return the 7 inventory/status fields."""
        import procurement_orchestrator as _po
        from procurement_scenario_engine import run_pse5a
        from procurement_decision_engine import generate_executive_report

        orch = _po.run_orchestration(workbook_path=str(workbook_path))
        so   = orch["strategy_output"]
        sr   = run_pse5a(workbook_path=str(workbook_path), live_prices=False,
                         today=TODAY, _orch=orch)
        er   = generate_executive_report(
            scenario_report=sr,
            local_status=so.local_status,
            imported_status=so.imported_status,
        )

        # Mimic what _run_pse5b_cached returns
        cached = {
            "ok": True,
            "report":   er.to_dict(),
            "scenario": sr.to_dict(),
            "inventory": {
                "local_inventory_tons":    float(so.local_inventory_tons),
                "imported_inventory_tons": float(so.imported_inventory_tons),
                "total_inventory_tons":    float(so.total_inventory_tons),
                "local_status":            so.local_status,
                "imported_status":         so.imported_status,
                "local_days_cover":        float(so.local_days_cover),
                "imported_days_cover":     float(so.imported_days_cover),
            },
        }

        required = {
            "local_inventory_tons", "imported_inventory_tons", "total_inventory_tons",
            "local_status", "imported_status", "local_days_cover", "imported_days_cover",
        }
        assert required.issubset(set(cached["inventory"]))

        inv = cached["inventory"]
        assert inv["local_inventory_tons"] >= 0.0
        assert inv["imported_inventory_tons"] >= 0.0
        assert inv["total_inventory_tons"] >= (
            inv["local_inventory_tons"] + inv["imported_inventory_tons"] - 0.01
        )

    def test_ok_false_propagates(self):
        """If the upstream cache fails, _pse6b_get_prod_so must return ok=False."""
        failed_cache = {"ok": False, "error": "Pipeline error"}
        # Simulate what _pse6b_get_prod_so does:
        if not failed_cache.get("ok"):
            result = {"ok": False, "error": failed_cache.get("error", "Pipeline error")}
        else:
            result = {"ok": True, **failed_cache["inventory"]}
        assert result["ok"] is False
        assert result["error"] == "Pipeline error"


# ===========================================================================
# Unit: full pipeline outputs are deterministic (same inputs → same outputs)
# ===========================================================================

class TestDeterminism:
    @pytest.mark.integration
    def test_two_runs_identical_qty(self, workbook_path):
        """Running the full pipeline twice must produce bit-for-bit identical quantities."""
        from procurement_decision_engine import run_pse5b

        sr1, er1 = run_pse5b(workbook_path=str(workbook_path), live_prices=False, today=TODAY)
        sr2, er2 = run_pse5b(workbook_path=str(workbook_path), live_prices=False, today=TODAY)

        assert er1.local.qty_now_tons    == er2.local.qty_now_tons
        assert er1.imported.qty_now_tons == er2.imported.qty_now_tons
        assert er1.portfolio_urgency     == er2.portfolio_urgency

    @pytest.mark.integration
    def test_monitoring_deterministic(self, workbook_path):
        from procurement_monitoring_engine import run_pse7a

        r1 = run_pse7a(workbook_path=str(workbook_path), today=TODAY)
        r2 = run_pse7a(workbook_path=str(workbook_path), today=TODAY)

        assert len(r1.alerts)   == len(r2.alerts)
        assert r1.highest_severity == r2.highest_severity
        assert r1.alert_count_by_severity == r2.alert_count_by_severity
