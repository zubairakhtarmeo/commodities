"""
test_supabase_repository.py
-----------------------------
PSE-4.3B.2 — Isolated tests for SupabaseInventoryRepository.

All tests mock the Supabase client. No live database is required.

Coverage:
    TestParseIsoDt          -- _parse_iso_datetime helper
    TestSnapshotCreation    -- get_snapshot() happy path
    TestSnapshotDataFrame   -- to_origin_summary_df() contract
    TestUnavailableHandling -- connection failures, empty table, query errors
    TestMalformedData       -- rows with missing / bad fields
    TestHealthCheck         -- health() in all status variants
    TestEngineCompatibility -- snapshot from Supabase drives the PSE chain
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_REPO_ROOT  = Path(__file__).parent.parent
SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from procurement_data_repository import (
    OrgInventoryEntry,
    InventorySnapshot,
    RepositoryHealth,
    RepositoryUnavailableError,
    SupabaseInventoryRepository,
    _parse_iso_datetime,
)

# ---------------------------------------------------------------------------
# Mock-client factory
# ---------------------------------------------------------------------------

def _response(data, count=None):
    """Create a mock APIResponse with .data and .count."""
    r = MagicMock()
    r.data  = data
    r.count = count
    return r


def _make_client(*responses):
    """Return a mock Supabase client that yields responses in sequence.

    Each positional argument is passed directly to _response() if it is a
    tuple of (data, count), or used as a raw mock response if it is already
    a MagicMock.

    The query builder chain (table().select().order().limit().eq().execute())
    is fully stubbed. execute() returns responses in the order provided.
    """
    resp_queue = list(responses)

    qb = MagicMock()
    qb.select.return_value = qb
    qb.order.return_value  = qb
    qb.limit.return_value  = qb
    qb.eq.return_value     = qb

    def execute():
        data, count = resp_queue.pop(0)
        return _response(data, count)

    qb.execute.side_effect = execute

    client = MagicMock()
    client.table.return_value = qb
    return client


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SNAPSHOT_UUID  = "550e8400-e29b-41d4-a716-446655440000"
_INGESTED_AT_TS = "2026-06-30T12:00:00+00:00"
_INGESTED_AT_DT = datetime(2026, 6, 30, 12, 0, 0)  # naive UTC

_META_ROW = [
    {
        "snapshot_id":  _SNAPSHOT_UUID,
        "ingested_at":  _INGESTED_AT_TS,
        "pipeline_run": "pipeline-2026-06-30",
    }
]

_ROWS_DATA = [
    {"org_name": "MTM", "origin": "LOCAL",    "tons": 9000.0},
    {"org_name": "MTM", "origin": "IMPORTED", "tons": 11000.0},
]

_AS_OF = date(2026, 6, 30)


def _repo_with_data(meta=None, rows=None):
    """Build a SupabaseInventoryRepository backed by a mock returning good data."""
    client = _make_client(
        (meta or _META_ROW, 1),
        (rows or _ROWS_DATA, len(rows or _ROWS_DATA)),
    )
    return SupabaseInventoryRepository(
        url="https://example.supabase.co",
        key="fake-key",
        _client=client,
    )


# ===========================================================================
# _parse_iso_datetime
# ===========================================================================

class TestParseIsoDt:
    def test_aware_utc_string(self):
        dt = _parse_iso_datetime("2026-06-30T12:00:00+00:00")
        assert dt == datetime(2026, 6, 30, 12, 0, 0)
        assert dt.tzinfo is None

    def test_aware_positive_offset_string(self):
        # +05:00 = UTC 07:00 after conversion
        dt = _parse_iso_datetime("2026-06-30T12:00:00+05:00")
        assert dt == datetime(2026, 6, 30, 7, 0, 0)
        assert dt.tzinfo is None

    def test_naive_string(self):
        dt = _parse_iso_datetime("2026-06-30T12:00:00")
        assert dt == datetime(2026, 6, 30, 12, 0, 0)
        assert dt.tzinfo is None

    def test_date_only_raises(self):
        with pytest.raises(Exception):
            _parse_iso_datetime("not-a-date")


# ===========================================================================
# Snapshot creation
# ===========================================================================

class TestSnapshotCreation:
    def test_returns_inventory_snapshot(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert isinstance(snap, InventorySnapshot)

    def test_as_of_date_set_correctly(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert snap.as_of_date == _AS_OF

    def test_ingested_at_parsed_correctly(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert snap.ingested_at == _INGESTED_AT_DT

    def test_pipeline_run_preserved(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert snap.pipeline_run == "pipeline-2026-06-30"

    def test_org_breakdowns_count(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 2

    def test_org_breakdowns_are_org_inventory_entries(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        for e in snap.org_breakdowns:
            assert isinstance(e, OrgInventoryEntry)

    def test_tons_values_correct(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        local_entry = next(e for e in snap.org_breakdowns if e.origin == "LOCAL")
        assert local_entry.tons == pytest.approx(9000.0)

    def test_data_quality_source_is_supabase(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert dq["source"] == "supabase"

    def test_data_quality_snapshot_id_recorded(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert dq["snapshot_id"] == _SNAPSHOT_UUID

    def test_data_quality_row_count_correct(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert dq["row_count"] == "2"

    def test_data_quality_pipeline_run_present(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert dq.get("pipeline_run") == "pipeline-2026-06-30"

    def test_data_quality_pipeline_run_absent_when_null(self):
        meta = [{"snapshot_id": _SNAPSHOT_UUID, "ingested_at": _INGESTED_AT_TS, "pipeline_run": None}]
        repo = _repo_with_data(meta=meta)
        snap = repo.get_snapshot(as_of=_AS_OF)
        dq = snap.data_quality_as_dict()
        assert "pipeline_run" not in dq

    def test_snapshot_is_frozen(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        with pytest.raises((AttributeError, TypeError)):
            snap.pipeline_run = "modified"  # type: ignore[misc]

    def test_default_as_of_is_today(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot()
        assert snap.as_of_date == date.today()


# ===========================================================================
# DataFrame contract
# ===========================================================================

class TestSnapshotDataFrame:
    def test_columns_are_correct(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        df = snap.to_origin_summary_df()
        assert list(df.columns) == ["org_name", "origin", "tons"]

    def test_tons_dtype_is_float64(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        df = snap.to_origin_summary_df()
        assert df["tons"].dtype == "float64"

    def test_string_columns_are_string_like(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        df = snap.to_origin_summary_df()
        assert pd.api.types.is_string_dtype(df["org_name"])
        assert pd.api.types.is_string_dtype(df["origin"])

    def test_row_values_match_source_data(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        df = snap.to_origin_summary_df()
        local_row = df[df["origin"] == "LOCAL"].iloc[0]
        assert local_row["org_name"] == "MTM"
        assert local_row["tons"]     == pytest.approx(9000.0)

    def test_total_tons_matches_source(self):
        repo = _repo_with_data()
        snap = repo.get_snapshot(as_of=_AS_OF)
        df = snap.to_origin_summary_df()
        assert df["tons"].sum() == pytest.approx(20000.0)

    def test_identical_to_workbook_repo_contract(self):
        """Supabase snapshot df must have the same shape/dtypes as workbook snapshot df."""
        from procurement_data_repository import WorkbookInventoryRepository, InventorySnapshot

        entries = (
            OrgInventoryEntry(org_name="MTM", origin="LOCAL",    tons=9000.0),
            OrgInventoryEntry(org_name="MTM", origin="IMPORTED", tons=11000.0),
        )
        workbook_snap = InventorySnapshot(
            as_of_date=_AS_OF,
            ingested_at=_INGESTED_AT_DT,
            pipeline_run=None,
            org_breakdowns=entries,
            data_quality=(),
        )
        workbook_df = workbook_snap.to_origin_summary_df()

        repo = _repo_with_data()
        supabase_snap = repo.get_snapshot(as_of=_AS_OF)
        supabase_df   = supabase_snap.to_origin_summary_df()

        assert list(supabase_df.columns) == list(workbook_df.columns)
        assert supabase_df["tons"].dtype == workbook_df["tons"].dtype
        assert len(supabase_df) == len(workbook_df)

    def test_empty_table_produces_empty_df(self):
        client = _make_client(
            ([{"snapshot_id": _SNAPSHOT_UUID, "ingested_at": _INGESTED_AT_TS, "pipeline_run": None}], 1),
            ([], 0),
        )
        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap = repo.get_snapshot(as_of=_AS_OF)
        df   = snap.to_origin_summary_df()
        assert list(df.columns) == ["org_name", "origin", "tons"]
        assert len(df) == 0
        assert df["tons"].dtype == "float64"


# ===========================================================================
# Unavailable handling
# ===========================================================================

class TestUnavailableHandling:
    def test_raises_when_meta_query_returns_empty(self):
        client = _make_client(([], 0))
        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        with pytest.raises(RepositoryUnavailableError, match="No inventory snapshots"):
            repo.get_snapshot(as_of=_AS_OF)

    def test_raises_when_client_connection_fails(self):
        client = MagicMock()
        client.table.side_effect = RuntimeError("Connection refused")
        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        with pytest.raises(RepositoryUnavailableError, match="Supabase query failed"):
            repo.get_snapshot(as_of=_AS_OF)

    def test_raises_when_execute_raises(self):
        qb = MagicMock()
        qb.select.return_value = qb
        qb.order.return_value  = qb
        qb.limit.return_value  = qb
        qb.execute.side_effect = ConnectionError("timeout")
        client = MagicMock()
        client.table.return_value = qb
        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        with pytest.raises(RepositoryUnavailableError):
            repo.get_snapshot(as_of=_AS_OF)

    def test_unavailable_error_is_runtime_error(self):
        assert issubclass(RepositoryUnavailableError, RuntimeError)

    def test_error_message_mentions_table_when_empty(self):
        client = _make_client(([], 0))
        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        with pytest.raises(RepositoryUnavailableError) as exc_info:
            repo.get_snapshot(as_of=_AS_OF)
        assert "procurement_inventory_snapshots" in str(exc_info.value)

    def test_re_raises_repository_unavailable_not_wrapped(self):
        """RepositoryUnavailableError must propagate unwrapped, not double-wrapped."""
        qb = MagicMock()
        qb.select.return_value = qb
        qb.order.return_value  = qb
        qb.limit.return_value  = qb

        call_count = [0]
        def execute():
            call_count[0] += 1
            if call_count[0] == 1:
                return _response([], 0)
            return _response([], 0)
        qb.execute.side_effect = execute
        client = MagicMock()
        client.table.return_value = qb

        repo = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        with pytest.raises(RepositoryUnavailableError) as exc_info:
            repo.get_snapshot(as_of=_AS_OF)
        assert exc_info.type is RepositoryUnavailableError


# ===========================================================================
# Malformed data handling
# ===========================================================================

class TestMalformedData:
    def test_missing_tons_field_skipped(self):
        rows = [
            {"org_name": "MTM", "origin": "LOCAL"},          # missing tons
            {"org_name": "MTM", "origin": "IMPORTED", "tons": 11000.0},
        ]
        client = _make_client((_META_ROW, 1), (rows, 2))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 1
        dq = snap.data_quality_as_dict()
        assert dq["malformed_rows"] == "1"

    def test_non_numeric_tons_skipped(self):
        rows = [
            {"org_name": "MTM", "origin": "LOCAL",    "tons": "bad"},
            {"org_name": "MTM", "origin": "IMPORTED", "tons": 11000.0},
        ]
        client = _make_client((_META_ROW, 1), (rows, 2))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 1
        assert snap.data_quality_as_dict()["malformed_rows"] == "1"

    def test_none_tons_skipped(self):
        rows = [
            {"org_name": "MTM", "origin": "LOCAL",    "tons": None},
            {"org_name": "MTM", "origin": "IMPORTED", "tons": 11000.0},
        ]
        client = _make_client((_META_ROW, 1), (rows, 2))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 1

    def test_all_rows_malformed_returns_empty_snapshot(self):
        rows = [{"bad_key": "x"}, {"also_bad": True}]
        client = _make_client((_META_ROW, 1), (rows, 2))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 0
        dq = snap.data_quality_as_dict()
        assert dq["malformed_rows"] == "2"
        assert dq["row_count"]      == "0"

    def test_valid_rows_coerce_int_tons_to_float(self):
        rows = [{"org_name": "MTM", "origin": "LOCAL", "tons": 9000}]  # int
        client = _make_client((_META_ROW, 1), (rows, 1))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert len(snap.org_breakdowns) == 1
        assert isinstance(snap.org_breakdowns[0].tons, float)

    def test_unknown_origin_accepted(self):
        rows = [{"org_name": "MTM", "origin": "UNKNOWN", "tons": 500.0}]
        client = _make_client((_META_ROW, 1), (rows, 1))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        assert snap.org_breakdowns[0].origin == "UNKNOWN"
        assert snap.data_quality_as_dict()["malformed_rows"] == "0"


# ===========================================================================
# Health check
# ===========================================================================

class TestHealthCheck:
    def _health_client(self, data, count):
        """Build a client for a single health() query."""
        client = _make_client((data, count))
        return client

    def test_health_returns_healthy_for_recent_data(self, monkeypatch):
        monkeypatch.setattr(
            "procurement_data_repository.datetime",
            type("_DT", (), {
                "now": staticmethod(lambda: datetime(2026, 6, 30, 13, 0, 0)),
                "fromisoformat": datetime.fromisoformat,
            })
        )
        data = [{"ingested_at": "2026-06-30T12:00:00+00:00"}]
        client = self._health_client(data, count=6)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.is_available is True
        assert h.status       == "healthy"
        assert h.backend      == "supabase"

    def test_health_returns_degraded_for_stale_data(self, monkeypatch):
        monkeypatch.setattr(
            "procurement_data_repository.datetime",
            type("_DT", (), {
                "now": staticmethod(lambda: datetime(2026, 7, 4, 0, 0, 0)),
                "fromisoformat": datetime.fromisoformat,
            })
        )
        data = [{"ingested_at": "2026-06-30T12:00:00+00:00"}]
        client = self._health_client(data, count=6)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.is_available is True
        assert h.status       == "degraded"

    def test_health_returns_unavailable_when_connection_fails(self):
        client = MagicMock()
        client.table.side_effect = RuntimeError("timeout")
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.is_available is False
        assert h.status       == "unavailable"
        assert "connection failed" in h.message.lower()

    def test_health_returns_unavailable_when_table_empty(self):
        client = self._health_client([], count=0)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.is_available is False
        assert h.status       == "unavailable"
        assert h.record_count == 0

    def test_health_never_raises(self):
        client = MagicMock()
        client.table.side_effect = Exception("anything")
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()   # must not raise
        assert isinstance(h, RepositoryHealth)

    def test_health_backend_is_supabase(self):
        data   = [{"ingested_at": _INGESTED_AT_TS}]
        client = self._health_client(data, count=4)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.backend == "supabase"

    def test_health_latency_is_positive(self):
        data   = [{"ingested_at": _INGESTED_AT_TS}]
        client = self._health_client(data, count=4)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.latency_ms is not None
        assert h.latency_ms >= 0.0

    def test_health_record_count_from_response(self):
        data   = [{"ingested_at": _INGESTED_AT_TS}]
        client = self._health_client(data, count=12)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert h.record_count == 12

    def test_health_message_is_nonempty_string(self):
        data   = [{"ingested_at": _INGESTED_AT_TS}]
        client = self._health_client(data, count=4)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        assert isinstance(h.message, str)
        assert len(h.message) > 0

    def test_health_is_frozen(self):
        data   = [{"ingested_at": _INGESTED_AT_TS}]
        client = self._health_client(data, count=4)
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        h      = repo.health()
        with pytest.raises((AttributeError, TypeError)):
            h.status = "healthy"  # type: ignore[misc]


# ===========================================================================
# Engine compatibility (synthetic Supabase data -> full PSE chain)
# ===========================================================================

class TestEngineCompatibility:
    def _run_chain(self, rows):
        client = _make_client((_META_ROW, 1), (rows, len(rows)))
        repo   = SupabaseInventoryRepository(url="https://x", key="k", _client=client)
        snap   = repo.get_snapshot(as_of=_AS_OF)
        df     = snap.to_origin_summary_df()

        from procurement_strategy_engine import build_strategy_output_v2
        from procurement_position_engine import assess_position
        from procurement_target_engine import define_strategy_target
        from procurement_gap_engine import analyze_gap
        from procurement_optimization_engine import optimize_portfolio
        from procurement_market_engine import assess_market_opportunity
        from procurement_strategy_assessment_engine import assess_strategy
        from procurement_execution_planning_engine import build_execution_plan

        so       = build_strategy_output_v2(df, run_date=_AS_OF.isoformat())
        pos      = assess_position(so, as_of=_AS_OF)
        tgt      = define_strategy_target(as_of=_AS_OF, desired_coverage_days=45.0)
        gap      = analyze_gap(pos, tgt)
        portfolio = optimize_portfolio(pos, tgt, gap)
        market   = assess_market_opportunity(
            market_price_inputs={"current_price_usd_per_lb": 0.78,
                                  "forecast_h1_usd_per_lb": 0.785},
            as_of=_AS_OF,
        )
        strategy = assess_strategy(portfolio, market, as_of=_AS_OF)
        plan     = build_execution_plan(pos, portfolio, market, strategy, as_of=_AS_OF)
        return so, pos, strategy, plan

    def test_full_pse_chain_runs_with_supabase_snapshot(self):
        so, pos, strategy, plan = self._run_chain(_ROWS_DATA)
        assert plan is not None
        assert plan.total_planned_quantity_tons >= 0.0

    def test_total_inventory_matches_supabase_rows(self):
        so, *_ = self._run_chain(_ROWS_DATA)
        assert so.total_inventory_tons == pytest.approx(20000.0, rel=1e-3)

    def test_critical_inventory_triggers_procurement(self):
        rows = [
            {"org_name": "MTM", "origin": "LOCAL",    "tons": 500.0},
            {"org_name": "MTM", "origin": "IMPORTED", "tons": 900.0},
        ]
        *_, plan = self._run_chain(rows)
        assert plan.total_planned_quantity_tons > 0

    def test_near_full_inventory_triggers_preservation(self):
        rows = [
            {"org_name": "MTM", "origin": "LOCAL",    "tons": 20000.0},
            {"org_name": "MTM", "origin": "IMPORTED", "tons": 22000.0},
        ]
        *rest, plan = self._run_chain(rows)
        strategy = rest[-1]
        assert strategy.overall_procurement_posture == "INVENTORY_PRESERVATION"

    def test_supabase_and_workbook_snapshots_produce_same_chain_output(self):
        """Two snapshots with identical data must produce identical PSE outputs."""
        # Build via Supabase mock
        so_sb, pos_sb, strategy_sb, plan_sb = self._run_chain(_ROWS_DATA)

        # Build via workbook-style InventorySnapshot (synthetic, no file I/O)
        from procurement_data_repository import InventorySnapshot
        from datetime import datetime as _dt
        entries = tuple(
            OrgInventoryEntry(org_name=r["org_name"], origin=r["origin"], tons=r["tons"])
            for r in _ROWS_DATA
        )
        wb_snap = InventorySnapshot(
            as_of_date=_AS_OF, ingested_at=_dt(2026, 6, 30, 12, 0, 0),
            pipeline_run=None, org_breakdowns=entries, data_quality=(),
        )
        from procurement_strategy_engine import build_strategy_output_v2
        so_wb = build_strategy_output_v2(wb_snap.to_origin_summary_df(), run_date=_AS_OF.isoformat())

        assert so_sb.total_inventory_tons == pytest.approx(so_wb.total_inventory_tons, rel=1e-6)
        assert so_sb.local_inventory_tons == pytest.approx(so_wb.local_inventory_tons, rel=1e-6)

    def test_impact_engine_accepts_supabase_derived_plan(self):
        from procurement_impact_engine import interpret_impact
        from procurement_position_engine import assess_position
        from procurement_optimization_engine import optimize_portfolio
        from procurement_market_engine import assess_market_opportunity

        so, pos, strategy, plan = self._run_chain(_ROWS_DATA)

        from procurement_target_engine import define_strategy_target
        from procurement_gap_engine import analyze_gap
        tgt      = define_strategy_target(as_of=_AS_OF, desired_coverage_days=45.0)
        gap      = analyze_gap(pos, tgt)
        portfolio = optimize_portfolio(pos, tgt, gap)
        market   = assess_market_opportunity(
            market_price_inputs={"current_price_usd_per_lb": 0.78,
                                  "forecast_h1_usd_per_lb": 0.785},
            as_of=_AS_OF,
        )
        impact = interpret_impact(plan, strategy, portfolio, market, pos)
        assert isinstance(impact.inventory_outlook, str)
        assert len(impact.inventory_outlook) > 0
