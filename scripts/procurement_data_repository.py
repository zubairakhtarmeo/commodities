"""
procurement_data_repository.py
---------------------------------
PSE-4.3B — Procurement Data Repository Layer.

The Repository is the single runtime data-access abstraction for the
Procurement Decision Engine pipeline. Its sole purpose is to supply
origin-classified, Kg-to-tons-converted inventory data so the engine chain
can execute without a direct filesystem dependency on Strategies.xlsx.

Design constraints (all frozen by PSE-4.3B architecture review):
    - The Repository NEVER calculates, optimises, forecasts, or makes
      procurement decisions.
    - InventorySnapshot.to_origin_summary_df() is the SOLE adapter between
      the Repository and build_strategy_output_v2(). All nine PSE engines
      (PSE-3.0 through PSE-4.1) are frozen and completely unchanged.
    - Daily consumption rates, lead times, reorder points, and storage
      capacity remain as constants in procurement_strategy_engine.py and
      procurement_orchestrator.py -- they are NOT stored in the snapshot.

Backends (PSE-4.3B.1 + PSE-4.3B.2):
    WorkbookInventoryRepository   reads Strategies.xlsx (local / pipeline)
    SupabaseInventoryRepository   reads from Supabase cloud database

Planned in future patches:
    FixtureInventoryRepository    returns hardcoded data for testing

Supabase table schema (run once in Supabase SQL Editor):

    CREATE TABLE procurement_inventory_snapshots (
        id           bigserial PRIMARY KEY,
        snapshot_id  uuid        NOT NULL DEFAULT gen_random_uuid(),
        pipeline_run text,
        as_of_date   date        NOT NULL,
        ingested_at  timestamptz NOT NULL DEFAULT now(),
        org_name     text        NOT NULL,
        origin       text        NOT NULL
                         CHECK (origin IN ('LOCAL', 'IMPORTED', 'UNKNOWN')),
        tons         double precision NOT NULL CHECK (tons >= 0)
    );

    CREATE INDEX idx_pis_snapshot_id ON procurement_inventory_snapshots (snapshot_id);
    CREATE INDEX idx_pis_ingested_at ON procurement_inventory_snapshots (ingested_at DESC);

Usage (local / pipeline):
    from procurement_data_repository import WorkbookInventoryRepository
    repo     = WorkbookInventoryRepository("data/strategy/Strategies.xlsx")
    snapshot = repo.get_snapshot()
    df       = snapshot.to_origin_summary_df()   # DataFrame(org_name, origin, tons)
    so       = build_strategy_output_v2(df)      # unchanged engine call

Usage (Streamlit Cloud / any environment with Supabase credentials):
    from procurement_data_repository import SupabaseInventoryRepository
    repo     = SupabaseInventoryRepository(url=SUPABASE_URL, key=SERVICE_ROLE_KEY)
    snapshot = repo.get_snapshot()
    df       = snapshot.to_origin_summary_df()
    so       = build_strategy_output_v2(df)
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ===========================================================================
# DOMAIN MODELS
# ===========================================================================

@dataclass(frozen=True)
class OrgInventoryEntry:
    """One row of origin-classified, Kg-to-tons-converted inventory data.

    Corresponds to one row of the origin_summary_converted DataFrame that
    procurement_orchestrator.run_orchestration() produces after the single
    approved Kg-to-tons conversion in convert_kg_to_tons().
    """

    org_name: str
    origin: str    # "LOCAL" | "IMPORTED" | "UNKNOWN"
    tons: float


@dataclass(frozen=True)
class InventorySnapshot:
    """Immutable snapshot of origin-classified inventory data from one pipeline run.

    Contains ONLY what the repository knows: raw inventory breakdowns per
    org/origin and ingestion metadata. Does NOT contain computed values
    (days-on-hand, reorder status, mix percentages) -- those are computed by
    the frozen PSE engines from this data.

    The sole coupling point to the engine pipeline is to_origin_summary_df().
    All other fields are repository metadata, not engine inputs.

    Immutability notes:
        frozen=True prevents field reassignment.
        data_quality is stored as tuple[tuple[str, str], ...] (not dict) so
        that frozen=True and __hash__ work correctly. Use data_quality_as_dict()
        for convenient key-value access.
    """

    as_of_date: date
    ingested_at: datetime
    pipeline_run: Optional[str]
    org_breakdowns: tuple[OrgInventoryEntry, ...]
    data_quality: tuple[tuple[str, str], ...]   # immutable key-value pairs

    # ------------------------------------------------------------------
    # Engine adapter -- the ONLY method the engine pipeline calls
    # ------------------------------------------------------------------

    def to_origin_summary_df(self) -> pd.DataFrame:
        """Reconstruct the DataFrame consumed by build_strategy_output_v2().

        Returns a DataFrame with exactly three columns:
            org_name  (object / str)
            origin    (object / str)  -- "LOCAL", "IMPORTED", or "UNKNOWN"
            tons      (float64)

        The shape, column order, and dtypes are guaranteed identical to the
        engine_input slice produced by run_orchestration():
            engine_input = origin_summary_converted[["org_name", "origin", "tons"]]

        This method is the sole coupling point between the Repository and the
        frozen engine pipeline. No other Repository method is ever called by
        an engine.
        """
        if not self.org_breakdowns:
            return pd.DataFrame(
                columns=["org_name", "origin", "tons"]
            ).astype({"org_name": "object", "origin": "object", "tons": "float64"})

        rows = [
            {"org_name": e.org_name, "origin": e.origin, "tons": float(e.tons)}
            for e in self.org_breakdowns
        ]
        df = pd.DataFrame(rows, columns=["org_name", "origin", "tons"])
        df["tons"] = df["tons"].astype("float64")
        return df

    # ------------------------------------------------------------------
    # Convenience helpers (not used by engines)
    # ------------------------------------------------------------------

    def data_quality_as_dict(self) -> dict[str, str]:
        """Convert the immutable data_quality tuple to a plain dict."""
        return dict(self.data_quality)


# ===========================================================================
# REPOSITORY HEALTH
# ===========================================================================

@dataclass(frozen=True)
class RepositoryHealth:
    """Immutable snapshot of repository availability and data freshness.

    Returned by ProcurementInventoryRepository.health() -- a non-blocking,
    low-cost check designed to run on every dashboard page render.

    The dashboard uses this to display a business-friendly status message
    instead of raw technical errors or workbook file-path messages.
    """

    backend: str              # "workbook" | "supabase" | "fixture" | "unavailable"
    status: str               # "healthy" | "degraded" | "unavailable"
    is_available: bool
    last_refresh: Optional[datetime]   # workbook mtime; Supabase ingested_at
    latency_ms: Optional[float]        # ms to fetch snapshot; None if not yet probed
    record_count: Optional[int]        # number of OrgInventoryEntry rows; None if not probed
    freshness_hours: Optional[float]   # hours since last_refresh; None if unknown
    message: str                       # one-line business-readable status


# ===========================================================================
# EXCEPTION
# ===========================================================================

class RepositoryUnavailableError(RuntimeError):
    """Raised by get_snapshot() when the backend cannot serve inventory data.

    The dashboard should catch this and delegate to health() to determine
    the user-facing message.
    """


# ===========================================================================
# ABSTRACT INTERFACE
# ===========================================================================

class ProcurementInventoryRepository(ABC):
    """Abstract data-access layer for procurement inventory.

    Implementors supply origin-classified, Kg-to-tons-converted inventory
    data to the engine pipeline. They must never calculate, optimise,
    forecast, or make procurement decisions.

    Contract:
        get_snapshot() returns InventorySnapshot or raises RepositoryUnavailableError.
        health()       returns RepositoryHealth without performing a full read.
    """

    @abstractmethod
    def get_snapshot(self, as_of: Optional[date] = None) -> InventorySnapshot:
        """Return the latest inventory snapshot.

        Args:
            as_of: Planning date attached to the snapshot as metadata.
                   Defaults to today. Does NOT filter which records are
                   returned -- all current inventory is always included.

        Raises:
            RepositoryUnavailableError: when the backend cannot serve data.
        """

    @abstractmethod
    def health(self) -> RepositoryHealth:
        """Check repository availability without a full data read.

        Must be non-blocking and inexpensive -- suitable for use in
        dashboard availability guards that run on every page render.
        Never raises; returns RepositoryHealth(status="unavailable") on error.
        """


# ===========================================================================
# WORKBOOK BACKEND
# ===========================================================================

class WorkbookInventoryRepository(ProcurementInventoryRepository):
    """Repository backend that reads inventory from Strategies.xlsx.

    Encapsulates the workbook-reading logic currently inlined in
    _run_pse5b_cached() and scattered through dashboard guard blocks.
    The engine behaviour is identical to today: run_orchestration() is called
    once, and its origin_summary_converted output is wrapped in an
    InventorySnapshot.

    Intended for:
        - Local development where Strategies.xlsx exists on disk
        - The pipeline self-hosted runner (ingestion + immediate analysis)
        - Backward compatibility during the PSE-4.3B.2 Supabase migration

    Not intended for:
        - Streamlit Cloud (Strategies.xlsx is gitignored; use Supabase backend)
    """

    def __init__(self, workbook_path: str | Path) -> None:
        self._workbook_path = Path(workbook_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_snapshot(self, as_of: Optional[date] = None) -> InventorySnapshot:
        """Run run_orchestration() and wrap the result in an InventorySnapshot.

        The returned snapshot's to_origin_summary_df() produces a DataFrame
        that is byte-for-byte identical to the engine_input slice currently
        passed to build_strategy_output_v2() from _run_pse5b_cached().

        Raises:
            RepositoryUnavailableError: if the workbook is absent or
                run_orchestration() raises any exception.
        """
        if not self._workbook_path.exists():
            raise RepositoryUnavailableError(
                f"Workbook not found: {self._workbook_path}"
            )

        as_of = as_of or date.today()
        t0 = time.monotonic()

        try:
            from procurement_orchestrator import run_orchestration
            result = run_orchestration(workbook_path=str(self._workbook_path))
        except Exception as exc:
            raise RepositoryUnavailableError(
                f"Workbook read failed ({self._workbook_path.name}): {exc}"
            ) from exc

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        origin_df: pd.DataFrame = result["origin_summary_converted"]
        cv: dict = result["conversion_validation"]
        sw: list = result.get("sanity_warnings", [])

        entries = tuple(
            OrgInventoryEntry(
                org_name=str(row["org_name"]),
                origin=str(row["origin"]),
                tons=float(row["tons"]),
            )
            for _, row in origin_df.iterrows()
        )

        dq: tuple[tuple[str, str], ...] = (
            ("source",                "workbook"),
            ("workbook",              self._workbook_path.name),
            ("row_count",             str(len(entries))),
            ("conversion_validated",  str(cv["all_passed"]).lower()),
            ("sanity_warning_count",  str(len(sw))),
            ("read_latency_ms",       f"{elapsed_ms:.1f}"),
        )

        return InventorySnapshot(
            as_of_date=as_of,
            ingested_at=datetime.now(),
            pipeline_run=None,
            org_breakdowns=entries,
            data_quality=dq,
        )

    def health(self) -> RepositoryHealth:
        """Check workbook presence by stat() only -- does not parse the file."""
        if not self._workbook_path.exists():
            return RepositoryHealth(
                backend="workbook",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=None,
                record_count=None,
                freshness_hours=None,
                message=f"Workbook not found: {self._workbook_path.name}",
            )

        try:
            mtime = datetime.fromtimestamp(self._workbook_path.stat().st_mtime)
        except OSError:
            return RepositoryHealth(
                backend="workbook",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=None,
                record_count=None,
                freshness_hours=None,
                message=f"Cannot stat workbook: {self._workbook_path.name}",
            )

        age_hours = (datetime.now() - mtime).total_seconds() / 3600.0
        status = "healthy" if age_hours < 48.0 else "degraded"
        ts_str = mtime.strftime("%Y-%m-%d %H:%M")

        if status == "healthy":
            message = f"Workbook available — last modified {ts_str}"
        else:
            message = (
                f"Workbook may be stale — last modified {ts_str} "
                f"({age_hours:.0f} h ago)"
            )

        return RepositoryHealth(
            backend="workbook",
            status=status,
            is_available=True,
            last_refresh=mtime,
            latency_ms=None,
            record_count=None,
            freshness_hours=round(age_hours, 1),
            message=message,
        )


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _parse_iso_datetime(ts: str) -> datetime:
    """Parse an ISO 8601 datetime string into a naive UTC datetime.

    Supabase returns timestamps as timezone-aware ISO strings
    (e.g. "2026-06-30T12:00:00+00:00"). This helper converts to a
    naive datetime in UTC for consistency with the rest of the codebase,
    which uses datetime.now() (naive local time) throughout.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ===========================================================================
# SUPABASE BACKEND
# ===========================================================================

class SupabaseInventoryRepository(ProcurementInventoryRepository):
    """Repository backend that reads inventory from a Supabase cloud database.

    Each pipeline run INSERTs one group of rows into the
    `procurement_inventory_snapshots` table, identified by a shared
    `snapshot_id` UUID. This backend retrieves the most recent snapshot and
    wraps it in an InventorySnapshot.

    The InventorySnapshot produced here is contract-identical to the one
    produced by WorkbookInventoryRepository: to_origin_summary_df() returns
    the same three-column, float64-tons DataFrame consumed by
    build_strategy_output_v2().

    Intended for:
        - Streamlit Cloud (Strategies.xlsx is gitignored; workbook backend
          cannot be used on Streamlit Cloud)
        - Any environment where Supabase credentials are available

    Not intended for:
        - The self-hosted Windows pipeline runner during the transition
          period (PSE-4.3B.3 Runtime Service will handle backend selection)

    Constructor Args:
        url:     Supabase project URL (SUPABASE_URL in secrets.toml)
        key:     Service-role key (SUPABASE_SERVICE_ROLE_KEY in secrets.toml)
        _client: Optional pre-built Supabase client, used for testing only.
                 Pass a mock client here instead of patching create_client.
    """

    _TABLE = "procurement_inventory_snapshots"
    _STALENESS_HOURS = 48.0

    def __init__(
        self,
        url: str,
        key: str,
        _client=None,          # injectable for testing; not part of public API
    ) -> None:
        self._url = url
        self._key = key
        self._injected_client = _client
        self._lazy_client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return the Supabase client, initialising lazily on first call."""
        if self._injected_client is not None:
            return self._injected_client
        if self._lazy_client is None:
            from supabase import create_client  # noqa: PLC0415
            self._lazy_client = create_client(self._url, self._key)
        return self._lazy_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_snapshot(self, as_of: Optional[date] = None) -> InventorySnapshot:
        """Fetch the latest inventory snapshot from Supabase.

        Executes two sequential queries:
          1. Identify the most-recent snapshot_id and its metadata.
          2. Fetch every org/origin row for that snapshot_id.

        The returned snapshot's to_origin_summary_df() is contract-identical
        to WorkbookInventoryRepository.get_snapshot().to_origin_summary_df().

        Raises:
            RepositoryUnavailableError: when the table is empty, the
                Supabase client cannot connect, or the query fails.
        """
        as_of = as_of or date.today()
        t0 = time.monotonic()

        try:
            client = self._get_client()

            # Query 1 — identify the latest snapshot
            meta_resp = (
                client.table(self._TABLE)
                .select("snapshot_id, ingested_at, pipeline_run")
                .order("ingested_at", desc=True)
                .limit(1)
                .execute()
            )
            if not meta_resp.data:
                raise RepositoryUnavailableError(
                    f"No inventory snapshots found in Supabase table '{self._TABLE}'"
                )

            meta         = meta_resp.data[0]
            snapshot_id  = meta["snapshot_id"]
            pipeline_run = meta.get("pipeline_run")
            ingested_at  = _parse_iso_datetime(meta["ingested_at"])

            # Query 2 — fetch all rows for that snapshot
            rows_resp = (
                client.table(self._TABLE)
                .select("org_name, origin, tons")
                .eq("snapshot_id", snapshot_id)
                .execute()
            )

        except RepositoryUnavailableError:
            raise
        except Exception as exc:
            raise RepositoryUnavailableError(
                f"Supabase query failed: {exc}"
            ) from exc

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        raw_rows = rows_resp.data or []
        entries: list[OrgInventoryEntry] = []
        malformed = 0

        for row in raw_rows:
            try:
                entries.append(OrgInventoryEntry(
                    org_name=str(row["org_name"]),
                    origin=str(row["origin"]),
                    tons=float(row["tons"]),
                ))
            except (KeyError, TypeError, ValueError):
                malformed += 1

        dq: tuple[tuple[str, str], ...] = (
            ("source",          "supabase"),
            ("table",           self._TABLE),
            ("snapshot_id",     str(snapshot_id)),
            ("row_count",       str(len(entries))),
            ("malformed_rows",  str(malformed)),
            ("read_latency_ms", f"{elapsed_ms:.1f}"),
        )
        if pipeline_run:
            dq = dq + (("pipeline_run", str(pipeline_run)),)

        return InventorySnapshot(
            as_of_date=as_of,
            ingested_at=ingested_at,
            pipeline_run=pipeline_run,
            org_breakdowns=tuple(entries),
            data_quality=dq,
        )

    def health(self) -> RepositoryHealth:
        """Check Supabase availability without a full snapshot read.

        Executes a single lightweight query that retrieves the latest
        ingested_at timestamp and the total row count. Never raises;
        returns RepositoryHealth(status="unavailable") on any error.
        """
        t0 = time.monotonic()

        try:
            client = self._get_client()
            resp = (
                client.table(self._TABLE)
                .select("ingested_at", count="exact")
                .order("ingested_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            return RepositoryHealth(
                backend="supabase",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=None,
                record_count=None,
                freshness_hours=None,
                message=f"Supabase connection failed: {exc}",
            )

        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)

        if not resp.data:
            return RepositoryHealth(
                backend="supabase",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=latency_ms,
                record_count=0,
                freshness_hours=None,
                message=f"Supabase reachable but '{self._TABLE}' contains no snapshots",
            )

        last_refresh = _parse_iso_datetime(resp.data[0]["ingested_at"])
        age_hours    = (datetime.now() - last_refresh).total_seconds() / 3600.0
        record_count = resp.count   # total rows before limit (from count="exact")
        status       = "healthy" if age_hours < self._STALENESS_HOURS else "degraded"
        ts_str       = last_refresh.strftime("%Y-%m-%d %H:%M")

        if status == "healthy":
            message = f"Supabase available — last ingested {ts_str}"
        else:
            message = (
                f"Supabase data may be stale — last ingested {ts_str} "
                f"({age_hours:.0f} h ago)"
            )

        return RepositoryHealth(
            backend="supabase",
            status=status,
            is_available=True,
            last_refresh=last_refresh,
            latency_ms=latency_ms,
            record_count=record_count,
            freshness_hours=round(age_hours, 1),
            message=message,
        )
