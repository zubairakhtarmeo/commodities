"""
procurement_runtime_service.py
---------------------------------
PSE-4.3B.3 -- Procurement Runtime Service.

This module is the SINGLE entry point between the dashboard and the
Procurement Decision Engine. The dashboard calls run() and receives a
ProcurementRuntimeResult; it never touches a repository, a workbook path,
or an individual engine directly.

Responsibilities:
    1. Obtain InventorySnapshot via the injected repository.
    2. Run the complete PSE-3.0 → PSE-4.1 engine chain.
    3. Run the PSE-5A/5B scenario and executive layers.
    4. Return one ProcurementRuntimeResult that contains every output the
       dashboard needs.

What this module does NOT do:
    - No business logic.
    - No procurement decisions.
    - No forecasting or optimisation.
    - No rendering.
    - No Streamlit.
    - No caching (caching belongs to the dashboard's @st.cache_data layer).

Engine execution order (frozen by PSE-4.3B architecture):
    build_strategy_output_v2   → StrategyOutput      (inventory basis)
    run_pse5a                  → ScenarioReport       (price-aware scenario)
    generate_executive_report  → ExecutiveReport      (PSE-5B decision)
    assess_position            → PositionSnapshot     (PSE-3.0)
    define_strategy_target     → StrategyTarget       (PSE-3.1)
    analyze_gap                → GapSnapshot          (PSE-3.2)
    optimize_portfolio         → PortfolioOptSnapshot (PSE-3.3)
    assess_market_opportunity  → MarketOppSnapshot    (PSE-3.4)
    assess_strategy            → StratAssessment      (PSE-3.5)
    build_execution_plan       → ExecutionPlan        (PSE-3.6)
    interpret_impact           → DecisionImpact       (PSE-4.1)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_data_repository import (
    InventorySnapshot,
    RepositoryHealth,
    RepositoryUnavailableError,
    ProcurementInventoryRepository,
)


# ===========================================================================
# RESULT DTO
# ===========================================================================

@dataclass
class ProcurementRuntimeResult:
    """Single immutable DTO produced by ProcurementRuntimeService.run().

    The dashboard consumes ONLY this object. It never calls repositories,
    engines, or orchestration code directly.

    Availability contract:
        Always check `is_available` before reading engine fields.
        When is_available=False, only `error_message`, `repository_health`,
        `as_of`, and `generated_at` are guaranteed to be non-None.

    Engine fields:
        All engine output objects are the frozen dataclass instances produced
        by the corresponding engine. They are never recalculated by the
        dashboard.

    Serialisation:
        Use to_legacy_dict() to obtain the dict format expected by the
        existing dashboard rendering helpers. This preserves backward
        compatibility without requiring any rendering code changes.
    """

    # Availability ─────────────────────────────────────────────────────────
    is_available: bool
    error_message: Optional[str]

    # Repository layer ──────────────────────────────────────────────────────
    repository_health: RepositoryHealth
    snapshot: Optional[InventorySnapshot]

    # PSE-3.x chain outputs ─────────────────────────────────────────────────
    position: Optional[object]          # PositionSnapshot (PSE-3.0)
    target: Optional[object]            # StrategyTarget   (PSE-3.1)
    gap: Optional[object]               # GapSnapshot      (PSE-3.2)
    portfolio: Optional[object]         # PortfolioOptimizationSnapshot (PSE-3.3)
    market: Optional[object]            # MarketOpportunitySnapshot     (PSE-3.4)
    strategy: Optional[object]          # StrategicProcurementAssessment (PSE-3.5)
    execution_plan: Optional[object]    # ProcurementExecutionPlan (PSE-3.6)
    impact: Optional[object]            # DecisionImpact (PSE-4.1)

    # PSE-5A/5B scenario/executive layer ────────────────────────────────────
    scenario_report: Optional[object]   # ScenarioReport  (PSE-5A)
    executive_report: Optional[object]  # ExecutiveReport (PSE-5B)

    # Shared strategy output (PSE-3B) ───────────────────────────────────────
    strategy_output: Optional[object]   # StrategyOutput from build_strategy_output_v2

    # Metadata ──────────────────────────────────────────────────────────────
    as_of: date
    generated_at: datetime

    # ------------------------------------------------------------------
    # Serialisation adapter
    # ------------------------------------------------------------------

    def to_legacy_dict(self) -> dict:
        """Convert this result to the dict format used by the dashboard.

        Produces the same structure as the old _run_pse5b_cached() return
        value, so all existing rendering helpers work without modification.

        Keys:
            ok:             bool
            error:          str  (only when ok=False)
            report:         dict (ExecutiveReport.to_dict())
            scenario:       dict (ScenarioReport.to_dict())
            execution_plan: dict (ProcurementExecutionPlan.to_dict())
            decision_impact:dict (DecisionImpact.to_dict())
            inventory:      dict (seven scalar fields from StrategyOutput)
        """
        if not self.is_available:
            return {
                "ok": False,
                "error": self.error_message or "Runtime service unavailable",
            }

        so = self.strategy_output
        sr = self.scenario_report
        er = self.executive_report

        return {
            "ok": True,
            "report":         er.to_dict() if er is not None else {},
            "scenario":       sr.to_dict() if sr is not None else {},
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else {},
            "decision_impact": self.impact.to_dict() if self.impact else {},
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

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def _make_unavailable(
        cls,
        error: str,
        health: RepositoryHealth,
        as_of: date,
    ) -> "ProcurementRuntimeResult":
        return cls(
            is_available=False,
            error_message=error,
            repository_health=health,
            snapshot=None,
            position=None,
            target=None,
            gap=None,
            portfolio=None,
            market=None,
            strategy=None,
            execution_plan=None,
            impact=None,
            scenario_report=None,
            executive_report=None,
            strategy_output=None,
            as_of=as_of,
            generated_at=datetime.now(),
        )


# ===========================================================================
# RUNTIME SERVICE
# ===========================================================================

class ProcurementRuntimeService:
    """Single orchestration boundary between the dashboard and the PSE engines.

    The dashboard creates one instance of this service (or calls the
    @st.cache_data adapter that wraps it), calls run(), and receives a
    ProcurementRuntimeResult that contains every output it needs.

    The service is stateless: all state lives in the repository and the
    engine layer. Multiple sequential run() calls are safe and independent.

    Dependency injection:
        Accepts any ProcurementInventoryRepository implementation.
        The dashboard never imports WorkbookInventoryRepository or
        SupabaseInventoryRepository directly — the Runtime Service owns
        that decision.

    Error handling:
        run() never raises. All errors are captured and returned inside
        ProcurementRuntimeResult(is_available=False, error_message=...).
    """

    def __init__(self, repository: ProcurementInventoryRepository) -> None:
        self._repository = repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, as_of: Optional[date] = None) -> ProcurementRuntimeResult:
        """Execute the complete PSE pipeline and return a RuntimeResult.

        Engine execution order is frozen:
            InventorySnapshot  (repository)
            StrategyOutput     (build_strategy_output_v2)
            ScenarioReport     (PSE-5A — for price inputs)
            ExecutiveReport    (PSE-5B)
            PositionSnapshot   (PSE-3.0)
            StrategyTarget     (PSE-3.1)
            GapSnapshot        (PSE-3.2)
            PortfolioDecision  (PSE-3.3)
            MarketAssessment   (PSE-3.4, uses PSE-5A price inputs)
            StrategyAssessment (PSE-3.5)
            ExecutionPlan      (PSE-3.6)
            DecisionImpact     (PSE-4.1)

        Returns:
            ProcurementRuntimeResult with is_available=True on success.
            ProcurementRuntimeResult with is_available=False on any failure;
            check error_message for a human-readable description.
        """
        as_of = as_of or date.today()

        # ── 1. Repository health (non-blocking; always attempt) ─────────────
        try:
            health = self._repository.health()
        except Exception as exc:
            health = RepositoryHealth(
                backend="unknown",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=None,
                record_count=None,
                freshness_hours=None,
                message=f"Health check failed: {exc}",
            )

        # ── 2. Inventory snapshot ────────────────────────────────────────────
        try:
            snapshot = self._repository.get_snapshot(as_of=as_of)
        except RepositoryUnavailableError as exc:
            return ProcurementRuntimeResult._make_unavailable(
                error=str(exc),
                health=health,
                as_of=as_of,
            )
        except Exception as exc:
            return ProcurementRuntimeResult._make_unavailable(
                error=f"Unexpected repository error: {exc}",
                health=health,
                as_of=as_of,
            )

        df = snapshot.to_origin_summary_df()

        # ── 3. Engine chain ──────────────────────────────────────────────────
        try:
            return self._run_engines(
                snapshot=snapshot,
                df=df,
                health=health,
                as_of=as_of,
            )
        except Exception as exc:
            import traceback
            return ProcurementRuntimeResult._make_unavailable(
                error=f"Engine pipeline error: {exc}\n{traceback.format_exc()}",
                health=health,
                as_of=as_of,
            )

    # ------------------------------------------------------------------
    # Engine orchestration (private)
    # ------------------------------------------------------------------

    def _run_engines(
        self,
        snapshot: InventorySnapshot,
        df,
        health: RepositoryHealth,
        as_of: date,
    ) -> ProcurementRuntimeResult:
        """Run every engine in the approved order. Called from run()."""
        from procurement_strategy_engine import build_strategy_output_v2
        from procurement_position_engine import assess_position
        from procurement_target_engine import define_strategy_target
        from procurement_gap_engine import analyze_gap
        from procurement_optimization_engine import optimize_portfolio
        from procurement_market_engine import assess_market_opportunity
        from procurement_strategy_assessment_engine import assess_strategy
        from procurement_execution_planning_engine import build_execution_plan
        from procurement_impact_engine import interpret_impact

        # Shared strategy output — used by both PSE-3.x chain and PSE-5A
        so = build_strategy_output_v2(df, run_date=as_of.isoformat())

        # PSE-5A: scenario engine produces price_inputs_used for PSE-3.4
        # Pass {"strategy_output": so} to avoid a second workbook read.
        sr, price_inputs = self._run_pse5a(so, as_of)

        # PSE-5B: executive report (depends on PSE-5A)
        er = self._run_pse5b(sr, so)

        # PSE-3.0 → PSE-4.1 chain ─────────────────────────────────────────
        position      = assess_position(so, as_of=as_of)
        target        = define_strategy_target(as_of=as_of)
        gap           = analyze_gap(position, target)
        portfolio     = optimize_portfolio(position, target, gap)
        market        = assess_market_opportunity(
            market_price_inputs=price_inputs,
            as_of=as_of,
        )
        strategy      = assess_strategy(portfolio, market, as_of=as_of)
        execution_plan = build_execution_plan(
            position, portfolio, market, strategy, as_of=as_of
        )
        impact        = interpret_impact(
            execution_plan, strategy, portfolio, market, position, as_of=as_of
        )

        return ProcurementRuntimeResult(
            is_available=True,
            error_message=None,
            repository_health=health,
            snapshot=snapshot,
            position=position,
            target=target,
            gap=gap,
            portfolio=portfolio,
            market=market,
            strategy=strategy,
            execution_plan=execution_plan,
            impact=impact,
            scenario_report=sr,
            executive_report=er,
            strategy_output=so,
            as_of=as_of,
            generated_at=datetime.now(),
        )

    def _run_pse5a(self, so, as_of: date):
        """Run the PSE-5A scenario engine. Returns (ScenarioReport, price_inputs).

        Passes {"strategy_output": so} as _orch to share the already-built
        inventory position without triggering a second workbook read.

        Non-fatal: if PSE-5A fails, returns (None, None). The PSE-3.4 market
        engine will then be called with market_price_inputs=None and will
        return market_data_quality="UNAVAILABLE" — the same behaviour as
        when no price data is available.
        """
        try:
            from procurement_scenario_engine import run_pse5a
            sr = run_pse5a(
                live_prices=False,
                today=as_of,
                _orch={"strategy_output": so},
            )
            price_inputs = getattr(sr, "price_inputs_used", None)
            return sr, price_inputs
        except Exception:
            return None, None

    def _run_pse5b(self, sr, so):
        """Run the PSE-5B executive engine. Returns ExecutiveReport or None.

        Non-fatal: if PSE-5B fails (e.g., because PSE-5A also failed),
        returns None. Affected dashboard sections show an appropriate
        unavailable state.
        """
        if sr is None:
            return None
        try:
            from procurement_decision_engine import generate_executive_report
            return generate_executive_report(
                scenario_report=sr,
                local_status=so.local_status,
                imported_status=so.imported_status,
            )
        except Exception:
            return None
