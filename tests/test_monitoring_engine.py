"""
test_monitoring_engine.py
--------------------------
Tests for procurement_monitoring_engine.py (PSE-7A).
Covers: all alert checkers, severity ordering, deduplication, watchlist,
        generate_monitoring_report API.
"""
from __future__ import annotations

from datetime import date

import pytest

from procurement_monitoring_engine import (
    generate_monitoring_report,
    MonitoringAlert, WatchlistItem, ExecutiveMonitoringReport,
    _SEV_RANK,
)
from procurement_strategy_engine import (
    SAFETY_STOCK_LOCAL_TONS, SAFETY_STOCK_IMPORTED_TONS,
    LOCAL_ROP_TONS, IMPORTED_ROP_TONS,
    DAILY_CONSUMPTION_LOCAL, DAILY_CONSUMPTION_IMPORTED,
    LOCAL_MIX_TARGET, IMPORTED_MIX_TARGET,
)

from conftest import make_exec_dict, make_scenario_dict

TODAY = date(2026, 6, 27)


# ===========================================================================
# Helpers
# ===========================================================================

def _run(
    local_inv, imported_inv, total_inv=None,
    exec_dict=None, scenario_dict=None,
    max_storage=None, best_sim=None, prod_score=None,
):
    if total_inv is None:
        total_inv = local_inv + imported_inv
    if exec_dict is None:
        exec_dict = make_exec_dict()
    if scenario_dict is None:
        scenario_dict = make_scenario_dict()
    return generate_monitoring_report(
        exec_report_dict=exec_dict,
        scenario_report_dict=scenario_dict,
        local_inventory_tons=local_inv,
        imported_inventory_tons=imported_inv,
        total_inventory_tons=total_inv,
        today=TODAY,
        max_storage_tons=max_storage,
        best_sim_score=best_sim,
        prod_score=prod_score,
    )


# ===========================================================================
# Output structure
# ===========================================================================

class TestMonitoringReportStructure:
    def test_returns_executive_monitoring_report(self):
        r = _run(4000.0, 8000.0)
        assert isinstance(r, ExecutiveMonitoringReport)

    def test_report_has_all_sections(self):
        r = _run(4000.0, 8000.0)
        assert isinstance(r.alerts, list)
        assert isinstance(r.watchlist, list)
        assert isinstance(r.upcoming_procurement_events, list)
        assert isinstance(r.upcoming_risk_events, list)
        assert isinstance(r.upcoming_savings_opportunities, list)
        assert isinstance(r.summary, str)
        assert isinstance(r.highest_severity, str)
        assert isinstance(r.alert_count_by_severity, dict)

    def test_alert_count_by_severity_complete(self):
        r = _run(4000.0, 8000.0)
        for sev in ("CRITICAL", "HIGH", "WARNING", "NOTICE", "INFO"):
            assert sev in r.alert_count_by_severity

    def test_highest_severity_consistent_with_alerts(self):
        r = _run(800.0, 2000.0)
        if r.alerts:
            severities = [_SEV_RANK[a.severity] for a in r.alerts]
            best_rank  = min(severities)
            best_label = [k for k, v in _SEV_RANK.items() if v == best_rank][0]
            assert r.highest_severity == best_label

    def test_to_dict_serialisable(self):
        r = _run(4000.0, 8000.0)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "alerts" in d and "watchlist" in d

    def test_no_alert_when_healthy(self):
        """Very healthy inventory should fire zero alerts (no ROP/SS/mix triggers)."""
        r = _run(
            local_inv=5000.0, imported_inv=10000.0,
            exec_dict=make_exec_dict(conf_score_local=90, conf_score_imported=90),
        )
        for a in r.alerts:
            assert a.severity not in ("CRITICAL",), \
                f"Unexpected CRITICAL alert when inventory is healthy: {a.title}"


# ===========================================================================
# Checker: ROP approach / breach
# ===========================================================================

class TestRopApproachAlerts:
    def test_below_rop_triggers_critical_or_high(self):
        """LOCAL inventory below ROP must trigger at minimum HIGH alert."""
        r = _run(800.0, 8000.0)   # 800t < LOCAL_ROP (1733t)
        local_rop_alerts = [a for a in r.alerts if "LOC_ROP" in a.alert_id]
        assert local_rop_alerts, "Expected LOCAL ROP alert when below ROP"
        severities = {a.severity for a in local_rop_alerts}
        assert severities & {"CRITICAL", "HIGH"}, (
            f"Expected CRITICAL or HIGH for below-ROP, got: {severities}"
        )

    def test_imported_below_rop_triggers_alert(self):
        r = _run(4000.0, 2000.0)  # 2000t < IMPORTED_ROP (6958t)
        imp_rop_alerts = [a for a in r.alerts if "IMP_ROP" in a.alert_id]
        assert imp_rop_alerts, "Expected IMPORTED ROP alert when below ROP"

    def test_safe_inventory_no_rop_alert(self):
        r = _run(4000.0, 10000.0)  # both above ROP
        rop_alerts = [a for a in r.alerts if "_ROP_" in a.alert_id]
        # ROP alerts should not fire when comfortably above ROP
        for a in rop_alerts:
            assert a.severity not in ("CRITICAL", "HIGH"), \
                f"Unexpected critical ROP alert with healthy inventory: {a}"


# ===========================================================================
# Checker: Safety stock breach
# ===========================================================================

class TestSafetyStockAlerts:
    def test_below_safety_stock_fires_critical(self):
        """LOCAL below safety stock (1237.5t) must fire CRITICAL or HIGH."""
        r = _run(1000.0, 8000.0)  # 1000t < 1237.5t SS_LOCAL
        ss_alerts = [a for a in r.alerts if "LOC_SS" in a.alert_id]
        assert ss_alerts, "Expected LOCAL safety stock alert"
        assert any(a.severity in ("CRITICAL", "HIGH") for a in ss_alerts), \
            f"Below-SS should fire CRITICAL/HIGH, got: {[a.severity for a in ss_alerts]}"

    def test_above_safety_stock_no_ss_critical(self):
        r = _run(5000.0, 10000.0)
        ss_critical = [a for a in r.alerts
                       if "_SS_" in a.alert_id and a.severity == "CRITICAL"]
        assert not ss_critical


# ===========================================================================
# Checker: Mix deviation
# ===========================================================================

class TestMixDeviationAlerts:
    def test_extreme_imbalance_fires_mix_alert(self):
        """Extreme LOCAL shortage → mix deviation should fire."""
        r = _run(500.0, 15000.0)   # local 500 / total 15500 = 3.2% << 45% target
        mix_alerts = [a for a in r.alerts if "MIX" in a.alert_id]
        assert mix_alerts, "Expected mix deviation alert with extreme imbalance"

    def test_balanced_mix_no_high_mix_alert(self):
        """Near-target mix should not fire HIGH or CRITICAL mix alert."""
        r = _run(4500.0, 5500.0)  # 45% / 55% target mix exactly
        mix_high = [a for a in r.alerts
                    if "MIX" in a.alert_id and a.severity in ("HIGH", "CRITICAL")]
        assert not mix_high


# ===========================================================================
# Checker: Storage capacity
# ===========================================================================

class TestStorageCapacityAlerts:
    def test_near_capacity_fires_alert(self):
        r = _run(
            local_inv=20000.0, imported_inv=23000.0, total_inv=43000.0,
            max_storage=45_000.0,
        )
        stor_alerts = [a for a in r.alerts if "STOR" in a.alert_id]
        assert stor_alerts, "Expected storage capacity alert when > 80% full"

    def test_low_stock_no_storage_alert(self):
        r = _run(800.0, 2000.0, max_storage=45_000.0)
        stor_alerts = [a for a in r.alerts if "STOR" in a.alert_id]
        assert not stor_alerts


# ===========================================================================
# Checker: Forecast confidence
# ===========================================================================

class TestForecastConfidenceAlerts:
    def test_low_confidence_fires_alert(self):
        exec_dict = make_exec_dict(conf_score_local=45, conf_score_imported=45)
        r = _run(4000.0, 8000.0, exec_dict=exec_dict)
        conf_alerts = [a for a in r.alerts if "CONF" in a.alert_id]
        assert conf_alerts, "Expected confidence alert when score < 50"

    def test_high_confidence_no_alert(self):
        exec_dict = make_exec_dict(conf_score_local=90, conf_score_imported=90)
        r = _run(4000.0, 8000.0, exec_dict=exec_dict)
        conf_alerts = [a for a in r.alerts if "CONF" in a.alert_id]
        assert not conf_alerts


# ===========================================================================
# Checker: Cost increase / saving opportunity
# ===========================================================================

class TestFinancialAlerts:
    def test_large_cost_increase_fires_alert(self):
        exec_dict = make_exec_dict(
            fi_type_local="COST_AVOIDANCE", fi_usd_local=1_200_000.0,
        )
        r = _run(4000.0, 8000.0, exec_dict=exec_dict)
        cost_alerts = [a for a in r.alerts if "COST" in a.alert_id]
        assert cost_alerts, "Expected cost-increase alert for $1.2M exposure"

    def test_large_saving_fires_alert(self):
        exec_dict = make_exec_dict(
            fi_type_imported="SAVING", fi_usd_imported=600_000.0,
        )
        r = _run(4000.0, 8000.0, exec_dict=exec_dict)
        sav_alerts = [a for a in r.alerts if "SAV" in a.alert_id]
        assert sav_alerts, "Expected saving-opportunity alert for $600k saving"


# ===========================================================================
# Checker: Simulation outperformance
# ===========================================================================

class TestSimulationOutperformanceAlert:
    def test_large_delta_fires_alert(self):
        r = _run(4000.0, 8000.0, best_sim=90, prod_score=65)
        sim_alerts = [a for a in r.alerts if "SIM" in a.alert_id]
        assert sim_alerts, "Expected sim-outperformance alert when delta >= 20"

    def test_small_delta_no_alert(self):
        r = _run(4000.0, 8000.0, best_sim=70, prod_score=68)
        sim_alerts = [a for a in r.alerts if "SIM" in a.alert_id]
        assert not sim_alerts, "No sim alert expected for delta < 8 pts"

    def test_none_sim_score_no_alert(self):
        r = _run(4000.0, 8000.0, best_sim=None, prod_score=None)
        sim_alerts = [a for a in r.alerts if "SIM" in a.alert_id]
        assert not sim_alerts


# ===========================================================================
# Alert deduplication
# ===========================================================================

class TestAlertDeduplication:
    def test_no_duplicate_alert_ids(self):
        r = _run(800.0, 2000.0)
        ids = [a.alert_id for a in r.alerts]
        assert len(ids) == len(set(ids)), f"Duplicate alert IDs found: {ids}"

    def test_most_severe_wins_per_prefix(self):
        """After dedup, no two alerts should share the same base-ID prefix."""
        r = _run(800.0, 2000.0)
        prefixes = []
        for a in r.alerts:
            # prefix = everything before the last underscore
            prefix = a.alert_id.rsplit("_", 1)[0]
            prefixes.append(prefix)
        assert len(prefixes) == len(set(prefixes)), \
            f"Multiple alerts with same prefix: {[p for p in prefixes if prefixes.count(p) > 1]}"


# ===========================================================================
# Severity ordering (lower rank = more severe)
# ===========================================================================

class TestSeverityRanking:
    def test_critical_more_severe_than_high(self):
        assert _SEV_RANK["CRITICAL"] < _SEV_RANK["HIGH"]

    def test_high_more_severe_than_warning(self):
        assert _SEV_RANK["HIGH"] < _SEV_RANK["WARNING"]

    def test_warning_more_severe_than_notice(self):
        assert _SEV_RANK["WARNING"] < _SEV_RANK["NOTICE"]

    def test_notice_more_severe_than_info(self):
        assert _SEV_RANK["NOTICE"] < _SEV_RANK["INFO"]

    def test_all_five_levels_present(self):
        assert set(_SEV_RANK) == {"CRITICAL", "HIGH", "WARNING", "NOTICE", "INFO"}


# ===========================================================================
# Alert metadata completeness
# ===========================================================================

class TestAlertMetadata:
    def test_all_alerts_have_required_fields(self):
        r = _run(800.0, 2000.0)
        required = {
            "alert_id", "title", "severity", "source",
            "reason", "business_impact", "recommended_action",
            "expected_urgency", "triggered_at",
        }
        for a in r.alerts:
            d = a.to_dict()
            for field in required:
                assert field in d and d[field], \
                    f"Alert {a.alert_id!r} missing or empty field: {field!r}"

    def test_severity_in_valid_vocab(self):
        r = _run(800.0, 2000.0)
        valid = set(_SEV_RANK)
        for a in r.alerts:
            assert a.severity in valid, \
                f"Alert {a.alert_id!r} has invalid severity {a.severity!r}"

    def test_source_in_valid_vocab(self):
        r = _run(800.0, 2000.0)
        valid = {"LOCAL", "IMPORTED", "PORTFOLIO", "SYSTEM"}
        for a in r.alerts:
            assert a.source in valid, \
                f"Alert {a.alert_id!r} has invalid source {a.source!r}"
