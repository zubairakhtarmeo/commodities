"""
Shared fixtures and helpers for the PSE test suite.

Deterministic inputs are used throughout so tests never depend on live Oracle
data, network calls, or the system clock.  The real workbook is only needed for
integration tests, which are guarded with @pytest.mark.integration and can be
deselected in CI with:   pytest -m "not integration"
"""
from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT   = Path(__file__).parent.parent
SCRIPTS_DIR  = _REPO_ROOT / "scripts"
WORKBOOK_PATH = _REPO_ROOT / "data" / "strategy" / "Strategies.xlsx"

sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures: deterministic run parameters
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def today() -> date:
    return date(2026, 6, 27)


@pytest.fixture(scope="session")
def pkr_rate() -> float:
    return 278.5


@pytest.fixture(scope="session")
def current_price() -> float:
    return 0.78


@pytest.fixture(scope="session")
def workbook_path() -> Path:
    """Return workbook path; skip integration test if file absent."""
    if not WORKBOOK_PATH.exists():
        pytest.skip("Strategies.xlsx not found — integration tests require the workbook")
    return WORKBOOK_PATH


# ---------------------------------------------------------------------------
# Scenario table (identical to pse5a_validation_v2.py / pse5b_validation.py)
# ---------------------------------------------------------------------------

SCENARIO_PARAMS = [
    # (label,              source,     local_inv,  imp_inv,  h1,    h3)
    ("S1: CRITICAL+RISING",  "LOCAL",    800.0,  8000.0, 0.97, 1.06),
    ("S2: CRITICAL+FALLING", "LOCAL",    800.0,  8000.0, 0.60, 0.60),
    ("S3: REORDER+RISING",   "IMPORTED", 4000.0, 2000.0, 0.97, 1.06),
    ("S4: REORDER+FALLING",  "IMPORTED", 4000.0, 2000.0, 0.60, 0.60),
    ("S5: SAFE+RISING",      "LOCAL",    4000.0, 8000.0, 0.97, 1.06),
    ("S6: SAFE+FALLING",     "LOCAL",    4000.0, 8000.0, 0.60, 0.60),
    ("S7: WATCH+RISING",     "LOCAL",    1900.0, 8000.0, 0.97, 1.06),
    ("S8: WATCH+FALLING",    "LOCAL",    1900.0, 8000.0, 0.60, 0.60),
]

# Expected PSE-5B outputs for each scenario (from validated harness run)
EXPECTED_URGENCY = {
    "S1: CRITICAL+RISING":  "CRITICAL",
    "S2: CRITICAL+FALLING": "CRITICAL",
    "S3: REORDER+RISING":   "HIGH",
    "S4: REORDER+FALLING":  "HIGH",
    "S5: SAFE+RISING":      "MEDIUM",
    "S6: SAFE+FALLING":     "LOW",
    "S7: WATCH+RISING":     "MEDIUM",
    "S8: WATCH+FALLING":    "LOW",
}

EXPECTED_RISK = {
    "S1: CRITICAL+RISING":  "CRITICAL",
    "S2: CRITICAL+FALLING": "CRITICAL",
    "S3: REORDER+RISING":   "HIGH",
    "S4: REORDER+FALLING":  "HIGH",
    "S5: SAFE+RISING":      "MEDIUM",
    "S6: SAFE+FALLING":     "LOW",
    "S7: WATCH+RISING":     "MEDIUM",
    "S8: WATCH+FALLING":    "LOW",
}

# Quantities from the golden snapshot (from validated harness run 2026-06-27)
EXPECTED_QTY = {
    "S1: CRITICAL+RISING":  {"qty_now": 3160.0, "qty_later": 0.0,    "mandatory": 933.0},
    "S2: CRITICAL+FALLING": {"qty_now": 3160.0, "qty_later": 0.0,    "mandatory": 933.0},
    "S3: REORDER+RISING":   {"qty_now": 4958.0, "qty_later": 0.0,    "mandatory": 4958.0},
    "S4: REORDER+FALLING":  {"qty_now": 4958.0, "qty_later": 0.0,    "mandatory": 4958.0},
    "S5: SAFE+RISING":      {"qty_now": 2930.0, "qty_later": 0.0,    "mandatory": 0.0},
    "S6: SAFE+FALLING":     {"qty_now": 0.0,    "qty_later": 1400.0, "mandatory": 0.0},
    "S7: WATCH+RISING":     {"qty_now": 4040.0, "qty_later": 0.0,    "mandatory": 0.0},
    "S8: WATCH+FALLING":    {"qty_now": 0.0,    "qty_later": 2555.0, "mandatory": 0.0},
}


# ---------------------------------------------------------------------------
# Helper: build a full ScenarioReport + ExecutiveReport for one scenario row
# ---------------------------------------------------------------------------

def build_reports(label, source, local_inv, imp_inv, h1, h3,
                  current_price=0.78, pkr_rate=278.5, run_date=date(2026, 6, 27)):
    """
    Run the full PSE-3D→4A→4B→5A→5B pipeline for synthetic inputs.
    Returns (ScenarioReport, ExecutiveReport, local_status, imp_status).
    No workbook I/O.
    """
    from procurement_strategy_engine import (
        DAILY_CONSUMPTION_LOCAL, DAILY_CONSUMPTION_IMPORTED,
        LOCAL_ROP_TONS, IMPORTED_ROP_TONS,
        SAFETY_STOCK_LOCAL_TONS, SAFETY_STOCK_IMPORTED_TONS,
        _classify_supply_status,
    )
    from procurement_planning_engine import run_pse3d
    from procurement_calendar_engine import compute_procurement_calendar
    from procurement_consolidation_engine import compute_order_consolidation
    from procurement_scenario_engine import compute_scenario_decision
    from procurement_decision_engine import generate_executive_report

    local_status = _classify_supply_status(local_inv, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS)
    imp_status   = _classify_supply_status(imp_inv,   SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS)
    total = local_inv + imp_inv

    so = types.SimpleNamespace(
        local_status=local_status,
        imported_status=imp_status,
        local_inventory_tons=local_inv,
        imported_inventory_tons=imp_inv,
        total_inventory_tons=total,
        local_days_cover=round(local_inv / DAILY_CONSUMPTION_LOCAL, 1),
        imported_days_cover=round(imp_inv / DAILY_CONSUMPTION_IMPORTED, 1),
    )

    plan = run_pse3d(
        local_inventory_tons=local_inv,
        imported_inventory_tons=imp_inv,
        local_status=local_status,
        imported_status=imp_status,
        total_inventory_tons=total,
        current_price_usd_per_lb=current_price,
        forecast_h1_usd_per_lb=h1,
        forecast_h3_usd_per_lb=h3,
        pkr_rate=pkr_rate,
        today=run_date,
    )
    cal  = compute_procurement_calendar(
        local_inventory_tons=local_inv,
        imported_inventory_tons=imp_inv,
        local_status=local_status,
        imported_status=imp_status,
        total_inventory_tons=total,
        today=run_date,
    )
    cons = compute_order_consolidation(calendar_result=cal)

    sr = compute_scenario_decision(
        strategy_output=so,
        procurement_plan=plan,
        calendar_result=cal,
        consolidation_result=cons,
        current_price_usd_per_lb=current_price,
        forecast_h1_usd_per_lb=h1,
        forecast_h3_usd_per_lb=h3,
        pkr_rate=pkr_rate,
        today=run_date,
    )
    er = generate_executive_report(
        scenario_report=sr,
        local_status=local_status,
        imported_status=imp_status,
        pkr_rate=pkr_rate,
    )
    return sr, er, local_status, imp_status


@pytest.fixture(scope="session")
def all_scenario_results():
    """Run all 8 scenarios once per session; return list of result dicts."""
    results = []
    for (label, src, li, ii, h1, h3) in SCENARIO_PARAMS:
        sr, er, ls, is_ = build_reports(label, src, li, ii, h1, h3)
        dec = er.local if src == "LOCAL" else er.imported
        pse5a_dec = sr.local if src == "LOCAL" else sr.imported
        results.append({
            "label": label, "source": src,
            "local_status": ls, "imp_status": is_,
            "sr": sr, "er": er, "dec": dec, "pse5a_dec": pse5a_dec,
        })
    return results


# ---------------------------------------------------------------------------
# Helper: minimal synthetic exec_report_dict for monitoring tests
# ---------------------------------------------------------------------------

def make_exec_dict(
    urgency_local="LOW",
    urgency_imported="LOW",
    action_local="HOLD",
    action_imported="HOLD",
    fi_type_local="NONE",
    fi_type_imported="NONE",
    fi_usd_local=0.0,
    fi_usd_imported=0.0,
    conf_score_local=80,
    conf_score_imported=80,
    price_delta_local=0.0,
    price_delta_imported=0.0,
):
    """Return a minimal serialised ExecutiveReport dict for monitoring tests."""
    def _dec(urgency, action, fi_type, fi_usd, conf_score, price_delta):
        return {
            "urgency": urgency,
            "action_code": action,
            "action": action.replace("_", " ").title(),
            "confidence": {"score": conf_score, "label": "HIGH" if conf_score >= 75 else "MEDIUM"},
            "risk": {"level": urgency},
            "expected_cost_impact": {
                "impact_type": fi_type,
                # Field names match ExecutiveDecision.to_dict() exactly
                "expected_savings_usd":  fi_usd if fi_type == "SAVING" else 0.0,
                "cost_avoidance_usd":    fi_usd if fi_type == "COST_AVOIDANCE" else 0.0,
                "expected_savings_pkr":  None,
                "cost_avoidance_pkr":    None,
                "narrative":             "",
            },
            "price_delta_pct": price_delta,
            "qty_now_tons": 0.0,
            "qty_later_tons": 0.0,
            "mandatory_tons": 0.0,
            "base_structural_tons": 0.0,
            "opportunistic_tons": 0.0,
            "deferred_tons": 0.0,
        }

    return {
        "local":    _dec(urgency_local,    action_local,    fi_type_local,    fi_usd_local,    conf_score_local,    price_delta_local),
        "imported": _dec(urgency_imported, action_imported, fi_type_imported, fi_usd_imported, conf_score_imported, price_delta_imported),
        "portfolio_urgency": max(urgency_local, urgency_imported,
                                  key=lambda u: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(u,9)),
    }


def make_scenario_dict(
    price_signal_local="PRICE_NEUTRAL",
    price_signal_imported="PRICE_NEUTRAL",
    price_delta_local=0.0,
    price_delta_imported=0.0,
):
    """Return a minimal serialised ScenarioReport dict for monitoring tests."""
    def _sd(source, price_signal, price_delta):
        return {
            "source": source,
            "price_signal": price_signal,
            "price_delta_pct": price_delta,
            "price_confidence": "HIGH",
            "final_action": "HOLD",
            "final_qty_now_tons": 0.0,
            "final_qty_later_tons": 0.0,
            "expected_savings_usd": None,
            "expected_avoided_cost_usd": None,
            "scenario_adjustment": {"adjustment_type": "NO_CHANGE_BALANCED"},
        }

    return {
        "local":    _sd("LOCAL",    price_signal_local,    price_delta_local),
        "imported": _sd("IMPORTED", price_signal_imported, price_delta_imported),
        "portfolio_action":     "HOLD",
        "portfolio_risk_level": "LOW",
    }
