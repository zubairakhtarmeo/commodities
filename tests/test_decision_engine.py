"""
test_decision_engine.py
-----------------------
Tests for procurement_decision_engine.py (PSE-5B).
Replicates all 6 checks from pse5b_validation.py and extends with edge cases.
"""
from __future__ import annotations

from datetime import date

import pytest

from procurement_scenario_engine import (
    ADJ_PULL_FORWARD_PERIODIC, ADJ_ADD_FORWARD_BUY,
    ADJ_DEFER_FULL, ADJ_DEFER_PART,
)
from procurement_decision_engine import generate_executive_report

from conftest import SCENARIO_PARAMS, EXPECTED_URGENCY, EXPECTED_RISK, EXPECTED_QTY, build_reports

TODAY     = date(2026, 6, 27)
PKR       = 278.5
CUR_PRICE = 0.78

# Required string fields on every ExecutiveDecision (V1)
_REQUIRED_STR_FIELDS = [
    "business_reason", "inventory_reason", "price_reason",
    "timing_reason", "quantity_reason", "recommendation_summary",
    "executive_summary", "expected_risk_if_ignored",
]

# Risk levels compatible with each urgency level (V5)
_RISK_COMPAT = {
    "CRITICAL": {"CRITICAL"},
    "HIGH":     {"HIGH"},
    "MEDIUM":   {"MEDIUM", "LOW"},
    "LOW":      {"LOW", "MEDIUM"},
}


# ===========================================================================
# V1 — All required string fields populated
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v1_all_string_fields_populated(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    missing = [f for f in _REQUIRED_STR_FIELDS if not getattr(dec, f, "")]
    assert not missing, f"{label}: missing fields: {missing}"


# ===========================================================================
# V2 — Urgency mapping
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v2_urgency_mapping(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    expected = EXPECTED_URGENCY[label]
    assert dec.urgency == expected, (
        f"{label}: urgency={dec.urgency!r}, expected={expected!r}"
    )


# ===========================================================================
# V3 — Confidence score in 0-100 range, label consistent
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v3_confidence_score(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    score = dec.confidence.score
    lbl   = dec.confidence.label

    assert 0 <= score <= 100, f"{label}: confidence score {score} out of range"
    if lbl == "HIGH":
        assert score >= 75, f"{label}: label=HIGH but score={score}"
    elif lbl == "MEDIUM":
        assert 50 <= score < 75, f"{label}: label=MEDIUM but score={score}"
    elif lbl == "LOW":
        assert score < 50, f"{label}: label=LOW but score={score}"
    else:
        pytest.fail(f"{label}: unexpected confidence label {lbl!r}")


# ===========================================================================
# V4 — Financial impact type consistent with adjustment type
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v4_financial_impact_consistency(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    pse5a_dec = sr.local if source == "LOCAL" else sr.imported

    adj     = pse5a_dec.scenario_adjustment.adjustment_type
    fi_type = dec.expected_cost_impact.impact_type

    if adj in (ADJ_PULL_FORWARD_PERIODIC, ADJ_ADD_FORWARD_BUY):
        assert fi_type in ("COST_AVOIDANCE", "NONE"), (
            f"{label}: forward adj expects COST_AVOIDANCE/NONE, got {fi_type!r}"
        )
    elif adj in (ADJ_DEFER_FULL, ADJ_DEFER_PART):
        assert fi_type in ("SAVING", "NONE"), (
            f"{label}: defer adj expects SAVING/NONE, got {fi_type!r}"
        )
    else:
        assert fi_type == "NONE", (
            f"{label}: no-change adj expects NONE, got {fi_type!r}"
        )


# ===========================================================================
# V5 — Risk level consistent with urgency
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v5_risk_consistent_with_urgency(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    compat = _RISK_COMPAT.get(dec.urgency, set())
    assert dec.risk.level in compat, (
        f"{label}: urgency={dec.urgency} but risk.level={dec.risk.level!r} "
        f"(expected one of {compat})"
    )


# ===========================================================================
# V6 — Quantity passthrough from PSE-5A
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_v6_quantity_passthrough(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec       = er.local if source == "LOCAL" else er.imported
    pse5a_dec = sr.local if source == "LOCAL" else sr.imported

    assert abs(dec.qty_now_tons   - pse5a_dec.final_qty_now_tons)   < 0.01, \
        f"{label}: qty_now mismatch {dec.qty_now_tons} vs {pse5a_dec.final_qty_now_tons}"
    assert abs(dec.qty_later_tons - pse5a_dec.final_qty_later_tons) < 0.01, \
        f"{label}: qty_later mismatch"
    assert abs(dec.mandatory_tons - pse5a_dec.mandatory_component_tons) < 0.01, \
        f"{label}: mandatory mismatch"


# ===========================================================================
# Golden urgency / risk snapshots
# ===========================================================================

@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_golden_urgency(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    assert dec.urgency == EXPECTED_URGENCY[label]


@pytest.mark.parametrize("label,source,local_inv,imp_inv,h1,h3", SCENARIO_PARAMS)
def test_golden_risk(label, source, local_inv, imp_inv, h1, h3):
    sr, er, ls, is_ = build_reports(label, source, local_inv, imp_inv, h1, h3)
    dec = er.local if source == "LOCAL" else er.imported
    assert dec.risk.level == EXPECTED_RISK[label]


# ===========================================================================
# generate_executive_report — structural tests
# ===========================================================================

class TestGenerateExecutiveReport:
    def test_portfolio_urgency_in_valid_set(self):
        sr, er, ls, is_ = build_reports(*SCENARIO_PARAMS[0])
        assert er.portfolio_urgency in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_executive_report_has_both_sources(self):
        for (label, src, li, ii, h1, h3) in SCENARIO_PARAMS[:2]:
            sr, er, ls, is_ = build_reports(label, src, li, ii, h1, h3)
            assert er.local    is not None
            assert er.imported is not None

    def test_critical_portfolio_urgency_when_local_critical(self):
        """If LOCAL is CRITICAL, portfolio urgency must be CRITICAL."""
        sr, er, ls, is_ = build_reports("S1: CRITICAL+RISING", "LOCAL", 800.0, 8000.0, 0.97, 1.06)
        assert er.portfolio_urgency == "CRITICAL"

    def test_to_dict_serialisable(self):
        sr, er, ls, is_ = build_reports(*SCENARIO_PARAMS[0])
        d = er.to_dict()
        assert isinstance(d, dict)
        assert "local" in d and "imported" in d

    def test_action_code_in_valid_vocab(self):
        valid = {"BUY_NOW", "BUY_FORWARD", "BUY_SPLIT", "DEFER", "HOLD"}
        for (label, src, li, ii, h1, h3) in SCENARIO_PARAMS:
            sr, er, ls, is_ = build_reports(label, src, li, ii, h1, h3)
            dec = er.local if src == "LOCAL" else er.imported
            assert dec.action_code in valid, (
                f"{label}: action_code {dec.action_code!r} not in valid vocab"
            )
