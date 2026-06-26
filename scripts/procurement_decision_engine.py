"""
procurement_decision_engine.py
--------------------------------
PSE-5B -- Executive Procurement Decision Intelligence Layer.

Converts PSE-5A SourceDecision / ScenarioReport outputs into structured
executive procurement intelligence consumable by the dashboard, management,
supply chain teams, and future AI assistants.

Does NOT recompute procurement quantities. Every numeric value is read
directly from PSE-5A and interpreted in business language.

Architecture (data flow):
    PSE-5A ScenarioReport
        └─> generate_executive_report(report, local_status, imported_status)
                ├─> _build_executive_decision(local,    local_status)
                └─> _build_executive_decision(imported, imported_status)
                         ├─> _classify_urgency()         CRITICAL / HIGH / MEDIUM / LOW
                         ├─> _score_confidence()         0-100 score + explanation
                         ├─> _build_inventory_reason()   inventory position in business terms
                         ├─> _build_price_reason()       market outlook in business terms
                         ├─> _build_timing_reason()      why now / why defer
                         ├─> _build_quantity_reason()    why this quantity / component narrative
                         ├─> _build_business_reason()    high-level "why are we doing this"
                         ├─> _build_cost_impact()        financial benefit narrative
                         ├─> _build_risk_assessment()    structured risk + if-ignored
                         ├─> _build_recommendation_summary()
                         └─> _build_executive_summary()  flagship management paragraph

Integration points:
    Streamlit dashboard : pass ExecutiveDecision fields into procurement action cards.
    Management reports  : print_executive_report() produces a formatted action sheet.
    AI assistant        : ExecutiveReport.to_dict() provides structured JSON context.
    Supply chain        : key_risks and recommended_actions lists for daily briefings.

Does not modify any upstream engine (PSE-3B/3C/3D/4A/4B/5A).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import (
    SAFETY_STOCK_LOCAL_TONS,
    SAFETY_STOCK_IMPORTED_TONS,
    LOCAL_ROP_TONS,
    IMPORTED_ROP_TONS,
    DAILY_CONSUMPTION_LOCAL,
    DAILY_CONSUMPTION_IMPORTED,
    LOCAL_LEAD_TIME_DAYS,
    IMPORTED_LEAD_TIME_DAYS,
    STATUS_CRITICAL,
    STATUS_REORDER,
    STATUS_WATCH,
    STATUS_SAFE,
)
from procurement_planning_engine import (
    PRICE_RISING,
    PRICE_FALLING,
    PRICE_NEUTRAL,
)
from procurement_consolidation_engine import (
    LOCAL_CONSOLIDATION_PERIOD_DAYS,
    S_T_LOCAL_CONSOLIDATED,
    TARGET_STOCK_IMPORTED,
)
from procurement_scenario_engine import (
    SourceDecision,
    ScenarioReport,
    ACTION_BUY_NOW,
    ACTION_BUY_FORWARD,
    ACTION_BUY_SPLIT,
    ACTION_DEFER,
    ACTION_HOLD,
    SCENARIO_STOCK_CRITICAL,
    SCENARIO_PRICE_RISING,
    SCENARIO_PRICE_FALLING,
    SCENARIO_BALANCED,
    ADJ_PULL_FORWARD_PERIODIC,
    ADJ_ADD_FORWARD_BUY,
    ADJ_DEFER_FULL,
    ADJ_DEFER_PART,
    ADJ_NO_CHANGE_STOCK_DOMINATES,
    ADJ_NO_CHANGE_COVERS_WINDOW,
    ADJ_NO_CHANGE_BALANCED,
)

__all__ = [
    "ConfidenceAssessment",
    "FinancialImpact",
    "RiskAssessment",
    "ExecutiveDecision",
    "ExecutiveReport",
    "generate_executive_report",
    "print_executive_report",
    "run_pse5b",
]

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ACTION_LABELS = {
    ACTION_BUY_NOW:     "Buy Now",
    ACTION_BUY_FORWARD: "Forward Purchase",
    ACTION_BUY_SPLIT:   "Split Order",
    ACTION_DEFER:       "Defer Purchase",
    ACTION_HOLD:        "Hold — No Action",
}

_URGENCY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ===========================================================================
# Output dataclasses
# ===========================================================================

@dataclass
class ConfidenceAssessment:
    """How confident should management be in this recommendation (0–100)?"""
    score: int           # 0-100
    label: str           # HIGH (>=75) / MEDIUM (50-74) / LOW (<50)
    explanation: str     # evidence-based reasoning, no invented probabilities

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinancialImpact:
    """Expected financial benefit of following the recommendation."""
    impact_type: str                        # COST_AVOIDANCE | SAVING | NONE
    cost_avoidance_usd: Optional[float]     # USD avoided by buying now vs later (rising price)
    cost_avoidance_pkr: Optional[float]
    expected_savings_usd: Optional[float]   # USD saved by deferring to lower price
    expected_savings_pkr: Optional[float]
    narrative: str                          # management-ready financial sentence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskAssessment:
    """Structured risk picture for this procurement source."""
    level: str       # CRITICAL / HIGH / MEDIUM / LOW
    headline: str    # one-line summary for dashboard badge
    detail: str      # fuller explanation for management
    if_ignored: str  # consequences of not following the recommendation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutiveDecision:
    """
    PSE-5B Executive Procurement Decision for one source (LOCAL or IMPORTED).

    All quantities pass through from PSE-5A without recomputation.
    This object adds urgency classification, confidence scoring,
    structured business reasoning, and an executive summary.
    """
    source: str

    # Action
    action: str        # human-readable label (e.g. "Buy Now")
    action_code: str   # PSE-5A code (BUY_NOW, DEFER, etc.) for programmatic use
    urgency: str       # CRITICAL / HIGH / MEDIUM / LOW

    # Confidence
    confidence: ConfidenceAssessment

    # Structured reasoning (business language, not programming language)
    business_reason: str   # Why are we making this recommendation?
    inventory_reason: str  # What does the inventory position tell us?
    price_reason: str      # What does the market price outlook tell us?
    timing_reason: str     # Why now? Why defer? Why hold?
    quantity_reason: str   # Why this quantity? What does each component represent?

    # Financial
    expected_cost_impact: FinancialImpact

    # Risk
    risk: RiskAssessment
    expected_risk_if_ignored: str   # shorthand for dashboard risk cards

    # Summaries
    recommendation_summary: str  # 1-2 sentence action summary
    executive_summary: str       # flagship management paragraph

    # Quantitative passthrough from PSE-5A (not recomputed here)
    qty_now_tons: float
    qty_later_tons: float
    mandatory_tons: float
    base_structural_tons: float
    opportunistic_tons: float
    deferred_tons: float
    order_date: Optional[str]
    expected_arrival: Optional[str]
    defer_until: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutiveReport:
    """PSE-5B Executive Procurement Intelligence Report — full portfolio view."""
    run_date: str
    local: ExecutiveDecision
    imported: ExecutiveDecision

    portfolio_urgency: str                     # worst-case urgency across sources
    portfolio_action_summary: str              # one-liner per source
    combined_financial_impact_narrative: str   # portfolio financial outlook
    key_risks: list                            # list[str] — for daily briefing
    recommended_actions: list                  # list[str] — for action tracker

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Private builders — each answers one management question
# ===========================================================================

def _classify_urgency(status: str, final_action: str, scenario: str) -> str:
    """What level of urgency does this decision carry?"""
    if status == STATUS_CRITICAL:
        return "CRITICAL"
    if status == STATUS_REORDER:
        return "HIGH"
    if status == STATUS_WATCH:
        if final_action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT):
            return "MEDIUM"
        return "LOW"
    # SAFE
    if scenario == SCENARIO_PRICE_RISING and final_action in (
        ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT
    ):
        return "MEDIUM"
    return "LOW"


def _score_confidence(dec: SourceDecision, status: str) -> ConfidenceAssessment:
    """
    Score the confidence in this recommendation (0–100).

    Based only on observable evidence: inventory position clarity,
    price signal strength, planning certainty. No invented probabilities.
    """
    score = 40
    factors: list[str] = []

    # Factor 1 — inventory position clarity
    _clarity = {
        STATUS_CRITICAL: (
            25,
            "Inventory is unambiguously below the minimum safety threshold, "
            "leaving no room for discretion.",
        ),
        STATUS_REORDER: (
            20,
            "Inventory has clearly reached the reorder point, "
            "making replenishment unambiguous.",
        ),
        STATUS_WATCH: (
            12,
            "Inventory is in the watch band; the procurement trigger is well-supported "
            "by current stock trends.",
        ),
        STATUS_SAFE: (
            6,
            "Inventory is comfortable; the recommendation depends primarily "
            "on the strength of the price signal.",
        ),
    }
    add, text = _clarity.get(status, (0, ""))
    score += add
    if text:
        factors.append(text)

    # Factor 2 — price signal strength
    delta = abs(dec.price_delta_pct or 0)
    if dec.price_signal in (PRICE_RISING, PRICE_FALLING):
        if delta >= 15:
            score += 20
            factors.append(
                f"The price forecast shows a strong directional move of "
                f"{dec.price_delta_pct:+.1f}%, well above the signal threshold."
            )
        elif delta >= 8:
            score += 13
            factors.append(
                f"The price forecast shows a clear directional move of {dec.price_delta_pct:+.1f}%."
            )
        else:
            score += 7
            factors.append(
                f"The price forecast shows a modest but above-threshold move "
                f"of {dec.price_delta_pct:+.1f}%."
            )
    elif dec.price_signal == PRICE_NEUTRAL:
        score += 3
        factors.append(
            "Price outlook is neutral; the recommendation rests on the "
            "inventory and calendar schedule."
        )
    else:  # UNAVAILABLE
        score -= 8
        factors.append(
            "Price forecast is currently unavailable; the recommendation "
            "is based on inventory position and the procurement calendar alone."
        )

    # Factor 3 — planning certainty
    adj = dec.scenario_adjustment.adjustment_type
    if adj in (ADJ_PULL_FORWARD_PERIODIC, ADJ_ADD_FORWARD_BUY):
        score += 10
        factors.append(
            "A confirmed scheduled procurement event provides a concrete basis "
            "for the pull-forward action."
        )
    elif adj in (ADJ_NO_CHANGE_STOCK_DOMINATES, ADJ_NO_CHANGE_COVERS_WINDOW):
        score += 8
        factors.append(
            "Stock security provides an unambiguous override; "
            "no price judgement is required."
        )
    elif adj in (ADJ_DEFER_FULL, ADJ_DEFER_PART):
        score += 5
        factors.append(
            "Deferral is supported by both inventory buffer and price direction, "
            "though the exact price trough carries inherent uncertainty."
        )
    else:
        score += 4
        factors.append("Procurement follows the established calendar schedule.")

    score = max(10, min(97, score))
    label = "HIGH" if score >= 75 else ("MEDIUM" if score >= 50 else "LOW")
    return ConfidenceAssessment(
        score=score,
        label=label,
        explanation=" ".join(factors),
    )


def _build_inventory_reason(dec: SourceDecision, status: str, source: str) -> str:
    """Explain the inventory position in plain business language."""
    rate = DAILY_CONSUMPTION_LOCAL if source == "LOCAL" else DAILY_CONSUMPTION_IMPORTED
    ss   = SAFETY_STOCK_LOCAL_TONS if source == "LOCAL" else SAFETY_STOCK_IMPORTED_TONS
    rop  = float(LOCAL_ROP_TONS    if source == "LOCAL" else IMPORTED_ROP_TONS)
    s_t  = S_T_LOCAL_CONSOLIDATED  if source == "LOCAL" else TARGET_STOCK_IMPORTED

    # Derive current inventory from cover_after_base and base_now
    inv        = round(dec.scenario_adjustment.days_of_cover_after_base * rate
                       - dec.base_plan.qty_now_tons, 0)
    inv_cover  = round(inv / rate, 1) if rate else 0.0

    if status == STATUS_CRITICAL:
        return (
            f"Current inventory of {inv:,.0f}t ({inv_cover:.0f} days of cover) has fallen "
            f"below the minimum safety threshold of {ss:,.0f}t. "
            f"At current consumption rates, the operational buffer has been exhausted. "
            f"Replenishment is non-negotiable."
        )
    if status == STATUS_REORDER:
        return (
            f"Current inventory of {inv:,.0f}t ({inv_cover:.0f} days of cover) has reached "
            f"the reorder point of {rop:,.0f}t. "
            f"Standard replenishment has been triggered to restore inventory toward "
            f"the operational target of {s_t:,.0f}t."
        )
    if status == STATUS_WATCH:
        return (
            f"Current inventory of {inv:,.0f}t ({inv_cover:.0f} days of cover) is in the "
            f"watch band — above the safety threshold of {ss:,.0f}t but approaching the "
            f"reorder point of {rop:,.0f}t. "
            f"The planned procurement maintains the operational buffer "
            f"and responds to current market conditions."
        )
    # SAFE
    return (
        f"Current inventory of {inv:,.0f}t ({inv_cover:.0f} days of cover) comfortably "
        f"exceeds the reorder point of {rop:,.0f}t. "
        f"Operational requirements are fully met. "
        f"The procurement recommendation is driven by price intelligence, not by stock necessity."
    )


def _build_price_reason(dec: SourceDecision, status: str) -> str:
    """Explain the market price outlook in business language."""
    delta = dec.price_delta_pct or 0.0

    if dec.price_signal == PRICE_RISING:
        if status in (STATUS_CRITICAL, STATUS_REORDER):
            return (
                f"Market forecasts indicate prices are expected to rise by {delta:+.1f}%. "
                f"While stock security is the primary driver here, purchasing now is also "
                f"the most cost-effective timing available given the price direction."
            )
        if abs(delta) >= 10:
            return (
                f"Market forecasts indicate prices are expected to rise by {delta:+.1f}%. "
                f"This is a material increase that raises the cost of any delay. "
                f"Securing supply at current prices is financially prudent."
            )
        return (
            f"Market forecasts indicate prices are expected to rise by {delta:+.1f}%. "
            f"Procuring now captures the current price before the forecast increase materialises."
        )

    if dec.price_signal == PRICE_FALLING:
        if status in (STATUS_CRITICAL, STATUS_REORDER):
            return (
                f"Market forecasts indicate prices may decline by {abs(delta):.1f}%. "
                f"However, with inventory below the reorder threshold, deferral is not an option. "
                f"The mandatory replenishment proceeds regardless of price direction."
            )
        return (
            f"Market forecasts indicate prices are expected to decline by {abs(delta):.1f}%. "
            f"Deferring to the forecast lower price is expected to reduce procurement cost "
            f"without material operational risk at current stock levels."
        )

    if dec.price_signal == PRICE_NEUTRAL:
        return (
            "Price forecasts indicate a neutral outlook with no material directional movement. "
            "There is no financial timing advantage available this cycle — "
            "the recommendation follows the standard procurement calendar."
        )

    # UNAVAILABLE
    return (
        "Price forecast data is currently unavailable. "
        "The recommendation is based on inventory position and the established procurement calendar. "
        "Price intelligence should be obtained before the next procurement review."
    )


def _build_timing_reason(dec: SourceDecision, status: str, source: str) -> str:
    """Explain why now, why defer, or why hold."""
    action = dec.final_action
    adj    = dec.scenario_adjustment.adjustment_type
    arrival = f", with delivery expected by {dec.expected_arrival_now}" if dec.expected_arrival_now else ""

    if status in (STATUS_CRITICAL, STATUS_REORDER):
        return (
            f"Immediate procurement is required. "
            f"The inventory position does not permit any deferral regardless of price direction. "
            f"The order should be placed today{arrival}."
        )

    if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD):
        if adj == ADJ_PULL_FORWARD_PERIODIC:
            next_date = dec.base_plan.next_event_order_date or "the next scheduled date"
            return (
                f"The next scheduled procurement event is on {next_date}. "
                f"Advancing this order to today locks in the current price before "
                f"the forecast increase arrives. "
                f"Current inventory is sufficient to support the pull-forward "
                f"without creating excess stock{arrival}."
            )
        if adj == ADJ_ADD_FORWARD_BUY:
            return (
                f"An opportunistic forward-buy tranche has been added to this cycle. "
                f"Purchasing the additional quantity now avoids placing a second order "
                f"at the forecast higher price{arrival}."
            )
        return (
            f"Procurement should proceed on the current schedule. "
            f"The order should be placed today{arrival}."
        )

    if action == ACTION_DEFER:
        return (
            f"The planned procurement can safely be deferred. "
            f"Current inventory supports operations while awaiting a more favourable price. "
            f"The latest safe order date is {dec.final_order_window_later_end}, "
            f"providing sufficient time to place the order and receive delivery "
            f"without disrupting supply."
        )

    if action == ACTION_BUY_SPLIT:
        return (
            f"The procurement has been split: the mandatory portion is ordered today "
            f"(delivery by {dec.expected_arrival_now}), "
            f"and the discretionary portion is deferred to "
            f"{dec.final_order_window_later_end} to capture a more favourable price."
        )

    # HOLD
    return (
        "No procurement action is required this cycle. "
        "Inventory and price conditions do not trigger any buying decision. "
        "The position will be reassessed at the next scheduled review."
    )


def _build_quantity_reason(dec: SourceDecision) -> str:
    """Explain the composition of the recommended quantity in business terms."""
    mandatory  = dec.mandatory_component_tons
    base_nm    = dec.base_non_mandatory_component_tons
    oppo       = dec.opportunistic_component_tons
    deferred   = dec.deferred_component_tons
    total_now  = dec.final_qty_now_tons
    total_later = dec.final_qty_later_tons

    if dec.final_action == ACTION_HOLD:
        return "No purchase is required this cycle."

    if dec.final_action == ACTION_DEFER:
        return (
            f"The full planned order of {deferred:,.1f}t has been deferred. "
            f"With no mandatory requirement and prices expected to decline, "
            f"the entire quantity will be purchased at the forecast lower price "
            f"before the latest safe order date."
        )

    # Compose narrative from whichever components are present
    if mandatory > 0 and base_nm > 0 and oppo > 0:
        return (
            f"The total order of {total_now:,.1f}t has three components: "
            f"{mandatory:,.1f}t of mandatory replenishment to restore inventory to the "
            f"reorder threshold (non-negotiable regardless of price), "
            f"{base_nm:,.1f}t of planned structural restocking to reach the operational target, "
            f"and {oppo:,.1f}t of opportunistic forward-buy timed to the current price."
        )
    if mandatory > 0 and base_nm > 0:
        return (
            f"The order of {total_now:,.1f}t covers {mandatory:,.1f}t of mandatory deficit "
            f"replenishment plus {base_nm:,.1f}t of planned structural restocking "
            f"to bring inventory toward the operational target. "
            f"No discretionary forward-buy has been added."
        )
    if mandatory > 0:
        return (
            f"This order of {total_now:,.1f}t covers the mandatory replenishment requirement only. "
            f"It restores inventory to the minimum acceptable threshold. "
            f"No additional opportunistic quantity has been included."
        )
    if base_nm > 0 and oppo > 0:
        return (
            f"The total order of {total_now:,.1f}t combines {base_nm:,.1f}t of planned "
            f"structural procurement per the calendar schedule, and {oppo:,.1f}t of "
            f"opportunistic forward-buy to lock in current pricing ahead of the forecast increase. "
            f"There is no mandatory deficit component — this is a proactive purchase."
        )
    if oppo > 0:
        return (
            f"This is a purely opportunistic forward-buy of {oppo:,.1f}t. "
            f"No structural procurement is due this cycle. "
            f"The order is timed entirely to the current price, "
            f"ahead of the forecast increase."
        )
    if base_nm > 0:
        return (
            f"This order of {total_now:,.1f}t represents the planned structural "
            f"procurement per the calendar schedule. "
            f"No mandatory deficit and no opportunistic addition are included."
        )

    return f"Total procurement this cycle: {total_now:,.1f}t."


def _build_business_reason(dec: SourceDecision, status: str, source: str) -> str:
    """High-level statement of why this recommendation is being made."""
    signal = dec.price_signal
    action = dec.final_action
    label  = "Local cotton" if source == "LOCAL" else "Imported cotton"

    if status == STATUS_CRITICAL:
        if signal == PRICE_RISING:
            return (
                f"{label} inventory has dropped below the minimum safety level. "
                f"Immediate replenishment is required to protect operational continuity. "
                f"With prices also forecast to rise, there is additional urgency to "
                f"secure supply now before costs increase further."
            )
        if signal == PRICE_FALLING:
            return (
                f"{label} inventory has dropped below the minimum safety level. "
                f"Replenishment is mandatory and cannot be deferred regardless of "
                f"price direction. Operational continuity takes absolute priority."
            )
        return (
            f"{label} inventory has dropped below the minimum safety level. "
            f"Immediate procurement is required to restore the operational buffer."
        )

    if status == STATUS_REORDER:
        if signal == PRICE_RISING:
            return (
                f"{label} inventory has reached the reorder threshold and prices are "
                f"forecast to rise. Mandatory replenishment is required; executing now "
                f"also represents the most cost-effective timing available."
            )
        if signal == PRICE_FALLING:
            return (
                f"{label} inventory has reached the reorder threshold. "
                f"Despite the expectation of lower future prices, mandatory replenishment "
                f"cannot be deferred — operational stock security takes priority."
            )
        return (
            f"{label} inventory has reached the reorder threshold. "
            f"Standard replenishment is required to maintain uninterrupted supply."
        )

    if status == STATUS_WATCH:
        if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD) and signal == PRICE_RISING:
            return (
                f"{label} inventory is in the watch band and prices are forecast to rise. "
                f"Advancing the planned procurement now addresses both the approaching "
                f"reorder threshold and the expected cost increase."
            )
        if action == ACTION_DEFER and signal == PRICE_FALLING:
            return (
                f"{label} inventory is in the watch band but sufficient for current operations. "
                f"With prices expected to decline, procurement can be safely deferred "
                f"to reduce acquisition cost."
            )
        return (
            f"{label} inventory is in the watch band. "
            f"Procurement is planned to maintain the operational buffer."
        )

    # SAFE
    if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD) and signal == PRICE_RISING:
        return (
            f"{label} inventory is comfortable and operationally sufficient, "
            f"but market forecasts indicate prices are rising. "
            f"Advancing the next planned order to today is expected to reduce "
            f"procurement cost by locking in current pricing."
        )
    if action == ACTION_DEFER and signal == PRICE_FALLING:
        return (
            f"{label} inventory comfortably exceeds operational requirements "
            f"and prices are expected to decline. "
            f"Deferring the planned purchase is expected to reduce procurement cost "
            f"without any operational risk."
        )
    return (
        f"{label} inventory is sufficient and price signals are neutral. "
        f"Procurement follows the planned calendar schedule."
    )


def _build_cost_impact(dec: SourceDecision, pkr_rate: float) -> FinancialImpact:
    """Translate PSE-5A cost figures into a management-ready financial narrative."""
    avoided_usd = dec.expected_avoided_cost_usd
    avoided_pkr = dec.expected_avoided_cost_pkr
    savings_usd = dec.expected_savings_usd
    savings_pkr = dec.expected_savings_pkr

    if avoided_usd and avoided_usd > 0:
        pkr_m = (avoided_pkr if avoided_pkr else avoided_usd * pkr_rate) / 1_000_000
        return FinancialImpact(
            impact_type="COST_AVOIDANCE",
            cost_avoidance_usd=avoided_usd,
            cost_avoidance_pkr=avoided_pkr,
            expected_savings_usd=None,
            expected_savings_pkr=None,
            narrative=(
                f"Procuring now is expected to avoid approximately "
                f"USD {avoided_usd:,.0f} (PKR {pkr_m:.1f}M) in additional procurement cost "
                f"compared to buying at the forecast higher price. "
                f"This represents the estimated benefit of advancing the purchase to today."
            ),
        )

    if savings_usd and savings_usd > 0:
        pkr_m = (savings_pkr if savings_pkr else savings_usd * pkr_rate) / 1_000_000
        return FinancialImpact(
            impact_type="SAVING",
            cost_avoidance_usd=None,
            cost_avoidance_pkr=None,
            expected_savings_usd=savings_usd,
            expected_savings_pkr=savings_pkr,
            narrative=(
                f"Deferring procurement is expected to save approximately "
                f"USD {savings_usd:,.0f} (PKR {pkr_m:.1f}M) compared to purchasing today. "
                f"This saving is contingent on the forecast price decline materialising."
            ),
        )

    return FinancialImpact(
        impact_type="NONE",
        cost_avoidance_usd=None,
        cost_avoidance_pkr=None,
        expected_savings_usd=None,
        expected_savings_pkr=None,
        narrative=(
            "No material financial advantage is associated with procurement timing in this cycle."
        ),
    )


def _build_risk_assessment(dec: SourceDecision, status: str) -> RiskAssessment:
    """Classify the supply risk and describe the consequence of inaction."""
    action = dec.final_action

    if status == STATUS_CRITICAL:
        return RiskAssessment(
            level="CRITICAL",
            headline="Inventory below minimum safety threshold — supply continuity at risk.",
            detail=(
                "Current inventory has fallen below the minimum safety stock level. "
                "Any further depletion without replenishment increases the risk of a "
                "supply gap affecting production. "
                "Immediate action is the only appropriate response."
            ),
            if_ignored=(
                "Continued inaction risks inventory reaching zero, potentially causing "
                "a production disruption. Emergency purchasing at spot or premium prices "
                "would likely result in significantly higher costs, "
                "and timely delivery cannot be guaranteed given procurement lead times."
            ),
        )

    if status == STATUS_REORDER:
        return RiskAssessment(
            level="HIGH",
            headline="Inventory at reorder threshold — mandatory replenishment required.",
            detail=(
                "Inventory has reached the reorder point. The current stock provides "
                "a buffer, but without replenishment it will fall toward the safety "
                "threshold within the procurement lead time. "
                "Standard replenishment is required to maintain uninterrupted supply."
            ),
            if_ignored=(
                "Without replenishment, inventory will approach the safety threshold "
                "before the next order can be received. "
                "If prices are also rising, delay further increases the cost "
                "of the eventual mandatory purchase."
            ),
        )

    if status == STATUS_WATCH:
        if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT):
            return RiskAssessment(
                level="MEDIUM",
                headline="Inventory in watch band — proactive procurement is warranted.",
                detail=(
                    "Inventory is above the safety threshold but approaching the reorder point. "
                    "Proactive procurement maintains a healthy buffer and, where prices "
                    "are rising, reduces cost exposure."
                ),
                if_ignored=(
                    "Continued drawdown without procurement will move inventory into "
                    "mandatory reorder territory, forcing a buy under less favourable conditions. "
                    "If prices are rising, the delayed purchase will cost more."
                ),
            )
        return RiskAssessment(
            level="LOW",
            headline="Inventory in watch band — safe to defer with close monitoring.",
            detail=(
                "Inventory is above the safety threshold. Deferral is supportable "
                "given current stock levels and the price outlook, "
                "but monitoring is recommended to ensure timely action "
                "before the reorder point is reached."
            ),
            if_ignored=(
                "The deferral window is time-limited. "
                "Monitoring should continue to ensure the reorder point is not reached "
                "without a plan in place."
            ),
        )

    # SAFE
    if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD):
        return RiskAssessment(
            level="MEDIUM",
            headline="No supply risk — recommendation is price-driven.",
            detail=(
                "Inventory is comfortable and operationally sufficient. "
                "The procurement recommendation is purely price-driven. "
                "Missing this window does not create a supply risk "
                "but does create a cost risk."
            ),
            if_ignored=(
                "Failure to act will result in purchasing the same quantity "
                "at the forecast higher price. "
                "There is no supply risk from inaction, "
                "but the cost of delay is material if the price forecast proves accurate."
            ),
        )

    return RiskAssessment(
        level="LOW",
        headline="Inventory is healthy — no immediate procurement risk.",
        detail=(
            "Inventory comfortably exceeds operational requirements. "
            "No supply risk is present. "
            "Following the deferral recommendation is expected to reduce cost "
            "without any supply impact."
        ),
        if_ignored=(
            "No material supply risk from proceeding with standard calendar procurement. "
            "Deferring captures an expected cost reduction; "
            "choosing not to defer results in a slightly higher procurement cost "
            "if the price forecast is accurate."
        ),
    )


def _build_recommendation_summary(dec: SourceDecision, urgency: str) -> str:
    """One to two sentence action summary for procurement team action trackers."""
    action    = dec.final_action
    total_now = dec.final_qty_now_tons
    total_later = dec.final_qty_later_tons

    urgency_statement = {
        "CRITICAL": "Immediate action required.",
        "HIGH":     "Action required this procurement cycle.",
        "MEDIUM":   "Action recommended this cycle.",
        "LOW":      "No urgent action required.",
    }.get(urgency, "")

    if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD):
        arrival = (
            f" Expected delivery: {dec.expected_arrival_now}."
            if dec.expected_arrival_now else ""
        )
        return (
            f"{urgency_statement} Place order for {total_now:,.1f}t today.{arrival}"
        ).strip()

    if action == ACTION_DEFER:
        return (
            f"{urgency_statement} Defer order of {total_later:,.1f}t to latest safe date "
            f"{dec.final_order_window_later_end}."
        ).strip()

    if action == ACTION_BUY_SPLIT:
        return (
            f"{urgency_statement} Order {total_now:,.1f}t today "
            f"(arrival {dec.expected_arrival_now}); "
            f"defer remaining {total_later:,.1f}t to {dec.final_order_window_later_end}."
        ).strip()

    return f"{urgency_statement} No procurement action required this cycle.".strip()


def _build_executive_summary(
    dec: SourceDecision,
    status: str,
    source: str,
    cost_impact: FinancialImpact,
) -> str:
    """
    Flagship management paragraph that answers all eight executive questions
    in three to four concise sentences.
    """
    signal = dec.price_signal
    action = dec.final_action
    label  = "Local cotton" if source == "LOCAL" else "Imported cotton"

    # Opening — inventory position
    opening = {
        STATUS_CRITICAL: (
            f"{label} inventory has fallen below the minimum safety threshold. "
            f"Immediate replenishment is non-negotiable."
        ),
        STATUS_REORDER: (
            f"{label} inventory has reached the reorder point, "
            f"triggering mandatory replenishment."
        ),
        STATUS_WATCH: (
            f"{label} inventory is in the watch band, "
            f"approaching the reorder threshold."
        ),
        STATUS_SAFE: f"{label} inventory is operationally sufficient.",
    }.get(status, f"{label} inventory position has been reviewed.")

    # Price context (omit for CRITICAL/REORDER if it would distract)
    if signal == PRICE_RISING and status not in (STATUS_CRITICAL, STATUS_REORDER):
        price_ctx = (
            f" Market forecasts indicate prices are expected to rise by "
            f"{dec.price_delta_pct:+.1f}%, making current pricing attractive."
        )
    elif signal == PRICE_FALLING and status not in (STATUS_CRITICAL, STATUS_REORDER):
        price_ctx = (
            f" Market forecasts indicate prices are expected to decline by "
            f"{abs(dec.price_delta_pct or 0):.1f}%, creating an opportunity to defer."
        )
    elif signal == PRICE_RISING and status in (STATUS_CRITICAL, STATUS_REORDER):
        price_ctx = " Rising price forecasts add further urgency to prompt procurement."
    else:
        price_ctx = ""

    # Action statement
    if action in (ACTION_BUY_NOW, ACTION_BUY_FORWARD):
        arrival = (
            f", with delivery expected by {dec.expected_arrival_now}"
            if dec.expected_arrival_now else ""
        )
        action_text = (
            f" Recommendation: procure {dec.final_qty_now_tons:,.1f}t today{arrival}."
        )
    elif action == ACTION_DEFER:
        action_text = (
            f" Recommendation: defer the planned order of {dec.final_qty_later_tons:,.1f}t "
            f"to the latest safe date of {dec.final_order_window_later_end}."
        )
    elif action == ACTION_BUY_SPLIT:
        action_text = (
            f" Recommendation: order {dec.final_qty_now_tons:,.1f}t immediately "
            f"and defer {dec.final_qty_later_tons:,.1f}t to "
            f"{dec.final_order_window_later_end}."
        )
    else:
        action_text = " No procurement action is required this cycle."

    # Financial benefit
    if cost_impact.impact_type == "COST_AVOIDANCE" and cost_impact.cost_avoidance_usd:
        pkr_m = (
            (cost_impact.cost_avoidance_pkr or cost_impact.cost_avoidance_usd * 281)
            / 1_000_000
        )
        financial_text = (
            f" Expected benefit: approximately USD {cost_impact.cost_avoidance_usd:,.0f} "
            f"(PKR {pkr_m:.1f}M) in procurement cost avoided."
        )
    elif cost_impact.impact_type == "SAVING" and cost_impact.expected_savings_usd:
        pkr_m = (
            (cost_impact.expected_savings_pkr or cost_impact.expected_savings_usd * 281)
            / 1_000_000
        )
        financial_text = (
            f" Expected saving: approximately USD {cost_impact.expected_savings_usd:,.0f} "
            f"(PKR {pkr_m:.1f}M) from deferred purchase at the forecast lower price."
        )
    else:
        financial_text = ""

    return (opening + price_ctx + action_text + financial_text).strip()


# ===========================================================================
# Per-source assembler
# ===========================================================================

def _build_executive_decision(
    dec: SourceDecision,
    status: str,
    pkr_rate: float,
) -> ExecutiveDecision:
    """Assemble all reasoning components into one ExecutiveDecision."""
    source    = dec.source
    urgency   = _classify_urgency(status, dec.final_action, dec.scenario)
    confidence = _score_confidence(dec, status)
    inv_reason = _build_inventory_reason(dec, status, source)
    price_reason = _build_price_reason(dec, status)
    timing_reason = _build_timing_reason(dec, status, source)
    qty_reason   = _build_quantity_reason(dec)
    biz_reason   = _build_business_reason(dec, status, source)
    cost_impact  = _build_cost_impact(dec, pkr_rate)
    risk         = _build_risk_assessment(dec, status)
    rec_summary  = _build_recommendation_summary(dec, urgency)
    exec_summary = _build_executive_summary(dec, status, source, cost_impact)

    return ExecutiveDecision(
        source=source,
        action=_ACTION_LABELS.get(dec.final_action, dec.final_action),
        action_code=dec.final_action,
        urgency=urgency,
        confidence=confidence,
        business_reason=biz_reason,
        inventory_reason=inv_reason,
        price_reason=price_reason,
        timing_reason=timing_reason,
        quantity_reason=qty_reason,
        expected_cost_impact=cost_impact,
        risk=risk,
        expected_risk_if_ignored=risk.if_ignored,
        recommendation_summary=rec_summary,
        executive_summary=exec_summary,
        qty_now_tons=dec.final_qty_now_tons,
        qty_later_tons=dec.final_qty_later_tons,
        mandatory_tons=dec.mandatory_component_tons,
        base_structural_tons=dec.base_non_mandatory_component_tons,
        opportunistic_tons=dec.opportunistic_component_tons,
        deferred_tons=dec.deferred_component_tons,
        order_date=dec.final_order_date_now,
        expected_arrival=dec.expected_arrival_now,
        defer_until=dec.final_order_window_later_end,
    )


# ===========================================================================
# Portfolio helpers
# ===========================================================================

def _portfolio_action_summary(local: ExecutiveDecision, imported: ExecutiveDecision) -> str:
    parts = []
    for dec in (local, imported):
        if dec.action_code == ACTION_HOLD:
            parts.append(f"{dec.source}: No action required.")
        else:
            qty = dec.qty_now_tons if dec.action_code != ACTION_DEFER else dec.qty_later_tons
            parts.append(f"{dec.source}: {dec.action} — {qty:,.1f}t.")
    return " | ".join(parts)


def _portfolio_financial(local: ExecutiveDecision, imported: ExecutiveDecision) -> str:
    total_avoided = sum(
        dec.expected_cost_impact.cost_avoidance_usd or 0
        for dec in (local, imported)
    )
    total_saved = sum(
        dec.expected_cost_impact.expected_savings_usd or 0
        for dec in (local, imported)
    )
    if total_avoided > 0 and total_saved > 0:
        return (
            f"Combined expected cost avoidance (forward procurement): "
            f"USD {total_avoided:,.0f}. "
            f"Combined expected savings (deferral): USD {total_saved:,.0f}."
        )
    if total_avoided > 0:
        return (
            f"Combined expected cost avoidance from forward procurement: "
            f"USD {total_avoided:,.0f}."
        )
    if total_saved > 0:
        return f"Combined expected procurement savings from deferral: USD {total_saved:,.0f}."
    return "No material combined financial impact from procurement timing in this cycle."


def _portfolio_key_risks(local: ExecutiveDecision, imported: ExecutiveDecision) -> list:
    risks = []
    for dec in (local, imported):
        if dec.risk.level in ("CRITICAL", "HIGH", "MEDIUM"):
            risks.append(f"{dec.source}: {dec.risk.headline}")
    return risks or ["No elevated procurement risks identified this cycle."]


def _portfolio_actions(local: ExecutiveDecision, imported: ExecutiveDecision) -> list:
    actions = []
    for dec in (local, imported):
        if dec.action_code != ACTION_HOLD:
            actions.append(f"{dec.source}: {dec.recommendation_summary}")
    return actions or ["No procurement actions required this cycle."]


# ===========================================================================
# PUBLIC API
# ===========================================================================

def generate_executive_report(
    scenario_report: ScenarioReport,
    local_status: str,
    imported_status: str,
    pkr_rate: Optional[float] = None,
) -> ExecutiveReport:
    """
    Generate a PSE-5B Executive Procurement Intelligence Report.

    Args:
        scenario_report : ScenarioReport from PSE-5A compute_scenario_decision().
        local_status    : Inventory status for LOCAL source (STATUS_CRITICAL /
                          STATUS_REORDER / STATUS_WATCH / STATUS_SAFE).
        imported_status : Inventory status for IMPORTED source.
        pkr_rate        : USD/PKR exchange rate. Defaults to the value stored in
                          scenario_report.price_inputs_used.

    Returns:
        ExecutiveReport with fully populated ExecutiveDecision objects for both sources.
    """
    _pkr = pkr_rate or scenario_report.price_inputs_used.get("pkr_rate", 281.0)

    local_dec    = _build_executive_decision(scenario_report.local,    local_status,    _pkr)
    imported_dec = _build_executive_decision(scenario_report.imported, imported_status, _pkr)

    portfolio_urgency = (
        local_dec.urgency
        if _URGENCY_RANK.get(local_dec.urgency, 9) <= _URGENCY_RANK.get(imported_dec.urgency, 9)
        else imported_dec.urgency
    )

    return ExecutiveReport(
        run_date=scenario_report.run_date,
        local=local_dec,
        imported=imported_dec,
        portfolio_urgency=portfolio_urgency,
        portfolio_action_summary=_portfolio_action_summary(local_dec, imported_dec),
        combined_financial_impact_narrative=_portfolio_financial(local_dec, imported_dec),
        key_risks=_portfolio_key_risks(local_dec, imported_dec),
        recommended_actions=_portfolio_actions(local_dec, imported_dec),
    )


# ===========================================================================
# FULL PIPELINE ENTRY POINT
# ===========================================================================

def run_pse5b(
    workbook_path: Optional[str] = None,
    input_path: Optional[str] = None,
    live_prices: bool = True,
    current_price_usd_per_lb: Optional[float] = None,
    forecast_h1_usd_per_lb: Optional[float] = None,
    forecast_h3_usd_per_lb: Optional[float] = None,
    forecast_h1_bounds=None,
    forecast_h3_bounds=None,
    pkr_rate: Optional[float] = None,
    horizon_days: int = 180,
    today: Optional[date] = None,
) -> tuple:
    """
    Run the full PSE pipeline through PSE-5B and return both reports.

    Returns:
        (ScenarioReport, ExecutiveReport)
    """
    from procurement_scenario_engine import run_pse5a
    from procurement_orchestrator import run_orchestration

    today = today or date.today()

    orch = run_orchestration(input_path=input_path, workbook_path=workbook_path)
    so   = orch["strategy_output"]

    scenario_report = run_pse5a(
        workbook_path=workbook_path,
        input_path=input_path,
        live_prices=live_prices,
        current_price_usd_per_lb=current_price_usd_per_lb,
        forecast_h1_usd_per_lb=forecast_h1_usd_per_lb,
        forecast_h3_usd_per_lb=forecast_h3_usd_per_lb,
        forecast_h1_bounds=forecast_h1_bounds,
        forecast_h3_bounds=forecast_h3_bounds,
        pkr_rate=pkr_rate,
        horizon_days=horizon_days,
        today=today,
        _orch=orch,  # reuse already-loaded orchestration; avoids second workbook read
    )

    executive_report = generate_executive_report(
        scenario_report=scenario_report,
        local_status=so.local_status,
        imported_status=so.imported_status,
    )

    return scenario_report, executive_report


# ===========================================================================
# REPORT PRINTER
# ===========================================================================

def _w(width: int = 78, char: str = "-") -> str:
    return char * width


def print_executive_report(report: ExecutiveReport) -> None:
    """Print the PSE-5B Executive Report in management action-sheet format."""

    print(_w(82, "="))
    print("PSE-5B  EXECUTIVE PROCUREMENT INTELLIGENCE REPORT")
    print(f"  Date             : {report.run_date}")
    print(f"  Portfolio Risk   : {report.portfolio_urgency}")
    print(_w(82, "="))

    print("\n  RECOMMENDED ACTIONS")
    for action in report.recommended_actions:
        print(f"    •  {action}")

    print("\n  KEY RISKS")
    for risk in report.key_risks:
        print(f"    •  {risk}")

    if report.combined_financial_impact_narrative:
        print("\n  FINANCIAL OUTLOOK")
        print(f"    {report.combined_financial_impact_narrative}")

    for dec in (report.local, report.imported):
        print()
        print(_w(82, "="))
        print(f"  {dec.source} COTTON  |  {dec.action.upper()}  |  {dec.urgency}")
        print(_w(82, "="))

        print(f"\n  Confidence: {dec.confidence.score}/100  [{dec.confidence.label}]")
        _wrap(dec.confidence.explanation, indent="    ")

        print("\n  EXECUTIVE SUMMARY")
        _wrap(dec.executive_summary, indent="    ")

        print("\n  BUSINESS RATIONALE")
        _wrap(dec.business_reason, indent="    ")

        print("\n  INVENTORY POSITION")
        _wrap(dec.inventory_reason, indent="    ")

        print("\n  PRICE INTELLIGENCE")
        _wrap(dec.price_reason, indent="    ")

        print("\n  TIMING")
        _wrap(dec.timing_reason, indent="    ")

        print("\n  QUANTITY BREAKDOWN")
        _wrap(dec.quantity_reason, indent="    ")
        print()
        print(f"    {'Mandatory replenishment':<28}: {dec.mandatory_tons:>9,.1f} t")
        print(f"    {'Structural (base plan)':<28}: {dec.base_structural_tons:>9,.1f} t")
        print(f"    {'Opportunistic (forward buy)':<28}: {dec.opportunistic_tons:>9,.1f} t")
        print(f"    {'Deferred to later window':<28}: {dec.deferred_tons:>9,.1f} t")
        print(f"    {_w(44, '-')}")
        print(f"    {'Total — order today':<28}: {dec.qty_now_tons:>9,.1f} t")
        if dec.qty_later_tons > 0:
            print(f"    {'Total — deferred':<28}: {dec.qty_later_tons:>9,.1f} t")

        print("\n  FINANCIAL IMPACT")
        _wrap(dec.expected_cost_impact.narrative, indent="    ")

        print(f"\n  RISK ASSESSMENT  [{dec.risk.level}]")
        print(f"    {dec.risk.headline}")
        _wrap(dec.risk.detail, indent="    ")
        print("\n    If this recommendation is not followed:")
        _wrap(dec.risk.if_ignored, indent="    ")

    print()
    print(_w(82, "="))
    print("END OF EXECUTIVE REPORT")
    print(_w(82, "="))


def _wrap(text: str, indent: str = "  ", width: int = 78) -> None:
    """Print text with simple sentence-level line breaks for readability."""
    sentences = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]
    for s in sentences:
        print(f"{indent}{s}")
