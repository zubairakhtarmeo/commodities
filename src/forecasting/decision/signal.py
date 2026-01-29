"""Signal generation from decision layer insights.

Rules-based logic to produce Buy / Hold / Delay signals based on:
- Scenario analysis (is market moving in favorable direction?)
- Impact attribution (which factors are driving the move?)
- User-defined thresholds
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    """Trading signal."""
    BUY = "BUY"
    HOLD = "HOLD"
    DELAY = "DELAY"


@dataclass
class SignalOutput:
    """One signal decision."""
    timestamp: str
    signal: Signal
    price_forecast: float
    confidence: float  # 0.0 to 1.0
    rationale: str  # human-readable explanation
    scenario_name: str  # which scenario triggered signal
    top_drivers: list[str]  # top impact variables


class SignalGenerator:
    """Generate buy/hold/delay signals from scenario and elasticity outputs."""

    def __init__(
        self,
        buy_threshold_pct: float = 2.0,  # if scenario shows >2% upside
        sell_threshold_pct: float = -1.5,  # if scenario shows >1.5% downside
        min_confidence: float = 0.6,  # min confidence to trade
    ):
        """
        Args:
            buy_threshold_pct: trigger buy if scenario shows upside >this
            sell_threshold_pct: trigger hold/delay if downside >this
            min_confidence: min confidence level required
        """
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct
        self.min_confidence = min_confidence

    def generate_signal(
        self,
        timestamp: str,
        ml_forecast_price: float,
        scenario_price: float,
        scenario_name: str,
        top_drivers: list[str],
        elasticity_coverage: float = 0.8,  # % of variance explained by elasticities
    ) -> SignalOutput:
        """Generate a signal based on scenario outcome.

        Args:
            timestamp: forecast timestamp
            ml_forecast_price: base ML forecast
            scenario_price: scenario-adjusted forecast
            scenario_name: name of the active scenario
            top_drivers: top impact variables (for context)
            elasticity_coverage: how much variance is explained (affects confidence)

        Returns:
            SignalOutput with decision and rationale.
        """
        price_change_pct = (
            (scenario_price - ml_forecast_price) / ml_forecast_price * 100.0
            if ml_forecast_price != 0
            else 0.0
        )

        # Confidence: based on scenario coverage and alignment
        confidence = elasticity_coverage * 0.7 + 0.3  # floor at 30%

        # Decision logic
        if price_change_pct > self.buy_threshold_pct and confidence >= self.min_confidence:
            signal = Signal.BUY
            rationale = f"Scenario '{scenario_name}' shows {price_change_pct:.2f}% upside. Top drivers: {', '.join(top_drivers[:2])}."
        elif price_change_pct < self.sell_threshold_pct:
            signal = Signal.DELAY
            rationale = f"Scenario '{scenario_name}' shows {price_change_pct:.2f}% downside. Caution advised."
        else:
            signal = Signal.HOLD
            rationale = f"Neutral signal. Scenario price within range. Monitor {', '.join(top_drivers[:2])}."

        return SignalOutput(
            timestamp=timestamp,
            signal=signal,
            price_forecast=ml_forecast_price,
            confidence=confidence,
            rationale=rationale,
            scenario_name=scenario_name,
            top_drivers=top_drivers,
        )

    def signal_batch(
        self, scenario_results: pd.DataFrame, elasticity_coverage: float = 0.8
    ) -> list[SignalOutput]:
        """Generate signals for multiple scenarios (bulk operation)."""
        # This is a placeholder; in practice, you'd iterate scenario_results rows
        # For now, just demonstrate the API.
        signals = []
        for _, row in scenario_results.iterrows():
            signal = self.generate_signal(
                timestamp=str(row.get("timestamp", "2026-01-21")),
                ml_forecast_price=row.get("base_price", 0),
                scenario_price=row.get("scenario_price", 0),
                scenario_name=row.get("scenario", "baseline"),
                top_drivers=["variable_1", "variable_2"],  # TODO: extract from impacts
                elasticity_coverage=elasticity_coverage,
            )
            signals.append(signal)
        return signals
