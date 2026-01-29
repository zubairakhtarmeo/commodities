"""Scenario simulation engine.

Given base case variable values and elasticities, compute price outcomes
under different scenarios (bull, bear, sideways, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .config import ElasticityConfig


@dataclass
class ScenarioInput:
    """One scenario definition."""
    name: str  # e.g., "Bull", "Bear", "Baseline"
    description: str = ""
    variable_values: Dict[str, float] = field(default_factory=dict)  # {var_name: level}
    variable_changes_pct: Dict[str, float] = field(
        default_factory=dict
    )  # {var_name: pct_change}


@dataclass
class ScenarioResult:
    """Outcome of a scenario."""
    scenario_name: str
    base_price: float  # ML forecast
    impacts: Dict[str, float]  # {variable: impact_pct}
    total_impact_pct: float  # sum of impacts
    scenario_price: float  # base_price * (1 + total_impact_pct/100)
    price_change_abs: float  # scenario_price - base_price
    price_change_pct: float  # (scenario_price - base_price) / base_price * 100


class ScenarioSimulator:
    """Run what-if scenarios on ML forecasts using elasticity."""

    def __init__(self, config: ElasticityConfig):
        self.config = config
        self.elasticity_dict = config.get_elasticity_dict()

    def simulate_scenario(
        self, base_price: float, scenario: ScenarioInput
    ) -> ScenarioResult:
        """Simulate one scenario.

        Args:
            base_price: ML forecast price (baseline scenario)
            scenario: ScenarioInput with variable changes

        Returns:
            ScenarioResult with price outcome.
        """
        impacts = {}
        total_impact_pct = 0.0

        # Use variable_changes_pct if provided, else estimate from levels
        changes = scenario.variable_changes_pct or {}

        for var_name, change_pct in changes.items():
            if var_name not in self.elasticity_dict:
                continue

            elasticity = self.elasticity_dict[var_name]
            impact_pct = elasticity * change_pct
            impacts[var_name] = impact_pct
            total_impact_pct += impact_pct

        scenario_price = base_price * (1.0 + total_impact_pct / 100.0)
        price_change_abs = scenario_price - base_price
        price_change_pct = (price_change_abs / base_price * 100.0) if base_price != 0 else 0.0

        return ScenarioResult(
            scenario_name=scenario.name,
            base_price=base_price,
            impacts=impacts,
            total_impact_pct=total_impact_pct,
            scenario_price=scenario_price,
            price_change_abs=price_change_abs,
            price_change_pct=price_change_pct,
        )

    def run_scenarios(
        self, base_price: float, scenarios: list[ScenarioInput]
    ) -> pd.DataFrame:
        """Run multiple scenarios and return comparison table."""
        results = [self.simulate_scenario(base_price, s) for s in scenarios]

        data = [
            {
                "scenario": r.scenario_name,
                "base_price": r.base_price,
                "scenario_price": r.scenario_price,
                "price_change_abs": r.price_change_abs,
                "price_change_pct": r.price_change_pct,
                "total_impact_pct": r.total_impact_pct,
            }
            for r in results
        ]
        return pd.DataFrame(data)

    def sensitivity_analysis(
        self,
        base_price: float,
        variable: str,
        pct_changes: list[float],
    ) -> pd.DataFrame:
        """1-way sensitivity: vary one variable, see impact on price.

        Args:
            base_price: ML forecast
            variable: variable name to vary
            pct_changes: list of % changes to test (e.g., [-2, -1, 0, 1, 2])

        Returns:
            DataFrame with sensitivity results.
        """
        if variable not in self.elasticity_dict:
            raise ValueError(f"Variable {variable} not found in elasticity dict")

        results = []
        elasticity = self.elasticity_dict[variable]

        for change in pct_changes:
            impact_pct = elasticity * change
            outcome_price = base_price * (1.0 + impact_pct / 100.0)
            results.append(
                {
                    "variable": variable,
                    "variable_change_pct": change,
                    "elasticity": elasticity,
                    "impact_pct": impact_pct,
                    "outcome_price": outcome_price,
                    "price_change_abs": outcome_price - base_price,
                }
            )

        return pd.DataFrame(results)
