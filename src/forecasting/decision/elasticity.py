"""Elasticity model for impact attribution.

Core logic:
- Given a base price forecast P₀ and variable changes Δx₁, Δx₂, ..., Δxₙ,
- Compute impact contribution: ΔP_i = E_i * Δx_i * P₀ / 100
  (where E_i is elasticity, assuming % changes).
- Aggregate to get total impact and residual.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .config import ElasticityConfig


@dataclass
class ImpactRow:
    """One row of impact attribution for a forecast horizon."""
    timestamp: pd.Timestamp
    horizon: int  # forecast horizon (steps ahead)
    base_price: float  # ML forecast price at this horizon
    variable: str  # name of the driver variable
    variable_change_pct: float  # % change in the variable
    elasticity: float  # elasticity coefficient
    impact_pct: float  # % impact on price
    impact_abs: float  # absolute impact on price
    cumulative_impact_pct: float = 0.0  # cumsum of impacts
    residual_pct: float = 0.0  # unexplained variance


@dataclass
class ImpactAttribution:
    """Complete impact attribution for one forecast."""
    asset_id: str
    timestamp: pd.Timestamp
    horizon: int
    base_price: float  # ML forecast
    total_impact_pct: float  # sum of all variable impacts
    total_impact_abs: float  # absolute price impact
    unexplained_pct: float  # residual
    rows: list[ImpactRow] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Export as DataFrame."""
        data = [
            {
                "timestamp": row.timestamp,
                "horizon": row.horizon,
                "base_price": row.base_price,
                "variable": row.variable,
                "variable_change_pct": row.variable_change_pct,
                "elasticity": row.elasticity,
                "impact_pct": row.impact_pct,
                "impact_abs": row.impact_abs,
            }
            for row in self.rows
        ]
        return pd.DataFrame(data)


class ElasticityModel:
    """Applies elasticity-based impact attribution to ML forecasts."""

    def __init__(self, config: ElasticityConfig):
        self.config = config
        self.elasticity_dict = config.get_elasticity_dict()

    def attribute_impacts(
        self,
        timestamp: pd.Timestamp,
        horizon: int,
        base_price_forecast: float,
        variable_changes: Dict[str, float],
        residual_pct: Optional[float] = None,
    ) -> ImpactAttribution:
        """Compute impact attribution.

        Args:
            timestamp: forecast date
            horizon: forecast horizon (steps ahead)
            base_price_forecast: ML price forecast at this horizon
            variable_changes: {variable_name: pct_change} (e.g., {"usd_index": 0.5})
            residual_pct: optional residual variance; if None, computed as remainder.

        Returns:
            ImpactAttribution with detailed breakdown.
        """
        rows = []
        total_impact_pct = 0.0
        total_impact_abs = 0.0

        for var_name, change_pct in variable_changes.items():
            if var_name not in self.elasticity_dict:
                continue  # Skip unknown variables

            elasticity = self.elasticity_dict[var_name]
            impact_pct = elasticity * change_pct
            impact_abs = (impact_pct / 100.0) * base_price_forecast

            total_impact_pct += impact_pct
            total_impact_abs += impact_abs

            rows.append(
                ImpactRow(
                    timestamp=timestamp,
                    horizon=horizon,
                    base_price=base_price_forecast,
                    variable=var_name,
                    variable_change_pct=change_pct,
                    elasticity=elasticity,
                    impact_pct=impact_pct,
                    impact_abs=impact_abs,
                )
            )

        # Residual: unexplained variance (typically from ML forecast adjustment or noise)
        if residual_pct is None:
            residual_pct = 0.0  # Assume full attribution by default

        return ImpactAttribution(
            asset_id=self.config.asset_id,
            timestamp=timestamp,
            horizon=horizon,
            base_price=base_price_forecast,
            total_impact_pct=total_impact_pct,
            total_impact_abs=total_impact_abs,
            unexplained_pct=residual_pct,
            rows=rows,
        )

    def rank_impacts(self, attribution: ImpactAttribution, top_n: int = 5) -> pd.DataFrame:
        """Return top N impacts by absolute value."""
        df = attribution.to_dataframe()
        df["abs_impact"] = df["impact_abs"].abs()
        return df.nlargest(top_n, "abs_impact")[
            ["variable", "variable_change_pct", "elasticity", "impact_pct", "impact_abs"]
        ]
