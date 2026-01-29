"""Elasticity configuration and schema.

Elasticity defines % impact of a variable on price. E.g., if USD index rises 1%,
EURUSD might decline 0.8% (elasticity = -0.8).

Schema structure:
  - variables: list of all drivers (local + international)
  - elasticities: mapping of variable name → elasticity coefficient
  - hierarchy: optional parent/child relationships for aggregation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class VariableDef:
    """Definition of a single variable (local or international)."""
    name: str  # e.g. "usd_index", "oil_price_wti"
    category: str  # "local" or "international"
    description: str = ""
    default_value: Optional[float] = None
    unit: str = ""  # e.g. "%", "bps", "USD/bbl"


@dataclass
class ElasticityDef:
    """Elasticity coefficient for a single variable."""
    variable: str  # reference to VariableDef.name
    elasticity: float  # % impact per 1% change in variable (unitless ratio)
    lag_periods: int = 0  # how many periods to lag (for forward-looking impacts)


@dataclass
class HierarchyNode:
    """Parent/child structure for hierarchical elasticity aggregation."""
    name: str
    parent: Optional[str] = None
    children: list = field(default_factory=list)


@dataclass
class ElasticityConfig:
    """Complete elasticity configuration for an asset."""
    asset_id: str
    variables: list[VariableDef]
    elasticities: list[ElasticityDef]
    hierarchy: Optional[list[HierarchyNode]] = None
    metadata: Dict = field(default_factory=dict)

    def get_elasticity_dict(self) -> Dict[str, float]:
        """Return {variable_name: elasticity_coefficient}."""
        return {e.variable: e.elasticity for e in self.elasticities}

    def get_variable_by_name(self, name: str) -> Optional[VariableDef]:
        """Lookup a variable definition."""
        return next((v for v in self.variables if v.name == name), None)


def load_elasticity_config(path: Path) -> ElasticityConfig:
    """Load elasticity config from YAML file.

    Expected YAML structure:
      asset_id: "EURUSD"
      variables:
        - name: "usd_index"
          category: "international"
          description: "US Dollar Index"
          unit: "index points"
          default_value: 103.5
        - name: "oil_price_wti"
          category: "international"
          description: "WTI Crude Oil"
          unit: "USD/bbl"
      elasticities:
        - variable: "usd_index"
          elasticity: -0.8
          lag_periods: 0
        - variable: "oil_price_wti"
          elasticity: 0.3
          lag_periods: 1
      hierarchy:
        - name: "energy_complex"
          parent: null
          children: ["oil_price_wti"]
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    asset_id = data["asset_id"]
    variables = [
        VariableDef(
            name=v["name"],
            category=v["category"],
            description=v.get("description", ""),
            default_value=v.get("default_value"),
            unit=v.get("unit", ""),
        )
        for v in data.get("variables", [])
    ]
    elasticities = [
        ElasticityDef(
            variable=e["variable"],
            elasticity=e["elasticity"],
            lag_periods=e.get("lag_periods", 0),
        )
        for e in data.get("elasticities", [])
    ]
    hierarchy = (
        [
            HierarchyNode(
                name=h["name"],
                parent=h.get("parent"),
                children=h.get("children", []),
            )
            for h in data.get("hierarchy", [])
        ]
        if "hierarchy" in data
        else None
    )
    metadata = data.get("metadata", {})

    return ElasticityConfig(
        asset_id=asset_id,
        variables=variables,
        elasticities=elasticities,
        hierarchy=hierarchy,
        metadata=metadata,
    )
