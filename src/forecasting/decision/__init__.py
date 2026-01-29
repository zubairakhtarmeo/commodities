"""Decision and elasticity layer.

This module applies elasticity-based impact attribution and scenario simulation
to ML-forecasted prices. It operates purely on ML outputs (forecasts) without
retraining or modifying core models.

Use case:
- Take a price forecast from the ML layer.
- Apply elasticity impacts from local/international variables.
- Attribute price changes to underlying drivers.
- Generate what-if scenarios.
- Produce buy/hold/delay signals based on scenario results.

Design:
- Rule-based, config-driven.
- No model training.
- Clean separation from ML training pipeline.
- Economic guardrails ensure outputs are management-safe.
"""

from .config import ElasticityConfig, load_elasticity_config  # noqa
from .elasticity import ElasticityModel  # noqa
from .guardrails import GuardrailConfig, GuardrailEngine, GuardrailResult  # noqa
from .scenario import ScenarioSimulator  # noqa
from .signal import SignalGenerator  # noqa

__all__ = [
    "ElasticityConfig",
    "load_elasticity_config",
    "ElasticityModel",
    "ScenarioSimulator",
    "SignalGenerator",
    "GuardrailConfig",
    "GuardrailEngine",
    "GuardrailResult",
]
