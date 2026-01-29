from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class CSVConnectorConfig(BaseModel):
    type: Literal["csv"]
    path: str
    time_col: str = "timestamp"
    value_col: str = "value"
    timezone: str | None = None


ConnectorConfig = CSVConnectorConfig


class SeriesSpec(BaseModel):
    role: str
    connector: ConnectorConfig

    # If a series is published with delay, shift it *backward* in availability
    # so that at time t we only allow values observed by then.
    # Example: availability_lag_steps=1 means value stamped at t is only usable at t+1,
    # so we shift the series by +1 step (downstream) to avoid leakage.
    availability_lag_steps: int = 0


class AssetSpec(BaseModel):
    asset_id: str
    target: SeriesSpec
    exogenous: list[SeriesSpec] = Field(default_factory=list)


class FrequencySpec(BaseModel):
    rule: str = "M"  # pandas offset alias
    timezone: str | None = None


class FeaturePackConfig(BaseModel):
    name: str
    enabled: bool = True
    # If true, the pack is allowed to be skipped when required inputs are missing.
    # This supports configs that enable packs which depend on optional exogenous series.
    optional: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class FeaturesConfig(BaseModel):
    packs: list[FeaturePackConfig] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    horizon_steps: list[int] = Field(default_factory=lambda: [1])
    lookback_steps: int = 12
    drop_na_target: bool = True


class ValidationConfig(BaseModel):
    strategy: Literal["walk_forward"] = "walk_forward"
    initial_train_periods: int = 60
    step_size: int = 1
    test_periods: int = 12
    embargo_steps: int = 0
    purge_steps: int = 0


class ModelSpec(BaseModel):
    name: str
    type: Literal["naive", "ridge", "hist_gbrt"]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class ExplainabilityConfig(BaseModel):
    enabled: bool = True
    method: Literal["permutation"] = "permutation"


class OutputConfig(BaseModel):
    artifacts_dir: str = "artifacts"


class ProjectMeta(BaseModel):
    name: str = "default"


class ProjectConfig(BaseModel):
    project: ProjectMeta = ProjectMeta()
    frequency: FrequencySpec = FrequencySpec()
    assets: list[AssetSpec]
    features: FeaturesConfig = FeaturesConfig()
    dataset: DatasetConfig = DatasetConfig()
    validation: ValidationConfig = ValidationConfig()
    model_ladder: list[ModelSpec] = Field(default_factory=list)
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)
