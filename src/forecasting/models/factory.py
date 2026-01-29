from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forecasting.config import ModelSpec

from .naive import LastValueNaive


@dataclass(frozen=True)
class BuiltModel:
    name: str
    estimator: object


def build_model(spec: ModelSpec, *, horizon_count: int) -> BuiltModel:
    if spec.type == "naive":
        return BuiltModel(name=spec.name, estimator=LastValueNaive(horizon_count=horizon_count))

    if spec.type == "ridge":
        base = Ridge(**spec.params)
        est = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", MultiOutputRegressor(base)),
            ]
        )
        return BuiltModel(name=spec.name, estimator=est)

    if spec.type == "hist_gbrt":
        base = HistGradientBoostingRegressor(**spec.params)
        est = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", MultiOutputRegressor(base)),
            ]
        )
        return BuiltModel(name=spec.name, estimator=est)

    raise ValueError(f"Unsupported model type: {spec.type}")
