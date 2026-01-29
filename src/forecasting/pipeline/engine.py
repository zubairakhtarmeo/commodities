from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecasting.artifacts import ArtifactStore
from forecasting.config import ProjectConfig
from forecasting.data import build_series_registry
from forecasting.dataset import build_supervised_dataset
from forecasting.features import FeatureBuilder
from forecasting.models import build_model
from forecasting.time import align_asset_frame
from forecasting.training import Trainer
from forecasting.validation import WalkForwardSplitter


@dataclass(frozen=True)
class ForecastRow:
    asset_id: str
    asof: pd.Timestamp
    horizon_steps: int
    target_time: pd.Timestamp
    y_pred: float


class ForecastingEngine:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.artifacts = ArtifactStore(Path(cfg.output.artifacts_dir))

    def _build_asset_dataset(self, asset_id: str):
        asset = next(a for a in self.cfg.assets if a.asset_id == asset_id)
        registry = build_series_registry(asset)
        base = align_asset_frame(registry, rule=self.cfg.frequency.rule)

        fb = FeatureBuilder.from_configs(self.cfg.features.packs)
        X = fb.build(base)

        ds = build_supervised_dataset(
            features=X,
            target=base["target"],
            cfg=self.cfg.dataset,
        )
        return base, ds

    def train(self) -> dict[str, list[str]]:
        """Train + evaluate each enabled model for each asset.

        Returns: dict asset_id -> list of trained model names
        """

        trained: dict[str, list[str]] = {}

        for asset in self.cfg.assets:
            base, ds = self._build_asset_dataset(asset.asset_id)

            splitter = WalkForwardSplitter(
                self.cfg.validation, max_horizon=max(self.cfg.dataset.horizon_steps)
            )
            trainer = Trainer(
                splitter=splitter, explainability_enabled=self.cfg.explainability.enabled
            )

            trained[asset.asset_id] = []
            for spec in self.cfg.model_ladder:
                if not spec.enabled:
                    continue

                built = build_model(spec, horizon_count=len(ds.horizons))
                result, est, importances = trainer.fit_walk_forward(
                    estimator=built.estimator,
                    X=ds.X,
                    y=ds.y,
                    times=ds.times,
                    model_name=built.name,
                )

                metrics_df = result.metrics_frame()
                self.artifacts.save_metrics(
                    asset_id=asset.asset_id, model_name=built.name, df=metrics_df
                )

                if est is not None:
                    self.artifacts.save_model(
                        asset_id=asset.asset_id, model_name=built.name, estimator=est
                    )

                if importances is not None:
                    self.artifacts.save_importances(
                        asset_id=asset.asset_id, model_name=built.name, df=importances
                    )

                trained[asset.asset_id].append(built.name)

        return trained

    def predict(self, *, asset_id: str, model_name: str, asof: pd.Timestamp) -> pd.DataFrame:
        """Predict for a single as-of timestamp.

        Safety:
          - Features are computed using data up to and including `asof` only.
          - If `asof` is not on the canonical index, we use the last available <= asof.
        """

        asset = next(a for a in self.cfg.assets if a.asset_id == asset_id)
        registry = build_series_registry(asset)
        base = align_asset_frame(registry, rule=self.cfg.frequency.rule)

        if asof not in base.index:
            base = base.loc[base.index <= asof]
            if len(base.index) == 0:
                raise ValueError("No data available on/before asof")
            asof = base.index[-1]

        fb = FeatureBuilder.from_configs(self.cfg.features.packs)
        X = fb.build(base)
        X_row = X.loc[[asof]]

        est = self.artifacts.load_model(asset_id=asset_id, model_name=model_name)
        pred = est.predict(X_row)

        horizons = tuple(int(h) for h in self.cfg.dataset.horizon_steps)
        rows: list[ForecastRow] = []
        for j, h in enumerate(horizons):
            target_time = base.index[base.index.get_loc(asof) + h] if (base.index.get_loc(asof) + h) < len(base.index) else asof + pd.tseries.frequencies.to_offset(self.cfg.frequency.rule) * h
            rows.append(
                ForecastRow(
                    asset_id=asset_id,
                    asof=asof,
                    horizon_steps=h,
                    target_time=pd.Timestamp(target_time),
                    y_pred=float(pred[0][j]),
                )
            )

        return pd.DataFrame([r.__dict__ for r in rows])
