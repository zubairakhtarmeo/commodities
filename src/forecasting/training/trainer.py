from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.evaluation import compute_metrics
from forecasting.explainability import permutation_importance_time_safe
from forecasting.validation import WalkForwardSplitter


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainResult:
    model_name: str
    fold_results: list[FoldResult]

    def metrics_frame(self) -> pd.DataFrame:
        rows = []
        for fr in self.fold_results:
            row = {
                "fold": fr.fold,
                "train_end": fr.train_end,
                "test_start": fr.test_start,
                "test_end": fr.test_end,
                **fr.metrics,
            }
            rows.append(row)
        return pd.DataFrame(rows)


class Trainer:
    def __init__(self, *, splitter: WalkForwardSplitter, explainability_enabled: bool = True):
        self.splitter = splitter
        self.explainability_enabled = explainability_enabled

    def fit_walk_forward(
        self,
        *,
        estimator,
        X: pd.DataFrame,
        y: pd.DataFrame,
        times: pd.DatetimeIndex,
        model_name: str,
    ) -> tuple[TrainResult, object, pd.DataFrame | None]:
        fold_results: list[FoldResult] = []
        last_estimator = None
        last_importances = None

        for split in self.splitter.split(times):
            X_train = X.iloc[split.train_idx]
            y_train = y.iloc[split.train_idx]
            X_test = X.iloc[split.test_idx]
            y_test = y.iloc[split.test_idx]

            est = estimator
            est.fit(X_train, y_train)
            pred = est.predict(X_test)
            y_pred = pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)

            metrics = compute_metrics(y_test, y_pred)
            fold_results.append(
                FoldResult(
                    fold=split.fold,
                    train_end=split.train_end,
                    test_start=split.test_start,
                    test_end=split.test_end,
                    metrics=metrics,
                )
            )

            # Compute explainability from train-only data (safe).
            if self.explainability_enabled and split.fold == 0:
                try:
                    last_importances = permutation_importance_time_safe(
                        estimator=est,
                        X_train=X_train,
                        y_train=y_train,
                    ).reset_index(drop=True)
                except Exception:
                    last_importances = None

            last_estimator = est

        return TrainResult(model_name=model_name, fold_results=fold_results), last_estimator, last_importances
