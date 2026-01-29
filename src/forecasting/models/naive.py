from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LastValueNaive:
    """Baseline: predict future equals current value.

    This is intentionally simple and interpretable.

    Required input column:
      - 'base_target' (the observed target at feature timestamp t)

    Leakage note:
      - Using base_target at t to predict t+h is safe if the forecast is made at time t.
    """

    horizon_count: int

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "base_target" not in X.columns:
            raise ValueError("LastValueNaive requires feature column 'base_target'")
        v = X["base_target"].to_numpy(dtype=float)
        pred = np.repeat(v.reshape(-1, 1), self.horizon_count, axis=1)
        return pred
