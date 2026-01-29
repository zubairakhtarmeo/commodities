from __future__ import annotations

import numpy as np
import pandas as pd


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    """Compute horizon-correct metrics per horizon column."""

    out: dict[str, float] = {}
    for col in y_true.columns:
        yt = y_true[col].to_numpy(dtype=float)
        yp = y_pred[col].to_numpy(dtype=float)

        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        if yt.size == 0:
            out[f"{col}/mae"] = float("nan")
            out[f"{col}/rmse"] = float("nan")
            out[f"{col}/smape"] = float("nan")
            continue

        out[f"{col}/mae"] = float(np.mean(np.abs(yp - yt)))
        out[f"{col}/rmse"] = float(np.sqrt(np.mean((yp - yt) ** 2)))
        out[f"{col}/smape"] = _smape(yt, yp)

    return out
