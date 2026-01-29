from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _multioutput_mae_scorer(estimator, X, y) -> float:
    # sklearn scoring: higher is better, so return negative MAE
    pred = estimator.predict(X)
    pred = np.asarray(pred)
    y_true = np.asarray(y)
    mae = np.mean(np.abs(pred - y_true))
    return -float(mae)


def permutation_importance_time_safe(
    *,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance using *train-only* data.

    Safety:
      - Uses only the training fold, avoiding any dependence on future/test labels.

    Returns:
      DataFrame with columns: feature, importance_mean, importance_std
    """

    result = permutation_importance(
        estimator,
        X_train,
        y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=_multioutput_mae_scorer,
    )

    imp = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return imp
