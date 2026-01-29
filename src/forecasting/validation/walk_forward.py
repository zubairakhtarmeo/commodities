from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.config import ValidationConfig


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """Expanding-window walk-forward split with purge + embargo.

    Definitions (on feature timestamps):
      - A sample at time t has labels at times t+h.
      - To prevent leakage, training labels must not extend into the test start.

    Safe rule enforced:
      - Keep only train samples where (t + max_horizon + purge_steps) < test_start.

    This is conservative and prevents the common overlap mistake where
    training uses labels that occur during the test period.
    """

    def __init__(self, cfg: ValidationConfig, *, max_horizon: int):
        self.cfg = cfg
        self.max_horizon = int(max_horizon)

    def split(self, times: pd.DatetimeIndex):
        n = len(times)
        init = self.cfg.initial_train_periods
        step = self.cfg.step_size
        test_len = self.cfg.test_periods

        if init <= 0 or test_len <= 0 or step <= 0:
            raise ValueError("initial_train_periods, test_periods, step_size must be > 0")

        if init + test_len > n:
            raise ValueError(
                f"Not enough history for walk-forward: n={n}, initial_train_periods={init}, test_periods={test_len}"
            )

        fold = 0
        yielded_any = False
        test_start_pos = init
        while test_start_pos + test_len <= n:
            test_end_pos = test_start_pos + test_len - 1

            test_start_time = times[test_start_pos]
            test_end_time = times[test_end_pos]

            # Embargo: stop training this many steps before test start.
            embargo_cut = test_start_pos - self.cfg.embargo_steps

            # Purge + horizon: stop training early enough that labels stay strictly before test start.
            # Condition: t_train + max_horizon + purge_steps < test_start
            # In index positions, approximate as: train_pos <= test_start_pos - (max_horizon + purge_steps) - 1
            safe_train_end_pos = test_start_pos - (self.max_horizon + self.cfg.purge_steps) - 1
            train_end_pos = min(embargo_cut - 1, safe_train_end_pos)

            if train_end_pos <= 0:
                break

            train_idx = np.arange(0, train_end_pos + 1)
            test_idx = np.arange(test_start_pos, test_end_pos + 1)

            yielded_any = True
            yield WalkForwardSplit(
                fold=fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=times[0],
                train_end=times[train_end_pos],
                test_start=test_start_time,
                test_end=test_end_time,
            )

            fold += 1
            test_start_pos += step

        if not yielded_any:
            raise ValueError(
                "Walk-forward produced zero folds. "
                "This typically means purge/embargo/max_horizon are too large for the available history. "
                "Reduce validation.purge_steps / embargo_steps, reduce max horizon, or provide more data."
            )
