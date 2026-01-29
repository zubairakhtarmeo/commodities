from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index (PSI).

    Use case:
      - compare training feature distribution (expected) vs current distribution (actual)

    Safety:
      - requires no labels; safe for production monitoring.
    """

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, quantiles)
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = np.clip(e_counts / max(1, expected.size), eps, 1.0)
    a_perc = np.clip(a_counts / max(1, actual.size), eps, 1.0)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
