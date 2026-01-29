from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Connector(ABC):
    """Loads a single time-indexed numeric series."""

    @abstractmethod
    def load_series(self) -> pd.Series:
        raise NotImplementedError
