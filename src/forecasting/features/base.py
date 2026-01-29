from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeaturePack(ABC):
    name: str

    @abstractmethod
    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe of features indexed like `base`.

        Safety rule: every feature at time t may depend only on values at times <= t.
        """

        raise NotImplementedError
