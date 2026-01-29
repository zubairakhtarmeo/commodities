from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CSVConnector:
    path: str
    time_col: str = "timestamp"
    value_col: str = "value"
    timezone: str | None = None

    def load_series(self) -> pd.Series:
        df = pd.read_csv(self.path)
        if self.time_col not in df.columns or self.value_col not in df.columns:
            raise ValueError(
                f"CSV missing required columns: {self.time_col!r}, {self.value_col!r}"
            )

        ts = pd.to_datetime(df[self.time_col], utc=False)
        if self.timezone is not None:
            # interpret as local time then convert to naive local timestamps
            ts = pd.to_datetime(df[self.time_col]).dt.tz_localize(self.timezone).dt.tz_convert(
                self.timezone
            )

        s = pd.Series(df[self.value_col].astype("float64").to_numpy(), index=ts)
        s = s.sort_index()

        # Safety: forbid duplicate timestamps unless user aggregates upstream
        if s.index.has_duplicates:
            dupes = s.index[s.index.duplicated()].unique()
            raise ValueError(
                f"Duplicate timestamps in {Path(self.path).name}: e.g. {dupes[:3].tolist()}"
            )

        s.name = Path(self.path).stem
        return s
