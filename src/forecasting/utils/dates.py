from __future__ import annotations

from datetime import datetime

import pandas as pd


def parse_date(value: str) -> pd.Timestamp:
    # Accepts YYYY-MM-DD and similar; returns normalized Timestamp.
    return pd.Timestamp(datetime.fromisoformat(value))
