"""Aggregation utilities for converting daily to monthly prices."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_daily_to_monthly(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    price_col: str = "price_rmb",
    method: Literal["mean", "last", "max", "min"] = "mean",
    freq: str = "ME",  # Month-end
) -> pd.DataFrame:
    """Aggregate daily commodity prices to monthly.
    
    Args:
        df: Input DataFrame with daily prices
        timestamp_col: Column name containing dates
        price_col: Column name containing prices
        method: Aggregation method
            - "mean": Monthly average (typical for spot prices)
            - "last": Last trading day of month
            - "max": Highest price in month
            - "min": Lowest price in month
        freq: Pandas frequency string ('ME' for month-end)
    
    Returns:
        DataFrame with monthly aggregated prices
        Columns: [timestamp, price_{original_currency}]
    
    Raises:
        ValueError: If required columns missing or invalid method
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in input DataFrame")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in input DataFrame")

    if method not in ["mean", "last", "max", "min"]:
        raise ValueError(f"Unknown aggregation method: {method}")

    logger.info(
        f"Aggregating {len(df)} daily records to monthly ({method}) from "
        f"{df[timestamp_col].min()} to {df[timestamp_col].max()}"
    )

    # Ensure timestamp is datetime
    df_agg = df.copy()
    df_agg[timestamp_col] = pd.to_datetime(df_agg[timestamp_col])

    # Set timestamp as index for grouping
    df_agg = df_agg.set_index(timestamp_col)

    # Aggregate by month
    if method == "mean":
        monthly = df_agg[price_col].resample(freq).mean()
    elif method == "last":
        monthly = df_agg[price_col].resample(freq).last()
    elif method == "max":
        monthly = df_agg[price_col].resample(freq).max()
    elif method == "min":
        monthly = df_agg[price_col].resample(freq).min()

    # Reset index to get timestamp column back
    result = monthly.reset_index()
    result.columns = [timestamp_col, price_col]

    # Remove any rows with NaN (months with no data)
    result = result.dropna()

    logger.info(
        f"Aggregated to {len(result)} monthly records "
        f"({result[timestamp_col].min()} to {result[timestamp_col].max()})"
    )

    return result


def align_monthly_dates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    align_to: Literal["month_start", "month_end"] = "month_end",
) -> pd.DataFrame:
    """Align timestamps to consistent month boundaries.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Column name containing dates
        align_to: "month_start" (1st) or "month_end" (last day)
    
    Returns:
        DataFrame with normalized timestamps
    """
    df_aligned = df.copy()
    df_aligned[timestamp_col] = pd.to_datetime(df_aligned[timestamp_col])

    if align_to == "month_start":
        df_aligned[timestamp_col] = df_aligned[timestamp_col].dt.to_period("M").dt.start_time
    elif align_to == "month_end":
        df_aligned[timestamp_col] = df_aligned[timestamp_col].dt.to_period("M").dt.end_time
    else:
        raise ValueError(f"Unknown alignment: {align_to}")

    return df_aligned
