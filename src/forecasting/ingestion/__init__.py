"""Data ingestion and aggregation utilities.

This module handles:
- Fetching commodity price data from various sources
- Converting daily/raw data to monthly aggregates
- Validating data quality and continuity
- Providing connector-based extensible design
"""

from .sunsirs_connector import SunSirsConnector  # noqa
from .aggregation import aggregate_daily_to_monthly  # noqa
from .validation import validate_price_series, DataValidationResult  # noqa
from .oracle_purchases import (  # noqa
    ingest_oracle_purchases_dir,
    clean_oracle_purchases_sheet,
    build_monthly_aggregates,
    MaterialClassifier,
)

__all__ = [
    "SunSirsConnector",
    "aggregate_daily_to_monthly",
    "validate_price_series",
    "DataValidationResult",
    "ingest_oracle_purchases_dir",
    "clean_oracle_purchases_sheet",
    "build_monthly_aggregates",
    "MaterialClassifier",
]
