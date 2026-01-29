"""Data validation and quality checks for commodity price series."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Results of validating a price series."""

    is_valid: bool
    num_records: int
    date_range: tuple[str, str] | None = None
    currency: str = ""
    commodity: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [
            f"{status} | {self.commodity} ({self.currency})",
            f"  Records: {self.num_records}",
        ]
        if self.date_range:
            lines.append(f"  Date range: {self.date_range[0]} to {self.date_range[1]}")
        if self.warnings:
            lines.append(f"  ⚠ Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.errors:
            lines.append(f"  ✗ Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


def validate_price_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str | None = None,
    commodity: str = "Unknown",
    expected_currency: str | None = None,
    min_records: int = 12,
) -> DataValidationResult:
    """Validate a commodity price series for quality and completeness.
    
    Checks:
    - Non-empty DataFrame
    - Required columns present
    - No missing values
    - Timestamps are sorted
    - No duplicate timestamps
    - Sufficient historical data
    - Prices within plausible range
    - Detectable currency from column names
    
    Args:
        df: Input DataFrame
        timestamp_col: Column name with dates
        value_col: Column name with prices. If None, auto-detect (*_usd, *_rmb, etc.)
        commodity: Commodity name for reporting
        expected_currency: Expected currency (usd, rmb, pkr). If None, auto-detect.
        min_records: Minimum required records to consider valid
    
    Returns:
        DataValidationResult with validation status and details
    """
    result = DataValidationResult(
        is_valid=True,
        num_records=len(df),
        commodity=commodity,
    )

    # Check 1: DataFrame not empty
    if df.empty:
        result.errors.append("DataFrame is empty")
        result.is_valid = False
        result.summary = "Empty dataset"
        return result

    # Check 2: Required columns exist
    if timestamp_col not in df.columns:
        result.errors.append(f"Missing required column: {timestamp_col}")
        result.is_valid = False

    # Check 3: Auto-detect or validate value column
    if value_col is None:
        value_col = _detect_price_column(df)
        if not value_col:
            result.errors.append(
                f"Could not auto-detect price column. Available: {df.columns.tolist()}"
            )
            result.is_valid = False
            result.summary = "Could not identify price column"
            return result

    if value_col not in df.columns:
        result.errors.append(f"Value column not found: {value_col}")
        result.is_valid = False
        result.summary = f"Missing column: {value_col}"
        return result

    # Extract currency from column name if not provided
    if expected_currency is None:
        expected_currency = _detect_currency_from_column(value_col)

    if expected_currency:
        result.currency = expected_currency.upper()

    # Check 4: Extract and validate timestamp column
    try:
        timestamps = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        result.errors.append(f"Could not parse timestamp column: {e}")
        result.is_valid = False
        result.summary = "Invalid date format"
        return result

    # Check 5: No missing timestamps
    if timestamps.isna().any():
        num_missing = timestamps.isna().sum()
        result.warnings.append(f"{num_missing} missing timestamps")

    # Check 6: Timestamps are sorted
    if not timestamps.is_monotonic_increasing:
        result.warnings.append("Timestamps are not sorted chronologically")

    # Check 7: No duplicate timestamps
    if timestamps.duplicated().any():
        num_dupes = timestamps.duplicated().sum()
        result.errors.append(f"{num_dupes} duplicate timestamps found")
        result.is_valid = False

    # Check 8: Extract price values
    try:
        prices = pd.to_numeric(df[value_col], errors="coerce")
    except Exception as e:
        result.errors.append(f"Could not parse price column: {e}")
        result.is_valid = False
        result.summary = "Invalid price format"
        return result

    # Check 9: No missing prices
    if prices.isna().any():
        num_missing = prices.isna().sum()
        pct_missing = 100 * num_missing / len(prices)
        result.errors.append(f"{num_missing} ({pct_missing:.1f}%) missing prices")
        result.is_valid = False

    # Check 10: Prices in plausible range (positive, not too extreme)
    if (prices > 0).all():
        pass  # Good
    else:
        num_non_positive = (prices <= 0).sum()
        result.warnings.append(f"{num_non_positive} non-positive prices")

    # Check 11: Date range
    if not timestamps.empty:
        date_min = timestamps.min().strftime("%Y-%m-%d")
        date_max = timestamps.max().strftime("%Y-%m-%d")
        result.date_range = (date_min, date_max)

    # Check 12: Minimum records
    if result.num_records < min_records:
        result.warnings.append(
            f"Only {result.num_records} records; recommended minimum: {min_records}"
        )

    # Summary
    if result.errors:
        result.summary = f"{len(result.errors)} validation error(s)"
    elif result.warnings:
        result.summary = f"Valid with {len(result.warnings)} warning(s)"
    else:
        result.summary = "All checks passed"

    return result


def _detect_price_column(df: pd.DataFrame) -> str | None:
    """Auto-detect price column by name pattern."""
    patterns = ["price", "value", "close", "spot"]
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in patterns):
            return col
    return None


def _detect_currency_from_column(col_name: str) -> str | None:
    """Extract currency code from column name (e.g., price_usd → usd)."""
    col_lower = col_name.lower()
    for currency in ["usd", "rmb", "pkr", "cny", "inr", "eur", "gbp", "jpy"]:
        if currency in col_lower:
            return currency
    return None


def validate_multiple_series(series_dict: dict[str, pd.DataFrame]) -> dict[str, DataValidationResult]:
    """Validate multiple commodity series at once.
    
    Args:
        series_dict: Dict of {commodity_name: DataFrame}
    
    Returns:
        Dict of {commodity_name: DataValidationResult}
    """
    results = {}
    for commodity, df in series_dict.items():
        results[commodity] = validate_price_series(df, commodity=commodity)
    return results


def print_validation_report(results: dict[str, DataValidationResult]) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 90)
    print("DATA VALIDATION REPORT")
    print("=" * 90)

    all_valid = all(r.is_valid for r in results.values())

    for commodity, result in results.items():
        print(f"\n{result}")

    print("\n" + "=" * 90)
    if all_valid:
        print("✓ All commodities validated successfully")
    else:
        num_invalid = sum(1 for r in results.values() if not r.is_valid)
        print(f"✗ {num_invalid}/{len(results)} commodities have validation errors")
    print("=" * 90 + "\n")
