import pandas as pd


class ValidationResult:
    def __init__(self, passed, warnings=None, errors=None):
        self.passed = passed
        self.warnings = warnings or []
        self.errors = errors or []


def validate_dataframe(df, commodity_key, expected_unit):
    warnings, errors = [], []

    if df is None or df.empty:
        errors.append(f"{commodity_key}: empty dataframe")
        return ValidationResult(False, warnings, errors)

    # Null check
    null_pct = float(df.isnull().mean().max())
    if null_pct > 0.1:
        warnings.append(f"{commodity_key}: {null_pct:.0%} nulls detected")

    # Duplicate dates
    if df.index.duplicated().any():
        errors.append(f"{commodity_key}: duplicate dates found")

    # Staleness (monthly data)
    try:
        days_since = int((pd.Timestamp.today() - pd.to_datetime(df.index.max())).days)
        if days_since > 60:
            warnings.append(f"{commodity_key}: last data point is {days_since} days old")
    except Exception:
        warnings.append(f"{commodity_key}: staleness check failed ({expected_unit})")

    passed = len(errors) == 0
    return ValidationResult(passed, warnings, errors)

