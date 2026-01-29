"""Quick validation script for existing commodity data."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from forecasting.ingestion import validate_price_series

def main():
    commodities = [
        ('Cotton USD', 'data/raw/cotton/cotton_usd_monthly.csv', 'usd'),
        ('Crude Oil Brent USD', 'data/raw/energy/crude_oil_brent_usd_monthly_clean.csv', 'usd'),
        ('Natural Gas USD', 'data/raw/energy/natural_gas_usd_monthly_clean.csv', 'usd'),
        ('Polyester RMB', 'data/raw/polyester/polyester_futures_monthly.csv', 'rmb'),
    ]

    print("=" * 100)
    print("COMMODITY DATA VALIDATION REPORT")
    print("=" * 100)

    all_valid = True
    for name, path, currency in commodities:
        try:
            df = pd.read_csv(path)
            result = validate_price_series(df, commodity=name, expected_currency=currency)
            status = "✓" if result.is_valid else "✗"
            
            date_start = result.date_range[0] if result.date_range else "N/A"
            date_end = result.date_range[1] if result.date_range else "N/A"
            
            print(f"{status} {name:25} | {result.num_records:3d} records | {date_start} to {date_end}")
            if not result.is_valid:
                all_valid = False
                for err in result.errors:
                    print(f"  ✗ {err}")
            if result.warnings:
                for warn in result.warnings:
                    print(f"  ⚠ {warn}")
        except Exception as e:
            print(f"✗ {name:25} | ERROR: {e}")
            all_valid = False

    print("=" * 100)
    if all_valid:
        print("✓ All commodities validated successfully")
    else:
        print("✗ Some commodities have validation errors")
    print("=" * 100)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
