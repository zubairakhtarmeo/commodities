"""Multi-commodity data ingestion and validation pipeline.

This script:
1. Validates existing commodity data (cotton, energy, polyester)
2. (Optional) Fetches and aggregates viscose data from SunSirs
3. Prepares all data for ML training
4. Reports any data quality issues
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from forecasting.ingestion import (
    SunSirsConnector,
    aggregate_daily_to_monthly,
    validate_price_series,
    DataValidationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class CommodityDataPipeline:
    """Multi-commodity data ingestion and validation pipeline."""

    def __init__(self, data_root: str = "data/raw"):
        """Initialize pipeline.
        
        Args:
            data_root: Root directory where commodity data is stored
        """
        self.data_root = Path(data_root)
        self.results = {}

    def validate_existing_commodity(
        self,
        commodity: str,
        filepath: str,
        expected_currency: Optional[str] = None,
    ) -> DataValidationResult:
        """Validate an existing commodity CSV file.
        
        Args:
            commodity: Commodity name (e.g., "cotton")
            filepath: Relative path to CSV (e.g., "cotton/cotton_usd_monthly.csv")
            expected_currency: Expected currency code
        
        Returns:
            DataValidationResult with validation details
        """
        full_path = self.data_root / filepath

        if not full_path.exists():
            logger.error(f"{commodity}: File not found at {full_path}")
            result = DataValidationResult(
                is_valid=False,
                num_records=0,
                commodity=commodity,
                errors=[f"File not found: {full_path}"],
                summary="File missing",
            )
            self.results[commodity] = result
            return result

        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            logger.error(f"{commodity}: Failed to read CSV: {e}")
            result = DataValidationResult(
                is_valid=False,
                num_records=0,
                commodity=commodity,
                errors=[f"Failed to read CSV: {e}"],
                summary="CSV read error",
            )
            self.results[commodity] = result
            return result

        result = validate_price_series(
            df,
            commodity=commodity,
            expected_currency=expected_currency,
        )
        self.results[commodity] = result
        logger.info(f"{commodity}: {result.summary}")
        return result

    def ingest_viscose_from_sunsirs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_daily: str = "viscose/viscose_daily.csv",
        output_monthly: str = "viscose/viscose_monthly.csv",
        aggregation_method: str = "mean",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch viscose data from SunSirs and aggregate to monthly.
        
        Args:
            start_date: ISO date (YYYY-MM-DD) or None for 1 year ago
            end_date: ISO date (YYYY-MM-DD) or None for today
            output_daily: Where to save raw daily data
            output_monthly: Where to save aggregated monthly data
            aggregation_method: "mean", "last", "max", or "min"
        
        Returns:
            Tuple of (daily_df, monthly_df)
        
        Raises:
            ConnectionError: If unable to reach SunSirs
            ValueError: If no data returned
        """
        logger.info("=" * 90)
        logger.info("VISCOSE DATA INGESTION (SunSirs)")
        logger.info("=" * 90)

        # Initialize connector
        connector = SunSirsConnector()

        # Fetch daily prices
        try:
            logger.info("Fetching daily viscose prices from SunSirs...")
            daily_df = connector.fetch_daily_prices(
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Failed to fetch from SunSirs: {e}")
            raise

        # Create output directories
        daily_path = self.data_root / output_daily
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        monthly_path = self.data_root / output_monthly
        monthly_path.parent.mkdir(parents=True, exist_ok=True)

        # Save daily data (no synthetic data; no gap filling)
        connector.save_daily_csv(daily_df, str(daily_path))
        logger.info(f"Saved daily data to {daily_path}")

        # Aggregate to monthly
        logger.info(f"Aggregating daily to monthly ({aggregation_method})...")
        monthly_df = aggregate_daily_to_monthly(
            daily_df,
            timestamp_col="timestamp",
            price_col="price_rmb",
            method=aggregation_method,
        )
        monthly_df.to_csv(monthly_path, index=False)
        logger.info(f"Saved monthly data to {monthly_path}")

        # Validate aggregated data
        result = validate_price_series(
            monthly_df,
            commodity="Viscose",
            expected_currency="rmb",
        )
        self.results["viscose"] = result
        logger.info(f"Viscose validation: {result.summary}")

        return daily_df, monthly_df

    def run_full_pipeline(
        self,
        validate_cotton: bool = True,
        validate_energy: bool = True,
        validate_polyester: bool = True,
        ingest_viscose: bool = True,
        viscose_start_date: Optional[str] = None,
        viscose_end_date: Optional[str] = None,
    ) -> dict[str, DataValidationResult]:
        """Run complete data pipeline.
        
        Args:
            validate_cotton: Validate cotton data
            validate_energy: Validate energy data (crude oil + natural gas)
            validate_polyester: Validate polyester data
            ingest_viscose: Fetch and aggregate viscose from SunSirs
            viscose_start_date: Viscose start date (if fetching)
            viscose_end_date: Viscose end date (if fetching)
        
        Returns:
            Dictionary of validation results for all commodities
        """
        logger.info("\n" + "=" * 90)
        logger.info("MULTI-COMMODITY DATA PIPELINE")
        logger.info("=" * 90)

        # Validate Cotton
        if validate_cotton:
            logger.info("\nValidating COTTON...")
            self.validate_existing_commodity("Cotton (USD)", "cotton/cotton_usd_monthly.csv", "usd")
            self.validate_existing_commodity("Cotton (PKR)", "cotton/cotton_pkr_monthly.csv", "pkr")

        # Validate Energy
        if validate_energy:
            logger.info("\nValidating ENERGY...")
            self.validate_existing_commodity(
                "Crude Oil Brent (USD)", "energy/crude_oil_brent_usd_monthly_clean.csv", "usd"
            )
            self.validate_existing_commodity(
                "Crude Oil Brent (PKR)", "energy/crude_oil_brent_pkr_monthly_clean.csv", "pkr"
            )
            self.validate_existing_commodity(
                "Natural Gas (USD)", "energy/natural_gas_usd_monthly_clean.csv", "usd"
            )
            self.validate_existing_commodity(
                "Natural Gas (PKR)", "energy/natural_gas_pkr_monthly_clean.csv", "pkr"
            )

        # Validate Polyester
        if validate_polyester:
            logger.info("\nValidating POLYESTER...")
            self.validate_existing_commodity(
                "Polyester (RMB)", "polyester/polyester_futures_monthly.csv", "rmb"
            )

        # Ingest Viscose
        if ingest_viscose:
            try:
                self.ingest_viscose_from_sunsirs(
                    start_date=viscose_start_date,
                    end_date=viscose_end_date,
                )
            except Exception as e:
                logger.error(f"Viscose ingestion failed: {e}")
                self.results["viscose"] = DataValidationResult(
                    is_valid=False,
                    num_records=0,
                    commodity="Viscose",
                    errors=[str(e)],
                    summary="Ingestion failed",
                )

        return self.results

    def print_summary(self) -> bool:
        """Print validation summary.
        
        Returns:
            True if all commodities valid, False otherwise
        """
        logger.info("\n" + "=" * 90)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 90)

        all_valid = True
        for commodity, result in self.results.items():
            status = "✓" if result.is_valid else "✗"
            logger.info(f"{status} {commodity:20} | {result.summary}")
            if not result.is_valid:
                all_valid = False

        logger.info("=" * 90)

        if all_valid:
            logger.info("✓ All commodities validated successfully")
        else:
            num_invalid = sum(1 for r in self.results.values() if not r.is_valid)
            logger.warning(f"✗ {num_invalid}/{len(self.results)} commodities have issues")

        return all_valid


def main():
    """Main entry point for data pipeline."""
    parser = argparse.ArgumentParser(description="Validate commodities and (optionally) ingest real viscose from SunSirs")
    parser.add_argument("--validate-cotton", action="store_true", help="Validate cotton monthly series")
    parser.add_argument("--validate-energy", action="store_true", help="Validate energy monthly series")
    parser.add_argument("--validate-polyester", action="store_true", help="Validate polyester monthly series")
    parser.add_argument("--ingest-viscose", action="store_true", help="Fetch real viscose daily prices from SunSirs and aggregate to monthly")
    parser.add_argument("--viscose-start-date", type=str, default=None, help="Start date (YYYY-MM-DD). Default: 1 year ago")
    parser.add_argument("--viscose-end-date", type=str, default=None, help="End date (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    pipeline = CommodityDataPipeline(data_root="data/raw")

    # If no validate flags are given, validate all frozen commodities by default.
    validate_any = args.validate_cotton or args.validate_energy or args.validate_polyester
    validate_cotton = args.validate_cotton or not validate_any
    validate_energy = args.validate_energy or not validate_any
    validate_polyester = args.validate_polyester or not validate_any

    # Run pipeline
    results = pipeline.run_full_pipeline(
        validate_cotton=validate_cotton,
        validate_energy=validate_energy,
        validate_polyester=validate_polyester,
        ingest_viscose=args.ingest_viscose,
        viscose_start_date=args.viscose_start_date,
        viscose_end_date=args.viscose_end_date,
    )

    # Print summary
    all_valid = pipeline.print_summary()

    # Exit with appropriate code
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
