═══════════════════════════════════════════════════════════════════════════════
    MULTI-COMMODITY DATA INFRASTRUCTURE
    Commodity Price Forecasting & Procurement Decision System
═══════════════════════════════════════════════════════════════════════════════

OVERVIEW
────────
This directory contains the data ingestion, validation, and preparation infrastructure
for a multi-commodity price forecasting system. The system handles:

  ✓ 4 Static Commodity Series (CSV-based, pre-validated)
    - Cotton (USD/PKR)
    - Crude Oil Brent (USD/PKR)
    - Natural Gas (USD/PKR)
    - Polyester (RMB)

  ✓ 1 Dynamic Commodity Series (Fetched on-demand)
    - Viscose Staple Fiber (RMB) – from SunSirs API

  ✓ Automated Validation Pipeline
  ✓ Connector-based Extensible Architecture
  ✓ Monthly Aggregation & Quality Checks

═══════════════════════════════════════════════════════════════════════════════

QUICK START
───────────

1. VALIDATE EXISTING DATA (Cotton, Energy, Polyester)

   python scripts/validate_data.py

   Expected Output:
   ──────────────
   ✓ Cotton USD                | 355 records | 1995-12-01 to 2025-06-01
   ✓ Crude Oil Brent USD       | 355 records | 1995-12-01 to 2025-06-01
   ✓ Natural Gas USD           | 355 records | 1995-12-01 to 2025-06-01
   ✓ Polyester RMB             |  69 records | 2021-05-19 to 2027-01-01
   ✓ All commodities validated successfully


2. GENERATE/FETCH VISCOSE DATA

  Fetch Real Data from SunSirs (requires internet)
  ───────────────────────────────────────────────
  python scripts/data_pipeline.py --ingest-viscose --viscose-start-date 2023-01-01


3. RUN FULL PIPELINE

   python scripts/data_pipeline.py

   Validates all 4 existing commodities + fetches/validates viscose.


4. PROCEED TO ML TRAINING

   Once validation passes:
   ──────────────────────
   python -m forecasting.cli train --config configs/cotton_monthly.yml

═══════════════════════════════════════════════════════════════════════════════

DATA STRUCTURE
──────────────

data/raw/
├── cotton/
│   ├── cotton_usd_monthly.csv        ✓ Validated (355 months)
│   └── cotton_pkr_monthly.csv        ✓ Validated (for reporting)
│
├── energy/
│   ├── crude_oil_brent_usd_monthly_clean.csv        ✓ Validated (355 months)
│   ├── crude_oil_brent_pkr_monthly_clean.csv
│   ├── natural_gas_usd_monthly_clean.csv            ✓ Validated (355 months)
│   └── natural_gas_pkr_monthly_clean.csv
│
├── polyester/
│   └── polyester_futures_monthly.csv ✓ Validated (69 months)
│
└── viscose/
  ├── viscose_daily.csv             (Real daily data from SunSirs; may be absent until fetched)
  └── viscose_monthly.csv           (Monthly aggregate of real daily data; may be absent until fetched)

═══════════════════════════════════════════════════════════════════════════════

COMMODITY METADATA
──────────────────

┌────────────────────────────────────────────────────────────────────────────┐
│ COTTON                                                                      │
├────────────────────────────────────────────────────────────────────────────┤
│ File:           cotton_usd_monthly.csv                                     │
│ Source:         ICE Futures U.S. Cotton (CT contracts)                     │
│ Currency:       USD/pound                                                  │
│ Frequency:      Monthly (mid-month settlement)                             │
│ Records:        355 months (Dec 1995 – Jun 2025)                           │
│ Reporting:      PKR/pound (for Pakistan procurement)                       │
│ Status:         ✓ Static, validated, ready for ML                          │
│ Role:           Primary commodity for forecasting                          │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ CRUDE OIL (Brent)                                                           │
├────────────────────────────────────────────────────────────────────────────┤
│ File:           crude_oil_brent_usd_monthly_clean.csv                      │
│ Source:         ICE Brent Crude Oil futures                                │
│ Currency:       USD/barrel                                                 │
│ Frequency:      Monthly (monthly average settlement)                       │
│ Records:        355 months (Dec 1995 – Jun 2025)                           │
│ Status:         ✓ Static, validated                                        │
│ Role:           Elasticity driver for polyester (petroleum-based)          │
│                 Freight cost proxy                                          │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ NATURAL GAS                                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ File:           natural_gas_usd_monthly_clean.csv                          │
│ Source:         ICE Natural Gas futures (Henry Hub)                        │
│ Currency:       USD/MMBtu                                                  │
│ Frequency:      Monthly                                                    │
│ Records:        355 months (Dec 1995 – Jun 2025)                           │
│ Status:         ✓ Static, validated                                        │
│ Role:           Production cost proxy (energy intensity of dyeing)         │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ POLYESTER                                                                   │
├────────────────────────────────────────────────────────────────────────────┤
│ File:           polyester_futures_monthly.csv                              │
│ Source:         CZCE (China Zhengzhou Commodity Exchange)                  │
│                 Polyester Staple Fiber Futures                             │
│ Currency:       RMB/ton                                                    │
│ Frequency:      Monthly (contract settlement)                              │
│ Records:        69 months (May 2021 – Jan 2027)                            │
│ Status:         ✓ Static, validated                                        │
│ Role:           Direct substitute for cotton (elasticity driver)           │
│ Note:           Short history (~3.5 years) but sufficient for recent analysis
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ VISCOSE STAPLE FIBER (VSF)                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ Files:          viscose_daily.csv (created on ingestion)                   │
│                 viscose_monthly.csv (aggregated from real daily data)      │
│ Source:         SunSirs – China's largest commodity price database         │
│                 https://www.sunsirs.com/uk/prodetail-1057.html             │
│ API:            ID 1057 (VSF spot benchmark)                               │
│ Currency:       RMB/ton                                                    │
│ Frequency:      Daily market prices → Monthly aggregation                  │
│ Records:        ~13 months typical (one year rolling window)               │
│ Status:         ⏳ On-demand ingestion (real-only)                          │
│ Update:         On-demand via Python script (not automated)                │
│ Role:           Alternative fiber elasticity driver                        │
│ Method:         Daily → Monthly (month-end average)                        │
│ Limitations:    • Spot prices only (no historical futures)                │
│                 • May have trading gaps (weekends, holidays)              │
│                 • Data lags (Chinese market hours)                         │
└────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

VALIDATION CHECKS
─────────────────

Every commodity series is automatically validated for:

  ✓ File existence
  ✓ CSV readability
  ✓ Required columns (timestamp, price_*)
  ✓ No missing values
  ✓ Unique timestamps (no duplicates)
  ✓ Chronological ordering
  ✓ Positive prices
  ✓ Currency detection (auto-detected from column name)
  ✓ Sufficient historical records (minimum 12 months)
  ✓ Date range consistency

Validation Output Example:
──────────────────────────
✓ VALID | Cotton (USD) (USD)
  Records: 355
  Date range: 1995-12-01 to 2025-06-01
  • All checks passed


═══════════════════════════════════════════════════════════════════════════════

INGESTION ARCHITECTURE
──────────────────────

Connector-Based Design (Extensible for New Sources)

┌─────────────────────────────────────────┐
│  Connector Base Class                   │
│  ────────────────────────────────────   │
│  • load_series() -> pd.Series          │
│  • validate()                           │
│  • name, frequency, currency            │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────────┐
        │                 │
   [CSVConnector]  [SunSirsConnector]
        │                 │
  ┌─────┘                 └─────────────────┐
  │                                         │
Static Files (Cotton, Energy, Polyester)   Dynamic (SunSirs API)
  │                                         │
  └─────────────────┬───────────────────────┘
                    │
            ┌───────┴────────┐
            │                │
        [Validate]      [Load to Memory]
            │                │
    (checks pass/fail)  (Series indexed)
            │                │
            └───────┬────────┘
                    │
            ┌───────┴────────┐
            │                │
        [Aggregate]     [Training Input]
      (daily→monthly)      (if needed)


═══════════════════════════════════════════════════════════════════════════════

PYTHON MODULES
──────────────

Location: src/forecasting/ingestion/

Modules:
────────
1. aggregation.py
   - aggregate_daily_to_monthly()
   - align_monthly_dates()
   Methods: mean, last, max, min

2. sunsirs_connector.py
   - SunSirsConnector.fetch_daily_prices()
   - Automatic pagination handling
   - Retry logic on network errors

3. validation.py
   - validate_price_series()
   - DataValidationResult class
   - print_validation_report()
   - Multiple validations in parallel

4. __init__.py
   - Public API for data ingestion

Usage Example:
──────────────
from forecasting.ingestion import (
    SunSirsConnector,
    aggregate_daily_to_monthly,
    validate_price_series,
)

# Fetch viscose
connector = SunSirsConnector()
daily = connector.fetch_daily_prices(
    start_date="2024-01-01",
    end_date="2025-01-31"
)

# Aggregate to monthly
monthly = aggregate_daily_to_monthly(
    daily,
    timestamp_col="timestamp",
    price_col="price_rmb",
    method="mean"
)

# Validate
result = validate_price_series(
    monthly,
    commodity="Viscose",
    expected_currency="rmb"
)

═══════════════════════════════════════════════════════════════════════════════

SCRIPTS
───────

scripts/validate_data.py
  Purpose:  Quickly validate all existing commodities
  Command:  python scripts/validate_data.py
  Output:   Validation report with status for each commodity
  

scripts/data_pipeline.py
  Purpose:  Full multi-commodity ingestion and validation pipeline
  Command:  python scripts/data_pipeline.py
            [--validate-cotton]
            [--validate-energy]
            [--validate-polyester]
            [--ingest-viscose]
            [--viscose-start-date YYYY-MM-DD]
            [--viscose-end-date YYYY-MM-DD]
  Output:   Complete validation report + viscose data ready for ML


═══════════════════════════════════════════════════════════════════════════════

COMMON WORKFLOWS
────────────────

Workflow 1: Validate Existing Data Only
  ────────────────────────────────────
  python scripts/validate_data.py
  
  When:  Starting the system, checking data integrity
  Time:  <1 second
  

Workflow 2: Fetch Real Viscose Data from SunSirs
  ───────────────────────────────────────────────
  python scripts/data_pipeline.py --ingest-viscose --viscose-start-date 2023-01-01
  
  When:  Setting up production system with real data
  Time:  30-60 seconds (depends on SunSirs response)
  Output: viscose_daily.csv + viscose_monthly.csv
  Requires: Internet access to SunSirs
  
Workflow 3: Full Pipeline
  ─────────────────────────
  python scripts/data_pipeline.py
  
  When:  Complete setup (validate all + fetch viscose)
  Time:  ~1 minute
  Output: Complete validation report + all data ready
  


═══════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING
───────────────

Issue: "File not found" error
───────────────────────────
Solution:
  ls -la data/raw/cotton/
  # If missing, data may not have been downloaded
  # See data documentation for source URLs

  
Issue: Validation fails with "Duplicate timestamps"
─────────────────────────────────────────────────────
Reason:  Two or more prices for same date
Solution:
  # Check for duplicates:
  pandas -c "import pandas as pd; df = pd.read_csv('data/raw/cotton/cotton_usd_monthly.csv'); print(df[df['timestamp'].duplicated()])"
  
  # If found, aggregate upstream (take mean):
  df.groupby('timestamp')['price_usd'].mean().reset_index()
  

Issue: Viscose fetch fails (ConnectionError)
──────────────────────────────────────────────
Reason:  SunSirs unreachable or rate-limited
Solution:
  1. Check internet: ping www.sunsirs.com
  2. Try again: Network delays are common
  3. Check SunSirs status: Visit https://www.sunsirs.com/uk/
  

Issue: "ImportError: No module named 'requests'"
──────────────────────────────────────────────────
Solution:
  pip install -r requirements.txt
  # or specifically:
  pip install requests>=2.31
  

═══════════════════════════════════════════════════════════════════════════════

PRE-TRAINING CHECKLIST
──────────────────────

Before running ML training, ensure:

  □ Validate existing data:
    python scripts/validate_data.py
    # All commodities should show ✓ VALID
    
  □ Viscose data ready (one of):
    • python scripts/data_pipeline.py --ingest-viscose --viscose-start-date YYYY-MM-DD  (real data)
    # data/raw/viscose/viscose_monthly.csv should exist
    
  □ No missing files:
    ls -la data/raw/cotton/*.csv
    ls -la data/raw/energy/*.csv
    ls -la data/raw/polyester/*.csv
    ls -la data/raw/viscose/*.csv
    # All CSVs should exist
    
  □ Data freshness:
    # Check latest date in cotton:
    pandas -c "import pandas as pd; df = pd.read_csv('data/raw/cotton/cotton_usd_monthly.csv'); print(df['timestamp'].max())"
    # Should be recent (within 1 month)
    
  □ Ready to train:
    python -m forecasting.cli train --config configs/cotton_monthly.yml
    # Should proceed without data errors


═══════════════════════════════════════════════════════════════════════════════

EXTENDING WITH NEW COMMODITIES
──────────────────────────────

To add a new commodity (e.g., Nylon fibers):

1. Prepare Data:
   • Download or source monthly prices
   • Format as CSV: timestamp, price_usd (or price_rmb, etc.)
   • Place in: data/raw/nylon/nylon_monthly.csv

2. Update Script:
   # In scripts/validate_data.py, add:
   ('Nylon USD', 'data/raw/nylon/nylon_monthly.csv', 'usd'),

3. Validate:
   python scripts/validate_data.py

4. Update ML Config:
   # Create: configs/nylon_monthly.yml
   # Copy from cotton_monthly.yml, change commodity name

5. Train:
   python -m forecasting.cli train --config configs/nylon_monthly.yml


═══════════════════════════════════════════════════════════════════════════════

DOCUMENTATION
──────────────

Detailed Guides:
  docs/data_ingestion.md  - Complete data infrastructure documentation
  

═══════════════════════════════════════════════════════════════════════════════

STATUS: ✅ READY FOR PRODUCTION

All existing commodity data:
  ✓ Validated
  ✓ Formatted consistently
  ✓ Ready for ML training

Viscose infrastructure:
  ✓ Connector built (SunSirsConnector)
  ✓ Aggregation pipeline ready
  ✓ Validation integrated

Next Steps:
  1. Validate data: python scripts/validate_data.py
  2. Prepare viscose: python scripts/data_pipeline.py --ingest-viscose --viscose-start-date YYYY-MM-DD
  3. Train models: python -m forecasting.cli train --config ...

═══════════════════════════════════════════════════════════════════════════════
