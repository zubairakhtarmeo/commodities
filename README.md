# Asset-agnostic price forecasting framework

Generic, leakage-safe, config-driven time-series forecasting for any numeric "price-like" series (commodities, FX, energy, crypto, macro indicators, internal business series).

## Core principles
- Strict time awareness (no future leakage)
- Forecast timestamp ≠ decision timestamp (explicit alignment)
- Configurable frequency (monthly by default)
- Explainability-first (interpretable baselines, stable explanations)
- Modular feature packs (roles, not asset-specific heuristics)
- Walk-forward validation with purge/embargo for overlapping windows

## Quickstart (after installing deps)
```powershell
python -m pip install -r requirements.txt
python -m pip install -e .

forecast train --config configs/example_monthly.yml
forecast predict --config configs/example_monthly.yml --asset-id GENERIC_ASSET --model-name linear_ridge --asof 2021-06-30
```

## Project layout
- `src/forecasting/`: core library
- `configs/`: YAML configs (assets, roles, features, models, split strategy)
- `data/`: optional local data (raw/interim/processed); connectors can also pull from APIs/DBs

## Invariants and extension points
Before onboarding real assets, read: `docs/invariants_and_extension_points.md`.
It defines which modules are invariant (must never be customized per asset) and the only allowed extension mechanisms.

## Data Sources and Freshness

| Commodity | Source | Type | Update Cadence | Notes |
|-----------|--------|------|---------------|-------|
| cotton_usd / cotton_pkr | FRED `PCOTTINDUSDM` | Live API | Monthly | FRED publishes ~4-6 weeks after month-end; 72d age is normal |
| crude_oil_usd / crude_oil_pkr | FRED `DCOILBRENTEU` | Live API | Daily → monthly | Very fresh; updates within ~24h |
| natural_gas_usd / natural_gas_pkr | FRED `DHHNGSP` | Live API | Daily → monthly | Very fresh |
| polyester_usd / polyester_pkr | `polyester_futures_monthly_clean.csv` | Futures CSV | Manual | Includes forward contract months (e.g. 2027); this is intentional futures data, not an error |
| viscose_usd / viscose_pkr | SunSirs + CSV baseline | Static/CSV | Manual | SunSirs now returns 2021 demo data; live scraping is dead. CSV ceiling: Jan 2026. Extend manually or find alternative source |

## Production Validation

Run the bootstrap health check before any deployment or after any data pipeline change:

```powershell
python scripts/bootstrap_production.py
python scripts/bootstrap_production.py --no-color   # CI / log files
python scripts/bootstrap_production.py --fail-fast  # stop on first failure
```

The script exits with code `0` (healthy) or `1` (critical failures present).

### Checks performed

| Section | What is checked |
|---------|----------------|
| Environment | SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, core Python deps |
| Connectivity | Supabase REST API reachable |
| Database tables | Existence + row counts for all 4 tables |
| Commodity data | Freshness per commodity, missing commodities, duplicate rows |
| Forecast records | Presence, horizons (1/3/6 months), null values, value sanity |
| Country cotton | Row counts, country diversity, prediction rows |
| Deployment files | Key scripts and source files present |

### Output levels

| Tag | Meaning |
|-----|---------|
| `[OK]` | Check passed |
| `[WARN]` | Non-critical issue — platform works but needs attention |
| `[FAIL]` | Critical issue — fix before relying on the platform |

### Common warnings and fixes

**`[FAIL] cotton_usd: latest data is 72d old`**
Cotton ingestion has not run recently. Trigger `python scripts/run_ingestion.py` or check the GitHub Actions `daily_ingestion.yml` workflow.

**`[FAIL] viscose_usd: latest data is 131d old`** / **`[WARN] viscose_pkr: no rows`**
Viscose ingestion pipeline needs a run: `python scripts/ingest_viscose.py`. The SunSirs scraper requires network access to the source site.

**`[WARN] cotton_country_predictions: empty`**
Country-level forecasts have not been generated. Run: `python scripts/run_cotton_country_forecasts.py`

**`[WARN] <commodity>: missing horizons [3, 6]`**
The last forecast run produced partial output. Re-run `python scripts/run_forecasts.py` and check for errors in the output.

**`[FAIL] <table>: HTTP 401`**
Key mismatch or RLS policy blocking access. Verify `SUPABASE_SERVICE_ROLE_KEY` in `.streamlit/secrets.toml`.

## Forecast Evaluation Methodology

Run `python scripts/evaluate_forecasts.py` to generate `artifacts/forecast_evaluation.json`.
Results appear in the **AI Predictions** dashboard under "Forecast Accuracy (Backtesting)".

**How it works:**
- Rolling walk-forward backtest over the last 18 months of historical data (configurable via `--backtest-months`)
- For each evaluation month `t` and horizon `h`: train on data `[0..t]`, predict `h` months ahead using actual values at intermediate steps (no error compounding), compare to observed price at `t+h`
- Metrics: MAE, RMSE, MAPE, directional accuracy (% correct direction)
- Baselines: last-value persistence (`last_value`) and 3-month moving average (`ma3`)

**Models evaluated:** linear_ridge, random_forest, ridge_returns, rf_returns

**Honest reporting:** if an ML model fails to outperform the last-value baseline, this is reported as-is.
Common causes: viscose/polyester are static/futures series with limited time variation;
natural gas has high volatility that ML struggles to predict without exogenous signals.

**Bootstrap integration:** `scripts/bootstrap_production.py` checks evaluation file age
and whether any ML model beats the baseline per commodity.

## Safety notes (common leakage traps)
- Never compute rolling stats using centered windows.
- Never fit scalers/imputers on full data; fit on train folds only.
- Never let the label horizon overlap the feature window without purge/embargo.
- Never generate features using future timestamps (e.g., forward-filled exogenous beyond as-of).
