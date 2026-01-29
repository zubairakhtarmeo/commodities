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

## Safety notes (common leakage traps)
- Never compute rolling stats using centered windows.
- Never fit scalers/imputers on full data; fit on train folds only.
- Never let the label horizon overlap the feature window without purge/embargo.
- Never generate features using future timestamps (e.g., forward-filled exogenous beyond as-of).
