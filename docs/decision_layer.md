# STEP 4 – Decision & Elasticity Layer

## Overview

The **decision layer** sits *downstream* of ML forecasts. It applies elasticity-based transformations to convert point forecasts into actionable insights.

**Key principle**: This layer does NOT retrain models or modify ML outputs. It _interprets_ them through the lens of economic elasticities.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ML Forecasting Layer (FROZEN)                          │
│  - Walk-forward validation                              │
│  - Models: baseline, ridge, hist_gbrt                   │
│  - Output: price forecast (point estimate)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Decision & Elasticity Layer (CONFIG-DRIVEN)            │
│  - Load elasticity config (YAML)                        │
│  - Compute impact attribution                           │
│  - Run scenario simulations                             │
│  - Generate trading signals                             │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────┴──────────────┐
        │                           │
        ▼                           ▼
   ┌─────────────┐          ┌──────────────┐
   │  Impact     │          │  Scenario    │
   │  Attribution│          │  Simulations │
   │  (Why)      │          │  (What-If)   │
   └─────────────┘          └──────────────┘
        │                           │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Trading Signal            │
        │  (Buy/Hold/Delay)          │
        └────────────────────────────┘
```

## Components

### 1. Elasticity Config (`data/decision/`)

**Location**: `data/decision/eurusd_elasticity.yml`

**Schema**:
```yaml
asset_id: "EURUSD"

variables:
  - name: "usd_index"              # variable identifier
    category: "international"      # "local" or "international"
    description: "..."             # human-readable
    unit: "%"                       # unit for reference
    default_value: 103.5            # baseline level

elasticities:
  - variable: "usd_index"
    elasticity: -0.8               # % impact per 1% change in variable
    lag_periods: 0                 # lead/lag relative to forecast

hierarchy:                          # optional parent/child structure
  - name: "macroeconomic_factors"
    parent: null
    children: ["usd_index", "us_inflation"]
```

**Interpretation**:
- Elasticity = -0.8 means: if USD index rises 1%, EURUSD falls ~0.8%.
- Elasticity = 0.2 means: if oil price rises 1%, EURUSD rises ~0.2%.

### 2. Impact Attribution Module (`src/forecasting/decision/elasticity.py`)

**Purpose**: Decompose price forecast into driver contributions.

**Input**:
- ML forecast price (point estimate from model)
- Variable changes (% moves in macro/market drivers)
- Elasticity coefficients from config

**Output**:
```
Base ML Forecast: 1.1100
Total Elasticity-Based Impact: -1.47%
Absolute Impact: -0.0163

Detailed Impacts:
  usd_index:      -0.80% impact (USD +1.0%)
  brent_crude:    -0.40% impact (oil -2.0%)
  eur_inflation:  +0.15% impact (inflation -0.5%)
  ...
```

**Answer to "WHY did the price move?"**

### 3. Scenario Simulation Module (`src/forecasting/decision/scenario.py`)

**Purpose**: Run what-if analyses under different market conditions.

**Scenarios**:
- **Bull (EUR Strength)**: Lower yields, rising stocks, oil rally → EURUSD +2.25%
- **Bear (USD Strength)**: Strong data, higher yields, oil selloff → EURUSD -3.35%
- **Baseline**: No macro moves → EURUSD flat

**One-Way Sensitivity**:
Vary USD index from -2% to +2%, see price outcomes across range.

**Answer to "WHAT-IF USD strengthens / weakens / etc?"**

### 4. Signal Generator Module (`src/forecasting/decision/signal.py`)

**Purpose**: Convert scenario analysis into trading decisions.

**Signals**:
- **BUY**: Scenario shows >2% upside, confidence >60%
- **HOLD**: Neutral zone, monitor drivers
- **DELAY**: Scenario shows >1.5% downside, caution advised

**Example**:
```
Bull Scenario:
  Signal: HOLD (upside 2.25%, at threshold; low confidence in driver visibility)
  Rationale: Monitor eur_stocks, us_10y_yield

Bear Scenario:
  Signal: DELAY (downside -3.35% likely)
  Rationale: Scenario shows material downside. Caution advised.
```

## Data Flow

### Minimal End-to-End Example

```python
from pathlib import Path
from forecasting.decision import ElasticityModel, ScenarioSimulator, SignalGenerator
from forecasting.decision.config import load_elasticity_config
from forecasting.decision.scenario import ScenarioInput

# 1. Load elasticity config
cfg = load_elasticity_config(Path("data/decision/eurusd_elasticity.yml"))
model = ElasticityModel(cfg)

# 2. Get ML forecast (e.g., from artifacts)
ml_forecast = 1.1100  # from model output

# 3. Impact Attribution (explain the forecast)
variable_changes = {
    "usd_index": 1.0,      # USD +1%
    "us_10y_yield": 0.30,  # Yields +30 bps
    "brent_crude": -2.0,   # Oil -2%
}
attribution = model.attribute_impacts(
    timestamp=pd.Timestamp("2026-02-28"),
    horizon=1,
    base_price_forecast=ml_forecast,
    variable_changes=variable_changes,
)
print(f"Total impact: {attribution.total_impact_pct:.2f}%")

# 4. Scenario Simulation (what-if analysis)
simulator = ScenarioSimulator(cfg)
bull_scenario = ScenarioInput(
    name="Bull",
    variable_changes_pct={"usd_index": -1.5, "us_10y_yield": -0.5, ...}
)
result = simulator.simulate_scenario(ml_forecast, bull_scenario)
print(f"Bull scenario price: {result.scenario_price:.4f}")

# 5. Signal Generation
signal_gen = SignalGenerator(buy_threshold_pct=2.0, sell_threshold_pct=-1.5)
signal = signal_gen.generate_signal(
    timestamp="2026-02-28",
    ml_forecast_price=ml_forecast,
    scenario_price=result.scenario_price,
    scenario_name="Bull",
    top_drivers=["usd_index", "us_10y_yield"],
)
print(f"Signal: {signal.signal.value}")
print(f"Rationale: {signal.rationale}")
```

## Files

| File | Purpose |
|------|---------|
| `src/forecasting/decision/__init__.py` | Module entry point |
| `src/forecasting/decision/config.py` | Elasticity schema & loader |
| `src/forecasting/decision/elasticity.py` | Impact attribution engine |
| `src/forecasting/decision/scenario.py` | Scenario simulation engine |
| `src/forecasting/decision/signal.py` | Signal generator (Buy/Hold/Delay) |
| `data/decision/eurusd_elasticity.yml` | Example EURUSD elasticity config |
| `scripts/demo_decision_layer.py` | End-to-end demo |

## Extension Points

### Add a New Asset Elasticity Config

1. Create `data/decision/{ASSET_ID}_elasticity.yml`
2. Define variables, elasticities, hierarchy
3. Load and use in your decision pipeline

Example:
```yaml
# data/decision/oil_elasticity.yml
asset_id: "BRENT"
variables:
  - name: "us_gdp_growth"
    category: "international"
    ...
elasticities:
  - variable: "us_gdp_growth"
    elasticity: 0.6
    ...
```

### Customize Signal Logic

Subclass `SignalGenerator` and override `generate_signal()`:

```python
class CustomSignalGen(SignalGenerator):
    def generate_signal(self, ...):
        # Custom logic here
        pass
```

### Add More Drivers

1. Update elasticity YAML with new variable
2. Provide market data (e.g., from data vendor)
3. Include in `variable_changes` dict when attributing impacts

## Design Principles

✅ **Rule-based**: No ML models, pure deterministic logic
✅ **Config-driven**: All elasticities, scenarios, thresholds in YAML
✅ **Separation of concerns**: Decision layer independent of ML training
✅ **Explainability**: Every output (impact, scenario, signal) has clear rationale
✅ **Scenario-focused**: Supports what-if analysis, not just point forecasts

## Limitations & Caveats

1. **Elasticities are static**: Updated periodically (not real-time ML)
2. **Linear assumption**: Elasticity * Change assumes linear impact (nonlinearity ignored)
3. **No covariance modeling**: Treats variable changes independently
4. **Signal thresholds are hardcoded**: Customize in `SignalGenerator.__init__()`
5. **Lag handling**: `lag_periods` is metadata; actual lags managed upstream

## Testing

Run the demo to validate all three layers:

```bash
python scripts/demo_decision_layer.py --demo all
```

Or test individual components:

```bash
python scripts/demo_decision_layer.py --demo impact
python scripts/demo_decision_layer.py --demo scenario
python scripts/demo_decision_layer.py --demo signal
```

## Summary

The decision layer transforms ML price forecasts into explainable, actionable insights via elasticity-based decomposition and scenario analysis. It remains completely independent from the ML training pipeline and can be updated (elasticities, thresholds, signals) without retraining models.

**Next Steps**:
1. Integrate actual market data feeds (USD index, yields, oil, equity indices)
2. Connect ML forecast outputs to decision layer inputs
3. Backtest signal performance on historical data
4. Add risk management layer (position sizing, stop losses)
