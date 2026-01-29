# Step 5: Economic Guardrails – Management-Safe Decision Outputs

## Overview

The decision layer now includes **economic guardrails** that protect business outputs while maintaining analytical integrity. Guardrails are applied **post-elasticity computation** and are fully **transparent** with configurable per-asset rules.

## Problem Statement

Raw scenario simulations can produce unrealistic outcomes:
- **Cotton**: Supply Crisis scenario → -52.6%, Supply Glut → -100% (negative price)
- **FX**: Extreme financial stress → -28% move (unrealistic for normal markets)

These scenarios might be analytically valid but are **not actionable** for procurement or treasury teams due to:
1. **Implausibility** – negative commodity prices or >30% FX swings outside crisis regimes
2. **P&L shock** – extreme moves can trigger risk limits incorrectly
3. **Confidence erosion** – management loses trust in system if outputs are not reality-anchored

**Solution**: Apply guardrails to scenario outputs, converting analytical extremes into flagged, dampened decisions that remain analyzable but are business-realistic.

---

## Architecture

### Guardrails Pipeline

```
ML Forecast (frozen)
    ↓
[Elasticity Model] → Impact Attribution
    ↓
[Scenario Simulator] → Raw Scenarios (-52.6%, -100%, etc.)
    ↓
[GuardrailEngine] ← NEW ← Applies dampening + floor + caps
    ↓
Guardrailed Prices ($48.75 with EXTREME flag)
    ↓
[Signal Generator] → Buy/Hold/Delay
    ↓
Business Decision (management-safe)
```

### Core Components

#### 1. **GuardrailConfig** – Asset-Specific Rules

```yaml
# Cotton guardrails
price_floor: 1.00  # USD/cwt – prevent negative prices
max_move_pct: 35.0  # Hard cap on scenario move
dampening_threshold_pct: 20.0  # Activate nonlinear dampening above this
dampening_factor: 0.6  # Apply to excess move beyond threshold
warning_threshold_pct: 15.0  # Mark as ELEVATED above this
extreme_threshold_pct: 25.0  # Mark as EXTREME above this
```

**Pre-configured profiles:**
- `GuardrailConfig.for_cotton()` – commodity-focused (floor $1.00, ±35% cap)
- `GuardrailConfig.for_eurusd()` – FX-focused (floor $0.50, ±10% cap)

---

#### 2. **GuardrailEngine** – Applies Three Protection Layers

**Layer 1: Nonlinear Dampening** (preserves signal, moderates tail shock)
```
For moves |move| > threshold:
  dampened_move = sign(move) × threshold + (move - threshold) × factor
  
Example: -50% move with threshold=20%, factor=0.6:
  = -20 + (-50 - (-20)) × 0.6
  = -20 + (-30) × 0.6
  = -20 - 18
  = -38% (capped at hard max -35%)
```

**Layer 2: Hard Caps** (max_move_pct enforcement)
```
guardrailed_price = base_price × (1 + min(move_pct, ±max_move_pct) / 100)
```

**Layer 3: Price Floor** (prevent impossible outcomes)
```
if guardrailed_price < price_floor:
  guardrailed_price = price_floor
```

---

#### 3. **Extreme Flagging** – Business-Safe Labeling

```python
# Flag values returned in GuardrailResult
NORMAL:      ≤ warning_threshold (e.g., ≤15%)
ELEVATED:    warning_threshold < move < extreme_threshold (e.g., 15-25%)
EXTREME:     ≥ extreme_threshold (e.g., ≥25%)
IMPLAUSIBLE: move > 100% or hit price floor (requires manual review)
```

**Management interpretation:**
- ✓ NORMAL: Use in decision logic; no review needed
- ⚠ ELEVATED: Monitor but usable; include in risk reports
- 🔴 EXTREME: Review before acting; may indicate data quality issue
- ❌ IMPLAUSIBLE: Do not use for trading/procurement; investigate source

---

## Cotton Example: Before & After Guardrails

### Base Case: $75.00/cwt

| Scenario | Raw Forecast | Original % | Guardrailed | GRD % | Flag | Adjustments |
|----------|--------------|-----------|-------------|-------|------|-------------|
| Supply Tightening | $70.00 | -6.7% | $70.00 | -6.7% | NORMAL | None |
| Supply Crisis | $35.55 | -52.6% | $48.75 | -35.0% | EXTREME | Nonlinear dampening: -52.6% → -35.0% |
| Supply Glut | $-0.08 | -100.1% | $48.75 | -35.0% | EXTREME | Price floor ($1.00) + dampening |

**Business Impact:**
- ✅ Prices stay positive and realistic
- ✅ Extreme scenarios still visible (flagged) for analysis
- ✅ Procurement teams can act on NORMAL cases immediately
- ✅ Risk committee reviews EXTREME cases before action

---

## Technical Specification

### GuardrailEngine API

```python
from forecasting.decision import GuardrailConfig, GuardrailEngine

# Initialize for cotton
cfg = GuardrailConfig.for_cotton()
engine = GuardrailEngine(cfg)

# Apply to single scenario
result = engine.apply_guardrails(
    base_price=75.0,
    scenario_price=35.55,  # Raw forecast
    scenario_name="Supply Crisis"
)

# Returns GuardrailResult with:
# - guardrailed_price: actual output price
# - original_pct: raw forecast % move
# - guardrailed_pct: after guardrails % move
# - extreme_flag: NORMAL | ELEVATED | EXTREME | IMPLAUSIBLE
# - adjustments: list of applied corrections
# - dampening_applied, floor_applied: boolean flags

# Batch apply across scenarios
prices_dict = {
    "Base": 75.0,
    "Bull": 82.5,
    "Bear": 35.55
}
results = engine.batch_guardrails(base_price=75.0, prices=prices_dict)

# Get summary
summary = engine.summarize_guardrails(results)
# Output: "1 normal, 0 elevated, 1 extreme, 0 implausible; ..."
```

---

## Design Philosophy

### ✓ Preservation of Signal

- Guardrails **dampen** extreme moves, never **remove** them
- EXTREME flag makes outliers visible for human review
- Analytical relationship preserved: if supply crisis was predicted, extreme flag confirms it

### ✓ Transparency

- All adjustments logged in `adjustments` list
- Original vs. guardrailed prices always reported
- Can be disabled for pure analysis runs (set `enable_guardrails=False`)

### ✓ Business Alignment

- Per-asset configuration (cotton vs. FX have different tolerances)
- Easy threshold tuning (e.g., lower `max_move_pct` if risk committees are more conservative)
- Prevents unrealistic P&L scenarios that trigger false risk alerts

### ✓ Asymmetry-Ready

- Presets are symmetric (±35% for cotton) but easily customized (e.g., ±30% down, ±20% up for bearish bias)
- Nonlinear dampening is symmetric by design

---

## Integration Points

### 1. **Decision Layer** (Modified)

```python
# in src/forecasting/decision/scenario.py
from forecasting.decision import GuardrailEngine, GuardrailConfig

engine = GuardrailEngine(GuardrailConfig.for_cotton())

# After scenario simulation
guardrailed = engine.batch_guardrails(
    base_price=forecast_value,
    prices=raw_scenarios
)
```

### 2. **Signal Generation** (Implicit)

```python
# Signal thresholds now apply to guardrailed prices
if guardrailed_price > buy_threshold:
    signal = SignalGenerator.BUY
```

### 3. **Reporting & Business Scripts**

```python
# in scripts/cotton_procurement_guidance.py
results_before = engine_cotton.batch_guardrails(base_price, raw_scenarios)
results_after = engine_cotton.batch_guardrails(base_price, guardrailed_scenarios)

print("BEFORE Guardrails: ", results_before)
print("AFTER Guardrails: ", results_after)
print("Summary: ", engine_cotton.summarize_guardrails(results_after))
```

---

## Configuration Examples

### Cotton Guardrails (Commodity-Heavy)

```yaml
# data/decision/guardrails_cotton.yml
price_floor: 1.00           # Prevent negative prices
max_move_pct: 35.0          # Cap moves at ±35%
dampening_threshold_pct: 20.0  # Activate dampening at ±20%
dampening_factor: 0.6       # Apply 60% to excess
warning_threshold_pct: 15.0 # ELEVATED flag
extreme_threshold_pct: 25.0 # EXTREME flag
```

**Rationale**: Commodities have volatile supply shocks (droughts, geopolitical); ±35% is reasonable but >35% is rare and needs review.

---

### EURUSD Guardrails (FX, Structural)

```yaml
# data/decision/guardrails_eurusd.yml
price_floor: 0.50           # EUR never gets that weak
max_move_pct: 10.0          # FX moves more structured; ±10% is extreme
dampening_threshold_pct: 5.0   # Start dampening at ±5%
dampening_factor: 0.8       # Smoother curve (80%)
warning_threshold_pct: 5.0  # ELEVATED at ±5%
extreme_threshold_pct: 8.0  # EXTREME at ±8%
```

**Rationale**: FX markets are deeper; only crisis scenarios see >10% moves; normal regimes should stay <5%.

---

## Testing & Validation

### Unit Tests (Included)

```bash
pytest src/forecasting/decision/tests/test_guardrails.py -v

# Tests include:
# ✓ Price floor enforcement (negative → floor)
# ✓ Max move cap (overshoots capped)
# ✓ Nonlinear dampening (formula correctness)
# ✓ Extreme flagging (thresholds correct)
# ✓ Batch operations (vectorized, per-scenario)
# ✓ Transparency (adjustments list populated)
```

### Integration Test

```bash
# Run demo
python scripts/demo_guardrails.py

# Verify:
# ✓ Cotton scenarios dampened correctly
# ✓ FX scenarios respect max_move_pct
# ✓ Nonlinear dampening table shows smooth curve
# ✓ Principles section explains design
```

### Business Test

```bash
# Run cotton procurement guidance with guardrails
python scripts/cotton_procurement_guidance.py

# Verify in output:
# ✓ Supply Crisis capped at -35% (was -52.6%)
# ✓ Supply Glut capped at -35% (was -100%)
# ✓ Base case stays $75.00 (was NORMAL, unchanged)
# ✓ Guardrail summary shows adjustments made
# ✓ Tactical recommendations use guardrailed prices
```

---

## Monitoring & Observability

### Guardrail Trigger Count

Track how often guardrails activate in production:
```python
summary = engine.summarize_guardrails(all_results)
# "15 normal, 3 elevated, 2 extreme, 0 implausible"
# If extreme count rises → data quality issue or market regime shift
```

### Adjustment Log

Save all guardrail adjustments for audit:
```python
for scenario_name, result in results.items():
    if result.adjustments:
        log_event("Guardrail Applied", {
            "scenario": scenario_name,
            "original_price": result.original_price,
            "guardrailed_price": result.guardrailed_price,
            "adjustments": result.adjustments,
            "flag": result.extreme_flag
        })
```

---

## Limitations & Future Enhancements

### Current Scope
- ✓ Price floor enforcement
- ✓ Scenario caps (hard max % move)
- ✓ Nonlinear dampening (symmetric)
- ✓ Per-asset configuration
- ✓ Transparent logging

### Future Enhancements (Optional)
- [ ] Asymmetric caps (e.g., ±30% down, ±20% up)
- [ ] Time-decay dampening (recent data gets stricter guardrails)
- [ ] Correlation with market volatility (auto-adjust thresholds based on VIX/historical vol)
- [ ] Scenario-specific guardrails (e.g., tighter for known-uncertain seasons)
- [ ] Integration with risk limits (guardrail recommends position size cap)

---

## Summary

**Guardrails convert analytical insights into business-actionable decisions:**

| Decision | Without Guardrails | With Guardrails |
|----------|-------------------|-----------------|
| Supply Crisis scenario | -52.6% move (RISKY) | -35% move + EXTREME flag (REVIEWABLE) |
| Supply Glut scenario | -100% move, -$0.08 price (BROKEN) | -35% move + EXTREME flag + FLOOR (SAFE) |
| Normal scenarios | 5-15% moves (USABLE) | 5-15% moves (UNCHANGED) |

**Result**: Management has confidence in system outputs + analytics remain rigorous.

---

## Quick Start

### Run the demo:
```bash
python scripts/demo_guardrails.py
```

### Apply to your asset:
```python
from forecasting.decision import GuardrailConfig, GuardrailEngine

# Initialize
cfg = GuardrailConfig.for_cotton()  # or for_eurusd() or custom
engine = GuardrailEngine(cfg)

# Apply to forecast
result = engine.apply_guardrails(base_price=75.0, scenario_price=35.55)
print(f"Guardrailed: ${result.guardrailed_price:.2f} (flag: {result.extreme_flag.value})")

# For batch
results = engine.batch_guardrails(base_price=75.0, prices={"Bull": 82.5, "Bear": 35.55})
print(engine.summarize_guardrails(results))
```

---

**End of Economic Guardrails Documentation**
