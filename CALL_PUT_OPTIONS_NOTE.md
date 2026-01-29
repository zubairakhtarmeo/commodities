# Call/Put Options - Implementation Status

## Current Status (Jan 28, 2026)

✅ **Basic hedging + compounding simulator is now available in the dashboard** (Market Intelligence page).

❌ **Full options pricing (Black–Scholes/Greeks, implied volatility, options chains) is NOT implemented.**

### What Are Call/Put Options?

**Call Options**: Right to BUY a commodity at a fixed price (strike price) on a future date
- **Use case**: Protect against price INCREASES
- **Example**: Buy call option for cotton at $0.80/lb. If price rises to $0.90, you can still buy at $0.80

**Put Options**: Right to SELL a commodity at a fixed price on a future date
- **Use case**: Protect against price DECREASES (if you already own the commodity)
- **Example**: Own cotton inventory. Buy put option at $0.75. If price drops to $0.65, you can still sell at $0.75

### Why It's Not in the Dashboard

1. **Not Part of ML Model**: Our framework focuses on price forecasting, not options pricing
2. **Different Financial Instrument**: Options require:
   - Strike price determination
   - Volatility modeling (Black-Scholes, etc.)
   - Time decay (theta) calculations
   - Greeks (delta, gamma, vega)
3. **Complexity**: Options trading requires specialized models beyond commodity forecasting

### What IS Available in Dashboard

✅ **Price Forecasts**: 1M, 3M, 6M, 9M, 12M, 18M horizons
✅ **Confidence Intervals**: Upper/lower bounds for risk assessment
✅ **Recommendations**: LOCK PRICES NOW, WAIT & MONITOR, etc.
✅ **Risk Alerts**: Identifies commodities with >5% forecast increases

✅ **New (Basic) Hedging + Bank Compounding Simulator**
- Compares: wait+invest in bank vs fixed-price agreement (forward-style) vs a call-ceiling-style contract
- Supports **declining interest rate** scenarios (month-by-month compounding)
- Uses forecast-based low/base/high scenarios for the selected horizon

### Mention in Code

There's ONE reference to "covered call" in `scripts/cotton_procurement_guidance.py` line 251:
```python
- Hedge: Consider 25% covered call at ${bull_price_grd:.2f} (supply crisis protection)
```

This is just a **suggestion** in text output, not an actual options pricing implementation.

---

## If You Need Options Trading Features

### Option 1: Simple Hedging Recommendations (Easy)
Add text-based recommendations to dashboard:
- "Consider buying call options if forecast shows +10% increase"
- "Put options may be useful if you hold inventory"

### Option 2: Basic Options Calculator (Medium)
Implement simple Black-Scholes calculator:
- Input: Current price, strike price, time to expiration, volatility
- Output: Estimated option premium (call/put price)

### Option 3: Full Options Module (Complex)
Build complete options trading system:
- Real-time options chain data
- Greeks calculation
- Multi-leg strategies (spreads, straddles)
- Risk management tools

---

## Recommendation for Your Use Case

**For Textile/Apparel Procurement:**

Instead of options, focus on:

1. **Forward Contracts**: Lock in prices directly with suppliers
   - Use our 3M/6M forecasts to negotiate fixed prices
   - More straightforward than options

2. **Supplier Agreements**: Flexible pricing clauses
   - "Price ceiling" contracts based on our upper bound forecasts
   - Easier to manage than financial derivatives

3. **Inventory Strategy**: Buy ahead when forecasts show increases
   - Use recommendations: "LOCK PRICES NOW" as trigger
   - Physical inventory hedging without derivatives complexity

4. **Budget Forecasting**: Use our predictions for budget planning
   - Upper bound = worst-case budget scenario
   - Lower bound = best-case savings opportunity

---

## Technical Requirements for Options Implementation

If still needed, would require:

```python
# Additional libraries
- quantlib or pyvollib (options pricing)
- scipy.stats (statistical distributions)
- Historical volatility calculations

# Additional data
- Options chain data (strike prices, premiums)
- Implied volatility from market
- Risk-free rate data

# New models
- Black-Scholes model for European options
- Binomial tree for American options
- Monte Carlo for exotic options
```

**Estimated effort**: 2-3 weeks for basic implementation

---

## Summary

✅ **Hedging + Compounding Planning**: Implemented as a simulator in dashboard
❌ **Call/Put Options Pricing Engine**: Not implemented
✅ **Price Forecasting**: Fully implemented with confidence intervals
✅ **Risk Management**: Recommendations and alerts available
💡 **Recommendation**: Use forward contracts + inventory strategy instead of options for textile procurement

---

*If options trading is critical requirement, we can scope a separate module. For now, focus on leveraging our forecasts for direct procurement decisions.*

**Last Updated**: January 28, 2026
