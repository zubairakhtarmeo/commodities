# Excel Export for Model Verification

## Purpose
Export comprehensive prediction data to Excel format for data analyst verification against ground realities.

## Quick Start

### Method 1: Run Export Script (Recommended for Team Lead)
```bash
python scripts/export_predictions_to_excel.py
```

This generates 3 professional Excel workbooks in the `exports/` folder:
1. **International_Market_Predictions_[timestamp].xlsx** - Cotton, Polyester, Viscose, Gas, Oil (USD)
2. **Pakistan_Local_Market_Predictions_[timestamp].xlsx** - Local commodities (PKR)
3. **All_Commodities_Comparison_[timestamp].xlsx** - Side-by-side comparison

### Method 2: Download from Dashboard
- Open any commodity tab (e.g., Cotton, Polyester)
- Scroll to bottom → "📥 Export Predictions for Verification"
- Click "📊 Download Excel" button
- Opens in Excel instantly

## What's Included in Each Export

### 1. Current Market Summary
- Commodity name & currency
- Current price
- Number of data points
- Date range (historical coverage)
- Last update timestamp

### 2. Historical Statistics
- Mean, Median, Standard Deviation
- Min/Max values
- 25th & 75th Percentiles
- Recent 6-month trend

### 3. Model Predictions (6 Time Horizons)
- **Forecast Period**: 1M, 3M, 6M, 9M, 12M, 18M
- **Predicted Price**: Model's price forecast
- **Change %**: Percentage change from current
- **Lower Bound**: Minimum expected price (confidence interval)
- **Upper Bound**: Maximum expected price (confidence interval)
- **Confidence %**: Model's confidence level (95% → 70%)
- **Recommendation**: Procurement action (LOCK PRICES NOW, WAIT & MONITOR, etc.)

### 4. Historical Data (Last 24 Months)
- Date-by-date price history
- Month-over-month % changes
- Allows verification against actual market prices

## How to Verify Model Accuracy

### For Data Analyst Review:

1. **Open Exported Workbook**
   ```
   exports/International_Market_Predictions_[date].xlsx
   ```

2. **Check Historical Data Sheet** (bottom section)
   - Compare model's historical data vs your records
   - Verify data sources are matching ground reality
   - Cross-check dates and prices

3. **Analyze Predictions Section**
   - Review predicted prices vs current market trends
   - Check if confidence intervals are reasonable
   - Validate recommendations against procurement strategy

4. **Compare Statistics**
   - Mean/Median should match your calculations
   - Std Dev indicates volatility (higher = riskier)
   - Recent trend shows if model captures market direction

5. **Test Model Logic**
   - Download multiple exports over weeks
   - Compare 1-month predictions vs actual prices
   - Calculate model accuracy: `(Predicted - Actual) / Actual * 100`

## Example Verification Workflow

```excel
# In your Excel sheet:

1. Open: International_Market_Predictions_[today].xlsx
2. Navigate to: Cotton → Historical Data section
3. Compare last month's price:
   - Model says: $0.74/lb (Dec 2025)
   - Your data says: $_____/lb
   - Match? ✓/✗

4. Check 1-month-old prediction:
   - Open last month's export
   - Find "1 Month" prediction
   - Compare to current actual price
   - Calculate error percentage

5. Quantify accuracy:
   Accuracy = 100% - |((Predicted - Actual) / Actual) × 100|
```

## Understanding the Recommendations

| Change % | Recommendation | Action |
|----------|---------------|--------|
| > +5% | LOCK PRICES NOW | Prices rising fast - secure contracts immediately |
| +2% to +5% | CONSIDER HEDGING | Moderate rise - partial hedging advisable |
| -2% to +2% | STABLE - MONITOR | Stable market - continue normal procurement |
| -5% to -2% | FAVORABLE ENTRY | Prices softening - good time to buy |
| < -5% | WAIT & MONITOR | Prices falling - wait for better entry |

## File Structure

```
exports/
├── International_Market_Predictions_20260127_143022.xlsx
│   ├── Cotton
│   ├── Polyester
│   ├── Viscose
│   ├── Natural Gas
│   └── Crude Oil (Brent)
│
├── Pakistan_Local_Market_Predictions_20260127_143022.xlsx
│   ├── Cotton (Local)
│   ├── Viscose (Local)
│   ├── Natural Gas (PKR)
│   └── Crude Oil (PKR)
│
└── All_Commodities_Comparison_20260127_143022.xlsx
    └── Summary Comparison (all commodities side-by-side)
```

## Professional Styling

All exports include:
- ✓ Navy blue headers with white text
- ✓ Auto-adjusted column widths
- ✓ Borders and cell formatting
- ✓ Professional fonts (Calibri 10-11pt)
- ✓ Ready for management presentations

## Automation

### Schedule Daily Exports
```bash
# Windows Task Scheduler
python scripts/export_predictions_to_excel.py

# Run at: 9:00 AM daily
# Location: exports/ folder
```

### Integration with Your Systems
```python
# In your analysis scripts:
import pandas as pd

# Load exported predictions
df = pd.read_excel('exports/International_Market_Predictions_[date].xlsx', 
                   sheet_name='Cotton')

# Compare with your actual data
your_actual_prices = pd.read_excel('your_data.xlsx')
accuracy = calculate_model_accuracy(df, your_actual_prices)
print(f"Model Accuracy: {accuracy:.2f}%")
```

## Questions for Team Lead Verification

### Critical Questions to Answer:
1. ✓ Do historical prices match your market data?
2. ✓ Are confidence intervals realistic (not too wide/narrow)?
3. ✓ Do recommendations align with market sentiment?
4. ✓ Is the trend direction (up/down/stable) accurate?
5. ✓ Can you replicate predictions using the statistics?

### Red Flags to Watch:
- ✗ Prices way off from market reality
- ✗ All predictions showing same trend (model bias)
- ✗ Confidence intervals too narrow (overconfidence)
- ✗ Historical data has gaps or outliers
- ✗ Statistics don't match your calculations

## Next Steps After Verification

### If Model is Accurate (>80% accuracy):
1. Use recommendations for procurement planning
2. Set alerts based on confidence levels
3. Integrate into decision workflow
4. Schedule regular exports

### If Model Needs Improvement (<80% accuracy):
1. Review data sources in `data/raw/`
2. Check if using latest market data
3. Analyze prediction errors by commodity
4. Consider model retraining

## Support

For questions about:
- **Data Sources**: Check `DATA_README.txt`
- **Model Logic**: Review `scripts/export_predictions_to_excel.py`
- **Dashboard**: See main `README.md`

---

**Generated for**: Data Analyst Verification  
**Purpose**: Quantify model predictions for ground reality validation  
**Format**: Professional Excel workbooks ready for analysis
