# Monthly Validation Report — April 2026

**Generated:** 2026-05-30 13:44  
**Reporting period:** April 2026 (01-Apr to 30-Apr)  
**Period days:** 30  
**Overall:** 5 PASS · 2 WARN · 0 FAIL

---

## Pipeline Steps

| Step | Name | Status | Detail |
|------|------|--------|--------|
| 1 | extract_inventory | ✓ PASS | Skipped extraction — using existing file (27 KB). |
| 2 | extract_consumption | ✓ PASS | Skipped extraction — using existing file (4811 KB). |
| 3 | clean_inventory | ✓ PASS | 169 rows, 7 orgs, 0 unmapped. |
| 4 | clean_consumption | ✓ PASS | 26987 transactions, 6 orgs, 0 unmapped, 0 return rows. |
| 5 | monthly_archive | ⚠ WARN | Archive already exists: C:\Users\Hamza Raza\OneDrive - mgapparel.com (1)\apparel\ml_models |
| 6 | update_workbook | ⚠ WARN | Workbook updated with 1 warning(s): 24 item+org pair(s) in fresh data are not in the workb |
| 7 | procurement_engine | ✓ PASS | 19 rows — BUY:10  HOLD:2  MONITOR:7 |

---

## Data Summary

### Inventory

- **Rows processed:** 169
- **Organisations:** 7
- **Unmapped item codes:** 0

**Inventory by Org (summary):**

| org_name          |   Cotton |   Cotton Waste |    Fiber |   Stretch Fiber |   Total Inventory |
|:------------------|---------:|---------------:|---------:|----------------:|------------------:|
| MSL - Fibres U3   |   386324 |              0 | 398809   |        18466.6  |  803600           |
| MSM - Spinning U1 |   859628 |             52 | 161556   |        75671.5  |       1.09691e+06 |
| MTM - Spinning U1 |   180439 |              0 | 235011   |            0    |  415450           |
| MTM - Spinning U2 |   789559 |              0 |  79758.3 |            0    |  869318           |
| MTM - Spinning U3 |   149610 |              0 |  31862   |        66196.5  |  247668           |
| MTM - Spinning U5 |   109215 |              0 | 304152   |            0    |  413367           |
| MTM - Spinning U6 |   501237 |              0 | 228001   |         2284.28 |  731523           |


### Consumption

- **Transactions processed:** 26,987
- **Organisations:** 6
- **Unmapped item codes:** 0
- **Return rows (positive qty):** 0

**Net Consumption by Org (summary):**

| org_name          |           Cotton |   Fiber |   Total Consumed |
|:------------------|-----------------:|--------:|-----------------:|
| MSM - Spinning U1 |      1.68983e+06 |  267868 |      1.95769e+06 |
| MTM - Spinning U1 | 230957           |  573239 | 804196           |
| MTM - Spinning U2 |      1.71626e+06 |   23940 |      1.7402e+06  |
| MTM - Spinning U3 |  25061.9         |  146616 | 171678           |
| MTM - Spinning U5 | 283845           |  241264 | 525109           |
| MTM - Spinning U6 |      1.07863e+06 |  403725 |      1.48235e+06 |

**Transaction Type Diagnostics:**

| txn_type                     |   positive_qty_count |   negative_qty_count |   zero_qty_count |   total_rows |
|:-----------------------------|---------------------:|---------------------:|-----------------:|-------------:|
| Direct Organization Transfer |                29964 |                29964 |                0 |        59928 |

---

## Coverage Checks

### Org Alignment

⚠ **In inventory only (no consumption):** MSL - Fibres U3
✓ No orgs appear in consumption only.

### Category Coverage

- Inventory categories: `Cotton, Cotton Waste, Fiber, Stretch Fiber`
- Consumption categories: `Cotton, Fiber`

---

## Procurement Recommendations

| Metric | Value |
|--------|-------|
| Total org-commodity pairs | 19 |
| BUY (shortfall > 0) | 10 |
| HOLD (stock adequate) | 2 |
| MONITOR (no consumption data) | 7 |
| Total procurement quantity (Kgs) | 6,547,679 |

**BUY Recommendations:**

| org_name          | commodity   |   inventory_qty |   monthly_consumption |     need_45_days |        shortfall |   days_cover | confidence   |
|:------------------|:------------|----------------:|----------------------:|-----------------:|-----------------:|-------------:|:-------------|
| MTM - Spinning U2 | Cotton      |          789559 |           1.71626e+06 |      2.57438e+06 |      1.78482e+06 |         13.8 | HIGH         |
| MSM - Spinning U1 | Cotton      |          859628 |           1.68983e+06 |      2.53474e+06 |      1.67511e+06 |         15.3 | HIGH         |
| MTM - Spinning U6 | Cotton      |          501237 |           1.07863e+06 |      1.61794e+06 |      1.11671e+06 |         13.9 | HIGH         |
| MTM - Spinning U5 | Cotton      |          109215 |      283845           | 425768           | 316553           |         11.5 | HIGH         |
| MTM - Spinning U1 | Cotton      |          180439 |      230957           | 346435           | 165997           |         23.4 | HIGH         |
| MTM - Spinning U1 | Fiber       |          235011 |      573239           | 859859           | 624848           |         12.3 | HIGH         |
| MTM - Spinning U6 | Fiber       |          228001 |      403725           | 605588           | 377586           |         16.9 | HIGH         |
| MSM - Spinning U1 | Fiber       |          161556 |      267868           | 401802           | 240246           |         18.1 | HIGH         |
| MTM - Spinning U3 | Fiber       |           31862 |      146616           | 219924           | 188062           |          6.5 | HIGH         |
| MTM - Spinning U5 | Fiber       |          304152 |      241264           | 361896           |  57744.4         |         37.8 | HIGH         |

---

## Engine Validation Checks

| check_category   | check_name                  | status   | detail                                                                    | affected                                                                                                                                                                                                                      |
|:-----------------|:----------------------------|:---------|:--------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| inventory        | no_negative_inventory       | PASS     | All inventory quantities are non-negative.                                |                                                                                                                                                                                                                               |
| inventory        | commodity_mapping           | PASS     | All inventory categories match known commodities.                         |                                                                                                                                                                                                                               |
| inventory        | category_universe           | PASS     | All consumption categories exist in inventory category universe.          |                                                                                                                                                                                                                               |
| inventory        | no_duplicate_rows           | PASS     | No duplicate org + commodity rows in inventory.                           |                                                                                                                                                                                                                               |
| inventory        | org_coverage                | WARN     | 1 org(s) in inventory but not in consumption — will be flagged MONITOR.   | MSL - Fibres U3                                                                                                                                                                                                               |
| consumption      | no_duplicate_rows           | PASS     | No duplicate org + commodity rows in consumption.                         |                                                                                                                                                                                                                               |
| consumption      | no_negative_net_consumption | PASS     | No negative net consumption values.                                       |                                                                                                                                                                                                                               |
| consumption      | abnormal_consumption        | PASS     | No statistical outliers in consumption values.                            |                                                                                                                                                                                                                               |
| consumption      | period_days                 | PASS     | period_days = 30 (valid calendar month length).                           |                                                                                                                                                                                                                               |
| procurement      | no_negative_shortfall       | PASS     | All shortfall values are non-negative.                                    |                                                                                                                                                                                                                               |
| procurement      | no_negative_procurement_qty | PASS     | All procurement_qty values are non-negative.                              |                                                                                                                                                                                                                               |
| procurement      | valid_days_cover            | PASS     | All days_cover values are non-negative.                                   |                                                                                                                                                                                                                               |
| procurement      | monitor_orgs                | WARN     | 7 org-commodity pair(s) have no consumption data and cannot be evaluated. | MSL - Fibres U3 / Cotton, MSL - Fibres U3 / Fiber, MSL - Fibres U3 / Stretch Fiber, MSM - Spinning U1 / Stretch Fiber, MTM - Spinning U3 / Stretch Fiber, MTM - Spinning U6 / Stretch Fiber, MSM - Spinning U1 / Cotton Waste |
