# Model card: four-week demand forecast

## Overview

| | |
|---|---|
| Task | Forecast the units each product will be ordered in the next four weeks, with a 10th to 90th percentile range |
| Data | UCI Online Retail II, all channels; 1,041 products sold in at least 70 of 104 complete weeks (7 Dec 2009 to 28 Nov 2011), about 63% of units and revenue |
| Model | One global `HistGradientBoostingRegressor` across all products: Poisson loss for the median, quantile losses for P10 and P90; features and target divided by each product's 13-week level; training weighted by that level |
| Intervals | Conformalized quantile regression: intervals widened by the 80% quantile of past normalized misses, calibrated sequentially in the backtest |
| Code | `src/retail_intelligence/demand.py`, notebook `notebooks/02_demand_forecasting.ipynb` |
| Outputs | `data/processed/demand_forecast.csv`, `reports/demand_metrics.json` |

## Features

Recent demand (lags 1 to 8, means of 4, 8 and 26 weeks, level), regularity (share of weeks sold, volatility, weeks since the last sale), product seasonality (same four weeks last year and the quarter before; missing in the first year), catalogue seasonality and calendar (catalogue-wide seasonal ratio and trend, week of the year). An ablation in the notebook shows what each group contributes.

## Evaluation

Rolling origin: 12 origins every two weeks from 6 June to 7 November 2011, retrained on past data only at each origin; target = units in the next four weeks.

| Method | Product WAPE | Bias | Catalogue WAPE |
|---|---|---|---|
| Last week x4 | 71.8% | -3.5% | 16.3% |
| Average of last 4 weeks | 52.7% | -6.1% | 11.5% |
| Average of last 13 weeks | 49.0% | -9.4% | 12.8% |
| Seasonal naive | 74.4% | +22.8% | 22.8% |
| **Global gradient boosting** | **47.5%** | **+3.6%** | **8.1%** |

80% interval coverage: 77% raw, 81% after conformal calibration.

## Intended use

Purchasing quantities and safety stock for established products, reviewed by a planner. The P90 is a reasonable order-up-to level for a one-in-ten stock-out risk per cycle.

## Limitations

- **December.** Only one past December exists in the data. The first five trading days of December 2011 suggest the December forecast is about 24% low.
- **Erratic demand.** Most products sell in lumpy wholesale orders; a large part of the product-level error is order-size noise no forecast removes.
- **Scope.** New and short-lived products (fewer than 70 selling weeks) are not covered; demand is gross orders (about 2% are later cancelled); product forecasts are not reconciled with category totals.
