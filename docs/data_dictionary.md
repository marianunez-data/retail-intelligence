# Data dictionary

## Source

UCI Online Retail II: all transactions of a UK-based online retailer of giftware, mostly sold to wholesalers, between 1 December 2009 and 9 December 2011. Two Excel sheets, 1,067,371 rows. License CC BY 4.0.

## Raw columns (after `retail_intelligence.ingest`)

| Column | Type | Description |
|---|---|---|
| invoice | string | Invoice number. Starts with `C` for a cancellation, `A` for a bad-debt adjustment |
| stock_code | string | Product code. Non-merchandise codes such as `POST`, `DOT`, `M`, `BANK CHARGES` also appear |
| description | string | Product name |
| quantity | integer | Units per line; negative for cancellations and stock adjustments |
| invoice_date | timestamp | Invoice date and time |
| price | float | Unit price in GBP |
| customer_id | integer, nullable | Customer number; missing for 22.8% of lines |
| country | string | Customer country (43 countries, United Kingdom 91% of customers) |

## Cleaning rules (`dbt/models/staging/stg_transactions.sql`)

| Rule | Removes | Reason |
|---|---|---|
| 1 | Exact duplicate lines (34,335) | System double-posts |
| 2 | Lines without customer ID (243,007) | Cannot be linked to a customer |
| 3 | Price zero or negative | Free samples and accounting entries, not sales |
| 4 | Invoices starting with `A` | Bad-debt adjustments |
| 5 | Codes in `non_product_codes` (`dbt_project.yml`) | Postage, carriage, fees, manual entries, tests |

Cancellations are kept and flagged (`is_cancellation`) to compute the return rate. About 74% of lines remain after cleaning.

## Feature table (`dbt/models/marts/customer_features.sql`)

One row per customer with at least one order before `cutoff_date`. Every feature uses transactions strictly before the cutoff.

| Column | Definition |
|---|---|
| CustomerID | Customer number |
| cutoff_date | Snapshot date |
| label | 1 if the customer placed a non-cancelled order in `[cutoff, cutoff + 90 days)`, else 0 |
| recency_days | Days between the last order and the cutoff |
| frequency | Distinct non-cancelled invoices |
| monetary | Net revenue in GBP (sales minus returns); customers with net spend of zero or less are excluded |
| avg_order_value | Net spend divided by number of orders, GBP |
| unique_products | Distinct stock codes bought |
| tenure_days | Days between the first order and the cutoff |
| return_rate | Cancelled invoices divided by orders (can exceed 1 when returns refer to older orders) |
| orders_last_90d | Orders in the 90 days before the cutoff |
| avg_days_between_orders | Average gap between consecutive orders; NULL when frequency is 1 |
| is_uk | 1 if the customer's country is United Kingdom |

## Data tests

dbt tests run on every build (42 across the three modules), including for the retention models: `not_null` and `unique` keys, accepted values for `label` and `is_uk`, a custom generic test that unit prices are positive, `assert_valid_feature_ranges` (non-negative recency and return rate, tenure at least as long as recency, at least one order, `orders_last_90d` not above `frequency`, empty gap only for one-order customers) and `assert_no_label_leakage` (no feature computed from transactions on or after the cutoff).

---

# Demand forecasting (notebook 02)

## `stg_sales_lines` (all channels)

Same rules as above except rule 2: guest lines (no customer ID) are kept and flagged with `channel = 'Guest'`, because they are real demand. Guest orders are 6.2% of units and 13.1% of revenue, at about twice the wholesale list price. `stg_transactions` is now a view on `stg_sales_lines` filtered to registered customers, so notebook 01 is unchanged.

## `dim_products` (`data/processed/dim_products.csv`)

| Column | Definition |
|---|---|
| product_id | Stock code |
| description | Most frequent description of the product |
| category | Merchandise category from `seeds/product_category_rules.csv`: keywords matched as whole words (plural allowed), lowest priority number first; `Other` when nothing matches (about 11% of revenue) |

## `product_weekly_demand` (`data/processed/product_weekly_demand.parquet`)

A complete product x week grid (weeks start on Monday) for products sold in at least `min_active_weeks` (70) of the 104 complete weeks: 1,041 products, about 63% of units and revenue.

| Column | Definition |
|---|---|
| product_id, week_start | Grain |
| units | Units ordered (non-cancelled lines, all channels); 0 when nothing sold, including the two Christmas weeks when the shop was closed |
| revenue, orders, customers | Revenue in GBP, distinct invoices, distinct registered customers |
| guest_units | Units ordered by guests |
| avg_unit_price | Revenue divided by units |
| list_price | Highest registered-channel price that covers at least 10% of the week's registered order lines (lower prices are volume tiers) |

The partial first week (data starts on a Tuesday) and last week (ends on a Friday) are excluded. Test `assert_demand_grid_complete` fails the build if any product misses a week.

## `demand_forecast.csv`

| Column | Definition |
|---|---|
| forecast_start, forecast_end | The four forecast weeks |
| p10, p50, p90 | 10th percentile, median and 90th percentile of units over the four weeks, intervals widened by conformal calibration |
| last_4_weeks, last_13_weeks_avg | Units in the last four weeks, and the 13-week average expressed per four weeks |

---

# Pricing and promotions (notebook 03)

## Source: dunnhumby *Breakfast at the Frat*

Three sheets downloaded by `python -m retail_intelligence.ingest_pricing`: transactions (524,950 rows, one per store, product and week ending on Wednesday), products (58) and stores (79 rows, 77 stores). (c) dunnhumby, not redistributed.

## Cleaning rules (`stg_frat_sales`, `stg_frat_stores`)

| Rule | Removes or fixes | Rows |
|---|---|---|
| 1 | Missing or zero shelf or base price | 209 |
| 2 | Zero units (out of stock or delisted) | 5 |
| 3 | More than 5 units per basket (bulk buying) | 67 |
| 4 | Shelf price above base price: kept, discount set to 0 | 6,047 kept |
| 5 | Stores 4503 and 17627 listed twice with different price tiers: one row kept (Mainstream) | 2 duplicate store rows |

## `pricing_store_weeks` (`data/processed/pricing_store_weeks.parquet`, not committed)

| Column | Definition |
|---|---|
| week_end, store_id, upc | Grain (unique, tested) |
| price_tier, state | Store attributes |
| description, manufacturer, category, sub_category | Product attributes |
| units, visits, households, spend | Weekly sales |
| price, base_price | Shelf price paid and regular price |
| discount_depth | 1 - price / base price, at least 0 |
| is_feature, is_display, is_tpr_only, is_promo | Circular feature, in-store display, shelf-tag-only price reduction, any of the three |

## `pricing_products.csv` (committed aggregate)

Per product: base price (median of the last 13 weeks on sale), stores, regular weekly chain units, promotion frequency, average discount when promoted, unpooled and pooled elasticities with standard errors, recommended base price change and recommendation at a 30% margin.
