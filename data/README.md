# Data

| Path | Content | In git |
|---|---|---|
| `raw/online_retail_II.xlsx`, `raw/online_retail_ii.parquet` | UCI Online Retail II (45 MB) and its Parquet copy written by `python -m retail_intelligence.ingest` | No |
| `raw/dunnhumby/*.parquet` | dunnhumby *Breakfast at the Frat* tables written by `python -m retail_intelligence.ingest_pricing` (downloads the source file) | No: (c) dunnhumby, all rights reserved |
| `processed/features_*.csv` | Three customer snapshots for the retention model (dbt) | Yes |
| `processed/product_weekly_demand.parquet`, `processed/dim_products.csv` | Weekly demand grid and product categories (dbt) | Yes |
| `processed/demand_forecast.csv` | Four-week forecast per product with P10, P50, P90 | Yes |
| `processed/pricing_products.csv`, `processed/pricing_model.json` | Aggregated pricing results: elasticities, promotion effects, validation | Yes |
| `processed/pricing_store_weeks.parquet` | Row-level pricing panel (dbt), derived from licensed data | No |
| `processed/warehouse.duckdb` | Local dbt warehouse | No |

Download both sources with `make data`. The committed outputs let the app and the Docker image run without any raw file. Rebuild everything with `make all`.
