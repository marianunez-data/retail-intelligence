"""Project settings in one place, shared by the pipeline, notebook, app and tests."""
import os
from pathlib import Path

# Repository root. RETAIL_HOME overrides it (used by the Docker image and the tests).
ROOT = Path(os.getenv("RETAIL_HOME") or Path(__file__).resolve().parents[2])
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
DBT_DIR = ROOT / "dbt"

RAW_EXCEL = DATA_RAW / "online_retail_II.xlsx"
RAW_PARQUET = DATA_RAW / "online_retail_ii.parquet"
PRICING_RAW = DATA_RAW / "dunnhumby"
MODEL_PATH = MODELS / "model.joblib"

# Snapshots. Features use data strictly before the cutoff; the label is
# "bought again within HORIZON_DAYS after the cutoff".
HORIZON_DAYS = 90
SNAPSHOTS = {
    # name: cutoff date
    "features_2011_06_01": "2011-06-01",   # earlier snapshot: out-of-time validation
    "features_2011_09_01": "2011-09-01",   # latest fully labeled snapshot: production training
    "features_current": "2011-12-10",      # day after the last transaction: customers to score
}
TRAIN_SNAPSHOT = "features_2011_09_01"
OOT_TRAIN_SNAPSHOT = "features_2011_06_01"
SCORING_SNAPSHOT = "features_current"

FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "unique_products",
    "tenure_days", "return_rate", "orders_last_90d", "avg_days_between_orders", "is_uk",
]
FEATURE_LABELS = {
    "recency_days": "Days since last purchase",
    "frequency": "Number of orders",
    "monetary": "Total spent (GBP)",
    "avg_order_value": "Average order value (GBP)",
    "unique_products": "Distinct products bought",
    "tenure_days": "Days as customer",
    "return_rate": "Return rate",
    "orders_last_90d": "Orders in last 90 days",
    "avg_days_between_orders": "Average days between orders",
    "is_uk": "UK customer",
}
TARGET = "label"
ID = "CustomerID"

# Business rules for segments and actions
LOYAL_THRESHOLD = 0.60
NURTURE_THRESHOLD = 0.30
HIGH_VALUE_QUANTILE = 0.80

RANDOM_STATE = 42

# Demand forecasting (notebook 02)
DEMAND_MART = DATA_PROCESSED / "product_weekly_demand.parquet"
PRODUCTS = DATA_PROCESSED / "dim_products.csv"
DEMAND_FORECAST = DATA_PROCESSED / "demand_forecast.csv"
DEMAND_HORIZON_WEEKS = 4          # replenishment cycle: forecast the next four weeks
DEMAND_BACKTEST_WEEKS = 26        # rolling-origin backtest over the last half year

# Pricing and promotions (notebook 03)
PRICING_MART = DATA_PROCESSED / "pricing_store_weeks.parquet"
PRICING_PRODUCTS = DATA_PROCESSED / "pricing_products.csv"
PRICING_MODEL = DATA_PROCESSED / "pricing_model.json"
DEFAULT_GROSS_MARGIN = 0.30       # assumed retailer margin at base price when no cost data exists
