"""Test setup: a synthetic project root with three feature snapshots, the
real training code run on it, and the Engine loaded from the result.
No real data, network or API key is needed."""
import os
import tempfile
from pathlib import Path

# Point the package at a scratch root BEFORE any package module is imported.
_ROOT = Path(tempfile.mkdtemp(prefix="retail-test-"))
os.environ["RETAIL_HOME"] = str(_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def synthetic_snapshot(n=800, seed=0, cutoff="2011-09-01", labeled=True):
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 600, n)
    freq = rng.integers(1, 40, n)
    monetary = (freq * rng.uniform(40, 400, n)).round(2)
    p = 1 / (1 + np.exp(0.012 * recency - 0.08 * freq))
    gap = np.where(freq > 1, rng.uniform(10, 200, n).round(1), np.nan)
    return pd.DataFrame({
        "CustomerID": np.arange(10000, 10000 + n),
        "cutoff_date": cutoff,
        "label": (rng.random(n) < p).astype(int) if labeled else 0,
        "recency_days": recency, "frequency": freq, "monetary": monetary,
        "avg_order_value": (monetary / freq).round(2),
        "unique_products": rng.integers(1, 200, n),
        "tenure_days": recency + rng.integers(0, 300, n),
        "return_rate": rng.uniform(0, 0.3, n).round(3),
        "orders_last_90d": np.where(recency < 90, rng.integers(1, 4, n), 0),
        "avg_days_between_orders": gap,
        "is_uk": (rng.random(n) < 0.85).astype(int),
    })


def write_demand_and_pricing_outputs():
    """Small stand-ins for the demand forecast and pricing exports the app reads."""
    import json

    from retail_intelligence.config import DEMAND_FORECAST, DEMAND_MART, PRICING_MODEL, PRICING_PRODUCTS, ROOT

    weeks = pd.date_range("2011-01-03", periods=20, freq="7D")
    pd.DataFrame([{"product_id": pid, "week_start": w, "units": float(i + 1)}
                  for pid in ["A1", "B2"] for i, w in enumerate(weeks)]).to_parquet(DEMAND_MART)
    pd.DataFrame({"product_id": ["A1", "B2"], "forecast_start": "2011-05-23", "forecast_end": "2011-06-19",
                  "p10": [10.0, 5.0], "p50": [20.0, 8.0], "p90": [30.0, 12.0], "last_4_weeks": [70.0, 70.0],
                  "last_13_weeks_avg": [60.0, 60.0], "description": ["RED HEART LANTERN", "PAPER CHAIN KIT"],
                  "category": ["Lighting and candles", "Christmas and seasonal"]}).to_csv(DEMAND_FORECAST, index=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "demand_metrics.json").write_text(json.dumps({
        "forecast_window": ["2011-05-23", "2011-06-19"], "products": 2, "backtest_period": ["2011-01-01", "2011-05-01"],
        "methods": {"Global gradient boosting": {"wape_product": 0.4}}, "best_baseline": "Average of last 13 weeks",
        "interval_coverage_calibrated": 0.8}))
    pd.DataFrame({"upc": ["111", "222"], "description": ["GM HONEY NUT CHEERIOS", "DIGRN PEPP PIZZA"],
                  "manufacturer": ["GENERAL MI", "TOMBSTONE"], "category": ["Cold cereal", "Frozen pizza"],
                  "sub_category": ["x", "y"], "base_price": [2.95, 5.80], "stores": [77, 77],
                  "weekly_units_regular": [3800.0, 1200.0], "promo_frequency": [0.2, 0.5],
                  "avg_discount_when_promoted": [0.2, 0.25], "weekly_revenue": [9000.0, 9000.0],
                  "elasticity_base": [-1.4, -2.8], "elasticity_promo": [-2.0, -3.3], "se_base": [0.1, 0.2],
                  "se_promo": [0.1, 0.2], "elasticity_base_pooled": [-1.4, -2.8],
                  "elasticity_promo_pooled": [-2.0, -3.3], "recommended_change": [0.1, 0.05],
                  "recommendation": ["Test an increase", "Test an increase"]}).to_csv(PRICING_PRODUCTS, index=False)
    effects = {"estimate": 0.4, "se": 0.05, "lift": 0.49}
    PRICING_MODEL.write_text(json.dumps({
        "category_elasticities": {"Cold cereal": {"elasticity_base": -1.35, "elasticity_promo": -1.7,
                                                  "between_product_sd_base": 0.8},
                                  "Frozen pizza": {"elasticity_base": -2.9, "elasticity_promo": -3.2,
                                                   "between_product_sd_base": 0.2}},
        "promo_effects": {c: {t: effects for t in ["is_feature", "is_display", "is_tpr_only"]}
                          for c in ["Cold cereal", "Frozen pizza"]},
        "validation": {"all_weeks": {"Seasonality + price and promotions": {"wape": 0.35, "bias": 0.0}}},
        "settings": {"default_gross_margin": 0.3, "price_range": 0.1}}))


@pytest.fixture(scope="session")
def engine():
    from retail_intelligence import train
    from retail_intelligence.config import DATA_PROCESSED, MODEL_PATH, SNAPSHOTS
    from retail_intelligence.engine import Engine

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for i, (name, cutoff) in enumerate(SNAPSHOTS.items()):
        synthetic_snapshot(seed=i, cutoff=cutoff, labeled=name != "features_current") \
            .to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
    train.main()
    write_demand_and_pricing_outputs()
    return Engine.load(MODEL_PATH)
