"""
Retention engine shared by the dashboard and the AI analyst.

Plain Python with no UI code, so it can be unit tested and exposed to the
agent as tools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from retail_intelligence.config import DEMAND_FORECAST, ID, MODEL_PATH, PRICING_PRODUCTS
from retail_intelligence.modeling import shap_values
from retail_intelligence.train import ACTIONS, SEGMENTS, assign_actions

SQL_ROW_LIMIT = 200
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|"
    r"export|import|call|set|truncate|replace|grant|revoke|vacuum|checkpoint)\b",
    re.IGNORECASE,
)

__all__ = ["Engine", "SQLGuardError", "SEGMENTS", "ACTIONS"]


class SQLGuardError(ValueError):
    """Raised when a query is not a single read-only SELECT."""


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class Engine:
    model: object
    features: list
    labels: dict
    metrics: dict
    gains: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    scored: pd.DataFrame

    # ---------------------------------------------------------------- setup
    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "Engine":
        b = joblib.load(path)
        return cls(b["model"], b["features"], b["feature_labels"], b["metrics"], b["gains"],
                   b["calibration"], b["importance"], b["scored"])

    def __post_init__(self):
        t = self.metrics["thresholds"]
        self.loyal_t, self.nurture_t, self.high_value_t = t["loyal"], t["nurture"], t["high_value_spend"]
        self.typical = self.scored[self.features].median()
        # Defense in depth: besides the query guard, the database itself cannot
        # touch files or the network, and its configuration is locked.
        self._db = duckdb.connect(database=":memory:")
        self._db.register("customers", self._sql_view())
        self.tables = ["customers"]
        for name, path, dtype in [("product_forecast", DEMAND_FORECAST, {"product_id": str}),
                                  ("product_pricing", PRICING_PRODUCTS, {"upc": str})]:
            if path.exists():
                self._db.register(name, pd.read_csv(path, dtype=dtype))
                self.tables.append(name)
        self._db.execute("SET enable_external_access = false")
        self._db.execute("SET lock_configuration = true")

    # ------------------------------------------------- demand and pricing
    @property
    def demand(self):
        """Demand forecast view (notebook 02), or None when it has not been built."""
        if not hasattr(self, "_demand"):
            from retail_intelligence.demand import DemandForecast
            self._demand = DemandForecast.load() if DEMAND_FORECAST.exists() else None
        return self._demand

    @property
    def pricing(self):
        """Pricing model view (notebook 03), or None when it has not been built."""
        if not hasattr(self, "_pricing"):
            from retail_intelligence.pricing import PricingModel
            self._pricing = PricingModel.load() if PRICING_PRODUCTS.exists() else None
        return self._pricing

    # -------------------------------------------------------------- scoring
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["probability"] = self.model.predict_proba(out[self.features].astype(float))[:, 1]
        return assign_actions(out, self.high_value_t).sort_values("probability", ascending=False)

    def customer(self, customer_id: int) -> pd.Series | None:
        hit = self.scored[self.scored[ID] == int(customer_id)]
        return None if hit.empty else hit.iloc[0]

    def explain(self, row: pd.Series | dict) -> tuple[pd.DataFrame, float]:
        """Exact TreeSHAP explanation of one score.

        Returns one row per feature with its SHAP contribution in log-odds
        (positive raises the probability), and the model's base probability:
        base log-odds plus all contributions equals this customer's score."""
        x = pd.DataFrame([pd.Series(row)[self.features].astype(float)])
        contrib = shap_values(self.model, x).iloc[0]
        out = pd.DataFrame({
            "feature": self.features,
            "label": [self.labels[f] for f in self.features],
            "value": x.iloc[0].values,
            "typical": self.typical.values,
            "shap": contrib[self.features].values.astype(float),
        }).sort_values("shap", key=abs, ascending=False)
        return out.reset_index(drop=True), float(_sigmoid(contrib["base_value"]))

    # ------------------------------------------------------------- analytics
    def summary(self) -> dict:
        s = self.scored
        seg = s["segment"].value_counts().reindex(SEGMENTS, fill_value=0)
        act = (s.groupby("action").agg(customers=(ID, "count"), past_spend=("monetary", "sum"))
                .reindex(ACTIONS).fillna(0))
        return {
            "customers": int(len(s)),
            "expected_repurchasers": round(float(s["probability"].sum())),
            "segments": {k: int(v) for k, v in seg.items()},
            "actions": {a: {"customers": int(r.customers), "past_spend": round(float(r.past_spend), 2)}
                        for a, r in act.iterrows()},
            "high_value_threshold": round(self.high_value_t, 2),
            "scoring_date": self.metrics["scoring_snapshot"]["cutoff"],
        }

    def find(self, segment: str | None = None, action: str | None = None,
             value_tier: str | None = None, uk_only: bool | None = None,
             min_spend: float = 0, min_recency_days: int = 0,
             sort_by: str = "probability", ascending: bool = False,
             limit: int = 50) -> pd.DataFrame:
        s = self.scored
        if segment:
            s = s[s["segment"] == segment]
        if action:
            s = s[s["action"] == action]
        if value_tier:
            s = s[s["value_tier"] == value_tier]
        if uk_only is not None:
            s = s[s["is_uk"] == int(uk_only)]
        s = s[(s["monetary"] >= min_spend) & (s["recency_days"] >= min_recency_days)]
        sort_by = sort_by if sort_by in s.columns else "probability"
        return s.sort_values(sort_by, ascending=ascending).head(max(1, min(int(limit), 500)))

    # ------------------------------------------------------------------- SQL
    def _sql_view(self) -> pd.DataFrame:
        v = self.scored.rename(columns={ID: "customer_id"}).copy()
        v["probability"] = v["probability"].round(4)
        return v[["customer_id", "probability", "segment", "value_tier", "action"] + self.features]

    @property
    def sql_schema(self) -> str:
        return "\n\n".join(self.SCHEMAS[t] for t in self.tables)

    SCHEMAS = {}
    SQL_SCHEMA = """Table `customers` (one row per customer, scored on the latest snapshot):
- customer_id INTEGER
- probability DOUBLE: probability of buying again in the next 90 days (0 to 1)
- segment VARCHAR: 'Loyal' (>= 0.6), 'Nurture' (0.3 to 0.6), 'At risk' (< 0.3)
- value_tier VARCHAR: 'High value' (top 20% by monetary) or 'Standard'
- action VARCHAR: 'Priority win-back', 'Win-back offer', 'Nurture: cross-sell', 'Loyalty reward'
- recency_days INTEGER: days since last purchase
- frequency INTEGER: number of orders
- monetary DOUBLE: net revenue in GBP
- avg_order_value DOUBLE: GBP per order
- unique_products INTEGER: distinct products bought
- tenure_days INTEGER: days since first purchase
- return_rate DOUBLE: cancelled invoices divided by orders
- orders_last_90d INTEGER: orders in the 90 days before the snapshot
- avg_days_between_orders DOUBLE: average gap between orders, NULL for one-order customers
- is_uk INTEGER: 1 if United Kingdom, else 0"""

    SCHEMAS["customers"] = SQL_SCHEMA
    SCHEMAS["product_forecast"] = """Table `product_forecast` (Online Retail wholesaler, one row per product,
units customers will order in the 4 weeks from forecast_start; demand notebook 02):
- product_id VARCHAR, description VARCHAR, category VARCHAR
- forecast_start DATE, forecast_end DATE
- p10, p50, p90 DOUBLE: 10th percentile, median and 90th percentile of units in the 4 weeks
- last_4_weeks DOUBLE: units ordered in the 4 weeks before the forecast
- last_13_weeks_avg DOUBLE: average units per 4 weeks over the last 13 weeks"""
    SCHEMAS["product_pricing"] = """Table `product_pricing` (a different business: a US grocery chain, dunnhumby data,
one row per product; pricing notebook 03; prices in USD):
- upc VARCHAR, description VARCHAR, manufacturer VARCHAR, category VARCHAR, sub_category VARCHAR
- base_price DOUBLE: current regular shelf price; stores INTEGER
- weekly_units_regular DOUBLE: chain units in a regular (non-promoted) week
- promo_frequency DOUBLE: share of store-weeks on promotion; avg_discount_when_promoted DOUBLE
- weekly_revenue DOUBLE
- elasticity_base_pooled DOUBLE: base price elasticity (e.g. -2 means +1% price -> -2% units)
- elasticity_promo_pooled DOUBLE: response to temporary discounts
- elasticity_base, se_base, elasticity_promo, se_promo DOUBLE: unpooled estimates and standard errors
- recommended_change DOUBLE: profit-maximising base price change at a 30% margin, capped at +-10%
- recommendation VARCHAR"""

    @staticmethod
    def guard_sql(query: str) -> str:
        """Allow exactly one read-only SELECT (or WITH ... SELECT)."""
        q = query.strip().rstrip(";").strip()
        if not q:
            raise SQLGuardError("Empty query.")
        if ";" in q:
            raise SQLGuardError("Only one statement is allowed.")
        if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
            raise SQLGuardError("Only SELECT queries are allowed.")
        if _FORBIDDEN.search(q):
            raise SQLGuardError("The query contains a forbidden keyword.")
        if re.search(r"read_(csv|parquet|json)|\bglob\s*\(|http", q, re.IGNORECASE):
            raise SQLGuardError("Queries may only read the registered tables.")
        return f"SELECT * FROM ({q}) AS result LIMIT {SQL_ROW_LIMIT}"

    def sql(self, query: str) -> pd.DataFrame:
        return self._db.execute(self.guard_sql(query)).df()
