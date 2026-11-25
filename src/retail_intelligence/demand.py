"""
Demand forecasting for the Online Retail catalogue (notebook 02).

Question: how many units of each product will customers order in the next four
weeks? The answer drives purchasing and stock levels, so the forecast comes with
an 80% interval (10th to 90th percentile) and is judged on a rolling-origin
backtest against simple methods a planner would otherwise use.

    python -m retail_intelligence.demand          # dbt build, backtest, final forecast

Outputs
    data/processed/demand_forecast.csv  next four weeks per product, P10 / P50 / P90
    reports/demand_metrics.json         backtest accuracy by method
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from retail_intelligence.config import (DEMAND_BACKTEST_WEEKS, DEMAND_FORECAST, DEMAND_HORIZON_WEEKS,
                              DEMAND_MART, PRODUCTS, RANDOM_STATE, ROOT)

H = DEMAND_HORIZON_WEEKS
MIN_HISTORY = 26          # weeks needed before the first training origin (the longest recent-demand window)
FULL_HISTORY = 65         # weeks needed for last-year features; before that they are missing
QUANTILES = (0.1, 0.9)    # 80% prediction interval
BASELINES = ["Last week x4", "Average of last 4 weeks", "Average of last 13 weeks", "Seasonal naive"]
MODEL = "Global gradient boosting"
FEATURE_GROUPS = {
    "Recent demand": ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8",
                      "mean_4", "mean_8", "mean_26", "log_level"],
    "Regularity and dormancy": ["share_weeks_sold_13", "volatility_13", "weeks_since_last_sale"],
    "Product seasonality (last year)": ["last_year_next_4", "last_year_prev_13", "last_year_seasonal_ratio"],
    "Catalogue seasonality and calendar": ["catalogue_seasonal_ratio", "catalogue_trend", "week_of_year"],
}
_excluded: set[str] = set()


@contextmanager
def excluding(*features: str):
    """Temporarily drop features, for ablation studies: with excluding(*cols): backtest(...)."""
    _excluded.update(features)
    try:
        yield
    finally:
        _excluded.difference_update(features)


# ------------------------------------------------------------------ data
def load_demand() -> pd.DataFrame:
    df = pd.read_parquet(DEMAND_MART)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values(["product_id", "week_start"]).reset_index(drop=True)


def load_products() -> pd.DataFrame:
    return pd.read_csv(PRODUCTS, dtype={"product_id": str})


def to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index, pd.DatetimeIndex]:
    """Products x weeks matrix of units (rows sorted by product id)."""
    m = df.pivot(index="product_id", columns="week_start", values="units").fillna(0)
    return m.to_numpy(dtype=float), m.index, pd.DatetimeIndex(m.columns)


def demand_class(units: pd.Series) -> str:
    """Syntetos-Boylan classification from the average interval between sales (ADI)
    and the squared coefficient of variation of non-zero sales (CV2)."""
    nz = units[units > 0]
    if len(nz) < 2:
        return "Lumpy"
    adi = len(units) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2
    if adi < 1.32:
        return "Smooth" if cv2 < 0.49 else "Erratic"
    return "Intermittent" if cv2 < 0.49 else "Lumpy"


# --------------------------------------------------------------- features
def _scale(Y: np.ndarray, t0: int) -> np.ndarray:
    """Recent demand level (13-week mean) used to put all products on one scale."""
    return Y[:, t0 - 13:t0].mean(1) + 0.5


def _weeks_since_sale(Y: np.ndarray, t0: int, cap: int = 52) -> np.ndarray:
    """Weeks since the product last sold, capped; dormant products tend to stay dormant."""
    recent = Y[:, max(0, t0 - cap):t0][:, ::-1] > 0
    return np.where(recent.any(1), recent.argmax(1), cap).astype(float)


def _last_year(Y: np.ndarray, t0: int, s: np.ndarray, total: np.ndarray) -> dict:
    """Same four weeks last year and the quarter before them. Missing (NaN) for
    origins in the first year, which gradient boosting handles natively; this lets
    the model train on the December 2010 slowdown even though no December 2009 exists."""
    if t0 < FULL_HISTORY:
        nan = np.full(len(Y), np.nan)
        return {k: nan for k in ["last_year_next_4", "last_year_prev_13", "last_year_seasonal_ratio",
                                 "catalogue_seasonal_ratio"]}
    nxt, prev = Y[:, t0 - 52:t0 - 48].mean(1), Y[:, t0 - 65:t0 - 52].mean(1)
    return {
        "last_year_next_4": nxt / s,
        "last_year_prev_13": prev / s,
        "last_year_seasonal_ratio": (nxt + 0.5) / (prev + 0.5),
        "catalogue_seasonal_ratio": np.full(len(Y), total[t0 - 52:t0 - 48].mean() / total[t0 - 65:t0 - 52].mean()),
    }


def make_features(Y: np.ndarray, weeks: pd.DatetimeIndex, t0: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Features known at forecast origin t0 (the first week to forecast).

    Every demand feature is divided by the product's recent level, so one model
    learns shapes shared by slow and fast sellers alike."""
    s = _scale(Y, t0)
    total = Y.sum(0)
    X = {f"lag_{lag}": Y[:, t0 - lag] / s for lag in range(1, 9)}
    X.update({
        "mean_4": Y[:, t0 - 4:t0].mean(1) / s,
        "mean_8": Y[:, t0 - 8:t0].mean(1) / s,
        "mean_26": Y[:, t0 - 26:t0].mean(1) / s,
        "share_weeks_sold_13": (Y[:, t0 - 13:t0] > 0).mean(1),
        "volatility_13": Y[:, t0 - 13:t0].std(1) / s,
        **_last_year(Y, t0, s, total),
        "catalogue_trend": np.full(len(Y), total[t0 - 4:t0].mean() / total[t0 - 13:t0].mean()),
        "week_of_year": np.full(len(Y), weeks[t0 - 1].isocalendar()[1] + 1),
        "log_level": np.log(s),
        "weeks_since_last_sale": _weeks_since_sale(Y, t0),
    })
    return pd.DataFrame(X).drop(columns=sorted(_excluded & set(X))), s


def target(Y: np.ndarray, t0: int) -> np.ndarray:
    """Units in the next H weeks (the replenishment quantity)."""
    return Y[:, t0:t0 + H].sum(1)


def baseline_forecasts(Y: np.ndarray, t0: int) -> dict[str, np.ndarray]:
    level_ratio = np.clip((Y[:, t0 - 13:t0].sum(1) + 1) / (Y[:, t0 - 65:t0 - 52].sum(1) + 1), 0.2, 5)
    return {
        "Last week x4": Y[:, t0 - 1] * H,
        "Average of last 4 weeks": Y[:, t0 - 4:t0].mean(1) * H,
        "Average of last 13 weeks": Y[:, t0 - 13:t0].mean(1) * H,
        # same four weeks last year, scaled by how much the product has grown since
        "Seasonal naive": Y[:, t0 - 52:t0 - 48].sum(1) * level_ratio,
    }


# ------------------------------------------------------------------ model
def fit_models(Y: np.ndarray, weeks: pd.DatetimeIndex, last_origin: int) -> dict[str, HistGradientBoostingRegressor]:
    """Train the point model (Poisson loss) and two quantile models on every
    origin whose four target weeks end before `last_origin`."""
    origins = range(MIN_HISTORY, last_origin - H + 1)
    parts = [make_features(Y, weeks, o) + (target(Y, o),) for o in origins]
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    s = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]) / s          # demand relative to the product's level
    common = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=100, random_state=RANDOM_STATE)
    models = {"p50": HistGradientBoostingRegressor(loss="poisson", **common)}
    for q in QUANTILES:
        models[f"p{int(q * 100)}"] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common)
    for m in models.values():
        m.fit(X, y, sample_weight=s)                        # large sellers weigh more, as in WAPE
    return models


def predict(models: dict, Y: np.ndarray, weeks: pd.DatetimeIndex, t0: int) -> pd.DataFrame:
    X, s = make_features(Y, weeks, t0)
    out = pd.DataFrame({k: np.clip(m.predict(X), 0, None) * s for k, m in models.items()})
    out["p10"] = np.minimum(out["p10"], out["p50"])
    out["p90"] = np.maximum(out["p90"], out["p50"])
    out["scale"] = s
    return out


def conformity_scores(actual, p10, p90, scale) -> np.ndarray:
    """How far the actual fell outside the raw interval, in units of the product's level
    (negative when inside). Used for conformalized quantile regression."""
    return np.maximum(np.asarray(p10) - actual, np.asarray(actual) - np.asarray(p90)) / np.asarray(scale)


def conformal_margin(scores: np.ndarray, level: float = 0.8) -> float:
    """Finite-sample quantile of past scores that makes the interval cover `level`."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n)))


def calibrate(p10, p90, scale, margin: float) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(np.asarray(p10) - margin * scale, 0, None), np.asarray(p90) + margin * scale


# --------------------------------------------------------------- backtest
def wape(y, yhat) -> float:
    """Weighted absolute percentage error: total absolute error over total demand."""
    return float(np.abs(np.asarray(y) - np.asarray(yhat)).sum() / np.asarray(y).sum())


def bias(y, yhat) -> float:
    """Total forecast minus total demand, as a share of demand (negative = under-forecast)."""
    return float((np.asarray(yhat).sum() - np.asarray(y).sum()) / np.asarray(y).sum())


def backtest(Y: np.ndarray, weeks: pd.DatetimeIndex, products: pd.Index, step: int = 2) -> pd.DataFrame:
    """Rolling-origin evaluation over the last DEMAND_BACKTEST_WEEKS weeks. At each
    origin the model is retrained on the past only, then forecasts the next H weeks."""
    T = Y.shape[1]
    rows = []
    for t0 in range(T - DEMAND_BACKTEST_WEEKS, T - H + 1, step):
        models = fit_models(Y, weeks, t0)
        actual = target(Y, t0)
        pred = predict(models, Y, weeks, t0)
        forecasts = {**baseline_forecasts(Y, t0), MODEL: pred["p50"].to_numpy()}
        for name, f in forecasts.items():
            frame = pd.DataFrame({"origin": weeks[t0], "product_id": products, "method": name,
                                  "actual": actual, "forecast": f})
            if name == MODEL:
                frame["p10"], frame["p90"] = pred["p10"].to_numpy(), pred["p90"].to_numpy()
                frame["scale"] = pred["scale"].to_numpy()
            rows.append(frame)
    bt = pd.concat(rows, ignore_index=True)
    return _sequential_calibration(bt)


def _sequential_calibration(bt: pd.DataFrame) -> pd.DataFrame:
    """Calibrate each origin's interval with scores from origins whose four target
    weeks had fully happened by then, exactly as it would run in production."""
    gb = bt[bt.method == MODEL].copy()
    gb["score"] = conformity_scores(gb.actual, gb.p10, gb.p90, gb.scale)
    for origin in gb.origin.unique():
        past = gb[gb.origin + pd.Timedelta(weeks=H) <= origin]
        if past.empty:
            continue
        idx = gb.origin == origin
        m = conformal_margin(past.score.to_numpy())
        gb.loc[idx, "p10_cal"], gb.loc[idx, "p90_cal"] = calibrate(gb.loc[idx, "p10"], gb.loc[idx, "p90"],
                                                                   gb.loc[idx, "scale"], m)
    return bt.join(gb[["score", "p10_cal", "p90_cal"]])


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per method at product level and for the catalogue total."""
    def per_method(d):
        total = d.groupby("origin")[["actual", "forecast"]].sum()
        return pd.Series({"wape_product": wape(d.actual, d.forecast), "bias": bias(d.actual, d.forecast),
                          "wape_total": wape(total.actual, total.forecast)})
    out = bt.groupby("method")[["origin", "actual", "forecast"]].apply(per_method)
    return out.reindex(BASELINES + [MODEL])


# ------------------------------------------------------------------- main
def main() -> None:
    from retail_intelligence.features import dbt_build

    dbt_build("+product_weekly_demand +dim_products")
    df = load_demand()
    Y, products, weeks = to_matrix(df)

    bt = backtest(Y, weeks, products)
    summary = summarize(bt)
    gb = bt[bt.method == MODEL]
    coverage = float(((gb.actual >= gb.p10) & (gb.actual <= gb.p90)).mean())
    cal = gb.dropna(subset=["p10_cal"])
    coverage_cal = float(((cal.actual >= cal.p10_cal) & (cal.actual <= cal.p90_cal)).mean())
    margin = conformal_margin(gb.score.to_numpy())

    # Final model on all history, forecasting the four weeks after the last complete week
    T = Y.shape[1]
    models = fit_models(Y, weeks, T)
    pred = predict(models, Y, weeks, T)
    pred["p10"], pred["p90"] = calibrate(pred.p10, pred.p90, pred.scale, margin)
    start = weeks[-1] + pd.Timedelta(weeks=1)
    forecast = pd.DataFrame({
        "product_id": products, "forecast_start": start.date(),
        "forecast_end": (start + pd.Timedelta(weeks=H) - pd.Timedelta(days=1)).date(),
        "p10": pred["p10"].round(1), "p50": pred["p50"].round(1), "p90": pred["p90"].round(1),
        "last_4_weeks": Y[:, T - 4:].sum(1), "last_13_weeks_avg": (Y[:, T - 13:].mean(1) * H).round(1),
    }).merge(load_products(), on="product_id", how="left")
    forecast.to_csv(DEMAND_FORECAST, index=False)

    best_baseline = summary.loc[BASELINES, "wape_product"].idxmin()
    metrics = {
        "horizon_weeks": H,
        "products": int(len(products)),
        "weeks_of_history": int(T),
        "backtest_origins": int(bt.origin.nunique()),
        "backtest_period": [str(bt.origin.min().date()), str(bt.origin.max().date())],
        "methods": {m: {k: round(float(v), 4) for k, v in r.items()} for m, r in summary.iterrows()},
        "best_baseline": best_baseline,
        "interval_coverage_raw": round(coverage, 4),
        "interval_coverage_calibrated": round(coverage_cal, 4),
        "conformal_margin": round(margin, 4),
        "forecast_window": [str(start.date()), str(forecast.forecast_end.iloc[0])],
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "demand_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(summary.round(3).to_string())
    print(f"80% interval coverage: raw {coverage:.1%}, conformal {coverage_cal:.1%}")
    print(f"Forecast for {start.date()} to {forecast.forecast_end.iloc[0]} saved to {DEMAND_FORECAST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- read API
class DemandForecast:
    """Read-only view of the exported forecast, used by the app and the agent."""

    def __init__(self, history: pd.DataFrame, forecast: pd.DataFrame, metrics: dict):
        self.history, self.forecast, self.metrics = history, forecast, metrics

    @classmethod
    def load(cls) -> "DemandForecast":
        forecast = pd.read_csv(DEMAND_FORECAST, dtype={"product_id": str})
        metrics = json.loads((ROOT / "reports" / "demand_metrics.json").read_text())
        return cls(load_demand(), forecast, metrics)

    def find(self, text: str | None = None, category: str | None = None,
             sort_by: str = "p50", limit: int = 20) -> pd.DataFrame:
        f = self.forecast
        if text:
            f = f[f.description.str.contains(text, case=False, regex=False, na=False)
                  | f.product_id.str.fullmatch(text, case=False)]
        if category:
            f = f[f.category.str.lower() == category.lower()]
        sort_by = sort_by if sort_by in f.columns else "p50"
        return f.sort_values(sort_by, ascending=False).head(max(1, min(int(limit), 200)))

    def product_history(self, product_id: str, weeks: int = 104) -> pd.DataFrame:
        h = self.history[self.history.product_id == str(product_id)]
        return h.tail(weeks)[["week_start", "units"]]

    def by_category(self) -> pd.DataFrame:
        f = self.forecast
        return (f.groupby("category")
                 .agg(products=("product_id", "size"), p50=("p50", "sum"), last_4_weeks=("last_4_weeks", "sum"))
                 .assign(change_vs_last_4_weeks=lambda t: t.p50 / t.last_4_weeks - 1)
                 .sort_values("p50", ascending=False))
