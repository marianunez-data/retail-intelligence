"""
Model definitions and evaluation helpers, shared by the notebook and train.py
so the numbers in the analysis and in production come from the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from retail_intelligence.config import FEATURES, RANDOM_STATE

# Business knowledge encoded as constraints: all else equal, a more recent,
# more frequent, higher spending customer should never look less likely to return.
MONOTONE = {"recency_days": -1, "frequency": 1, "orders_last_90d": 1, "monetary": 1}


def _log1p_clip(x):
    return np.log1p(np.clip(x, 0, None))


def make_logistic() -> object:
    """Interpretable baseline: log-transformed, scaled features into a logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_log1p_clip),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def make_xgboost(monotone: bool = True, **overrides) -> XGBClassifier:
    """Production model: shallow, regularised gradient boosting.
    Missing values (customers with a single order have no inter-order gap)
    are handled natively by XGBoost."""
    params = dict(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=2,
    )
    if monotone:
        params["monotone_constraints"] = tuple(MONOTONE.get(f, 0) for f in FEATURES)
    params.update(overrides)
    return XGBClassifier(**params)


def capture_at(y_true, y_prob, share: float = 0.2) -> float:
    """Share of all actual repurchasers found in the top `share` of customers by score."""
    order = np.argsort(-np.asarray(y_prob))
    y = np.asarray(y_true)[order]
    k = max(1, int(round(len(y) * share)))
    return float(y[:k].sum() / max(y.sum(), 1))


def evaluate(y_true, y_prob) -> dict:
    """Ranking, probability quality and business metrics in one place."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    cap = capture_at(y_true, y_prob, 0.2)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "capture_top20": cap,
        "lift_top20": cap / 0.2,
        "base_rate": float(y_true.mean()),
        "mean_prediction": float(y_prob.mean()),
    }


def gains_table(y_true, y_prob, points: int = 51) -> pd.DataFrame:
    """Cumulative gains: share of repurchasers captured vs share of customers contacted."""
    order = np.argsort(-np.asarray(y_prob))
    y = np.asarray(y_true)[order]
    contacted = np.arange(1, len(y) + 1) / len(y)
    captured = np.cumsum(y) / max(y.sum(), 1)
    idx = np.linspace(0, len(y) - 1, points).astype(int)
    out = pd.DataFrame({"pct_contacted": contacted[idx], "pct_captured": captured[idx]})
    return pd.concat([pd.DataFrame({"pct_contacted": [0.0], "pct_captured": [0.0]}), out],
                     ignore_index=True)


def calibration_table(y_true, y_prob, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed repurchase rate by score decile."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    return (df.groupby("bin")
              .agg(predicted=("p", "mean"), observed=("y", "mean"), customers=("y", "size"))
              .reset_index(drop=True))


def shap_values(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions computed by XGBoost itself (log-odds scale).
    The last column of XGBoost's output is the bias term, returned as `base_value`."""
    import xgboost as xgb

    contrib = model.get_booster().predict(xgb.DMatrix(X[FEATURES]), pred_contribs=True)
    out = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)
    out["base_value"] = contrib[:, -1]
    return out
