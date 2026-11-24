"""
Step 3, training: reproduce the model chosen in the notebook, validate it,
retrain it on the latest labeled snapshot and score today's customers.

    python -m retail_intelligence.train

Outputs
    models/model.joblib     model + everything the app and agent need
    reports/metrics.json    validation metrics (also quoted in the README)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from retail_intelligence.config import (FEATURE_LABELS, FEATURES, HIGH_VALUE_QUANTILE, ID, LOYAL_THRESHOLD,
                              MODEL_PATH, NURTURE_THRESHOLD, OOT_TRAIN_SNAPSHOT, RANDOM_STATE,
                              ROOT, SCORING_SNAPSHOT, SNAPSHOTS, TARGET, TRAIN_SNAPSHOT)
from retail_intelligence.features import load_snapshot
from retail_intelligence.modeling import (calibration_table, evaluate, gains_table, make_logistic,
                                make_xgboost, shap_values)

SEGMENTS = ["Loyal", "Nurture", "At risk"]
ACTIONS = ["Priority win-back", "Win-back offer", "Nurture: cross-sell", "Loyalty reward"]


def assign_actions(scored: pd.DataFrame, high_value_threshold: float) -> pd.DataFrame:
    """Turn a probability into a segment and a next-best action."""
    p = scored["probability"]
    scored["segment"] = np.where(p >= LOYAL_THRESHOLD, "Loyal",
                                 np.where(p >= NURTURE_THRESHOLD, "Nurture", "At risk"))
    hv = scored["monetary"] >= high_value_threshold
    scored["value_tier"] = np.where(hv, "High value", "Standard")
    scored["action"] = np.select(
        [(scored["segment"] == "At risk") & hv, scored["segment"] == "At risk",
         scored["segment"] == "Nurture"],
        ACTIONS[:3], default=ACTIONS[3])
    return scored


def _round(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items()}


def main() -> None:
    labeled = load_snapshot(TRAIN_SNAPSHOT)
    earlier = load_snapshot(OOT_TRAIN_SNAPSHOT)
    current = load_snapshot(SCORING_SNAPSHOT)

    # 1. Holdout validation: 80/20 stratified split of the training snapshot
    train, test = train_test_split(labeled, test_size=0.2, stratify=labeled[TARGET],
                                   random_state=RANDOM_STATE)
    xgb_holdout = make_xgboost().fit(train[FEATURES], train[TARGET])
    p_test = xgb_holdout.predict_proba(test[FEATURES])[:, 1]
    logit = make_logistic().fit(train[FEATURES], train[TARGET])
    p_logit = logit.predict_proba(test[FEATURES])[:, 1]
    recency_rule = -test["recency_days"].to_numpy()  # "most recent first" heuristic

    # 2. Out-of-time validation: train on the June snapshot, test on September
    xgb_oot = make_xgboost().fit(earlier[FEATURES], earlier[TARGET])
    p_oot = xgb_oot.predict_proba(labeled[FEATURES])[:, 1]

    # 3. Production model: all labeled customers of the latest snapshot
    model = make_xgboost().fit(labeled[FEATURES], labeled[TARGET])

    # 4. Score today's customers and explain the scores
    scored = current[[ID] + FEATURES].copy()
    scored["probability"] = model.predict_proba(scored[FEATURES])[:, 1]
    high_value_t = float(scored["monetary"].quantile(HIGH_VALUE_QUANTILE))
    scored = assign_actions(scored, high_value_t).sort_values("probability", ascending=False)

    contrib = shap_values(model, scored)
    importance = (contrib[FEATURES].abs().mean().rename("mean_abs_shap").reset_index()
                  .rename(columns={"index": "feature"}))
    importance["label"] = importance["feature"].map(FEATURE_LABELS)
    importance = importance.sort_values("mean_abs_shap", ascending=False)

    holdout = evaluate(test[TARGET], p_test)
    metrics = {
        "model": "XGBoost, depth 3, 300 trees, monotone constraints",
        "training_snapshot": {"name": TRAIN_SNAPSHOT, "cutoff": SNAPSHOTS[TRAIN_SNAPSHOT],
                              "customers": int(len(labeled)),
                              "repurchase_rate": round(float(labeled[TARGET].mean()), 4)},
        "scoring_snapshot": {"name": SCORING_SNAPSHOT, "cutoff": SNAPSHOTS[SCORING_SNAPSHOT],
                             "customers": int(len(scored))},
        "holdout": _round(holdout),
        "holdout_logistic": _round(evaluate(test[TARGET], p_logit)),
        "holdout_recency_rule": {"auc": round(float(roc_auc_score(test[TARGET], recency_rule)), 4)},
        "out_of_time": _round(evaluate(labeled[TARGET], p_oot)),
        "thresholds": {"loyal": LOYAL_THRESHOLD, "nurture": NURTURE_THRESHOLD,
                       "high_value_spend": round(high_value_t, 2)},
        "segments": scored["segment"].value_counts().reindex(SEGMENTS, fill_value=0).astype(int).to_dict(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "metrics": metrics,
        "gains": gains_table(test[TARGET], p_test),
        "calibration": calibration_table(test[TARGET], p_test),
        "importance": importance,
        "scored": scored.reset_index(drop=True),
    }, MODEL_PATH)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    h, o = metrics["holdout"], metrics["out_of_time"]
    print(f"Holdout      AUC {h['auc']:.3f} | AP {h['average_precision']:.3f} | Brier {h['brier']:.3f} | "
          f"top 20% captures {h['capture_top20']:.1%} (lift {h['lift_top20']:.2f})")
    print(f"Out-of-time  AUC {o['auc']:.3f} | mean prediction {o['mean_prediction']:.3f} vs actual {o['base_rate']:.3f}")
    print(f"Scored {len(scored):,} customers: {metrics['segments']}")
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} and reports/metrics.json")


if __name__ == "__main__":
    main()
