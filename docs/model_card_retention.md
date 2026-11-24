# Model card: 90-day repurchase model

## Overview

| | |
|---|---|
| Task | Binary classification: will a customer place at least one non-cancelled order in the 90 days after the scoring date? |
| Model | XGBoost classifier, 300 trees, depth 3, learning rate 0.03, subsampling 0.8, monotone constraints on recency (decreasing), frequency, recent orders and spend (increasing) |
| Code | `src/retail_intelligence/modeling.py` (definition), `src/retail_intelligence/train.py` (training and scoring) |
| Training data | Snapshot of 1 September 2011: 5,224 customers, 42.8% repurchased |
| Scoring data | Snapshot of 10 December 2011: 5,852 customers |
| Output | Probability, segment (Loyal 0.60+, Nurture 0.30 to 0.60, At risk below 0.30), value tier, recommended action, SHAP explanation |

## Intended use

- Prioritising retention campaigns for registered business customers of an online gift wholesaler.
- Explaining to account managers why a customer is at risk before a contact.

Not intended for: credit, pricing or any decision that withholds service from a customer; anonymous or guest buyers (they have no customer ID); predicting order value.

## Features

Ten behavioural features computed in dbt from transactions strictly before the cutoff date: days since last purchase, number of orders, net spend, average order value, distinct products, days as customer, return rate, orders in the last 90 days, average days between orders (missing for one-order customers, handled natively), UK flag. Definitions: [data dictionary](data_dictionary.md).

## Evaluation

| Validation | AUC | Average precision | Brier | Top 20% capture | Notes |
|---|---|---|---|---|---|
| Repeated 5-fold CV (15 fits, training part) | 0.802 ± 0.015 | 0.773 | 0.177 | | Used for model selection |
| Stratified 20% holdout | 0.786 | 0.749 | 0.185 | 37.6% (1.88x lift) | Touched once |
| Out-of-time (train Jun 2011, test Sep 2011) | 0.794 | 0.765 | 0.196 | 38.6% | Mean prediction 30.9% vs 42.8% actual |
| Baselines on the holdout | Logistic regression 0.780, recency rule 0.748 | | | | |

The learned models are within cross-validation noise of each other; XGBoost was preferred for monotone constraints, native missing-value handling and exact TreeSHAP explanations.

## Limitations

- **Seasonal calibration.** Ranking is stable across periods, but absolute probabilities follow the season's base rate (33% for a June cutoff, 43% for September in 2011). The production model is trained on the autumn snapshot and scores a winter quarter, so segment sizes are likely optimistic. Use the score primarily as a ranking.
- **Prediction, not causation.** A high score means the customer is likely to buy, not that a campaign will change their behaviour. Measure campaign impact with random control groups.
- **Data period.** Two years of history from one retailer, ending December 2011. Behaviour may differ for other businesses or periods.
- **Registered customers only.** About 23% of raw lines have no customer ID and are excluded.

## Ethical considerations

The model uses purchasing behaviour and a country flag only; no personal or demographic data. The UK flag has almost no influence on scores (lowest mean absolute SHAP value). The recommended use is to offer more help (rewards, offers) to customers, not to deny anything.

## Monitoring and retraining

Rebuild the snapshots and retrain when a new period is fully labeled (`python -m retail_intelligence.features && python -m retail_intelligence.train`). Track feature drift (population stability index), realised repurchase rate per segment, and the AUC on the latest labeled period; retrain or recalibrate when the base rate moves.
