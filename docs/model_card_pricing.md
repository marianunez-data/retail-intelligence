# Model card: price elasticity and promotion model

## Overview

| | |
|---|---|
| Task | Estimate how weekly unit sales respond to base price, temporary discounts and promotional support, per product |
| Data | dunnhumby *Breakfast at the Frat*: 524,674 store-product-weeks after cleaning, 55 products, 4 categories, 77 stores, January 2009 to January 2012 (USD) |
| Model | Log-log regression per category with store x product and category x week fixed effects (pyfixest), product-specific base price and discount elasticities, category-level feature, display and shelf-tag effects; standard errors clustered by category-week |
| Pooling | Empirical Bayes shrinkage of product elasticities towards the category mean (DerSimonian-Laird between-product variance) |
| Code | `src/retail_intelligence/pricing.py`, notebook `notebooks/03_pricing_and_promotions.ipynb` |
| Outputs | `data/processed/pricing_products.csv` (per product), `data/processed/pricing_model.json` (effects, validation, settings) |

## Intended use

- Ranking products by price sensitivity and choosing which ones to put into a controlled price test.
- Comparing promotion mechanics (discount depth, circular feature, in-store display) and screening promotions that are unlikely to pay.
- Scenario simulation for one product and one week at chain level.

Not intended for: setting prices automatically without a test; products, stores or price ranges outside the data; categories other than the four modelled.

## Results

| Category | Base price elasticity | Discount elasticity | Circular lift | Display lift |
|---|---|---|---|---|
| Bag snacks | -1.24 | -0.67 | +41% | +41% |
| Cold cereal | -1.35 | -1.74 | +50% | +59% |
| Frozen pizza | -2.91 | -3.25 | +21% | +41% |
| Oral hygiene products | -1.05 | -1.22 | +42% | +20% |

Other findings: discounts under 10% produce no measurable lift; no post-promotion dip (no pantry loading at store-week level); cannibalization of about 20% (cereal) and 10% (pretzels) when all rival products are promoted.

## Validation

Out-of-time: fit on all weeks before 13 July 2011, predict the last 26 weeks (82,947 store-product-weeks).

| Method | WAPE, all weeks | WAPE, promoted weeks | Bias |
|---|---|---|---|
| Store-product average, last 13 weeks | 46.6% | 49.5% | +4.6% |
| Seasonality only (fixed effects, no prices) | 48.2% | 50.3% | +10.3% |
| Seasonality + price and promotions | **34.9%** | **38.6%** | **-0.8%** |

## Simulator assumptions

- Chain-level regular week = average non-promoted store-week in each product's last year, summed over stores.
- Unit cost is not in the data: cost = base price x (1 - assumed gross margin), 30% by default and adjustable.
- Base price changes are limited to +-10% (inside the observed price variation); discounts under 10% get no volume response.
- Recommendations use the Lerner rule (price = cost x e / (1 + e)) under constant elasticity; they are candidates for testing, not decisions.

## Limitations

- Observational data: fixed effects remove permanent store and product differences and weekly shocks, not every reason a price was chosen.
- Product-level elasticities include switching to other products in the same category; category profit needs a cross-price model.
- No supplier funding, no competitor prices, one elasticity per product across all stores.

## Data licence

The source data is (c) dunnhumby, free to use for learning and case studies, all rights reserved. It is downloaded by `python -m retail_intelligence.ingest_pricing` and never committed; only aggregated model outputs are.
