"""
Price elasticity, promotion effects and a price and promotion simulator (notebook 03),
on dunnhumby's "Breakfast at the Frat" store-week panel.

Model, estimated separately for each category with two-way fixed effects:

    log(units) = b_base[product] * log(base price)
               + b_promo[product] * log(shelf price / base price)
               + feature + display + shelf-tag-only
               + store x product effect + category x week effect + error

The store x product effect absorbs everything permanent about a product in a store
(location, shelf space, local taste); the category x week effect absorbs seasonality,
holidays and anything that moves the whole category in a week. What is left to
identify the price coefficients is price variation across stores and over time
relative to the rest of the category.

    python -m retail_intelligence.pricing         # dbt build, estimate, validate, export

Outputs (aggregates only, safe to commit)
    data/processed/pricing_products.csv   elasticities, baselines and recommendations per product
    data/processed/pricing_model.json     promotion effects, validation, settings
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from retail_intelligence.config import (DEFAULT_GROSS_MARGIN, PRICING_MART, PRICING_MODEL, PRICING_PRODUCTS,
                              ROOT)

PROMO_TERMS = ["is_feature", "is_display", "is_tpr_only"]
PROMO_LABELS = {"is_feature": "Featured in circular", "is_display": "In-store display",
                "is_tpr_only": "Shelf tag only (beyond the price cut)"}
MIN_EFFECTIVE_DISCOUNT = 0.10   # discounts below 10% showed no measurable lift (notebook 03, section 7)
PRICE_RANGE = 0.10          # simulations stay within +-10% of today's base price, well inside observed variation
HOLDOUT_WEEKS = 26


# ------------------------------------------------------------------ data
def load_panel() -> pd.DataFrame:
    d = pd.read_parquet(PRICING_MART)
    d["week_end"] = pd.to_datetime(d["week_end"])
    return add_model_columns(d)


def add_model_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_units"] = np.log(d["units"])
    d["log_base_price"] = np.log(d["base_price"])
    d["log_price_ratio"] = np.log(d["price"] / d["base_price"]).clip(upper=0)   # 0 when not discounted
    for c in PROMO_TERMS + ["is_promo"]:
        d[c] = d[c].astype(int)
    d["store_product"] = d["upc"] + "_" + d["store_id"].astype(str)
    d["category_week"] = d["category"] + "_" + d["week_end"].dt.strftime("%Y-%m-%d")
    d["category_week_of_year"] = d["category"] + "_" + d["week_end"].dt.isocalendar().week.astype(str)
    d["category_year"] = d["category"] + "_" + d["week_end"].dt.year.astype(str)
    return d


# ------------------------------------------------------------- estimation
FORMULA = "log_units ~ i(upc, log_base_price) + i(upc, log_price_ratio) + {promo} | {fe}"


def _feols(formula: str, data: pd.DataFrame, cluster: str | None = "category_week"):
    import pyfixest as pf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vcov = {"CRV1": cluster} if cluster else "iid"
        return pf.feols(formula, data=data, vcov=vcov)


def estimate(d: pd.DataFrame, fe: str = "store_product + category_week") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product elasticities and category promotion effects, one regression per category.
    Standard errors are clustered by category-week, since promotions are planned by week."""
    products, promos = [], []
    formula = FORMULA.format(promo=" + ".join(PROMO_TERMS), fe=fe)
    for category, g in d.groupby("category"):
        tidy = _feols(formula, g).tidy()
        for name, r in tidy.iterrows():
            if name.startswith("upc::"):
                upc, var = name.split("::")[1].split(":")
                products.append({"category": category, "upc": upc, "term": var,
                                 "estimate": r["Estimate"], "se": r["Std. Error"]})
            else:
                promos.append({"category": category, "term": name, "estimate": r["Estimate"],
                               "se": r["Std. Error"], "lift": np.exp(r["Estimate"]) - 1})
    est = (pd.DataFrame(products)
             .pivot_table(index=["category", "upc"], columns="term", values=["estimate", "se"]))
    est.columns = [f"{'elasticity' if a == 'estimate' else 'se'}_{'base' if b == 'log_base_price' else 'promo'}"
                   for a, b in est.columns]
    return est.reset_index(), pd.DataFrame(promos)


def shrink(estimate: pd.Series, se: pd.Series) -> tuple[pd.Series, float, float]:
    """Empirical Bayes partial pooling towards the precision-weighted category mean.
    Noisy product estimates move most; precise ones barely move. Between-product
    variance tau^2 uses the DerSimonian-Laird estimator."""
    w = 1 / se ** 2
    mean = float((w * estimate).sum() / w.sum())
    q = float((w * (estimate - mean) ** 2).sum())
    tau2 = max(0.0, (q - (len(estimate) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    weight = tau2 / (tau2 + se ** 2)
    return weight * estimate + (1 - weight) * mean, mean, tau2


def pooled_elasticities(est: pd.DataFrame) -> pd.DataFrame:
    out = est.copy()
    for kind in ["base", "promo"]:
        parts = []
        for _, g in out.groupby("category"):
            s, _, _ = shrink(g[f"elasticity_{kind}"], g[f"se_{kind}"])
            parts.append(s)
        out[f"elasticity_{kind}_pooled"] = pd.concat(parts)
    return out


# ------------------------------------------------------------- validation
def validate(d: pd.DataFrame) -> dict:
    """Out-of-time test: fit on all but the last HOLDOUT_WEEKS weeks, predict the holdout.

    Week effects for future weeks are unknown, so the predictive version replaces the
    category x week effect with category x week-of-year and category x year effects."""
    cut = np.sort(d["week_end"].unique())[-HOLDOUT_WEEKS]
    train, test = d[d.week_end < cut], d[d.week_end >= cut].copy()
    recent = train[train.week_end >= np.sort(train.week_end.unique())[-13]]
    test["Store-product average, last 13 weeks"] = test.store_product.map(recent.groupby("store_product").units.mean())
    fe = "store_product + category_week_of_year + category_year"
    specs = {"Seasonality only": f"log_units ~ 1 | {fe}",
             "Seasonality + price and promotions": FORMULA.format(promo=" + ".join(PROMO_TERMS), fe=fe)}
    for name, formula in specs.items():
        preds = []
        for _, g in train.groupby("category"):
            m = _feols(formula, g, cluster=None)
            smear = float(np.exp(m.resid()).mean())          # Duan smearing: back from logs to units
            h = test[test.category == g.category.iloc[0]]
            with warnings.catch_warnings():                  # unseen store-products predict as NaN
                warnings.simplefilter("ignore")
                preds.append(pd.Series(np.exp(m.predict(newdata=h)) * smear, index=h.index))
        test[name] = pd.concat(preds)
    methods = ["Store-product average, last 13 weeks"] + list(specs)
    ok = test.dropna(subset=methods)

    def score(frame):
        return {m: {"wape": float(np.abs(frame.units - frame[m]).sum() / frame.units.sum()),
                    "bias": float((frame[m].sum() - frame.units.sum()) / frame.units.sum())} for m in methods}

    return {"holdout_start": str(pd.Timestamp(cut).date()), "rows": int(len(ok)),
            "rows_without_history": int(len(test) - len(ok)),
            "all_weeks": score(ok), "promoted_weeks": score(ok[ok.is_promo == 1])}


# ------------------------------------------------------------- baselines
def product_baselines(d: pd.DataFrame, weeks: int = 52) -> pd.DataFrame:
    """Chain-level starting point for simulations, from the last year of data:
    weekly units across all stores in weeks without promotion, today's base price,
    and how the product has been promoted."""
    last_week = d.groupby("upc").week_end.transform("max")
    last = d[d.week_end > last_week - pd.Timedelta(weeks=weeks)]           # each product's last year on sale
    recent = d[d.week_end > last_week - pd.Timedelta(weeks=13)]
    # regular weekly units across the chain = average non-promoted store-week, summed over stores
    per_store = last[last.is_promo == 0].groupby(["upc", "store_id"]).units.mean()
    regular = per_store.groupby("upc").sum()
    attrs = d.groupby("upc")[["description", "manufacturer", "category", "sub_category"]].first()
    promoted = last[last.is_promo == 1]
    return attrs.assign(
        base_price=recent.groupby("upc").base_price.median(),
        stores=last.groupby("upc").store_id.nunique(),
        weekly_units_regular=regular,
        promo_frequency=last.groupby("upc").is_promo.mean(),
        avg_discount_when_promoted=promoted.groupby("upc").discount_depth.mean(),
        weekly_revenue=last.groupby("upc").spend.sum() / last.week_end.nunique(),
    ).reset_index()


# -------------------------------------------------------------- simulator
def simulate(product: pd.Series | dict, promo_effects: dict, base_price_change: float = 0.0,
             discount: float = 0.0, feature: bool = False, display: bool = False,
             gross_margin: float = DEFAULT_GROSS_MARGIN) -> dict:
    """Weekly chain-level units, revenue and gross profit for one scenario, against
    today's regular week (current base price, no promotion).

    Unit cost is unknown, so it is derived from an assumed gross margin at today's
    base price: cost = base price x (1 - margin). Discounts under 10% get no volume
    response, as measured in the data, so they only cost margin."""
    p = pd.Series(product)
    change = float(np.clip(base_price_change, -PRICE_RANGE, PRICE_RANGE))
    p0, q0 = float(p.base_price), float(p.weekly_units_regular)
    cost = p0 * (1 - gross_margin)
    new_base = p0 * (1 + change)
    shelf = new_base * (1 - discount)
    cat = promo_effects[p.category]
    effective = discount >= MIN_EFFECTIVE_DISCOUNT          # shoppers ignore cuts under 10%
    lift = np.exp(cat["is_feature"] * feature + cat["is_display"] * display
                  + cat["is_tpr_only"] * (effective and not (feature or display)))
    discount_response = (1 - discount) ** p.elasticity_promo_pooled if effective else 1.0
    units = q0 * (1 + change) ** p.elasticity_base_pooled * discount_response * lift
    out = {"base_price": new_base, "shelf_price": shelf, "units": units, "revenue": units * shelf,
           "gross_profit": units * (shelf - cost)}
    base = {"units": q0, "revenue": q0 * p0, "gross_profit": q0 * (p0 - cost)}
    for k, v in base.items():
        out[f"{k}_change"] = out[k] / v - 1 if v else np.nan
        out[f"{k}_baseline"] = v
    return out


def recommend(product: pd.Series, gross_margin: float = DEFAULT_GROSS_MARGIN) -> tuple[float, str]:
    """Profit-maximising base price change under constant elasticity (Lerner rule:
    price = cost x e / (1 + e)), kept within the +-10% range the data supports.

    A small price increase raises profit when |elasticity| < 1 / margin, so the
    answer depends on the margin; the app lets the user change it."""
    e = float(product.elasticity_base_pooled)
    if e >= -1:
        return PRICE_RANGE, "Test an increase: demand is inelastic"
    target = (1 - gross_margin) * e / (1 + e) - 1
    change = float(np.clip(target, -PRICE_RANGE, PRICE_RANGE))
    if change > 0.02:
        return change, "Test an increase: margin gained outweighs volume lost"
    if change < -0.02:
        return change, "Test a decrease: extra volume outweighs margin given up"
    return change, "Hold: close to the profit-maximising price"


class PricingModel:
    """Read-only view of the exported pricing results, used by the app and the agent."""

    def __init__(self, products: pd.DataFrame, model: dict):
        self.products = products
        self.model = model
        self.promo_effects = {c: {t: v["estimate"] for t, v in terms.items()}
                              for c, terms in model["promo_effects"].items()}

    @classmethod
    def load(cls) -> "PricingModel":
        return cls(pd.read_csv(PRICING_PRODUCTS, dtype={"upc": str}), json.loads(PRICING_MODEL.read_text()))

    def product(self, upc: str) -> pd.Series | None:
        hit = self.products[self.products.upc == str(upc)]
        return None if hit.empty else hit.iloc[0]

    def find(self, text: str) -> pd.DataFrame:
        mask = self.products.description.str.contains(text, case=False, regex=False) | \
            self.products.manufacturer.str.contains(text, case=False, regex=False) | \
            self.products.category.str.contains(text, case=False, regex=False)
        return self.products[mask]

    def simulate(self, upc: str, **kwargs) -> dict:
        return simulate(self.product(upc), self.promo_effects, **kwargs)


# ------------------------------------------------------------------- main
def main() -> None:
    from retail_intelligence.features import dbt_build

    dbt_build("+pricing_store_weeks")
    d = load_panel()
    est, promos = estimate(d)
    est = pooled_elasticities(est)
    validation = validate(d)

    products = product_baselines(d).merge(est.drop(columns="category"), on="upc", how="inner")
    rec = products.apply(lambda r: recommend(r), axis=1, result_type="expand")
    products["recommended_change"], products["recommendation"] = rec[0], rec[1]
    products = products.sort_values(["category", "manufacturer", "description"])
    products.round(4).to_csv(PRICING_PRODUCTS, index=False)

    pooled = {}
    for c, g in est.groupby("category"):
        _, mean_b, tau_b = shrink(g.elasticity_base, g.se_base)
        _, mean_p, tau_p = shrink(g.elasticity_promo, g.se_promo)
        pooled[c] = {"elasticity_base": round(mean_b, 4), "elasticity_promo": round(mean_p, 4),
                     "between_product_sd_base": round(float(np.sqrt(tau_b)), 4)}
    model = {
        "data": {"rows": int(len(d)), "stores": int(d.store_id.nunique()), "products": int(d.upc.nunique()),
                 "weeks": int(d.week_end.nunique()),
                 "period": [str(d.week_end.min().date()), str(d.week_end.max().date())]},
        "category_elasticities": pooled,
        "promo_effects": {c: {r.term: {"estimate": round(r.estimate, 4), "se": round(r.se, 4),
                                       "lift": round(r.lift, 4)} for r in g.itertuples()}
                          for c, g in promos.groupby("category")},
        "validation": validation,
        "settings": {"default_gross_margin": DEFAULT_GROSS_MARGIN, "price_range": PRICE_RANGE},
    }
    PRICING_MODEL.write_text(json.dumps(model, indent=2))

    print(pd.DataFrame(pooled).T.to_string())
    v = validation["all_weeks"]
    print("Holdout WAPE: " + ", ".join(f"{k} {m['wape']:.1%}" for k, m in v.items()))
    print(f"Saved {PRICING_PRODUCTS.relative_to(ROOT)} and {PRICING_MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
