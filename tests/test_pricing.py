import numpy as np
import pandas as pd
import pytest

from retail_intelligence import pricing as pr

EFFECTS = {"Cold cereal": {"is_feature": 0.4, "is_display": 0.5, "is_tpr_only": -0.05}}
PRODUCT = pd.Series({"category": "Cold cereal", "base_price": 3.0, "weekly_units_regular": 1000.0,
                     "elasticity_base_pooled": -2.0, "elasticity_promo_pooled": -2.5})


def test_regular_week_is_the_baseline():
    r = pr.simulate(PRODUCT, EFFECTS)
    assert r["units"] == pytest.approx(1000) and r["gross_profit_change"] == pytest.approx(0)


def test_price_increase_follows_elasticity():
    r = pr.simulate(PRODUCT, EFFECTS, base_price_change=0.05)
    assert r["units"] == pytest.approx(1000 * 1.05 ** -2.0)
    assert r["units_change"] < 0 < r["gross_profit_change"]          # |e| < 1/margin, so profit rises


def test_price_change_is_capped():
    assert pr.simulate(PRODUCT, EFFECTS, base_price_change=0.5)["base_price"] == pytest.approx(3.0 * 1.10)


def test_small_discounts_get_no_volume():
    r = pr.simulate(PRODUCT, EFFECTS, discount=0.05)
    assert r["units"] == pytest.approx(1000) and r["gross_profit_change"] < 0


def test_support_adds_lift():
    plain = pr.simulate(PRODUCT, EFFECTS, discount=0.2)
    supported = pr.simulate(PRODUCT, EFFECTS, discount=0.2, feature=True, display=True)
    assert supported["units"] == pytest.approx(1000 * 0.8 ** -2.5 * np.exp(0.9))
    assert supported["units"] > plain["units"]


def test_recommendation_follows_margin():
    elastic = PRODUCT.copy(); elastic["elasticity_base_pooled"] = -4.0
    assert pr.recommend(elastic, 0.5)[0] < 0                          # lower when |e| > 1/margin
    assert pr.recommend(elastic, 0.2)[0] > 0
    inelastic = PRODUCT.copy(); inelastic["elasticity_base_pooled"] = -0.5
    assert pr.recommend(inelastic, 0.3)[0] == pytest.approx(pr.PRICE_RANGE)


def test_shrinkage_moves_noisy_estimates_most():
    est = pd.Series([-1.0, -1.2, -5.0])
    se = pd.Series([0.05, 0.05, 1.5])
    pooled, mean, tau2 = pr.shrink(est, se)
    assert abs(pooled[2] - est[2]) > abs(pooled[0] - est[0])
    assert pooled[2] > est[2]
