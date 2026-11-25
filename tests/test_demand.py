import numpy as np
import pandas as pd
import pytest

from retail_intelligence import demand as dm


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2009-12-07", periods=104, freq="7D")
    season = 1 + 0.8 * np.sin(2 * np.pi * (np.arange(104) - 30) / 52).clip(0)
    level = rng.uniform(5, 50, 40)[:, None]
    Y = rng.poisson(level * season).astype(float)
    return Y, weeks


def test_features_only_use_the_past(series):
    Y, weeks = series
    t0 = 80
    X1, s1 = dm.make_features(Y, weeks, t0)
    Y2 = Y.copy()
    Y2[:, t0:] = 999                      # change the future
    X2, s2 = dm.make_features(Y2, weeks, t0)
    pd.testing.assert_frame_equal(X1, X2)
    assert np.allclose(s1, s2)


def test_last_year_features_missing_in_first_year(series):
    Y, weeks = series
    X, _ = dm.make_features(Y, weeks, 30)
    assert X["last_year_next_4"].isna().all()
    X, _ = dm.make_features(Y, weeks, 80)
    assert X["last_year_next_4"].notna().all()


def test_target_is_next_four_weeks(series):
    Y, _ = series
    assert np.allclose(dm.target(Y, 50), Y[:, 50:54].sum(1))


def test_model_forecasts_are_ordered_and_beat_naive(series):
    Y, weeks = series
    models = dm.fit_models(Y, weeks, 90)
    pred = dm.predict(models, Y, weeks, 90)
    assert (pred.p10 <= pred.p50).all() and (pred.p50 <= pred.p90).all() and (pred.p10 >= 0).all()
    actual = dm.target(Y, 90)
    naive = dm.baseline_forecasts(Y, 90)["Last week x4"]
    assert dm.wape(actual, pred.p50) < dm.wape(actual, naive)


def test_conformal_margin_widens_to_target_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 2000)
    lo, hi, scale = np.full_like(y, -1.0), np.full_like(y, 1.0), np.ones_like(y)   # far too narrow
    m = dm.conformal_margin(dm.conformity_scores(y, lo, hi, scale))
    lo2, hi2 = lo - m, hi + m
    assert ((y >= lo2) & (y <= hi2)).mean() == pytest.approx(0.8, abs=0.03)


def test_wape_and_bias():
    assert dm.wape([10, 10], [12, 8]) == pytest.approx(0.2)
    assert dm.bias([10, 10], [12, 12]) == pytest.approx(0.2)


@pytest.mark.parametrize("units,expected", [
    ([5, 6, 5, 6, 5, 6], "Smooth"),
    ([1, 30, 2, 40, 1, 25], "Erratic"),
    ([5, 0, 0, 5, 0, 0, 5, 0], "Intermittent"),
    ([1, 0, 0, 40, 0, 0, 3, 0], "Lumpy"),
])
def test_demand_classes(units, expected):
    assert dm.demand_class(pd.Series(units, dtype=float)) == expected
