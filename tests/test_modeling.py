import numpy as np
import pytest

from retail_intelligence.modeling import calibration_table, capture_at, evaluate, gains_table


def test_capture_perfect_and_random_ranking():
    y = np.array([1] * 20 + [0] * 80)
    assert capture_at(y, -np.arange(100), 0.2) == 1.0          # perfect ranking
    assert capture_at(y, np.arange(100), 0.2) == 0.0           # worst ranking


def test_evaluate_keys_and_lift():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    m = evaluate(y, np.clip(y * 0.6 + rng.uniform(0, 0.4, 500), 0, 1))
    assert m["auc"] > 0.9
    assert m["lift_top20"] == pytest.approx(m["capture_top20"] / 0.2)


def test_gains_curve_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(1)
    g = gains_table(rng.integers(0, 2, 300), rng.random(300))
    assert g["pct_captured"].is_monotonic_increasing
    assert g["pct_captured"].iloc[-1] == pytest.approx(1.0)


def test_calibration_table_has_ten_bins():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    t = calibration_table((rng.random(1000) < p).astype(int), p)
    assert len(t) == 10 and t["customers"].sum() == 1000
