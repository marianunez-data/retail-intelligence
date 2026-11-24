import numpy as np
import pandas as pd
import pytest

from retail_intelligence.engine import ACTIONS, SEGMENTS, Engine, SQLGuardError


def test_every_customer_scored_and_valid(engine):
    s = engine.scored
    assert s["probability"].between(0, 1).all()
    assert set(s["segment"]) <= set(SEGMENTS)
    assert set(s["action"]) <= set(ACTIONS)


def test_segments_follow_thresholds(engine):
    s = engine.scored
    assert (s.loc[s.segment == "Loyal", "probability"] >= engine.loyal_t).all()
    assert (s.loc[s.segment == "At risk", "probability"] < engine.nurture_t).all()


def test_priority_winback_is_high_value_and_at_risk(engine):
    pw = engine.scored[engine.scored.action == "Priority win-back"]
    assert (pw.segment == "At risk").all() and (pw.value_tier == "High value").all()


@pytest.mark.parametrize("feature,better", [("recency_days", 1), ("frequency", 200),
                                            ("orders_last_90d", 20), ("monetary", 1e6)])
def test_monotone_constraints_hold(engine, feature, better):
    """More recent, more frequent, higher spending never lowers the score."""
    rows = engine.scored[engine.features].head(50).copy()
    base = engine.score(rows)["probability"].sort_index()
    improved = rows.assign(**{feature: better})
    assert (engine.score(improved)["probability"].sort_index() >= base - 1e-9).all()


def test_missing_gap_for_one_order_customers_is_scored(engine):
    row = engine.scored[engine.features].iloc[[0]].assign(frequency=1, avg_days_between_orders=np.nan)
    assert 0 <= engine.score(row).iloc[0]["probability"] <= 1


def test_shap_explanation_adds_up_to_the_score(engine):
    row = engine.scored.iloc[3]
    exp, base = engine.explain(row)
    assert set(exp["feature"]) == set(engine.features)
    logit = np.log(base / (1 - base)) + exp["shap"].sum()
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(row["probability"], abs=1e-4)


def test_find_filters(engine):
    df = engine.find(segment="At risk", min_spend=500, limit=10)
    assert len(df) <= 10
    assert (df.segment == "At risk").all() and (df.monetary >= 500).all()


def test_summary_counts_add_up(engine):
    s = engine.summary()
    assert sum(s["segments"].values()) == s["customers"] == len(engine.scored)


def test_sql_aggregation(engine):
    df = engine.sql("SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment")
    assert df["n"].sum() == len(engine.scored)


def test_sql_row_limit(engine):
    assert len(engine.sql("SELECT * FROM customers")) <= 200


@pytest.mark.parametrize("q", [
    "DROP TABLE customers",
    "DELETE FROM customers",
    "SELECT 1; DROP TABLE customers",
    "SELECT * FROM read_csv('/etc/passwd')",
    "INSTALL httpfs",
    "COPY customers TO 'out.csv'",
    "",
])
def test_sql_guard_rejects(q):
    with pytest.raises(SQLGuardError):
        Engine.guard_sql(q)


def test_database_cannot_read_files_even_if_guard_bypassed(engine):
    with pytest.raises(Exception):
        engine._db.execute("SELECT * FROM '/etc/passwd'").df()
