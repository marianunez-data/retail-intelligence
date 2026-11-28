"""End-to-end pipelines on synthetic raw data: dbt builds with all data tests
(including the label-leakage and demand-grid tests), then training, scoring
and pricing estimation."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from make_raw_sample import make, make_pricing

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("dbt") is None, reason="dbt-duckdb not installed")


def run(args, root, cwd=None):
    env = {**os.environ, "RETAIL_HOME": str(root)}
    return subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("project")
    shutil.copytree(REPO / "dbt", tmp / "dbt", ignore=shutil.ignore_patterns("target", "logs", "*.duckdb"))
    make(tmp)
    make_pricing(tmp)
    return tmp


def test_retention_pipeline(root):
    features = run([sys.executable, "-m", "retail_intelligence.features"], root)
    assert features.returncode == 0, features.stdout + features.stderr
    snap = pd.read_csv(root / "data" / "processed" / "features_2011_09_01.csv")
    assert snap["CustomerID"].is_unique
    assert 0 < snap["label"].mean() < 1
    assert (snap["monetary"] > 0).all()

    trained = run([sys.executable, "-m", "retail_intelligence.train"], root)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    metrics = json.loads((root / "reports" / "metrics.json").read_text())
    assert 0.5 < metrics["holdout"]["auc"] <= 1


def test_demand_models_build_with_tests(root):
    dbt = shutil.which("dbt")
    result = run([dbt, "build", "--profiles-dir", ".", "--select", "+product_weekly_demand", "+dim_products",
                  "--vars", '{"min_active_weeks": 5}'], root, cwd=root / "dbt")
    assert result.returncode == 0, result.stdout[-3000:]
    grid = pd.read_parquet(root / "data" / "processed" / "product_weekly_demand.parquet")
    weeks = grid.groupby("product_id").week_start.nunique()
    assert weeks.nunique() == 1 and len(grid) == len(weeks) * weeks.iloc[0]


@pytest.mark.skipif(pytest.importorskip("pyfixest") is None, reason="pyfixest not installed")
def test_pricing_pipeline_recovers_true_elasticity(root):
    result = run([sys.executable, "-m", "retail_intelligence.pricing"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    products = pd.read_csv(root / "data" / "processed" / "pricing_products.csv")
    model = json.loads((root / "data" / "processed" / "pricing_model.json").read_text())
    # the synthetic data was generated with an elasticity of -2 for every product
    assert products.elasticity_base_pooled.between(-2.6, -1.4).all()
    assert set(model["promo_effects"]) == {"Bag snacks", "Cold cereal", "Frozen pizza", "Oral hygiene products"}
