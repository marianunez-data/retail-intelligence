"""
Step 2, features: run the dbt project once per snapshot.

Each snapshot freezes time at a cutoff date: features use only data before it,
and the label is whether the customer ordered in the following 90 days.

    python -m retail_intelligence.features
"""
import json
import shutil
import subprocess
import sys

import pandas as pd

from retail_intelligence.config import DATA_PROCESSED, DBT_DIR, HORIZON_DAYS, SNAPSHOTS


def dbt_build(select: str, variables: dict | None = None) -> None:
    """Run `dbt build` (models, seeds and data tests) for a selection of models."""
    dbt = shutil.which("dbt") or sys.exit("dbt not found. Install with: pip install -e '.[pipeline]'")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cmd = [dbt, "build", "--profiles-dir", ".", "--select", *select.split()]
    if variables:
        cmd += ["--vars", json.dumps(variables)]
    result = subprocess.run(cmd, cwd=DBT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"dbt build failed for '{select}':\n{result.stdout[-3000:]}")


def build_snapshot(name: str, cutoff: str) -> pd.DataFrame:
    dbt_build("+customer_features",
              {"cutoff_date": cutoff, "horizon_days": HORIZON_DAYS, "output_name": name})
    return load_snapshot(name)


def load_snapshot(name: str) -> pd.DataFrame:
    # Sorted so that random splits are reproducible regardless of dbt's write order
    df = pd.read_csv(DATA_PROCESSED / f"{name}.csv", parse_dates=["cutoff_date"])
    return df.sort_values("CustomerID").reset_index(drop=True)


def main() -> None:
    for name, cutoff in SNAPSHOTS.items():
        df = build_snapshot(name, cutoff)
        rate = f"repurchase rate {df['label'].mean():.1%}" if df["label"].any() else "unlabeled (scoring)"
        print(f"{name}: cutoff {cutoff}, {len(df):,} customers, {rate}")


if __name__ == "__main__":
    main()
