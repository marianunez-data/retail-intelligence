"""Writes a small synthetic data/raw/online_retail_ii.parquet shaped like
Online Retail II, so the full dbt pipeline can run without the real dataset.

    python tests/make_raw_sample.py [project_root]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(root: Path, customers: int = 500, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    rows = []
    for cid in range(12000, 12000 + customers):
        loyal = rng.random() < 0.45
        first = start + pd.Timedelta(days=int(rng.integers(0, 600)))
        last_active = end if loyal else first + pd.Timedelta(days=int(rng.integers(30, 400)))
        for _ in range(int(rng.integers(2, 25 if loyal else 6))):
            day = first + pd.Timedelta(days=int(rng.integers(0, max((min(last_active, end) - first).days, 1))))
            cancel = rng.random() < 0.04
            invoice = ("C" if cancel else "") + str(500000 + len(rows))
            country = "United Kingdom" if rng.random() < 0.9 else "France"
            for _ in range(int(rng.integers(1, 5))):
                qty = int(rng.integers(1, 24)) * (-1 if cancel else 1)
                rows.append([invoice, str(int(rng.integers(20000, 23000))), "ITEM", qty,
                             day + pd.Timedelta(hours=10), round(float(rng.uniform(0.4, 12)), 2), cid, country])
        # data quality issues the staging model must remove
        rows.append(["600000", "POST", "POSTAGE", 1, first, 18.0, cid, "United Kingdom"])
    rows.append(["600001", "22000", "ANONYMOUS", 3, start, 2.5, None, "United Kingdom"])
    rows.append(["600002", "22001", "FREE SAMPLE", 3, start, 0.0, 12000, "United Kingdom"])
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity",
                                     "invoice_date", "price", "customer_id", "country"])
    df = pd.concat([df, df.head(5)])  # exact duplicates
    for col in ["invoice", "stock_code", "description", "country"]:
        df[col] = df[col].astype("string")
    df["customer_id"] = df["customer_id"].astype("Int64")
    out = root / "data" / "raw" / "online_retail_ii.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def make_pricing(root: Path, weeks: int = 70, stores: int = 8, seed: int = 0, elasticity: float = -2.0) -> Path:
    """Synthetic dunnhumby-shaped tables with a known price elasticity, so the
    pricing pipeline can be tested end to end and checked against the truth."""
    rng = np.random.default_rng(seed)
    cats = ["BAG SNACKS", "COLD CEREAL", "FROZEN PIZZA", "ORAL HYGIENE PRODUCTS"]
    products = pd.DataFrame([{"UPC": 1000 + 10 * i + j, "DESCRIPTION": f"PRODUCT {i}{j}", "MANUFACTURER": f"MAKER {j}",
                              "CATEGORY": c, "SUB_CATEGORY": c, "PRODUCT_SIZE": "1 OZ"}
                             for i, c in enumerate(cats) for j in range(3)])
    store_ids = [100 + s for s in range(stores)]
    store_tab = pd.DataFrame({"STORE_ID": store_ids, "STORE_NAME": [f"S{s}" for s in store_ids],
                              "ADDRESS_CITY_NAME": "CITY", "ADDRESS_STATE_PROV_CODE": "TX", "MSA_CODE": 1,
                              "SEG_VALUE_NAME": rng.choice(["UPSCALE", "MAINSTREAM", "VALUE"], stores),
                              "PARKING_SPACE_QTY": np.nan, "SALES_AREA_SIZE_NUM": 50000,
                              "AVG_WEEKLY_BASKETS": 20000.0})
    rows = []
    week_ends = pd.date_range("2009-01-14", periods=weeks, freq="7D")
    for p in products.itertuples():
        base0 = rng.uniform(2, 6)
        for s in store_ids:
            level = rng.uniform(10, 40)
            for w in week_ends:
                base = base0 * rng.choice([0.95, 1.0, 1.05])
                promo = rng.random() < 0.25
                disc = rng.uniform(0.1, 0.35) if promo else 0.0
                feature, display = promo and rng.random() < 0.4, promo and rng.random() < 0.4
                price = round(base * (1 - disc), 2)
                mu = level * (price / base0) ** elasticity * np.exp(0.4 * feature + 0.3 * display)
                units = max(1, rng.poisson(mu))
                rows.append([w, s, p.UPC, units, units, units, round(units * price, 2), price, round(base, 2),
                             int(feature), int(display), int(promo and not (feature or display))])
    tx = pd.DataFrame(rows, columns=["WEEK_END_DATE", "STORE_NUM", "UPC", "UNITS", "VISITS", "HHS", "SPEND",
                                     "PRICE", "BASE_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY"])
    out = root / "data" / "raw" / "dunnhumby"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {"transactions": tx, "products": products, "stores": store_tab}.items():
        df.columns = [c.lower() for c in df.columns]
        df.to_parquet(out / f"{name}.parquet", index=False)
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    print(f"Wrote {make(root)} and {make_pricing(root)}")
