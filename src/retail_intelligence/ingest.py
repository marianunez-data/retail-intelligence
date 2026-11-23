"""
Step 1, ingestion: read the two yearly sheets of the UCI Online Retail II
Excel file and store them as one typed Parquet file that dbt reads as its source.

    python -m retail_intelligence.ingest            # uses data/raw/online_retail_II.xlsx
    python -m retail_intelligence.ingest path.xlsx
"""
import sys
from pathlib import Path

import pandas as pd

from retail_intelligence.config import RAW_EXCEL, RAW_PARQUET

COLUMNS = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
    "Customer ID": "customer_id", "Country": "country",
}


def load_excel(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True).rename(columns=COLUMNS)
    # Invoice and stock codes mix numbers and text ("C489449" is a cancellation)
    for col in ["invoice", "stock_code", "description", "country"]:
        df[col] = df[col].astype("string")
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df[list(COLUMNS.values())]


def main(path: str | None = None) -> None:
    src = Path(path) if path else RAW_EXCEL
    if not src.exists():
        sys.exit(f"Raw file not found: {src}\nDownload it from "
                 "https://archive.ics.uci.edu/dataset/502/online+retail+ii into data/raw/")
    df = load_excel(src)
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET, index=False)
    print(f"Ingested {len(df):,} rows "
          f"({df['invoice_date'].min():%Y-%m-%d} to {df['invoice_date'].max():%Y-%m-%d}) -> {RAW_PARQUET}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
