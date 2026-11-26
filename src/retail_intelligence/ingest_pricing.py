"""
Pricing data ingestion: download dunnhumby's "Breakfast at the Frat" source file
and store its three tables as Parquet files that dbt reads as sources.

The data is (c) dunnhumby, all rights reserved, and free to use for learning and
case studies. It is downloaded on demand and never committed to the repository.

    python -m retail_intelligence.ingest_pricing                 # download and convert
    python -m retail_intelligence.ingest_pricing path/to.xlsx    # convert a local copy
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from retail_intelligence.config import PRICING_RAW

URL = ("https://downloads.ctfassets.net/psj0p18eh7z1/7B4xI3FKcgMLak9Wm95IHC/"
       "3b63fe3d37bc7122d13524514be4ab51/dunnhumby_Breakfast-at-the-Frat.zip")
SHEETS = {"dh Transaction Data": "transactions", "dh Products Lookup": "products",
          "dh Store Lookup": "stores"}


def _workbook(path: str | None) -> io.BytesIO | Path:
    if path:
        return Path(path)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=300) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    return io.BytesIO(archive.read(name))


def main(path: str | None = None) -> None:
    book = _workbook(path)
    PRICING_RAW.mkdir(parents=True, exist_ok=True)
    for sheet, name in SHEETS.items():
        df = pd.read_excel(book, sheet_name=sheet, header=1)   # row 0 holds the workbook title
        df.columns = [c.strip().lower() for c in df.columns]
        df.to_parquet(PRICING_RAW / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows -> {PRICING_RAW / name}.parquet")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
