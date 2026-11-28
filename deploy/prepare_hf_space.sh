#!/usr/bin/env bash
# Copies only the files a Hugging Face Docker Space needs into a folder.
#     bash deploy/prepare_hf_space.sh ../my-space
# Then commit and push that folder to the Space repository (see DEPLOY.md).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?Usage: bash deploy/prepare_hf_space.sh <space-folder>}"
mkdir -p "$OUT/data/processed" "$OUT/reports"
cp -r Dockerfile .dockerignore pyproject.toml src app .streamlit "$OUT/"
cp data/processed/*.csv data/processed/product_weekly_demand.parquet data/processed/pricing_model.json "$OUT/data/processed/"
cp reports/demand_metrics.json "$OUT/reports/"
cp deploy/HF_SPACE_README.md "$OUT/README.md"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
echo "Space files ready in $OUT"
