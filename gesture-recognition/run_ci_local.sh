#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run CI checks locally (no cloud) - same as ml-validate.yml
# Usage: ./run_ci_local.sh   or   bash run_ci_local.sh
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

echo "CI (local): data validation + config/imports"
echo ""

echo "[1/2] Validate data..."
python validate_data.py

echo ""
echo "[2/2] Config and imports..."
python -c "
from mlops_config import load_config
c = load_config()
assert 'random_seed' in c
print('config OK')
"
python -c "from trainer import build_model; build_model(); print('trainer OK')"
python -c "from export_tflite import convert_to_tflite; print('export_tflite OK')"

echo ""
echo "CI passed (local). Safe to push."
