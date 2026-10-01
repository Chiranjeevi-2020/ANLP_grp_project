#!/usr/bin/env bash
# Full mid-eval pipeline. Usage: ./run_all.sh [configs/full.json]
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-configs/full.json}"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"
[ -x "$PY" ] || PY=python3
echo "Using $PY ($($PY --version))"
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
"$PY" -m pytest -q tests
"$PY" -m mideval.cli build-data --config "$CFG" --overwrite
"$PY" -m mideval.cli audit-data --config "$CFG"
for s in gen-decoder-data train-decoder build-bank smoke audit-smoke train-rl evaluate audit-final; do
  "$PY" -m mideval.cli "$s" --config "$CFG"
done
echo "Done. See results/mideval/$("$PY" -c "import json;print(json.load(open('$CFG'))['run_name'])")/"
