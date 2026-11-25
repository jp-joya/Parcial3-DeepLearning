#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HOURS="${1:-8}"
OUT="${2:-runs/overnight}"
mkdir -p -- "$OUT"
if [[ ! -x .venv/bin/python ]]; then
  uv sync --locked
fi
# tee conserva el log y pipefail propaga fallos o presupuesto agotado (exit 2).
.venv/bin/python -m har.experiment run --config configs/overnight.json --out "$OUT" --hours "$HOURS" 2>&1 | tee -a "$OUT/console.log"
if [[ -x .tools/tectonic ]] || command -v pdflatex >/dev/null || command -v tectonic >/dev/null; then
  bash scripts/build_report.sh "$OUT/analysis/report"
fi
