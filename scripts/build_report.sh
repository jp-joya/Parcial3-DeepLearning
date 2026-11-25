#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
LOCAL_TECTONIC="$PWD/.tools/tectonic"
REPORT="${1:-runs/overnight/analysis/report}"
if [[ ! -f "$REPORT/main.tex" ]]; then
  echo "Falta el informe generado: $REPORT/main.tex" >&2
  exit 1
fi
cd -- "$REPORT"
if [[ -x "$LOCAL_TECTONIC" ]]; then
  "$LOCAL_TECTONIC" main.tex
elif command -v tectonic >/dev/null; then
  tectonic main.tex
elif command -v pdflatex >/dev/null; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
else
  echo 'Instala Tectonic/TeX Live o sube analysis/informe_ieee.zip a Overleaf (main.tex).' >&2
  exit 1
fi
if command -v pdfinfo >/dev/null; then
  PAGES="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
  echo "Informe: $PAGES páginas"
  if (( PAGES > 2 )); then
    echo 'El informe supera el máximo de 2 páginas; ajusta el contenido antes de entregar.' >&2
    exit 1
  fi
fi
