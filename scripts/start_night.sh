#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
HOURS="${1:-8}"
OUT="${2:-runs/overnight}"
mkdir -p -- "$OUT"
nohup bash scripts/run_night.sh "$HOURS" "$OUT" > "$OUT/launcher.log" 2>&1 < /dev/null &
NIGHT_PID=$!
echo "$NIGHT_PID" > "$OUT/launcher.pid"
echo "Proceso lanzado: $NIGHT_PID"
echo "Log: $OUT/console.log (errores de arranque: $OUT/launcher.log)"
echo "Consultar: tail -f '$OUT/console.log'"
echo 'Mantén el equipo encendido, conectado y sin suspensión automática.'
