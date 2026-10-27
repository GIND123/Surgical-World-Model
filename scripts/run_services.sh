#!/usr/bin/env bash
# Start the long-running pipeline as systemd user units so they survive the end of an interactive session.
# Each unit logs to outputs/logs/<unit>.log; inspect with `systemctl --user status swm-*`.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=$ROOT/.venv/bin/python
start() {   # name, command...
  local name=$1; shift
  systemctl --user reset-failed "$name" 2>/dev/null || true
  systemctl --user is-active --quiet "$name" && { echo "$name already running"; return; }
  systemd-run --user --unit="$name" --working-directory="$ROOT" --collect \
    -p StandardOutput=append:"$ROOT/outputs/logs/$name.log" -p StandardError=append:"$ROOT/outputs/logs/$name.log" "$@"
  echo "started $name"
}
mkdir -p outputs/logs
for svc in "${@:-cache encode detect cond ui queue download}"; do
  for s in $svc; do
    case $s in
      cache)    start swm-cache    $PY scripts/data/build_surgvu_cache.py --workers 18 --encoder nvenc ;;
      cachegpu) start swm-cache-gpu $PY scripts/data/build_surgvu_cache.py --workers 4 --encoder nvenc --nvdec --reverse ;;
      encode)   start swm-encode   $PY scripts/data/encode_latents.py --decoders 4 --batch 8 --wait ;;
      detect)   start swm-detect   $PY scripts/data/detect_instruments.py --wait ;;
      cond)     start swm-cond     $PY scripts/data/build_conditioning.py --wait ;;
      ui)       start swm-ui       $PY scripts/data/ui_readout.py --wait ;;
      queue)    start swm-queue    -p OOMPolicy=continue -p MemoryMax=72G $PY scripts/queue/run_queue.py ;;
      download) start swm-download -p Restart=on-failure -p RestartSec=120 $PY scripts/data/download_external.py open_h_srth_chole ;;
    esac
  done
done
