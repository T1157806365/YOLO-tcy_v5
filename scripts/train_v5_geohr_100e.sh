#!/usr/bin/env bash
# One experiment per invocation. The original project trainer launches DDP as needed.
set -euo pipefail
MODE="${1:-spatial}"
case "$MODE" in control|spatial|spatial_guard) ;; *) echo 'Usage: bash scripts/train_v5_geohr_100e.sh {control|spatial|spatial_guard}' >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="configs/experiments/v5_geohr_spatial_retrain/uavcb_hvad_${MODE}_officialPT_100e.yaml"
[ -f "$CFG" ] || { echo "Missing $CFG" >&2; exit 1; }
[ -f 'configs/weights/yolo26/yolo26m.pt' ] || { echo 'Missing official yolo26m.pt' >&2; exit 1; }
mkdir -p logs pid
LOG="logs/UAVCB_HVAD_${MODE}_officialPT_100e.log"
PID="pid/UAVCB_HVAD_${MODE}_officialPT_100e.pid"
if [ -f "$PID" ]; then
  PREV="$(cat "$PID")"
  if kill -0 "$PREV" 2>/dev/null; then
    echo "Training already appears to be running (PID $PREV). Stop or use a new run name." >&2
    exit 3
  fi
fi
nohup python -u train_rlsfa_gtga_bhlr_v2.py --cfg "$CFG" >"$LOG" 2>&1 &
echo $! > "$PID"
echo "[$MODE] PID=$(cat "$PID") log=$LOG"
echo "Follow: tail -f $LOG"
