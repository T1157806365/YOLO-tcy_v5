#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-spatial}"
SPLIT="${2:-val}"
DEVICE="${3:-5}"
case "$MODE" in control|spatial|spatial_guard) ;; *) echo 'Usage: bash scripts/val_v5_geohr_100e.sh {control|spatial|spatial_guard} [val|test] [device]' >&2; exit 2;; esac
case "$SPLIT" in val|test) ;; *) echo 'Split must be val or test' >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
W="runs/v5_geohr_spatial_retrain/yolo26m_UAVCB_P2_HVAD_${MODE}_officialPT_100e_seed0/weights/best.pt"
[ -s "$W" ] || { echo "Checkpoint missing: $W" >&2; exit 1; }
python val_rlsfa_gtga_bhlr_v2.py \
  --weights "$W" --split "$SPLIT" --device "$DEVICE" \
  --batch 4 --workers 4 --conf 0.001 --iou 0.70 \
  --save-dir "runs/v5_geohr_spatial_retrain/eval_${MODE}_${SPLIT}"
