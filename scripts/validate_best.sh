#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
python val_rlsfa_gtga_bhlr_v2.py \
  --weights "runs/rlsfa_gtga_bhlr_gmls_v2/yolo26m_UAVCB_GMLSv2_DMLS-Correction_Fine_Scale_BHLRv2-P3_ADD_rgb1920_sem640_tir640_seed0_scratch_bs32_r6_bhlrnew_newbest/weights/best.pt" \
  --rgb configs/datasets/UAV-CB-RGB.yaml \
  --tir configs/datasets/UAV-CB-TIR.yaml \
  --split val --device "${1:-3}" --batch 4 --workers 4 \
  --conf 0.001 --iou 0.70 --max-det 300 \
  --save-dir runs/v5_validation
