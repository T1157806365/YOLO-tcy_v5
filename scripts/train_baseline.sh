#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
python train_rlsfa_gtga_bhlr_v2.py --cfg configs/experiments/best_baseline.yaml
