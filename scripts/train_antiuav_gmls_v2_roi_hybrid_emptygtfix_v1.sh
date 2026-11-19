#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
cd "$ROOT"

MODE="${1:-amp}"
case "$MODE" in
  amp)
    CFG=configs/experiments/gmls_v2_roi/antiuav_gmls_v2_roi_single_head_hybrid_officialpt_top2_100e_emptygtfix_v1.yaml
    LOG=logs/AntiUAV_GMLS_Hybrid_EmptyGTFixV1_AMP.log
    PID=pid/AntiUAV_GMLS_Hybrid_EmptyGTFixV1_AMP.pid
    ;;
  fp32)
    CFG=configs/experiments/gmls_v2_roi/antiuav_gmls_v2_roi_single_head_hybrid_officialpt_top2_100e_emptygtfix_v1_fp32.yaml
    LOG=logs/AntiUAV_GMLS_Hybrid_EmptyGTFixV1_FP32.log
    PID=pid/AntiUAV_GMLS_Hybrid_EmptyGTFixV1_FP32.pid
    ;;
  *) echo "Usage: bash scripts/train_antiuav_gmls_v2_roi_hybrid_emptygtfix_v1.sh [amp|fp32]" >&2; exit 2;;
esac

[[ -f "$CFG" ]] || { echo "Missing YAML: $CFG" >&2; exit 1; }
[[ -f gmls_roi_onehead_v2/runner.py && -f gmls_roi_onehead_v2/model.py ]] || { echo "Missing gmls_roi_onehead_v2 module; recover it before training." >&2; exit 1; }
[[ -f gmls_roi_single_head/model.py ]] || { echo "Missing gmls_roi_single_head/model.py" >&2; exit 1; }
if ! grep -Eq 'GENERAL_EMPTY_GT_GRAD_FIX_V1|EMPTY_GT_GRAD_FIX_V1' gmls_roi_single_head/model.py; then
  echo 'Missing Empty-GT gradient fix in gmls_roi_single_head/model.py.' >&2
  echo 'Run: python integrate_empty_gt_all_datasets.py --target "$PWD"' >&2
  exit 1
fi
if ! grep -q 'AMP_GRAD_GUARD_V2' gmls_roi_onehead_v2/runner.py; then
  echo 'WARNING: AMP Guard V2 marker not found; additional AMP settings may be ignored.' >&2
fi
python - "$CFG" <<'PY'
from pathlib import Path
import sys,yaml
p=Path(sys.argv[1]); c=yaml.safe_load(p.read_text(encoding='utf8'))
assert c['model']['name']=='yolo26m' and c['model']['roi_single_head']['topk']==2
assert 'Anti-UAV' in c['data']['rgb'] and 'Anti-UAV' in c['data']['tir']
out=Path(c['output']['project'])/c['output']['name']
if (out/'weights/last.pt').exists():
 raise SystemExit(f'Refusing to overwrite checkpoint at {out}; select a different output.name or explicitly resume')
print('[READY]',p,'output=',out,flush=True)
PY
python -c 'from gmls_roi_onehead_v2.runner import run_train; print("[IMPORT OK] Hybrid V2 runner")'
mkdir -p logs pid
nohup python -u train_gmls_v2_roi_hybrid.py --cfg "$CFG" > "$LOG" 2>&1 &
echo $! > "$PID"
echo "[START] PID=$(cat "$PID") MODE=$MODE LOG=$LOG"
tail -n 40 -f "$LOG"
