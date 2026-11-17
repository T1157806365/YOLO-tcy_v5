#!/usr/bin/env python3
"""Summarize validation history for the four controlled BHLR Dual-HR runs."""
from pathlib import Path
import json

root=Path("runs/bhlr_dual_hr")
names={
    "control":"yolo26m_UAVCB_P2_HVAD_Control_DualHR_100e_seed0",
    "clean":"yolo26m_UAVCB_P2_HVAD_CleanResidual_100e_seed0",
    "dual":"yolo26m_UAVCB_P2_HVAD_DualHR_100e_seed0",
    "guided":"yolo26m_UAVCB_P2_HVAD_GuidedDualHR_100e_seed0",
}
print(f"{'Experiment':<12} {'Epoch':>6} {'AP50':>8} {'AP75':>8} {'AP50:95':>9} {'gamma':>10} {'delta_l1':>12}")
print("-"*73)
for key,run_name in names.items():
    file=root/run_name/'history_stage1.json'
    if not file.exists():
        print(f"{key:<12} {'(not trained)':>48}")
        continue
    hist=json.loads(file.read_text())
    if not hist:
        print(f"{key:<12} {'(empty)':>48}")
        continue
    best=max(hist,key=lambda v:float(v.get('map5095',-1)))
    def x(k):return float(best.get(k,float('nan')))
    print(f"{key:<12} {int(best['epoch']):6d} {100*x('map50'):8.3f} {100*x('map75'):8.3f} {100*x('map5095'):9.3f} {x('dualhr_gamma'):10.5f} {x('dualhr_delta_l1'):12.6f}")
