#!/usr/bin/env python3
"""Official pretrained YOLO26m, one shared detect, GMLS-v2 global aux + native Top-2 ROIs."""
import argparse
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default=None)
    p.add_argument('--resume',default=None)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke:
        from gmls_roi_single_head.launch import auto_launch
        auto_launch(args,Path(__file__).resolve())
    from gmls_roi_onehead_v2.runner import run_train
    run_train(args)
