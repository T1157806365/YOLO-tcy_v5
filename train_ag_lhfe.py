#!/usr/bin/env python3
"""Unified E0/E1/E2/E3 trainer using the project's original 100e/300+200 trainer.

No KD/Mono/PCGrad in this controlled structural experiment. The original
trainer runs one optimizer update per batch (no gradient accumulation).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ag_lhfe.preflight import check_config,smoke


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg',required=True)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--smoke-test',action='store_true')
    ap.add_argument('--smoke-device',default='3')
    args=ap.parse_args()
    cfg=check_config(Path(args.cfg))
    if args.check:
        print('[CHECK PASS] Configuration parsed; no model launched')
        return
    if args.smoke_test:
        smoke(cfg,args.smoke_device)
        return
    from train_rlsfa_gtga_bhlr_v2 import _auto_launch_ddp
    from engine.trainer import train_experiment
    if _auto_launch_ddp(args.cfg,cfg):
        return
    train_experiment(cfg)

if __name__=='__main__':main()
