#!/usr/bin/env python3
"""Unified 01/10 from-official-PT, 100/200-epoch joint dual-head training."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default=None)
    p.add_argument('--resume',default=None)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke:
        from gmls_roi_hybrid.launch import auto_launch
        auto_launch(args,Path(__file__).resolve())
    from gmls_roi_hybrid.runner import run_train
    run_train(args)

if __name__=='__main__':main()
