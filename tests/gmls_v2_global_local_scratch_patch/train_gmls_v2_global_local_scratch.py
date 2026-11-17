#!/usr/bin/env python3
"""Scratch joint GMLS-v2 global + local RGB-T training."""
import argparse
from global_local_scratch.runner import run_train

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default=None)
    p.add_argument('--smoke',action='store_true')
    run_train(p.parse_args())
