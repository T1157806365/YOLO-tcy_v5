#!/usr/bin/env python3
"""Scratch GMLS-v2 global + local RGB-T joint training, auto-DDP aware."""
import argparse

from global_local_scratch.ddp_launch import auto_ddp_launch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', default=None, help='Resume from existing last.pt checkpoint (not initial pretrained weights)')
    args = parser.parse_args()

    # With train.device: '5,6', a normal `nohup python -u` command now
    # automatically relaunches itself under two torchrun ranks.
    auto_ddp_launch(args, __file__)

    from global_local_scratch.runner import run_train
    run_train(args)
