#!/usr/bin/env python3
"""Inspect whether P3/P4/P5 new reliability fusion gate learned from training.

Requires no GPU or repository imports; reads checkpoint file with torch.
"""
from __future__ import annotations
import argparse
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    args=p.parse_args()
    ckpt=torch.load(args.weights,map_location='cpu',weights_only=False)
    if not isinstance(ckpt,dict) or 'model_state_dict' not in ckpt:
        raise KeyError('Checkpoint missing model_state_dict')
    cfg=ckpt.get('config',{}) or {}
    print('variant:',(cfg.get('model',{}) or {}).get('variant'))
    sd=ckpt['model_state_dict']
    if sd and all(str(k).startswith('module.') for k in sd):
        sd={str(k)[7:]:v for k,v in sd.items()}
    for scale in (3,4,5):
        prefix=f'reliability_gates.{scale}.net.4.'
        w=sd.get(prefix+'weight')
        b=sd.get(prefix+'bias')
        if w is None or b is None:
            raise KeyError(f'P{scale} gate not found in checkpoint: {prefix}*')
        print(f'P{scale}: last_conv_weight_l2={float(w.float().norm()):.8g}, '
              f'bias={float(b.float().mean()):+.8g}, '
              f'changed_from_init={bool(w.float().abs().max()>0 or b.float().abs().max()>0)}')
    print('NOTE: Changed weights establish optimization, not AP improvement.')

if __name__=='__main__':main()
