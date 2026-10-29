#!/usr/bin/env python3
"""One batch dry-run checks loss, gradients and stage1 parameter freeze."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset, build_dual_resolution_dataloader,
)
from engine.runtime import load_config, preprocess_batch, select_device
from rgb_preserve_v1.model import build_rgb_preserve_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--phase',choices=['stage1','stage2'],default='stage1')
    args=p.parse_args()
    device=select_device(args.device)
    cfg=load_config(args.cfg)
    d=cfg['data']
    model=build_rgb_preserve_model(cfg,init_single_modal=True,verbose=False).to(device)
    model.set_training_phase(args.phase)
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split='train',
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    dl=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=0,
        shuffle=False,pin_memory=False)
    batch=preprocess_batch(next(iter(dl)),device)
    model.zero_grad(set_to_none=True)
    raw,items=model(batch)
    loss=raw.sum()
    if not torch.isfinite(loss).item():
        raise SystemExit('FAIL: non-finite training loss')
    loss.backward()
    alpha_grad=model.p3_residual_raw.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).all().item():
        raise SystemExit('FAIL: P3 residual not connected to detection loss')
    frozen_rgb=[(name,param) for name,param in model.named_parameters()
                if name.startswith('model.') and not param.requires_grad]
    bad_frozen=[name for name,p in frozen_rgb if p.grad is not None]
    if bad_frozen:
        raise SystemExit('FAIL: frozen RGB params received grads: '+repr(bad_frozen[:8]))
    aux_grad=sum(p.grad is not None for n,p in model.named_parameters()
                 if p.requires_grad and not n.startswith('model.'))
    print('Phase:',args.phase, 'Loss:',float(loss.detach()),
          'alpha grad:',float(alpha_grad.detach()), 'trainable aux grads:',aux_grad,
          'frozen RGB tensors:',len(frozen_rgb))
    if aux_grad<2:
        raise SystemExit('FAIL: too few non-RGB gradients; inspect alignment and TIR path')
    print('PASS: stage loss and backward graph are valid.')

if __name__=='__main__': main()
