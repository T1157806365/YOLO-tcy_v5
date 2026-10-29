#!/usr/bin/env python3
"""Strict stage-0 functional check: RGB-only and new model produce same logits."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ultralytics import YOLO
from engine.runtime import load_config, select_device, preprocess_batch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from rgb_preserve_v1.model import build_rgb_preserve_model


def flatten_tensors(obj):
    if isinstance(obj, torch.Tensor): return [obj]
    if isinstance(obj, (tuple, list)):
        ret = []
        for v in obj: ret += flatten_tensors(v)
        return ret
    if isinstance(obj, dict):
        ret = []
        for k in sorted(obj): ret += flatten_tensors(obj[k])
        return ret
    return []

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-batches', type=int, default=2)
    p.add_argument('--tolerance', type=float, default=1e-5)
    a = p.parse_args()
    cfg=load_config(a.cfg)
    d=cfg['data']
    device=select_device(a.device)
    m=build_rgb_preserve_model(cfg, init_single_modal=True, verbose=False).to(device).eval()
    m.set_training_phase('identity')
    rgb=YOLO(cfg['rgb_preserve']['rgb_weights']).model.to(device).eval()
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'], tir_yaml=d['tir'], split='val',
        rgb_high_imgsz=int(d['rgb_imgsz']), rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']), pair_mode=d.get('pair_mode','relative'),
        augment=False, tir_channels=3, strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=a.batch,workers=a.workers,shuffle=False,pin_memory=True)
    worst=0.0
    for bi,b in enumerate(loader):
        if bi>=a.max_batches: break
        b=preprocess_batch(b,device)
        rgbx=b['rgb_semantic_img']
        reference=flatten_tensors(rgb(rgbx))
        candidate=flatten_tensors(m(b['rgb_img'],b['tir_img'],rgb_semantic=rgbx))
        if len(reference)!=len(candidate):
            raise RuntimeError(f'Prediction format mismatch: RGB tensors={len(reference)}, RGBT tensors={len(candidate)}')
        for j,(u,v) in enumerate(zip(reference,candidate)):
            if u.shape!=v.shape:
                raise RuntimeError(f'Pred[{j}] shape mismatch: {u.shape} != {v.shape}')
            delta=(u.float()-v.float()).abs().max().item()
            worst=max(worst,delta)
            print(f'[batch={bi}] tensor[{j}] shape={tuple(u.shape)} max_abs={delta:.8g}')
    print(f'[STAGE0] worst max abs = {worst:.8g}; tolerance={a.tolerance}; alpha={m.p3_residual_raw.item():.6g}')
    if worst > a.tolerance:
        raise SystemExit('FAIL: RGB-only detection was not preserved; DO NOT TRAIN')
    print('PASS: same RGB prediction tensors at zero residual (without training).')

if __name__=='__main__': main()
