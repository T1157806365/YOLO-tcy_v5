"""Real-val single-sample identity check without old RGB-T student checkpoint.
Builds a freshly initialized student; checks gate's zero residual.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from engine.runtime import select_device,preprocess_batch,load_config
from ga_monoalign_pcgrad.factory import build

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default='3')
    args=p.parse_args()
    cfg=load_config(args.cfg)
    if not cfg.get('model',{}).get('tri_gate',{}).get('enabled'):
        raise ValueError('Preflight requires B/C/D enabled gate')
    device=select_device(str(args.device))
    # Neither old student nor single-modal teacher weights needed for this
    # structural test. Actual train uses build(...,load_pretrained=True).
    net=build(cfg,device=device,load_pretrained=False).eval()
    core=net.core
    d=cfg['data']
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split='val',
         rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
         tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
         augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=1,workers=0,shuffle=False,pin_memory=(device.type=='cuda'))
    batch=preprocess_batch(next(iter(loader)),device)
    kw={'rgb_semantic':batch['rgb_semantic_img']}
    fnew=core.forward_features(batch['rgb_img'],batch['tir_img'],**kw)
    p3=int(core.scale_to_index[3]); a=fnew['fused_backbone'][p3].detach().float()
    core.tri_gate_enabled=False
    try:
        fold=core.forward_features(batch['rgb_img'],batch['tir_img'],**kw)
    finally:
        core.tri_gate_enabled=True
    b=fold['fused_backbone'][p3].detach().float()
    delta=float((a-b).abs().max().item())
    print('[PREFLIGHT] max |new P3 - original Post-Fusion P3| =',delta)
    if delta>1e-5:
        raise AssertionError(f'Gate is not identity at initialization: {delta}')
    print('[PASS] Zero-init attention preserves Post-Fusion P3 exactly')

if __name__=='__main__':main()
