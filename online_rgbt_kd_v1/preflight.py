"""Separate static config check and real-data gradient smoke test."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from engine.runtime import load_config,select_device,preprocess_batch
from online_rgbt_kd_v1.factory import validate_cfg,build_model_from_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--full',action='store_true',help='Real first train batch forward+backward')
    p.add_argument('--device',default='3')
    a=p.parse_args()
    cfg=validate_cfg(load_config(a.cfg))
    print('[PASS] YAML, official weight, model/data configuration validated')
    if not a.full:return
    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
    import torch
    d=cfg['data']
    device=select_device(a.device)
    dataset=build_dual_resolution_dataset(
       rgb_yaml=d['rgb'],tir_yaml=d['tir'],split='train',
       rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
       tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
       augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(dataset,batch_size=1,workers=0,shuffle=False,pin_memory=False)
    model=build_model_from_config(cfg,verbose=False).to(device).train()
    batch=preprocess_batch(next(iter(loader)),device)
    raw,items=model(batch)
    total=raw.sum()
    if not torch.isfinite(total):raise RuntimeError(f'Nonfinite loss: {total}')
    total.backward()
    prefixes=['aux_rgb','aux_tir','aux_fused','reliability_net','alignment']
    report={k:sum(1 for name,v in model.named_parameters() if name.startswith(k) and v.grad is not None) for k in prefixes}
    print('Loss',float(total.detach()),'grad tensor counts',report)
    for k in prefixes:
        if report[k]==0:raise RuntimeError(f'No gradient: {k}')
    print('[PASS] forward/backward with native paired data')

if __name__=='__main__':main()
