"""Config and real-data one-batch smoke test for the AG-LHFE experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import yaml


def check_config(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    variant = cfg.get('model', {}).get('variant')
    if variant not in {'rlsfa_gtga_centered_ag_lhfe_v1', 'rlsfa_gtga_centered_bhlr_v2'}:
        raise ValueError('Unexpected model variant: ' + str(variant))
    d=cfg['data']
    assert d['rgb_imgsz'] == 3*d['rgb_semantic_imgsz'], 'Expected exactly 1920:640 ratio'
    assert cfg['model']['fusion']=='add'
    assert cfg['model']['rlsfa_gtga_centered']['enabled'] is True
    assert cfg['train'].get('resume') in (None, 'null')
    if variant.endswith('ag_lhfe_v1'):
        a=cfg['model']['ag_lhfe']
        assert a['enabled'] and not cfg['model']['bhlr']['enabled']
        assert a['topk']>0 and a['positive_radius']>0
    # Current original clean trainer has no gradient accumulation.
    if int(cfg['train'].get('accumulate',1)) != 1:
        raise ValueError('Original engine.trainer lacks accumulate: remove this key or use 1')
    try:
        from engine.runtime import parse_device_ids
        devices=parse_device_ids(cfg['train'].get('device'))
    except ImportError:
        devices=[int(x) for x in str(cfg['train'].get('device','3')).split(',')]
    if len(devices)>1 and cfg['train']['batch_mode']=='global':
        assert cfg['train']['batch']%len(devices)==0
    print(f"[CHECK] {path.name}: variant={variant} devices={devices} global_batch={cfg['train']['batch']}")
    for k in ('rgb','tir'):
        if not Path(d[k]).is_file():
            print(f'[WARNING] Dataset YAML not found here: {d[k]} (expected on training server)')
    return cfg


def smoke(cfg, device_str):
    from datasets.dual_resolution_rgbt_dataset import (
        build_dual_resolution_dataset, build_dual_resolution_dataloader)
    from engine.runtime import preprocess_batch, autocast_context
    from models.gtga_bhlr_factory import build_model_from_config
    device=torch.device('cuda:'+device_str if torch.cuda.is_available() and device_str!='cpu' else 'cpu')
    data=cfg['data']
    ds=build_dual_resolution_dataset(
        rgb_yaml=data['rgb'], tir_yaml=data['tir'], split='val',
        rgb_high_imgsz=int(data['rgb_imgsz']), rgb_semantic_imgsz=int(data['rgb_semantic_imgsz']),
        tir_imgsz=int(data['tir_imgsz']), pair_mode=data.get('pair_mode','relative'),
        augment=False, tir_channels=3, strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=1,workers=0,shuffle=False,pin_memory=(device.type=='cuda'))
    batch=preprocess_batch(next(iter(loader)),device)
    model=build_model_from_config(cfg,pretrained=bool(cfg['model'].get('pretrained',True)),verbose=False).to(device)
    model.train()
    amp=(device.type=='cuda')
    with autocast_context(device,amp):
        raw,items=model(batch)
        loss=raw.sum()
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite smoke-test loss: {loss.item()}')
    loss.backward()
    if hasattr(model,'ag_lhfe'):
        diagnostics={}
        for name,p in model.ag_lhfe.named_parameters():
            if p.grad is not None:
                diagnostics[name]=float(p.grad.detach().float().norm().item())
        if not diagnostics:
            raise RuntimeError('No AG-LHFE trainable gradients found')
        print('[SMOKE] AG-LHFE gradient L2 (first 12):',list(diagnostics.items())[:12])
        print('[SMOKE] ROI metrics:',{k:float(v.detach().cpu()) if torch.is_tensor(v) else v
              for k,v in items.items() if k.startswith('lhfe_')})
    print(f'[SMOKE PASS] loss={loss.item():.6f} device={device}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--device',default='3')
    args=p.parse_args()
    cfg=check_config(Path(args.cfg))
    if args.smoke_test:
        smoke(cfg,args.device)

if __name__=='__main__':main()
