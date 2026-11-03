"""One real-data graph preflight, no optimizer step / no checkpoint modification."""
from __future__ import annotations
import argparse
import math
import torch

from engine.runtime import load_config,select_device,preprocess_batch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from ga_monoalign_pcgrad.factory import build


def grad_stats(named_params,pred):
    values=[]
    for name,p in named_params:
        if pred(name) and p.grad is not None:
            g=p.grad.detach().float()
            if not torch.isfinite(g).all():raise RuntimeError(f'Nonfinite gradient: {name}')
            values.append((name,float(g.norm())))
    return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--split',default='val')
    p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    if __import__('os').environ.get('AGHLR_EVAL_MODE','on') != 'on':
        raise RuntimeError('Preflight must run AGHLR_EVAL_MODE=on')
    dev=select_device(a.device)
    cfg=load_config(a.cfg)
    d=cfg['data']
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=a.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    dl=build_dual_resolution_dataloader(ds,batch_size=1,workers=a.workers,
                                        shuffle=False,pin_memory=dev.type=='cuda')
    batch=preprocess_batch(next(iter(dl)),dev)
    # DO NOT load teachers or 61.16% student weights: graph check only.
    model=build(cfg,device=dev,load_pretrained=False)
    if not hasattr(model,'aghlr'):raise RuntimeError('AGHLRStudent factory was not activated')
    model.train()
    model.set_progress('stage1',1)  # KD schedule 0, no teacher model involved
    # The AGHLR head is zero-initialized for identity at Epoch 1. Probe the
    # full path using a temporary, tiny nonzero regression weight in memory.
    with torch.no_grad():
        torch.nn.init.normal_(model.aghlr.regression.weight,mean=0,std=1e-3)
        torch.nn.init.zeros_(model.aghlr.regression.bias)
    model.zero_grad(set_to_none=True)
    vec,logs=model(batch)
    total=vec.sum()/max(1,int(batch['rgb_img'].shape[0]))
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f'Nonfinite loss before backward: {total}')
    total.backward()
    named=list(model.named_parameters())
    checks={
        'AGHLR regression head':lambda n:n.startswith('aghlr.regression'),
        'AGHLR detail encoder':lambda n:n.startswith('aghlr.pre_s2d'),
        'original BHLR detail bank':lambda n:'core.bhlr.detail_stem' in n,
        'original BHLR selector':lambda n:'core.bhlr.adapters' in n and '.selector.' in n,
        'original YOLO Detect P3 regression':lambda n:'core.model.' in n and '.cv2.0.' in n,
    }
    for label,fn in checks.items():
        v=grad_stats(named,fn)
        norm=sum(x[1] for x in v)
        print(f'[GRAD] {label:32s} tensors={len(v):3d} norm_sum={norm:.6g}')
        if not v or norm <= 0:
            raise RuntimeError(f'No connected gradient for {label}; abort training')
    if model._aghlr_last.get('delta_max_abs',0.0) <= 0:
        raise RuntimeError('AGHLR did not change P3 regression')
    print('[AGHLR PRE-FLIGHT PASS] real paired-image forward/backward connected; '
          'YOLO P3 regression reached; no optimizer step performed.')

if __name__=='__main__': main()
