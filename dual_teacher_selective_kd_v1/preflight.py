"""Static validation or real-data GPU teacher/student backward smoke test."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from engine.runtime import load_config,select_device,preprocess_batch
from .factory import validate_cfg,build_model_from_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--full',action='store_true')
    p.add_argument('--device',default='3')
    a=p.parse_args()
    cfg=validate_cfg(load_config(a.cfg))
    print('[PASS] dual teachers + student official pretrained + YAML paths found')
    if not a.full:return
    import torch
    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
    d=cfg['data'];device=select_device(a.device)
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split='train',
      rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
      tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
      augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=1,workers=0,shuffle=False,pin_memory=False)
    student=build_model_from_config(cfg,verbose=False).to(device).train()
    assert all('teacher' not in k for k in student.state_dict()),'Teacher leaked into state_dict!'
    batch=preprocess_batch(next(iter(loader)),device)
    vec,logs=student(batch)
    loss=vec.sum()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward()
    check=['rgb_adapter','tir_adapter','fused_adapter','rgb_response','tir_response',
           'fused_response','quality','tir_projection','alignment']
    for key in check:
        seen=sum(1 for n,p in student.named_parameters() if n.startswith(key) and p.grad is not None)
        if seen==0:raise RuntimeError(f'No gradient connection: {key}')
        print('grad',key,seen)
    for name,t in logs.items():
        if isinstance(t,torch.Tensor) and not torch.isfinite(t).all():
            raise RuntimeError(f'Nonfinite logged metric: {name}')
    print('TOTAL LOSS',float(loss.detach()))
    print('TEACHER TP RGB/TIR',float(logs['dtkd_teacher_rgb_tp_rate']),float(logs['dtkd_teacher_tir_tp_rate']))
    print('[PASS] full student + two teachers forward/backward; no teacher weights serialized')

if __name__=='__main__':main()
