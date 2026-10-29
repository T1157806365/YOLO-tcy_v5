"""Dedicated RGB-coordinate AP evaluation: teachers NOT loaded or required."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from engine.evaluator import validate_metrics
from engine.runtime import select_device
from .factory import build_model_from_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--device',default='3')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=.001)
    p.add_argument('--iou',type=float,default=.70)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--save-dir',default=None)
    a=p.parse_args()
    dev=select_device(a.device)
    ck=torch.load(a.weights,map_location='cpu',weights_only=False)
    if 'model_state_dict' not in ck or 'config' not in ck:raise ValueError('Invalid student checkpoint')
    cfg=ck['config'];m=build_model_from_config(cfg,pretrained=False,verbose=False,attach_teachers=False).to(dev)
    m.load_state_dict(ck['model_state_dict'],strict=True)
    assert m._frozen_dual_teachers is None,'No teachers should be loaded at inference'
    m.eval();d=cfg['data']
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=a.split,
      rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
      tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
      augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=a.batch,workers=a.workers,
       shuffle=False,pin_memory=(dev.type=='cuda'))
    result=validate_metrics(m,loader,dev,amp=a.amp,conf_thres=a.conf,iou_thres=a.iou,
       max_det=300,preview_num=0)
    saved=Path(a.save_dir) if a.save_dir else Path(a.weights).resolve().parent.parent/f'validation_{a.split}'
    saved.mkdir(parents=True,exist_ok=True)
    data={'checkpoint':str(Path(a.weights).resolve()),'split':a.split,**result}
    (saved/'metrics.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    print('[DUAL TEACHER KD] Test student only; no teachers loaded:\n'+json.dumps(data,indent=2,ensure_ascii=False))
    print('Metrics saved:',saved/'metrics.json')

if __name__=='__main__':main()
