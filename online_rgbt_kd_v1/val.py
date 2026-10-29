"""RGB-coordinate detection AP evaluation for online RGB-T checkpoints."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
import torch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.evaluator import validate_metrics
from engine.runtime import select_device
from online_rgbt_kd_v1.factory import build_model_from_config


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
    args=p.parse_args()
    device=select_device(args.device)
    ck=torch.load(args.weights,map_location='cpu',weights_only=False)
    if 'model_state_dict' not in ck or 'config' not in ck:
        raise ValueError('Checkpoint missing model_state_dict or config')
    cfg=ck['config']
    model=build_model_from_config(cfg,pretrained=False,verbose=False).to(device)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.eval()
    d=cfg['data']
    ds=build_dual_resolution_dataset(
        rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']), rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(
        ds,batch_size=args.batch,workers=args.workers,shuffle=False,
        pin_memory=(device.type=='cuda'))
    results=validate_metrics(model,loader,device,amp=args.amp,
                             conf_thres=args.conf,iou_thres=args.iou,max_det=300,
                             preview_num=0)
    print('Online RGB-T %s metrics: %s'%(args.split,json.dumps(results,ensure_ascii=False,indent=2)))
    out_dir=Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent.parent/f'validation_{args.split}'
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'metrics.json').write_text(json.dumps({'checkpoint':str(Path(args.weights).resolve()),'split':args.split,**results},indent=2),encoding='utf-8')
    print('Metrics saved:',out_dir/'metrics.json')

if __name__=='__main__':main()
