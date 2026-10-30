"""Strict validation of new student and two single-modal auxiliary heads."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from .factory import load_checkpoint
from .metrics import mono_metrics,geometry_iou


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--device',default='0');p.add_argument('--batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=.001);p.add_argument('--iou',type=float,default=.70)
    p.add_argument('--save-dir',default=None)
    p.add_argument('--geometry-all',action='store_true',help='Calculate alignment IoU on entire split')
    args=p.parse_args()
    device=select_device(args.device)
    model,ck,cfg=load_checkpoint(args.weights,device)
    d=cfg['data']
    data=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(data,batch_size=args.batch,workers=args.workers,
        shuffle=False,pin_memory=device.type=='cuda')
    main=validate_metrics(model,loader,device,amp=device.type=='cuda',
        conf_thres=args.conf,iou_thres=args.iou,max_det=300,preview_num=0)
    mono=mono_metrics(model,loader,device,amp=device.type=='cuda')
    geom=geometry_iou(model,loader,device,max_images=len(data) if args.geometry_all else 96)
    res={'split':args.split,'checkpoint':str(Path(args.weights).resolve()),
         'student_scratch':True,'gmlsv2':True,**main,**mono,**geom}
    path=Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent.parent/f'val_{args.split}_monoalign'
    if path.exists() and (path/'metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite metrics: {path}/metrics.json')
    path.mkdir(parents=True,exist_ok=True)
    (path/'metrics.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
