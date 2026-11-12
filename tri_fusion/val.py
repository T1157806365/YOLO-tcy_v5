"""Fixed-checkpoint Val/Test with true in-place P3 ablations.
Same weights, no reloading alternative run; controls RGB/TIR/HR/guidance separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from ga_monoalign_pcgrad.factory import load_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='val',choices=['train','val','test'])
    p.add_argument('--device',default='3')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=0.001)
    p.add_argument('--iou',type=float,default=0.70)
    p.add_argument('--ablation',default='none',
                   choices=['none','off','rgb_off','tir_off','hr_off','no_alignment'])
    p.add_argument('--save-dir',required=True)
    args=p.parse_args()
    output=Path(args.save_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing {output}')
    device=select_device(args.device)
    model,ck,cfg=load_checkpoint(args.weights,device)
    gate=getattr(model.core,'tri_gate',None)
    if gate is None and args.ablation!='none':
        raise ValueError('Original baseline has no gate; ablations are only for B/C/D')
    totals={};seen={'n':0}
    if gate is not None:
        gate.ablation=args.ablation
        def accumulate(m,inp,out):
            seen['n']+=1
            for k,v in m.last_stats.items():
                totals[k]=totals.get(k,0.0)+float(v)
        handle=gate.register_forward_hook(accumulate)
    else: handle=None
    d=cfg['data']
    ds=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=args.workers,
                                           shuffle=False,pin_memory=(device.type=='cuda'))
    metric=validate_metrics(model,loader,device,amp=(device.type=='cuda'),
                            conf_thres=args.conf,iou_thres=args.iou,max_det=300,preview_num=0)
    if handle:handle.remove()
    gate_avg={k: v/max(seen['n'],1) for k,v in totals.items()}
    result={'split':args.split,'weights':str(Path(args.weights).resolve()),
            'ablation':args.ablation,
            'gate_mode':(gate.mode if gate else 'original_identity'),
            'gate_calls':seen['n'],**metric,**gate_avg}
    output.mkdir(parents=True,exist_ok=False)
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
