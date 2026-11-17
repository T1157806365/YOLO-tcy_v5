"""Same-checkpoint P2 selective HR ablation; P3 TriGate remains ON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from models.gtga_bhlr_factory import load_model_from_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',choices=('val','test','train'),default='test')
    p.add_argument('--device',default='2')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=0.001)
    p.add_argument('--iou',type=float,default=0.70)
    p.add_argument('--save-dir',default=None)
    args=p.parse_args()
    device=select_device(args.device)
    model,ck,cfg=load_model_from_checkpoint(args.weights,device=device,strict=True,verbose=False)
    if not hasattr(model,'set_p2_hr_disabled'):
        raise RuntimeError('Checkpoint is not P2SelectiveHR model')
    if hasattr(model,'set_tri_gate_disabled'):
        model.set_tri_gate_disabled(False)
    d=cfg['data']
    dataset=build_dual_resolution_dataset(
        rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(dataset,batch_size=args.batch,workers=args.workers,
                                            shuffle=False,pin_memory=(device.type=='cuda'))
    scores={}
    for key,disabled in [('full',False),('p2_hr_off',True)]:
        model.set_p2_hr_disabled(disabled)
        scores[key]=validate_metrics(model,loader,device,amp=device.type=='cuda',
                                      conf_thres=args.conf,iou_thres=args.iou,max_det=300,preview_num=0)
        print(f'[P2HR VAL] {key}: AP50:95={scores[key]["map5095"]:.6f}, AP75={scores[key]["map75"]:.6f}')
    out={'checkpoint':str(Path(args.weights).resolve()),'split':args.split,
         'modes':scores,
         'p2hr_delta_ap5095':scores['full']['map5095']-scores['p2_hr_off']['map5095'],
         'note':'Same checkpoint; P3 TriGate, GTGA and BHLR ON for both modes'}
    path=Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent.parent / f'p2_hr_val_{args.split}'
    path.mkdir(parents=True,exist_ok=True)
    (path/'metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
