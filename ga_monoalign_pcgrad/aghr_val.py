"""Full validation/test of one AGHR checkpoint with fixed-weight inference switches.

Use val to select the model; reserve test for final reporting. Modes operate
on the SAME weights. Each mode performs a complete dataloader pass.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset, build_dual_resolution_dataloader,
)
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from .factory import load_checkpoint

MODES = ('identity', 'hr_only', 'no_hr', 'no_guidance', 'no_tir', 'full')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--split', default='val', choices=('val','test'))
    p.add_argument('--modes', default='full',
                   help='Comma-separated: identity,hr_only,no_hr,no_guidance,no_tir,full')
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.70)
    p.add_argument('--save-dir', required=True)
    args = p.parse_args()
    modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    if not modes or any(x not in MODES for x in modes):
        raise ValueError('Allowed modes: ' + ', '.join(MODES))
    if len(modes) != len(set(modes)):
        raise ValueError('Duplicate modes are not allowed')
    destination = Path(args.save_dir)
    if destination.exists() and (destination/'metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite {destination}/metrics.json')
    device = select_device(args.device)
    model,checkpoint,cfg = load_checkpoint(args.weights, device)
    if not bool(getattr(model,'aghr_enabled',False)):
        raise RuntimeError('Checkpoint does not contain enabled model.aghr_refine')
    d = cfg['data']
    data = build_dual_resolution_dataset(
        rgb_yaml=d['rgb'], tir_yaml=d['tir'], split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']), pair_mode=d.get('pair_mode','relative'),
        augment=False, tir_channels=3, strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        data, batch_size=args.batch, workers=args.workers,
        shuffle=False, pin_memory=device.type=='cuda',
    )
    results={}
    for mode in modes:
        model.aghr_mode=mode
        # Evaluator invokes model.forward and refines *postprocessed* xyxy.
        res=validate_metrics(model,loader,device,amp=device.type=='cuda',
                             conf_thres=args.conf,iou_thres=args.iou,
                             max_det=300,preview_num=0)
        results[mode]=res
        print(f'[AGHR {mode}] AP50:95={res["map5095"]:.6f}, '
              f'AP75={res["map75"]:.6f}, '
              f'identity_AP5095={res.get("aghr_identity_map5095",float("nan")):.6f}, '
              f'DeltaAP5095={res.get("aghr_delta_map5095",float("nan")):+.6f}, '
              f'delta_abs_px={res.get("aghr_delta_abs_px",0):.4f}', flush=True)
    if 'identity' in results:
        baseline=results['identity']['map5095']
        for mode in modes:
            results[mode]['delta_map5095_vs_identity']=(
                results[mode]['map5095']-baseline)
    if 'full' in results:
        for mode in ('hr_only','no_hr','no_guidance','no_tir'):
            if mode in results:
                results[mode]['delta_map5095_vs_full']=(
                    results[mode]['map5095']-results['full']['map5095'])
    out={'split':args.split,'weights':str(Path(args.weights).resolve()),
         'modes':modes,'conf':args.conf,'iou':args.iou,'images':len(data),
         'metrics':results}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'metrics.json').write_text(
        json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[SAVED]',destination/'metrics.json')

if __name__=='__main__':
    main()
