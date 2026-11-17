"""Validation: same trained checkpoint, HVAD ON versus OFF.

python -m hvad.val --weights runs/hvad_geohr_v1/.../weights/best.pt --split test --device 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset, build_dual_resolution_dataloader,
)
from engine.evaluator import validate_metrics
from engine.runtime import preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=['val','test','train'], default='test')
    p.add_argument('--device', default='2')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.70)
    p.add_argument('--save-dir', default=None)
    args = p.parse_args()
    dev = select_device(args.device)
    model, _, cfg = load_model_from_checkpoint(
        args.weights, device=dev, strict=True, verbose=False,
    )
    if not hasattr(model, 'hvad'):
        raise RuntimeError('Checkpoint config has no enabled HVAD module')
    d = cfg['data']
    dataset = build_dual_resolution_dataset(
        rgb_yaml=d['rgb'], tir_yaml=d['tir'], split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']), pair_mode=d.get('pair_mode','relative'),
        augment=False, tir_channels=3, strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        dataset, batch_size=args.batch, workers=args.workers,
        shuffle=False, pin_memory=(dev.type=='cuda'),
    )
    results = {}
    for enabled, label in ((True, 'hvad_on'), (False, 'hvad_off')):
        model.hvad.enabled_runtime = enabled
        print(f'\n======== {label} ========', flush=True)
        result = validate_metrics(
            model=model, loader=loader, device=dev,
            amp=(dev.type=='cuda'), conf_thres=args.conf,
            iou_thres=args.iou, max_det=300, preview_num=0,
        )
        results[label] = result
        print(f'[{label}] AP50:95={result["map5095"]:.6f} AP75={result["map75"]:.6f}')

    results['delta_on_minus_off'] = {
        key: results['hvad_on'][key] - results['hvad_off'][key]
        for key in ('map5095','map75','map50','precision','recall')
    }
    # Oracle selection diagnostics: labels ONLY used to evaluate candidates.
    model.hvad.enabled_runtime = True
    oracle_n = 0
    oracle_correct = 0.0
    oracle_err = 0.0
    model.eval()
    print('\n======== candidate diagnostics ========', flush=True)
    with torch.inference_mode():
        for batch in loader:
            batch = preprocess_batch(batch, dev)
            model(batch['rgb_img'], batch['tir_img'], rgb_semantic=batch['rgb_semantic_img'])
            _, stat = model.hvad.ranking_loss(batch)
            n = int(stat['hvad_valid_pairs'])
            oracle_n += n
            oracle_correct += float(stat['hvad_rank_accuracy'])*n
            oracle_err += float(stat['hvad_oracle_err_px'])*n
    results['selection'] = {
        'valid_pairs': oracle_n,
        'best_gt_candidate_selection_accuracy': oracle_correct/max(oracle_n,1),
        'oracle_candidate_distance_px_p3': oracle_err/max(oracle_n,1),
    }
    save_dir = Path(args.save_dir) if args.save_dir else (
        Path(args.weights).expanduser().resolve().parent.parent/f'hvad_ablation_{args.split}'
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir/'hvad_ablation.json'
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print('[SAVED]', out)

if __name__=='__main__':
    main()
