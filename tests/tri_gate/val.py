"""Evaluate the same trained checkpoint with final gate ON and OFF.

This ablation switches ONLY the new final P3 three-way gate. GTGA and BHLR
remain enabled in both modes. Results are not independent-training ablations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from models.gtga_bhlr_factory import load_model_from_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.70)
    p.add_argument('--save-dir', default=None)
    a = p.parse_args()
    device = select_device(a.device)
    model, _, cfg = load_model_from_checkpoint(
        a.weights, device=device, strict=True, verbose=False
    )
    if not hasattr(model, 'set_tri_gate_disabled'):
        raise RuntimeError('Checkpoint is not a final TriGate model')
    d = cfg['data']
    data = build_dual_resolution_dataset(
        rgb_yaml=d['rgb'], tir_yaml=d['tir'], split=a.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),
        pair_mode=d.get('pair_mode', 'relative'),
        augment=False, tir_channels=3, strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        data, batch_size=a.batch, workers=a.workers,
        shuffle=False, pin_memory=(device.type == 'cuda'),
    )
    scores = {}
    for mode, disabled in [('full', False), ('gate_off', True)]:
        model.set_tri_gate_disabled(disabled)
        scores[mode] = validate_metrics(
            model, loader, device, amp=(device.type == 'cuda'),
            conf_thres=a.conf, iou_thres=a.iou, max_det=300, preview_num=0
        )
        print(f'[FINAL TRI-FUSION VAL] {mode}: '
              f'AP50:95={scores[mode]["map5095"]:.6f} '
              f'AP75={scores[mode]["map75"]:.6f}')
    delta = float(scores['full']['map5095'] - scores['gate_off']['map5095'])
    out = {
        'checkpoint': str(Path(a.weights).resolve()),
        'split': a.split, 'modes': scores,
        'tri_gate_delta_map5095': delta,
        'note': 'Same checkpoint; gate_off retains GTGA and BHLR',
    }
    save_dir = Path(a.save_dir) if a.save_dir else Path(a.weights).resolve().parent.parent / f'tri_gate_val_{a.split}'
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / 'metrics.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[VAL SAVED]', path)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
