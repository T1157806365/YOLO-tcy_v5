"""Evaluate original or module-only checkpoint with the unchanged forward graph."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from .runtime import load_student, choose_device, make_loader, write_json, json_safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf', type=float, default=.001)
    p.add_argument('--iou', type=float, default=.70)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--geometry-all', action='store_true')
    p.add_argument('--mono', action='store_true')
    p.add_argument('--save-dir', required=True)
    args = p.parse_args()
    dest = Path(args.save_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {dest}')
    device = choose_device(args.device)
    model, meta, cfg = load_student(args.weights, device)
    data, loader = make_loader(cfg, {'batch': args.batch, 'workers': args.workers}, args.split)
    from engine.evaluator import validate_metrics
    amp = device.type == 'cuda' and not args.no_amp
    with torch.inference_mode():
        metrics = validate_metrics(model, loader, device, amp=amp,
            conf_thres=args.conf, iou_thres=args.iou, max_det=300, preview_num=0)
        if args.geometry_all:
            from ga_monoalign_pcgrad.metrics import geometry_iou
            metrics.update(geometry_iou(model, loader, device, amp=False, max_images=len(data)))
        if args.mono:
            from ga_monoalign_pcgrad.metrics import mono_metrics
            metrics.update(mono_metrics(model, loader, device, amp=amp))
    result = {'weights': str(Path(args.weights).resolve()), 'split': args.split,
              'images': len(data), 'module_only_epoch': meta.get('module_only_epoch'),
              'best_is_original_source': meta.get('module_only_epoch') == 0,
              **metrics}
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / 'metrics.json', result)
    import json
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
