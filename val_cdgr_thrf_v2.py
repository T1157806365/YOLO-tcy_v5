#!/usr/bin/env python3
"""Self-contained strict CDGR benchmark; no changes to legacy validator required."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from engine.runtime import load_config, select_device
from gmls_roi_hybrid.runner import build_loader, evaluate
from gmls_roi_hybrid.benchmark import (InferenceMeter, model_parameters, make_row,
                                        report_paths, save_results, _write_previews)
from cdgr_thrf_v2.model import CDGRTHRFV2


def main():
    p = argparse.ArgumentParser(description='CDGR-THRF-V2 GeoBox strict val/test benchmark')
    p.add_argument('--cfg', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--device', default='6')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--plots', action='store_true')
    p.add_argument('--preview-num', type=int, default=8)
    p.add_argument('--no-flops', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.cfg)
    path = Path(args.weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not (cfg.get('model', {}).get('geometry_refinement', {}).get('enabled') and cfg.get('model', {}).get('thrf_v2', {}).get('enabled')):
        raise ValueError('Wrong YAML: CDGR and THRF must both be enabled')
    device = select_device(args.device)
    net = CDGRTHRFV2(cfg, verbose=True).to(device)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != CDGRTHRFV2.architecture:
        raise ValueError(f'Wrong checkpoint architecture={checkpoint.get("architecture")!r}; '
                         f'expected {CDGRTHRFV2.architecture!r}')
    state = checkpoint['model_state_dict']
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    print(f'[CDGR-THRF-V2 STRICT LOADED] epoch={checkpoint.get("epoch")}; '
          f'architecture={checkpoint.get("architecture")}', flush=True)
    net.eval()
    run_dir = path.parent.parent if path.parent.name == 'weights' else Path(cfg['output']['project']) / cfg['output']['name']
    paths = report_paths(run_dir, f"thrf_v2/{args.split}/{path.stem}")
    loader, _ = build_loader(cfg, args.split, args.batch, args.workers)
    if args.plots:
        paths['plots'].mkdir(parents=True, exist_ok=True)
        # Keep unrelated previews; overwrite same-named files only.
    meter = InferenceMeter(net, device, warmup=args.warmup,
                           count_flops=not args.no_flops).start()
    if args.plots:
        timed_predict = net.predict
        counters = {'images': 0}
        def with_previews(batch, *a, **kw):
            pred, ex = timed_predict(batch, *a, **kw)
            if counters['images'] < args.preview_num:
                _write_previews(batch, pred, ex, paths['plots'], args.preview_num, counters)
            return pred, ex
        net.predict = with_previews
    try:
        report = evaluate(net, loader, device, cfg, rank=0, world=1, smoke=False)
    finally:
        meter.stop()
    row = make_row(model_parameters(net), meter, report['metrics'])
    summary = {
        'checkpoint': str(path), 'epoch': checkpoint.get('epoch'),
        'architecture': checkpoint.get('architecture'),
        'split': args.split, 'image_count': int(report['images']),
        'batch': int(args.batch), 'parameters': model_parameters(net),
        'flops': {'gflops_per_image': meter.gflops_per_image, 'method': 'forward_hooks_conv_linear_2mac'},
        'inference': {'fps': meter.fps, 'ms_per_image': meter.latency_ms},
        'flops_definition': 'Estimated 2*MACs for Conv2d, ConvTranspose2d and Linear. Excludes grid_sample, local attention correlations/weighted aggregation, softmax, NMS.',
        'fps_definition': 'Synchronized model.predict including global alignment, refined warp, two ROIs, local RGB-T detect, and NMS; excludes data load.',
        'evaluation': report,
    }
    lines = save_results(paths, summary, row)
    lines = [line.replace('GMLS-v2 UNIFIED DUAL HEAD HYBRID TEST BENCHMARK',
                         'CDGR-THRF-V2 PHASE-PRESERVING RGB P2 DETAIL BENCHMARK') for line in lines]
    paths['text'].write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines), flush=True)
    print(f'[SAVED] {paths["json"]}', flush=True)


if __name__ == '__main__':
    main()
