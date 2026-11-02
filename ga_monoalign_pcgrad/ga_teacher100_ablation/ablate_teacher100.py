#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-checkpoint functional ablation for ga_monoalign_pcgrad.

Compatible with YOLO-tcy_v5/backup-20261031-teacher100. No weights or
project source files are modified.

Experiments:
  full                : original trained inference
  no_bhlr             : suppress BHLR forward enhancement only
  no_warp_keep_guide  : bypass TIR spatial warp; retain computed alignment
                        support maps for BHLR (NOT a full no-alignment model)
  no_align_no_bhlr    : bypass BOTH alignment and BHLR
  mono_rgb / mono_tir : original trained auxiliary single-modal heads

Key causal comparisons:
  full - no_bhlr                  : incremental BHLR contribution (alignment on)
  no_bhlr - no_align_no_bhlr      : alignment contribution (BHLR off)
  full - no_warp_keep_guide       : warp contribution with BHLR guidance held
                                    at its originally computed values

Cannot infer causal effects of KD training from a single checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _positive_int(x: str) -> int:
    n = int(x)
    if n < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return n


@contextlib.contextmanager
def ablation_mode(core, mode: str) -> Iterator[None]:
    """Only transient object attributes/forward routing; no state_dict edits."""
    if mode not in {'full', 'no_bhlr', 'no_warp_keep_guide', 'no_align_no_bhlr'}:
        raise ValueError(f'Unsupported ablation: {mode}')
    old_align = bool(core.alignment_enabled)
    old_bhlr = bool(core.bhlr_enabled)
    align_obj = core.alignment
    original_align_pyramid = getattr(align_obj, 'align_pyramid', None)
    try:
        if mode == 'no_bhlr':
            core.bhlr_enabled = False
        elif mode == 'no_align_no_bhlr':
            # With BHLR disabled, no learned alignment information is needed.
            # forward_features() then uses raw TIR features as RGB-aligned TIR.
            core.alignment_enabled = False
            core.bhlr_enabled = False
        elif mode == 'no_warp_keep_guide':
            if not old_align or align_obj is None or original_align_pyramid is None:
                raise RuntimeError('Warp ablation requires an enabled alignment module')

            def identity_warp_with_original_guidance(*args, **kwargs):
                # Run the ORIGINAL estimator to preserve its 3-channel support
                # maps/quality/debug. Only replace warped features with raw TIR.
                _aligned, info = original_align_pyramid(*args, **kwargs)
                raw_tir = kwargs.get('tir_features', None)
                if raw_tir is None:
                    if len(args) < 2:
                        raise RuntimeError('Cannot resolve original TIR features')
                    raw_tir = args[1]
                if len(_aligned) != len(raw_tir):
                    raise RuntimeError('Unexpected feature-pyramid lengths')
                return list(raw_tir), info

            # Temporary instance override (not written to state_dict).
            align_obj.align_pyramid = identity_warp_with_original_guidance
        yield
    finally:
        core.alignment_enabled = old_align
        core.bhlr_enabled = old_bhlr
        if mode == 'no_warp_keep_guide' and align_obj is not None:
            # Restore exact previously bound method. Clearing instance shadow
            # is preferable where method came from the class descriptor.
            if 'align_pyramid' in vars(align_obj):
                del align_obj.align_pyramid
            if original_align_pyramid is not None and getattr(align_obj, 'align_pyramid', None) != original_align_pyramid:
                align_obj.align_pyramid = original_align_pyramid


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float,)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if hasattr(obj, 'item'):
        return clean_json(obj.item())
    return str(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True, help='Trained ga_monoalign_pcgrad best.pt')
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    ap.add_argument('--device', default='3')
    ap.add_argument('--batch', type=_positive_int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.70)
    ap.add_argument('--output', default='runs/ablation/ga_teacher100_val')
    ap.add_argument('--repo-root', default='.', help='Path to the restored YOLO-tcy_v5 repo')
    ap.add_argument('--skip-mono', action='store_true')
    ap.add_argument('--verify-restore', action='store_true', help='Rerun full after all ablations')
    ap.add_argument('--overwrite', action='store_true', help='Allow rewriting result JSON/CSV')
    args = ap.parse_args()

    if args.workers < 0 or not (0 <= args.conf <= 1) or not (0 < args.iou <= 1):
        ap.error('Invalid workers/conf/iou argument')
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        ap.error(f'Checkpoint missing: {weights}')

    root = Path(args.repo_root).expanduser().resolve()
    if not (root / 'ga_monoalign_pcgrad' / 'factory.py').is_file():
        ap.error('repo-root is not a YOLO-tcy_v5 repository directory')
    sys.path.insert(0, str(root))

    import torch
    from datasets.dual_resolution_rgbt_dataset import (build_dual_resolution_dataset,
                                                      build_dual_resolution_dataloader)
    from engine.runtime import select_device
    from engine.evaluator import validate_metrics
    from ga_monoalign_pcgrad.factory import load_checkpoint
    from ga_monoalign_pcgrad.metrics import MonoEval, TirGTLoader

    device = select_device(args.device)
    model, ck, cfg = load_checkpoint(str(weights), device)
    model.eval()
    core = model.core
    if not getattr(core, 'alignment_enabled', False):
        raise RuntimeError('Checkpoint already has alignment disabled; wrong baseline for this study')
    if not getattr(core, 'bhlr_enabled', False):
        raise RuntimeError('Checkpoint already has BHLR disabled; wrong baseline for this study')
    if args.split == 'test':
        print('[CAUTION] Test must NOT be used to select ablation variants. Use val for development.', flush=True)

    d = cfg['data']
    dataset = build_dual_resolution_dataset(
        rgb_yaml=d['rgb'], tir_yaml=d['tir'], split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),
        pair_mode=d.get('pair_mode', 'relative'),
        augment=False, tir_channels=3, strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        dataset, batch_size=args.batch, workers=args.workers,
        shuffle=False, pin_memory=device.type == 'cuda',
    )
    out = Path(args.output).expanduser().resolve()
    json_out, csv_out = out / 'ablation.json', out / 'ablation.csv'
    if (json_out.exists() or csv_out.exists()) and not args.overwrite:
        raise FileExistsError(f'Output exists: {out}; choose a new --output or add --overwrite')

    # Keep all conditions exactly matched. Running same model.eval() throughout
    # is important so BN/Dropout do not drift across conditions.
    methods = ['full', 'no_bhlr', 'no_warp_keep_guide', 'no_align_no_bhlr']
    results = {}
    evaluate = lambda net, batches: validate_metrics(
        net, batches, device, amp=device.type == 'cuda',
        conf_thres=args.conf, iou_thres=args.iou, max_det=300, preview_num=0)

    print('=' * 78, flush=True)
    print('[CHECKPOINT]', weights, flush=True)
    print('[EPOCH]', ck.get('epoch'), '[SPLIT]', args.split,
          '[IMAGES]', len(dataset), '[DEVICE]', str(device), flush=True)
    print('[BHLR SCALES]', core.bhlr_scales, '[ALIGN SCALES]',
          cfg['model']['rlsfa_gtga_centered']['apply_scales'], flush=True)
    print('=' * 78, flush=True)

    for mode in methods:
        print('\n[ABLATE]', mode, flush=True)
        with ablation_mode(core, mode):
            res = evaluate(model, loader)
        results[mode] = res
        print(f"[RESULT] {mode}: P={res['precision']:.6f} "
              f"R={res['recall']:.6f} AP50={res['map50']:.6f} "
              f"AP75={res['map75']:.6f} AP50:95={res['map5095']:.9f}", flush=True)

    if not args.skip_mono:
        # Auxiliary heads (from teachers) are evaluated on their native
        # modalities, with the same conf/iou settings as RGB-T fusion.
        for modality in ('rgb', 'tir'):
            name = 'mono_' + modality
            print('\n[MONO]', modality, flush=True)
            mono_model = MonoEval(model, modality).to(device).eval()
            mono_loader = loader if modality == 'rgb' else TirGTLoader(loader)
            res = evaluate(mono_model, mono_loader)
            results[name] = res
            print(f'[RESULT] {name}: AP50:95={res["map5095"]:.9f}', flush=True)

    if args.verify_restore:
        print('\n[CHECK] Full model after restoring switches', flush=True)
        again = evaluate(model, loader)
        diff = abs(again['map5095'] - results['full']['map5095'])
        print(f'[RESTORE CHECK] abs(AP delta)={diff:.12g}', flush=True)
        if diff > 1e-8:
            raise RuntimeError(f'Ablation restore changed original model prediction AP by {diff}')

    pp = lambda a, b: 100.0 * (results[a]['map5095'] - results[b]['map5095'])
    deltas = {
        'BHLR_incremental_pp_full_minus_no_bhlr': pp('full', 'no_bhlr'),
        'Alignment_incremental_pp_bhlr_off': pp('no_bhlr', 'no_align_no_bhlr'),
        'Warp_incremental_pp_guidance_fixed': pp('full', 'no_warp_keep_guide'),
        'Both_modules_incremental_pp_full_minus_double_off': pp('full', 'no_align_no_bhlr'),
    }
    if not args.skip_mono:
        deltas['Fusion_minus_RGB_aux_pp'] = pp('full', 'mono_rgb')
        deltas['Fusion_minus_TIR_aux_pp'] = pp('full', 'mono_tir')

    report = {
        'checkpoint': str(weights), 'checkpoint_epoch': ck.get('epoch'),
        'split': args.split, 'device': str(device), 'batch': args.batch,
        'conf': args.conf, 'iou': args.iou, 'images': len(dataset),
        'bhlr_scales': list(core.bhlr_scales),
        'alignment_scales': cfg['model']['rlsfa_gtga_centered']['apply_scales'],
        'source_root': str(root),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'results': results, 'deltas_pp': deltas,
        'interpretation': {
            'no_bhlr': 'BHLR forward disabled; trained alignment intact',
            'no_warp_keep_guide': 'Spatial TIR warp bypassed; original alignment support/debug maps still drive BHLR. NOT full alignment-off',
            'no_align_no_bhlr': 'Both alignment and BHLR disabled; original raw TIR sent to add fusion',
            'mono_rgb_tir': 'Trained auxiliary single-modal heads, not the standalone teacher checkpoints',
            'KD': 'KD is training-only. Same-checkpoint inference ablations do NOT measure causal KD effect',
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'precision', 'recall', 'map50', 'map75', 'map5095',
                         'delta_vs_full_pp', 'images', 'instances'])
        for name, vals in results.items():
            writer.writerow([name] + [vals.get(k, '') for k in
                         ('precision', 'recall', 'map50', 'map75', 'map5095')] +
                         [100*(vals['map5095'] - results['full']['map5095']),
                          vals.get('images', ''), vals.get('instances', '')])

    print('\n' + '=' * 78, flush=True)
    for k, v in deltas.items():
        print(f'[DELTA] {k} = {v:+.6f} pp', flush=True)
    print('[SAVED]', json_out, flush=True)
    print('[SAVED]', csv_out, flush=True)
    print('Note: positive BHLR delta => BHLR helps on this fixed checkpoint.', flush=True)
    print('Note: positive alignment delta => learned warp helps with BHLR disabled.', flush=True)


if __name__ == '__main__':
    main()
