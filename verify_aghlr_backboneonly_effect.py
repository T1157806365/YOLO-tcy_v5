#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-checkpoint functional ablations for YOLO-tcy_v5 AG-HLR RegP3.

No training, no checkpoint changes. Test four conditions:
  - full_real:         AG-HLR on, actual RGB high resolution.
  - aghlr_off:         AG-HLR off, actual RGB high resolution.
  - no_real_hr:        AG-HLR on, high RGB replaced by RGB_semantic upsampled.
  - no_align_guidance: AG-HLR on, learned alignment *support* set to zero;
                       geometric RGB-T warp stays intact.
Also compare learned geometric AlignIoU to unwarped paired-GT box IoU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from ga_monoalign_pcgrad.factory import load_checkpoint
from ga_monoalign_pcgrad.metrics import geometry_iou
from ga_monoalign_pcgrad.teachers import xywh2xyxy, box_iou_single


class RGBHighUpsampleLoader:
    """Preserve semantic RGB/TIR/GT; remove *extra* resolution only."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for batch in self.base:
            b = dict(batch)
            high = b['rgb_img']
            sem = b['rgb_semantic_img']
            # Images produced by this repo's dataset are uint8 in range 0..255.
            # preprocess_batch later converts /255, as usual.
            up = F.interpolate(
                sem.float(), size=high.shape[-2:],
                mode='bilinear', align_corners=False,
            ).round().clamp(0, 255)
            b['rgb_img'] = up.to(high.dtype)
            yield b


def unwarped_pair_gt_iou(loader):
    """GT box overlap when the TIR-to-RGB transform is the identity."""
    values = []
    for batch in loader:
        bsz = int(batch['rgb_semantic_img'].shape[0])
        for i in range(bsz):
            rb = batch['rgb_semantic_bboxes'][batch['rgb_batch_idx'] == i]
            tb = batch['tir_bboxes'][batch['tir_batch_idx'] == i]
            if len(rb) != 1 or len(tb) != 1:
                continue
            rxyxy = xywh2xyxy(rb.float())[0]
            txyxy = xywh2xyxy(tb.float())
            values.append(float(box_iou_single(rxyxy, txyxy).max().item()))
    return {
        'unaligned_identity_iou': sum(values) / len(values) if values else None,
        'identity_gt_pairs': len(values),
    }


def zero_support_hook(module, args, out):
    """Only ablate guidance support; retain the learned warp itself."""
    if not isinstance(out, (tuple, list)) or not out:
        raise RuntimeError('Unsupported alignment_guidance_fusion output signature')
    if not torch.is_tensor(out[0]):
        raise RuntimeError('Guidance fusion support must be a Tensor')
    z = torch.zeros_like(out[0])
    return (z, *out[1:]) if isinstance(out, tuple) else [z, *out[1:]]


def delta(a, b, key):
    return 100.0 * (float(a[key]) - float(b[key]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.70)
    p.add_argument('--output', default='runs/AG_HLR/verified_effectiveness.json')
    p.add_argument('--skip-geometry', action='store_true')
    args = p.parse_args()

    device = select_device(args.device)
    model, checkpoint, cfg = load_checkpoint(args.weights, device)
    if not hasattr(model, '_aghlr_mode'):
        raise RuntimeError('Checkpoint is not an AGHLRStudent. Are the AG-HLR patches installed?')
    if model._aghlr_mode != 'on':
        print('[NOTE] Resetting AG-HLR mode to on for four-way ablation')
        model._aghlr_mode = 'on'
    if not hasattr(model.core, 'alignment_guidance_fusion') or '3' not in model.core.alignment_guidance_fusion:
        raise RuntimeError('Missing alignment-guidance fusion P3 module')
    d = cfg['data']
    data = build_dual_resolution_dataset(
        rgb_yaml=d['rgb'], tir_yaml=d['tir'], split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),
        pair_mode=d.get('pair_mode', 'relative'),
        augment=False, tir_channels=3, strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        data, batch_size=args.batch, workers=args.workers,
        shuffle=False, pin_memory=device.type == 'cuda',
    )
    results = {
        'weights': str(Path(args.weights).resolve()),
        'split': args.split,
        'images_in_dataset': len(data),
        'conditions': {},
        'notes': {
            'aghlr_off': 'Disables AG-HLR P3 box correction; original BHLR generic P3 injection stays bypassed.',
            'no_real_hr': 'RGB high replaced by 640 semantic RGB upsampled to 1920; semantic RGB and TIR untouched.',
            'no_align_guidance': 'P3 learned alignment support replaced by zero; spatial alignment warp is still applied.',
            'comparison': 'All conditions use exactly the same checkpoint and detector evaluation protocol.',
        },
    }
    cases = [
        ('full_real', 'on', False, False),
        ('aghlr_off', 'off', False, False),
        ('no_real_hr', 'on', True, False),
        ('no_align_guidance', 'on', False, True),
    ]
    for name, mode, upsample, suppress_guide in cases:
        model._aghlr_mode = mode
        hook = None
        try:
            if suppress_guide:
                hook = model.core.alignment_guidance_fusion['3'].register_forward_hook(
                    zero_support_hook, prepend=True,
                )
            active_loader = RGBHighUpsampleLoader(loader) if upsample else loader
            print('\n' + '=' * 65)
            print('[CASE]', name, '| AGHLR=', mode,
                  '| remove_real_HR=', upsample,
                  '| remove_guidance=', suppress_guide, flush=True)
            with torch.inference_mode():
                r = validate_metrics(
                    model, active_loader, device,
                    amp=device.type == 'cuda',
                    conf_thres=args.conf, iou_thres=args.iou,
                    max_det=300, preview_num=0,
                )
            results['conditions'][name] = r
            print('[RESULT]', name,
                  'AP50=%.5f AP75=%.5f AP50:95=%.5f' %
                  (r['map50'], r['map75'], r['map5095']), flush=True)
        finally:
            if hook is not None:
                hook.remove()

    full = results['conditions']['full_real']
    comparisons = {
        'AGHLR_vs_OFF_pp': {k: delta(full, results['conditions']['aghlr_off'], k)
                            for k in ('map50', 'map75', 'map5095')},
        'real1920_vs_upsampled640_pp': {k: delta(full, results['conditions']['no_real_hr'], k)
                                       for k in ('map50', 'map75', 'map5095')},
        'align_guidance_vs_zero_pp': {k: delta(full, results['conditions']['no_align_guidance'], k)
                                      for k in ('map50', 'map75', 'map5095')},
    }
    results['delta_pp'] = comparisons

    if not args.skip_geometry:
        print('\n[GEOMETRY] Evaluating paired GT identity overlap and learned warp ...', flush=True)
        results['identity_geometry'] = unwarped_pair_gt_iou(loader)
        model._aghlr_mode = 'off'
        with torch.inference_mode():
            results['learned_geometry'] = geometry_iou(
                model, loader, device, amp=False, max_images=len(data),
            )
        unaligned = results['identity_geometry']['unaligned_identity_iou']
        learned = results['learned_geometry'].get('align_iou')
        results['alignment_iou_improvement_pp'] = (
            100.0 * (learned - unaligned)
            if isinstance(learned, (int, float)) and isinstance(unaligned, (int, float))
            and math.isfinite(learned) and math.isfinite(unaligned)
            else None
        )

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing result: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n' + '=' * 65)
    print('[SUMMARY] AP delta in percentage points:')
    for contrast, values in comparisons.items():
        print(' ', contrast, ' | '.join('%s=%+.4f' % (k, v) for k,v in values.items()))
    if not args.skip_geometry:
        print(' [GEOMETRY] identity IoU=%.5f, learned IoU=%.5f, change=%+.4f pp' % (
            results['identity_geometry']['unaligned_identity_iou'],
            results['learned_geometry']['align_iou'],
            results['alignment_iou_improvement_pp']))
    print('[SAVED]', out)


if __name__ == '__main__':
    main()
