#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-retrain, fixed-ROI local TIR ablation for GMLS-v2 OneHead Hybrid v2.

The global encoder, GMLS-v2, ROI guide, Top-K centers, native RGB crops, and
postprocessing are IDENTICAL across the three inference variants.

A: full_rgbt       -- original GTGA-aligned local TIR (checkpoint baseline)
B: local_tir_zero  -- zero local TIR *pixels* (not a retrained RGB-only network)
C: local_tir_raw   -- TIR sampled with q=p rather than q=S*p+t, same ROI pixels

This quantifies LOCAL TIR and LOCAL geometric sampling contributions
conditional on the ORIGINAL RGB-T-dependent ROI selection. It does NOT
isolate TIR's contribution to the global ROI selector itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Any

import numpy as np
import torch
import torch.nn.functional as F

MODES = ('full_rgbt', 'local_tir_zero', 'local_tir_raw')
MODE_LABELS = {
    'full_rgbt': 'Full RGB-T: GTGA-aligned TIR',
    'local_tir_zero': 'Fixed ROI: zero local TIR pixels',
    'local_tir_raw': 'Fixed ROI: raw TIR q=p (no local GTGA)',
}


def _meta_arrays(metas, reference):
    """Use the same letterbox metadata convention as geometry.sample_roi_pair."""
    scale, pad = [], []
    for m in metas:
        r = m['ratio']
        if isinstance(r, (tuple, list)):
            if len(r) != 2 or abs(float(r[0]) - float(r[1])) > 1e-5:
                raise ValueError('Expected isotropic letterbox ratio')
            r = r[0]
        scale.append(float(r))
        pad.append([float(v) for v in m['pad']])
    return reference.new_tensor(scale).reshape(-1, 1), reference.new_tensor(pad)


def sample_identity_tir_with_fixed_roi(batch, extraction):
    """Return TIR crops with q=p, at the exact SAME high-RGB ROI origins.

    This deliberately ignores S and t, without running the ROI selector again.
    Alignment coords follow the original implementation's align_corners=True.
    """
    geo = extraction['roi']
    k = int(extraction['rois_per_image'])
    b = int(batch['rgb_img'].shape[0])
    size = int(geo.size)
    n = b * k
    assert geo.xy0.shape == (n, 2), 'Unexpected crop geometry shape'
    ref = geo.xy0.float()
    hh, ww = batch['rgb_img'].shape[-2:]
    sh, sw = batch['rgb_semantic_img'].shape[-2:]
    high_meta = [m for m in batch['rgb_ratio_pad'] for _ in range(k)]
    sem_meta = [m for m in batch['rgb_semantic_ratio_pad'] for _ in range(k)]
    rh, ph = _meta_arrays(high_meta, ref)
    rs, ps = _meta_arrays(sem_meta, ref)
    rh = rh[:, None]
    rs = rs[:, None]
    ph = ph[:, None, None, :]
    ps = ps[:, None, None, :]
    yy, xx = torch.meshgrid(
        torch.arange(size, device=ref.device, dtype=torch.float32),
        torch.arange(size, device=ref.device, dtype=torch.float32),
        indexing='ij',
    )
    pixel_xy = torch.stack((xx, yy), dim=-1).unsqueeze(0) + ref[:, None, None, :]
    sem_xy = (pixel_xy - ph) / rh.unsqueeze(-1) * rs.unsqueeze(-1) + ps
    p = sem_xy / sem_xy.new_tensor([max(sw - 1, 1), max(sh - 1, 1)])
    tir = batch['tir_img'].repeat_interleave(k, dim=0)
    out = F.grid_sample(
        tir.float(), 2 * p - 1, mode='bilinear',
        padding_mode='border', align_corners=True,
    )
    return out.to(dtype=tir.dtype)


def _center_bbox_iou(a, b):
    alo = a[:2] - a[2:] / 2
    ahi = a[:2] + a[2:] / 2
    blo = b[:2] - b[2:] / 2
    bhi = b[:2] + b[2:] / 2
    inter = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod()
    union = (a[2:].clamp_min(0).prod() + b[2:].clamp_min(0).prod() - inter)
    return float((inter / union.clamp_min(1e-9)).item())


def _paired_geometry_stats(batch, ex):
    """Match runner.evaluate semantics: only images with exactly 1 RGB/TIR GT."""
    info = ex['alignment_info']
    bs = int(batch['rgb_img'].shape[0])
    raw_sum = final_sum = center_sum = 0.
    n = 0
    h, w = batch['rgb_semantic_img'].shape[-2:]
    for bi in range(bs):
        rmask = batch['rgb_batch_idx'] == bi
        tmask = batch['tir_batch_idx'] == bi
        if int(rmask.sum()) != 1 or int(tmask.sum()) != 1:
            continue
        rb = batch['rgb_semantic_bboxes'][rmask][0].float()
        tb = batch['tir_bboxes'][tmask][0].float()
        s = info['sampling_scale'][bi].float().reshape(2).clamp_min(1e-6)
        t = info['sampling_translation'][bi].float().reshape(2)
        back = torch.cat(((tb[:2] - t) / s, tb[2:] / s))
        raw_sum += _center_bbox_iou(rb, tb)
        final_sum += _center_bbox_iou(rb, back)
        center_sum += float(torch.linalg.norm((back[:2] - rb[:2]) * rb.new_tensor([w, h])).item())
        n += 1
    return {'raw_iou_sum': raw_sum, 'align_iou_sum': final_sum,
            'align_center_error_sum': center_sum, 'matched_pairs': n}


def _coverage_by_image(batch, geo, k, min_visible=.6):
    """Purely diagnostic labels, never used to choose ROI centers."""
    b = batch['rgb_img'].shape[0]
    h, w = batch['rgb_img'].shape[-2:]
    index = batch['rgb_batch_idx'].long()
    out = []
    for bi in range(b):
        bb = batch['rgb_bboxes'][index == bi]
        if not len(bb):
            out.append((0, 0))
            continue
        px = bb * bb.new_tensor([w, h, w, h])
        lo = px[:, :2] - px[:, 2:] * .5
        hi = px[:, :2] + px[:, 2:] * .5
        vmax = torch.zeros(len(bb), dtype=px.dtype, device=px.device)
        for ki in range(k):
            origin = geo.xy0[bi*k+ki].detach()
            inter_lo = torch.maximum(lo, origin)
            inter_hi = torch.minimum(hi, origin + geo.size)
            frac = (inter_hi - inter_lo).clamp_min(0).prod(-1) / px[:, 2:].prod(-1).clamp_min(1e-8)
            vmax = torch.maximum(vmax, frac)
        out.append((int((vmax >= float(min_visible)).sum().item()), len(bb)))
    return out


def _to_bgr(x):
    import cv2
    arr = (x.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def _save_preview(preview_path, batch, ex, raw_tir, image_index, seq_number):
    """Unresized native crops; labels outside the image in a separate strip."""
    import cv2
    b = int(image_index)
    k = int(ex['rois_per_image'])
    preview_path.mkdir(parents=True, exist_ok=True)
    for ki in range(k):
        ri = b*k+ki
        rgb = _to_bgr(ex['rgb_roi'][ri])
        aligned = _to_bgr(ex['tir_roi'][ri])
        unaligned = _to_bgr(raw_tir[ri])
        blend = cv2.addWeighted(rgb, .5, aligned, .5, 0)
        pieces = [rgb, aligned, unaligned, blend]
        labels = ['RGB native 640 ROI', 'TIR GTGA aligned', 'TIR raw q=p', 'RGB/TIR blend']
        tiles = []
        for img, label in zip(pieces, labels):
            header = np.full((34, img.shape[1], 3), 18, dtype=np.uint8)
            cv2.putText(header, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate((header,img), axis=0))
        panel = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(preview_path/f'{seq_number:04d}_roi{ki+1}_rgb_tir_geometry.png'), panel)


def _safe_json(x):
    if isinstance(x, dict):
        return {str(k): _safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_json(v) for v in x]
    if isinstance(x, (np.number,)):
        return _safe_json(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@torch.inference_mode()
def run(args):
    from tqdm import tqdm
    from ultralytics.utils.metrics import DetMetrics
    from engine.runtime import load_config, select_device, preprocess_batch, autocast_context
    from engine.evaluator import prepare_semantic_ground_truth, process_single_image, postprocess_predictions
    from gmls_roi_onehead_v2.model import GMLSV2OneHeadHybrid
    from gmls_roi_onehead_v2.runner import build_loader, _load_checkpoint
    from gmls_roi_onehead_v2.multi_roi import per_image_merge

    cfg = load_config(args.cfg)
    device = select_device(args.device)
    weights = Path(args.weights or (Path(cfg['output']['project']) / cfg['output']['name'] / 'weights' / 'best.pt')).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights}')
    if weights.parent.name != 'weights':
        raise ValueError('--weights must point to an experiment weights/best.pt or weights/last.pt')
    run_dir = weights.parent.parent
    output = run_dir/'evaluation'/'fixed_roi_ablation'/args.split
    output.mkdir(parents=True, exist_ok=True)
    suffix = '_smoke' if args.smoke else ''
    amp = bool(cfg['train'].get('amp', True)) and device.type == 'cuda'
    conf = float(cfg.get('eval', {}).get('conf', .001)) if args.conf is None else float(args.conf)
    iou = float(cfg.get('eval', {}).get('iou', .70)) if args.iou is None else float(args.iou)
    print('[FIXED ROI] ROI selection once per batch; baseline, TIR zero, and TIR identity share exact RGB crops.', flush=True)
    print(f'[OUTPUT] {output}', flush=True)
    net = GMLSV2OneHeadHybrid(cfg, verbose=True).to(device)
    ckpt = _load_checkpoint(weights, net)
    net.eval()
    print(f'[LOADED] epoch={ckpt.get("epoch")} checkpoint={weights}', flush=True)
    loader, _ = build_loader(cfg, args.split, int(args.batch), int(args.workers), rank=0, world=1)
    metric_objects = {m: DetMetrics(names={0: 'UAV'}) for m in MODES}
    counts = {m: 0 for m in MODES}
    geo_totals = {'raw_iou_sum': 0., 'align_iou_sum': 0., 'align_center_error_sum': 0., 'matched_pairs': 0}
    hit = total = samples = 0
    rec_rows = []
    n_preview = 0
    t0 = time.perf_counter()
    ecfg = cfg.get('eval', {})
    max_det = int(ecfg.get('max_det', 300))
    nms_iou = float(ecfg.get('crop_nms_iou', .5))
    min_visible = float(cfg['model']['roi_single_head'].get('min_gt_visible', .6))
    iouv = torch.linspace(.5, .95, 10, device=device)
    for batch_id, batch in enumerate(tqdm(loader, desc='Fixed-ROI ablation', ncols=110)):
        batch = preprocess_batch(batch, device)
        with autocast_context(device, enabled=amp):
            # Execute global feature extraction, GMLS, and ROI Guide exactly ONCE.
            ex = net.encode_and_crop(batch, return_debug=False, train_rois=False)
            k = int(ex['rois_per_image'])
            bs = int(batch['rgb_img'].shape[0])
            # Idem ROI image coordinates and RGB pixels for every counterfactual.
            rgb = ex['rgb_roi']
            aligned_tir = ex['tir_roi']
            raw_tir = sample_identity_tir_with_fixed_roi(batch, ex)
            modalities = {
                'full_rgbt': aligned_tir,
                'local_tir_zero': torch.zeros_like(aligned_tir),
                'local_tir_raw': raw_tir,
            }
            predictions = {}
            for mode in MODES:
                loc_raw = net.detector.predict(rgb, modalities[mode])
                loc_preds = postprocess_predictions(loc_raw, net.detector,
                                                    conf_thres=conf, iou_thres=iou)
                predictions[mode] = per_image_merge(loc_preds, ex['roi'], batch, k,
                                                   conf=conf, nms_iou=nms_iou, max_det=max_det)
        coverage = _coverage_by_image(batch, ex['roi'], k, min_visible=min_visible)
        stat = _paired_geometry_stats(batch, ex)
        for key in geo_totals:
            geo_totals[key] += stat[key]
        hit += sum(x[0] for x in coverage)
        total += sum(x[1] for x in coverage)
        for bi in range(bs):
            gt = prepare_semantic_ground_truth(batch, bi)
            target_cls = gt['cls'].detach().float().cpu().numpy()
            row = {'index': samples+bi, 'image': str(batch['rgb_path'][bi]),
                   'gt_count': len(target_cls), 'roi_hit': coverage[bi][0],
                   'roi_gt_count': coverage[bi][1]}
            for mode in MODES:
                pred = predictions[mode][bi]
                tp = process_single_image(pred, gt, iouv)
                metric_objects[mode].update_stats({
                    'tp': tp,
                    'conf': pred['conf'].detach().float().cpu().numpy(),
                    'pred_cls': pred['cls'].detach().float().cpu().numpy(),
                    'target_cls': target_cls,
                    'target_img': np.unique(target_cls),
                    'im_name': Path(batch['rgb_path'][bi]).name,
                })
                row[f'{mode}_preds'] = int(len(pred['bboxes']))
                row[f'{mode}_tp50'] = int(tp[:,0].sum()) if getattr(tp, 'ndim', 0)==2 else 0
                row[f'{mode}_maxconf'] = float(pred['conf'].max().item()) if len(pred['conf']) else 0.
            rec_rows.append(row)
        if args.preview_num and n_preview < args.preview_num:
            for bi in range(min(bs, args.preview_num-n_preview)):
                _save_preview(output/'previews',batch,ex,raw_tir,bi,n_preview+bi)
            n_preview += min(bs,args.preview_num-n_preview)
        samples += bs
        if args.smoke:
            break
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    output_modes = {}
    for mode, metric in metric_objects.items():
        metric.process(plot=False)
        p, r, ap50, ap5095 = metric.mean_results()
        output_modes[mode] = {
            'P': float(p), 'R': float(r), 'AP50': float(ap50),
            'AP75': float(metric.box.map75), 'AP50:95': float(ap5095),
            'detections': int(sum(x[f'{mode}_preds'] for x in rec_rows)),
        }
    baseline = output_modes['full_rgbt']
    for m in MODES:
        output_modes[m]['delta_AP50:95_pp'] = (output_modes[m]['AP50:95'] - baseline['AP50:95'])*100
        output_modes[m]['delta_AP75_pp'] = (output_modes[m]['AP75'] - baseline['AP75'])*100
    geo = {
        'original_rgb_tir_box_iou': geo_totals['raw_iou_sum']/max(geo_totals['matched_pairs'],1),
        'gtga_aligned_box_iou': geo_totals['align_iou_sum']/max(geo_totals['matched_pairs'],1),
        'align_iou_gain': (geo_totals['align_iou_sum']-geo_totals['raw_iou_sum'])/max(geo_totals['matched_pairs'],1),
        'center_error_sem_px': geo_totals['align_center_error_sum']/max(geo_totals['matched_pairs'],1),
        'matched_pairs': geo_totals['matched_pairs'],
        'roi_gt_coverage': hit/max(total,1), 'roi_hit':hit, 'roi_gt_total':total,
    }
    report = {
        'checkpoint': str(weights), 'epoch': ckpt.get('epoch'),
        'config': str(Path(args.cfg).resolve()), 'split': args.split,
        'image_count': samples, 'batch': args.batch, 'amp': amp, 'conf': conf, 'iou': iou,
        'smoke': bool(args.smoke), 'rois_per_image': int(net.num_rois),
        'method': 'One encode_and_crop per batch, three local-TIR alternatives at identical ROI coordinates',
        'limitation': 'Fixed ROI still depends on original RGB-T global selector; zero-TIR is inference occlusion, not retrained RGB-only.',
        'geometry': geo, 'modes': output_modes, 'wall_seconds_including_data_and_metrics': elapsed,
    }
    path_json = output/f'metrics{suffix}.json'
    path_csv = output/f'metrics{suffix}.csv'
    path_txt = output/f'metrics{suffix}.txt'
    path_images = output/f'per_image{suffix}.csv'
    path_json.write_text(json.dumps(_safe_json(report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    keys = ['mode', 'P (%)','R (%)','AP50 (%)','AP75 (%)','AP50:95 (%)','Delta AP50:95 (pp)','Delta AP75 (pp)','Predictions']
    with path_csv.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
        for mode in MODES:
            r=output_modes[mode]
            wr.writerow({'mode':mode, 'P (%)':r['P']*100, 'R (%)':r['R']*100,
                         'AP50 (%)':r['AP50']*100, 'AP75 (%)':r['AP75']*100,
                         'AP50:95 (%)':r['AP50:95']*100,
                         'Delta AP50:95 (pp)':r['delta_AP50:95_pp'],
                         'Delta AP75 (pp)':r['delta_AP75_pp'],
                         'Predictions':r['detections']})
    if rec_rows:
        with path_images.open('w',encoding='utf-8-sig',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rec_rows[0].keys()));wr.writeheader();wr.writerows(rec_rows)
    lines=['='*94, 'FIXED TOP-2 ROI — LOCAL TIR / GTGA ABLATION (NO RETRAIN)',
           f'Split={args.split}  images={samples}  checkpoint epoch={ckpt.get("epoch")}',
           '-'*94, f'{"Mode":<19} {"P (%)":>9} {"R (%)":>9} {"AP50":>9} {"AP75":>9} {"AP50:95":>10} {"Delta pp":>10}']
    for mode in MODES:
        rr=output_modes[mode]
        lines.append(f'{mode:<19} {rr["P"]*100:9.3f} {rr["R"]*100:9.3f} {rr["AP50"]*100:9.3f} {rr["AP75"]*100:9.3f} {rr["AP50:95"]*100:10.3f} {rr["delta_AP50:95_pp"]:+10.3f}')
    lines.extend(['-'*94,
                  f'Original box IoU (no alignment): {geo["original_rgb_tir_box_iou"]:.5f}',
                  f'GTGA final AlignIoU:             {geo["gtga_aligned_box_iou"]:.5f}',
                  f'Geometry IoU improvement:         {geo["align_iou_gain"]:+.5f}',
                  f'ROI union coverage:               {geo["roi_gt_coverage"]:.5f} ({hit}/{total} GT objects)',
                  f'Center error (640 semantic px):   {geo["center_error_sem_px"]:.2f}',
                  'IMPORTANT: GT does NOT select ROIs; zero-TIR is not equivalent to trained RGB-only.',
                  'IMPORTANT: all modes retain the ORIGINAL RGB-T-dependent ROI guidance.',
                  '='*94])
    path_txt.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    for p in (path_json,path_csv,path_txt,path_images):print(f'[SAVED] {p}',flush=True)
    if args.preview_num:print(f'[PREVIEW] {output/"previews"}',flush=True)
    if args.smoke:print('[SMOKE ONLY] Metrics represent one batch, NOT the complete validation/test set.',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg',required=True)
    p.add_argument('--weights',default=None)
    p.add_argument('--split',default='val',choices=('val','test'))
    p.add_argument('--device',default='4')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=None,help='Default YAML eval.conf')
    p.add_argument('--iou',type=float,default=None,help='Default YAML eval.iou')
    p.add_argument('--preview-num',type=int,default=4,help='Sample previews, 0=off')
    p.add_argument('--smoke',action='store_true',help='Run only first batch')
    args=p.parse_args()
    if args.batch<1 or args.workers<0 or args.preview_num<0:
        p.error('--batch must be >=1; --workers and --preview-num must be >=0')
    run(args)


if __name__=='__main__':
    main()
