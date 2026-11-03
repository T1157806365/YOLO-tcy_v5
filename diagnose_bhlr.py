#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only BHLR information-utilization diagnosis for YOLO-tcy_v5 teacher100.

Works on the original and GuidanceGrad010 weights. Runs two inference forwards
per paired batch (BHLR ON, BHLR OFF), records input/feature/detail/ROI statistics,
computes the original YOLO dataset-wide AP metrics for both, and saves examples.

Does not edit model weights, code, YAML, or training state on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--repo-root', default='.', help='YOLO-tcy_v5_teacher100 source checkout')
    p.add_argument('--weights', required=True, help='Original 61.16%% or GuidanceGrad checkpoint best.pt')
    p.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    p.add_argument('--device', default='3')
    p.add_argument('--batch', type=int, default=1, help='1 strongly recommended to limit 1920 debug tensor memory')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--conf', type=float, default=0.001, help='AP calculation: same as original evaluator')
    p.add_argument('--iou', type=float, default=0.70, help='AP postprocess IoU threshold')
    p.add_argument('--diagnostic-conf', type=float, default=0.25, help='Per-image prediction IoU threshold')
    p.add_argument('--roi-pad-cells', type=int, default=2, help='Extra feature cells around the GT bbox for ROI diagnostics')
    p.add_argument('--max-images', type=int, default=0, help='0=whole split; use 24 for a quick smoke test')
    p.add_argument('--vis-num', type=int, default=12, help='Number of 8-panel jpg examples')
    p.add_argument('--output', required=True, help='New output folder; refuses to overwrite existing summary')
    p.add_argument('--no-amp', action='store_true', help='Use FP32 forward to investigate precision issues')
    return p.parse_args()


def roi_mask(boxes: torch.Tensor, height: int, width: int, pad: int, device: torch.device):
    """GT normalized xywh in RGB semantic coordinates -> boolean feature map ROI.

    Pixel positions are scaled separately per grid. Tiny boxes are kept at
    least one cell. The padded ROI is a diagnostic region, not object GT.
    """
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for box in boxes:
        xc, yc, bw, bh = [float(v) for v in box.detach().cpu().reshape(-1)[:4]]
        x0 = max(0, min(width - 1, math.floor((xc - bw * .5) * width) - pad))
        y0 = max(0, min(height - 1, math.floor((yc - bh * .5) * height) - pad))
        x1 = max(x0 + 1, min(width, math.ceil((xc + bw * .5) * width) + pad))
        y1 = max(y0 + 1, min(height, math.ceil((yc + bh * .5) * height) + pad))
        mask[y0:y1, x0:x1] = True
    return mask


def energy_map(tensor: torch.Tensor):
    """B,C,H,W -> B,1,H,W mean channel squared magnitude (FP32)."""
    x = tensor.detach().float()
    return x.square().mean(dim=1, keepdim=True)


def region_stats(e: torch.Tensor, mask: torch.Tensor, prefix: str):
    """Input energy H,W and ROI mask H,W. Return scalar diagnostics."""
    e = e.float()
    total = e.numel()
    area = int(mask.sum().item())
    background = ~mask
    target = float(e[mask].mean().item()) if area else float('nan')
    bg = float(e[background].mean().item()) if bool(background.any()) else float('nan')
    mean = float(e.mean().item())
    return {
        prefix+'_roi_mse': target,
        prefix+'_bg_mse': bg,
        prefix+'_global_mse': mean,
        prefix+'_roi_bg_energy_ratio': (target+EPS)/(bg+EPS) if math.isfinite(target) and math.isfinite(bg) else float('nan'),
        prefix+'_roi_area_fraction': area / max(1, total),
    }


def safe_rms_ratio(a: torch.Tensor, b: torch.Tensor, mask=None):
    """Relative RMS of tensor a to tensor b; same B,C,H,W spatial alignment."""
    a = energy_map(a)[0, 0]
    b = energy_map(b)[0, 0]
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(torch.sqrt((a.mean()+EPS)/(b.mean()+EPS)).item())


def make_base_tile(rgb_chw: torch.Tensor, wh: int = 320):
    im = rgb_chw.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()
    im = np.clip(im*255, 0,255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(im,cv2.COLOR_RGB2BGR),(wh,wh),interpolation=cv2.INTER_AREA)


def draw_roi(frame: np.ndarray, boxes: torch.Tensor, pad_features=0):
    h,w = frame.shape[:2]
    for box in boxes.detach().float().cpu():
        xc,yc,bw,bh = box.tolist()[:4]
        x0=int(np.clip((xc-bw*.5)*w,0,w-1));y0=int(np.clip((yc-bh*.5)*h,0,h-1))
        x1=int(np.clip((xc+bw*.5)*w,0,w-1));y1=int(np.clip((yc+bh*.5)*h,0,h-1))
        if (x1-x0)<5 or (y1-y0)<5:
            px=int(np.clip(xc*w,0,w-1));py=int(np.clip(yc*h,0,h-1))
            cv2.drawMarker(frame,(px,py),(45,255,65),cv2.MARKER_CROSS,12,1)
        cv2.rectangle(frame, (x0,y0),(max(x0+1,x1),max(y0+1,y1)),(45,255,65),1)
    return frame


def heat_tile(e_hw: torch.Tensor, size=320):
    vals=e_hw.detach().float().cpu().numpy()
    vals=np.nan_to_num(vals,nan=0,posinf=0,neginf=0)
    vals=np.maximum(vals,0)
    # The same map-specific normalization is applied per panel; colors are NOT
    # comparable as absolute magnitude between different panels.
    p=float(np.percentile(vals,99))
    scaled=np.clip(vals/max(p,1e-12),0,1)
    out=np.clip(scaled*255,0,255).astype(np.uint8)
    out=cv2.resize(out,(size,size),interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(out,cv2.COLORMAP_TURBO)


def add_title(image, title):
    out=image.copy()
    h,w=out.shape[:2]
    cv2.rectangle(out,(0,0),(w,29),(0,0,0),-1)
    cv2.putText(out,title[:41],(7,20),cv2.FONT_HERSHEY_SIMPLEX,0.53,(255,255,255),1,cv2.LINE_AA)
    return out


def save_visual(path, name, boxes, rgb, maps, row):
    order=[
        ('RGB 640 | GT',None),
        ('1920 residual (energy)',maps['input_residual']),
        ('detail before pooling',maps['detail_raw']),
        ('detail after pooling',maps['detail_scale']),
        ('alignment support',maps['support']),
        ('selector | multiplier-1',maps['selector_effect']),
        ('BHLR injected energy',maps['injected']),
        ('RGB P3 raw energy',maps['raw_p3']),
    ]
    panels=[]
    for title,emap in order:
        tile=make_base_tile(rgb) if emap is None else heat_tile(emap)
        draw_roi(tile,boxes)
        panels.append(add_title(tile,title))
    top=np.concatenate(panels[:4],axis=1)
    bottom=np.concatenate(panels[4:],axis=1)
    board=np.concatenate([top,bottom],axis=0)
    foot=f"{name} | gamma={row['gamma']:.5f} inject/rgb={row['inject_p3_rms_ratio']:.4f} | padded ROI={row['roi_fraction_p3']:.4f}"
    cv2.rectangle(board,(0,board.shape[0]-27),(board.shape[1]-1,board.shape[0]-1),(0,0,0),-1)
    cv2.putText(board,foot[:170],(7,board.shape[0]-7),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),board):
        raise IOError(f'Failed to write {path}')


def pred_image_stats(pred, gt, threshold):
    from ultralytics.utils.metrics import box_iou
    total=int(pred['conf'].numel())
    keep=pred['conf'] >= float(threshold)
    selected=int(keep.sum().item())
    if selected == 0:
        return {'num_dets':total,'num_dets_conf25':0,'best_gt_iou_conf25':0.0,'max_conf':0.0}
    boxes=pred['bboxes'][keep]
    maxconf=float(pred['conf'][keep].max().item())
    if not gt['bboxes'].numel():
        best=float('nan')
    else:
        best=float(box_iou(gt['bboxes'], boxes).max().item())
    return {'num_dets':total,'num_dets_conf25':selected,'best_gt_iou_conf25':best,'max_conf':maxconf}


def update_ap(metrics, pred, gt, sample_path, iouv):
    from engine.evaluator import process_single_image
    tp=process_single_image(pred,gt,iouv)
    target_cls=gt['cls'].detach().float().cpu().numpy()
    if pred['cls'].numel():
        conf=pred['conf'].detach().float().cpu().numpy()
        cls=pred['cls'].detach().float().cpu().numpy()
    else:
        conf=np.zeros(0,dtype=np.float32)
        cls=np.zeros(0,dtype=np.float32)
    metrics.update_stats({
        'tp':tp,'conf':conf,'pred_cls':cls,'target_cls':target_cls,
        'target_img':np.unique(target_cls),'im_name':Path(sample_path).name,
    })


def finalize_ap(metrics, seen):
    metrics.process(plot=False)
    p,r,ap50,ap=metrics.mean_results()
    return {'precision':float(p),'recall':float(r),'map50':float(ap50),
            'map75':float(metrics.box.map75),'map5095':float(ap),'images':seen}


def describe(values):
    x=np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))],dtype=np.float64)
    if x.size==0:return {'count':0}
    return { 'count':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),
             'p10':float(np.percentile(x,10)),'p90':float(np.percentile(x,90)),
             'min':float(x.min()),'max':float(x.max()) }


def save_csv(path, rows):
    if not rows: raise ValueError('Cannot save empty diagnostics')
    keys=list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='raise')
        w.writeheader();w.writerows(rows)



def make_report(summary):
    stat=summary['per_image_statistic_summary']
    def med(key):return float(stat[key].get('median',float('nan')))
    on=summary['full'];off=summary['no_bhlr']
    lines=[
        '# BHLR 高分辨率信息利用诊断报告',
        '',
        f'- Checkpoint：`{summary["checkpoint"]}`',
        f'- 分割：`{summary["split"]}`；诊断图像：{summary["num_images"]} / {summary["dataset_images"]}',
        f'- 模型Epoch：{summary["epoch"]}；BHLR-P3所在Backbone层：{summary["p3_layer_index"]}',
        '',
        '## 1. 同权重BHLR ON/OFF检测表现',
        '',
        '| 指标 | 完整模型 | 暂时关闭BHLR | 差值（pp） |',
        '|---|---:|---:|---:|',
    ]
    for key,label in [('map50','AP50'),('map75','AP75'),('map5095','AP50:95')]:
        lines.append(f'| {label} | {100*on[key]:.4f}% | {100*off[key]:.4f}% | {100*(on[key]-off[key]):+.4f} |')
    lines += [
        '',
        '## 2. 累积信息与细节注入',
        '',
        '| 指标（逐图中位数） | 结果 | 解释 |',
        '|---|---:|---|',
        f'| BHLR Gamma | {med("gamma"):.6f} | tanh后的实际残差系数 |',
        f'| 注入特征 / 原始P3（全图RMS） | {med("inject_p3_rms_ratio"):.5f} | 残差幅度相对P3基特征 |',
        f'| Full/NoBHLR融合P3差值 / 原融合P3（全图RMS） | {med("fusion_p3_change_rms_ratio"):.5f} | 检查细节是否真正改变融合特征 |',
        f'| 注入特征 / 原始P3（目标ROI RMS） | {med("inject_roi_rms_ratio"):.5f} | 无人机附近区域 |',
        f'| 注入特征 / 原始P3（背景RMS） | {med("inject_bg_rms_ratio"):.5f} | ROI以外 |',
        f'| 压缩前细节ROI/背景能量比 | {med("detail_before_pool_roi_bg_energy_ratio"):.4f} | 320特征网格 |',
        f'| 压缩后细节ROI/背景能量比 | {med("detail_after_pool_roi_bg_energy_ratio"):.4f} | P3特征网格 |',
        f'| 压缩后/前区域对比度比值 | {med("compression_contrast_relative"):.4f} | 仅指响应对比度，非精确信息保留率 |',
        f'| 注入响应相对于原P3的目标富集程度 | {med("injected_contrast_vs_raw_p3"):.4f} | >1表示注入特征相对更集中目标ROI |',
        f'| Selector目标ROI均值 | {med("selector_roi_mean"):.4f} | 无人机附近的细节选择倍率 |',
        f'| Selector背景均值 | {med("selector_bg_mean"):.4f} | 目标与背景选择差异 |',
        f'| 对齐引导目标ROI均值 | {med("support_roi_mean"):.4f} | Alignment支持图是否聚焦目标 |',
        f'| 对齐引导背景均值 | {med("support_bg_mean"):.4f} | Alignment支持图背景响应 |',
        '',
        '## 3. 判断路径',
        '',
        '1. 先看全图与ROI注入RMS比值：如果很小，说明BHLR注入的有效幅度偏小。',
        '2. 再看压缩前后ROI/背景能量比：如果压缩后明显低于压缩前，提示目标局部响应的相对集中程度下降；**不能直接证明细节丢失**。',
        '3. 再看Selector目标/背景均值：如果差别不大，说明其空间选择性有限（但不代表它在通道上毫无作用）。',
        '4. 最后看AP与逐图最高IoU：如果注入幅度不低，但检测变化很小，应重点检查下游融合/检测头是否使用这部分信息。',
        '',
        '## 4. 数据与方法的边界',
        '',
        '- 目标区域为GT bbox在不同特征尺度上映射并扩展的ROI，**不是模型预测出来的目标位置**。',
        '- 前后细节特征的数值尺度、通道数、非线性不同，因此能量比只适合做空间分布诊断。',
        '- `per_image.csv`中每图最大GT IoU使用另一个检测置信度阈值，用于定位案例；并非逐图AP。',
        '- 可视化每张热力图独立归一化颜色，不要跨图比较绝对能量。',
        '- 同权重ON/OFF只检验现有训练权重对BHLR的依赖，不等于分别训练两种模型。',
    ]
    if summary['partial_split']:
        lines += ['', '> 警告：仅分析部分图像，表中的AP是此子集指标，不能与完整Val/Test AP直接比较。']
    return '\n'.join(lines)+'\n'


def main():
    args=parse_args()
    if args.batch<1 or args.workers<0 or args.max_images<0 or args.roi_pad_cells<0 or args.vis_num<0:
        raise ValueError('batch must be >=1; workers/max-images/roi-pad-cells/vis-num must be >=0')
    root=Path(args.repo_root).expanduser().resolve()
    if not (root/'ga_monoalign_pcgrad'/'factory.py').is_file():
        raise FileNotFoundError('repo-root must contain ga_monoalign_pcgrad/factory.py: '+str(root))
    sys.path.insert(0,str(root))
    os.chdir(root)

    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
    from engine.runtime import select_device,preprocess_batch,autocast_context
    from engine.evaluator import postprocess_predictions,prepare_semantic_ground_truth
    from ga_monoalign_pcgrad.factory import load_checkpoint
    from ultralytics.utils.metrics import DetMetrics

    weights=Path(args.weights).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing nonempty output directory: {output}')
    device=select_device(args.device)
    student,ck,cfg=load_checkpoint(weights,device)
    student.eval();core=student.core
    if not bool(core.bhlr_enabled) or core.bhlr is None:
        raise RuntimeError('Checkpoint has BHLR disabled; no meaningful diagnostic.')
    if 3 not in tuple(core.bhlr_scales):
        raise RuntimeError(f'This tool diagnoses P3; got BHLR scales={core.bhlr_scales}')
    if bool(getattr(core.bhlr, 'detach_external_guidance',False)):
        raise RuntimeError('BHLR internally detaches external guidance; use the matching original core.')
    # Keep geometry routing exactly as used in ordinary evaluation.
    routing=os.environ.get('GMLS_V2_EVAL_MODE','auto').strip().lower()
    if routing!='auto':
        raise RuntimeError('Unset GMLS_V2_EVAL_MODE or set auto to diagnose the trained default routing.')

    d=cfg['data']
    dataset=build_dual_resolution_dataset(
        rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),
        pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(
        dataset,batch_size=args.batch,workers=args.workers,
        shuffle=False,pin_memory=(device.type=='cuda'))

    names=student.names
    if not isinstance(names,dict): names={i:n for i,n in enumerate(names)}
    names={int(k):str(v) for k,v in names.items()}
    metr_on=DetMetrics(names=names)
    metr_off=DetMetrics(names=names)
    iouv=torch.linspace(.50,.95,10,device=device)
    rows=[]; seen=0; count_vis=0; start=time.time()
    analysis_size=min(len(dataset),args.max_images or len(dataset))
    vis_indices=set(int(i) for i in np.linspace(0,max(0,analysis_size-1),min(args.vis_num,analysis_size),dtype=int))
    idx=int(core.scale_to_index[3])
    print(f'[LOAD] checkpoint={weights} epoch={ck.get("epoch")} split={args.split} n={len(dataset)} device={device}',flush=True)
    print(f'[MODEL] gamma_P3_initial={float(torch.tanh(core.bhlr.adapters[str(idx)].gamma_raw).item()):.7f} raw_layer={idx}',flush=True)
    print('[CHECK] Same checkpoint, same GT, same model; ON vs runtime BHLR OFF. No retraining.',flush=True)
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                if args.max_images and seen>=args.max_images:break
                batch=preprocess_batch(raw_batch,device)
                n=int(batch['rgb_img'].shape[0])
                with autocast_context(device,not args.no_amp):
                    core.bhlr_enabled=True
                    full=core.predict(batch['rgb_img'],batch['tir_img'],
                        rgb_semantic=batch['rgb_semantic_img'],
                        return_features=True,return_bhlr_debug=True)
                    if not isinstance(full,dict) or 'bhlr_debug' not in full:
                        raise RuntimeError('Core did not return BHLR debug; incompatible source checkout.')
                    raw_on=full['pred']
                    core.bhlr_enabled=False
                    try:
                        baseline=core.predict(batch['rgb_img'],batch['tir_img'],
                            rgb_semantic=batch['rgb_semantic_img'],return_features=True)
                        raw_off=baseline['pred']
                    finally:
                        core.bhlr_enabled=True
                preds_on=postprocess_predictions(raw_on,student,args.conf,args.iou,300)
                preds_off=postprocess_predictions(raw_off,student,args.conf,args.iou,300)
                dbg_map=full['bhlr_debug']['scales']
                dbg=dbg_map.get(3,dbg_map.get('3'))
                if dbg is None: raise KeyError('P3 BHLR debug absent; got '+str(list(dbg_map)))
                bank=full['bhlr_debug']['detail_bank']
                resid=full['bhlr_debug']['resolution_residual']
                raw_p3=full['rgb_backbone_raw'][idx]
                fused_on=full['fused_backbone'][idx]
                fused_off=baseline['fused_backbone'][idx]
                fused_delta=fused_on.float()-fused_off.float()
                detail=dbg['detail_scale']
                usable=dbg['usable_detail']
                sel=dbg['selector_multiplier']
                support=dbg['targetness']
                gamma=dbg['gamma'].float()
                injected=gamma*usable
                if raw_p3.shape != injected.shape: raise RuntimeError(f'BHLR injection shape mismatch {tuple(raw_p3.shape)} {tuple(injected.shape)}')
                if not torch.isfinite(injected).all(): raise FloatingPointError('BHLR injected feature nonfinite')
                if not torch.isfinite(raw_p3).all(): raise FloatingPointError('RGB P3 feature nonfinite')

                e_bank=energy_map(bank)
                e_resid=energy_map(resid)
                e_detail=energy_map(detail)
                e_inj=energy_map(injected)
                e_raw=energy_map(raw_p3)
                e_sup=support.detach().float()
                e_sel=(sel.detach().float()-1).square()
                p3_hw=tuple(raw_p3.shape[-2:])
                e_resid_p3=F.adaptive_avg_pool2d(e_resid,p3_hw)
                for i in range(n):
                    if args.max_images and seen>=args.max_images: break
                    boxes=batch['rgb_semantic_bboxes'][batch['rgb_batch_idx']==i]
                    name=Path(batch['rgb_path'][i]).name
                    gt=prepare_semantic_ground_truth(batch,i)
                    update_ap(metr_on,preds_on[i],gt,name,iouv)
                    update_ap(metr_off,preds_off[i],gt,name,iouv)
                    p_on=pred_image_stats(preds_on[i],gt,args.diagnostic_conf)
                    p_off=pred_image_stats(preds_off[i],gt,args.diagnostic_conf)
                    mask_p3=roi_mask(boxes,*p3_hw,args.roi_pad_cells,device)
                    mask_bank=roi_mask(boxes,*bank.shape[-2:],args.roi_pad_cells*4,device)
                    mask_resid=roi_mask(boxes,*resid.shape[-2:],args.roi_pad_cells*24,device)
                    row={
                        'image':name,'index':seen,'boxes':int(len(boxes)),
                        'gamma':float(gamma.item()),
                        'roi_fraction_p3':float(mask_p3.float().mean().item()),
                        'inject_p3_rms_ratio':safe_rms_ratio(injected[i:i+1],raw_p3[i:i+1]),
                        'fusion_p3_change_rms_ratio':safe_rms_ratio(fused_delta[i:i+1],fused_off[i:i+1]),
                        'inject_roi_rms_ratio':safe_rms_ratio(injected[i:i+1],raw_p3[i:i+1],mask_p3),
                        'inject_bg_rms_ratio':safe_rms_ratio(injected[i:i+1],raw_p3[i:i+1],~mask_p3),
                        'selector_mean':float(sel[i].float().mean().item()),
                        'selector_roi_mean':float(sel[i,0].float()[mask_p3].mean().item()),
                        'selector_bg_mean':float(sel[i,0].float()[~mask_p3].mean().item()),
                        'support_roi_mean':float(e_sup[i,0][mask_p3].mean().item()),
                        'support_bg_mean':float(e_sup[i,0][~mask_p3].mean().item()),
                        'full_best_iou_conf25':p_on['best_gt_iou_conf25'],
                        'off_best_iou_conf25':p_off['best_gt_iou_conf25'],
                        'full_max_conf':p_on['max_conf'],
                        'off_max_conf':p_off['max_conf'],
                        'full_num_det_conf25':p_on['num_dets_conf25'],
                        'off_num_det_conf25':p_off['num_dets_conf25'],
                    }
                    for prefix,emap,mask in [
                        ('pixel_residual',e_resid[i,0],mask_resid),
                        ('detail_before_pool',e_bank[i,0],mask_bank),
                        ('detail_after_pool',e_detail[i,0],mask_p3),
                        ('bhlr_injected',e_inj[i,0],mask_p3),
                        ('rgb_p3',e_raw[i,0],mask_p3),
                    ]:
                        row.update(region_stats(emap,mask,prefix))
                    row['compression_contrast_relative']=(row['detail_after_pool_roi_bg_energy_ratio']+EPS)/(row['detail_before_pool_roi_bg_energy_ratio']+EPS)
                    row['injected_contrast_vs_raw_p3']=(row['bhlr_injected_roi_bg_energy_ratio']+EPS)/(row['rgb_p3_roi_bg_energy_ratio']+EPS)
                    row['full_minus_off_iou'] = row['full_best_iou_conf25']-row['off_best_iou_conf25']
                    rows.append(row)
                    if seen in vis_indices:
                        # Use actual prepool grid for spatial heatmap rather than its
                        # P3 average; both bboxes are in the same normalized frame.
                        maps={
                            'input_residual':e_resid_p3[i,0],
                            'detail_raw':e_bank[i,0],
                            'detail_scale':e_detail[i,0],
                            'support':e_sup[i,0],
                            'selector_effect':e_sel[i,0],
                            'injected':e_inj[i,0],
                            'raw_p3':e_raw[i,0],
                        }
                        vis_path=output/'visualizations'/f'{seen:04d}_{Path(name).stem}.jpg'
                        save_visual(vis_path,name,boxes,batch['rgb_semantic_img'][i],maps,row)
                        count_vis+=1
                    seen+=1
                if seen%50==0 or seen==len(dataset):
                    print(f'[PROGRESS] {seen}/{min(len(dataset),args.max_images or len(dataset))} elapsed={time.time()-start:.1f}s',flush=True)
                # Guard against holding 1920 debug tensors between batches.
                del full,baseline,raw_on,raw_off,bank,resid,raw_p3,fused_on,fused_off,fused_delta,detail,usable,sel,support,injected
                del e_bank,e_resid,e_detail,e_inj,e_raw,e_sup,e_sel,e_resid_p3
    finally:
        core.bhlr_enabled=True
    if not rows:raise RuntimeError('No images processed')
    metric_on=finalize_ap(metr_on,seen)
    metric_off=finalize_ap(metr_off,seen)
    keys=[k for k in rows[0] if k not in ('image','index','boxes')]
    summary={
        'checkpoint':str(weights),'epoch':ck.get('epoch'),'split':args.split,
        'num_images':seen,'dataset_images':len(dataset),'partial_split':seen!=len(dataset),
        'device':str(device),'amp':not args.no_amp,
        'eval_conf':args.conf,'eval_iou':args.iou,
        'diagnostic_conf':args.diagnostic_conf,'roi_pad_cells':args.roi_pad_cells,
        'p3_layer_index':idx,'bhlr_enabled_in_checkpoint':True,
        'bhlr_scales':list(core.bhlr_scales),
        'full':metric_on,'no_bhlr':metric_off,
        'delta_ap5095_pp':100*(metric_on['map5095']-metric_off['map5095']),
        'delta_ap50_pp':100*(metric_on['map50']-metric_off['map50']),
        'delta_ap75_pp':100*(metric_on['map75']-metric_off['map75']),
        'per_image_statistic_summary':{k:describe([r[k] for r in rows]) for k in keys},
        'interpretation_notes':[
            'ROI means use padded feature-cell GT boxes; ROI sizes differ by map resolution.',
            'Raw input residual, prepool detail and postpool detail have different feature channels/scales: energy ratios are NOT an exact information-retention measure.',
            'Heatmaps are each normalized by their own 99th percentile; colors cannot be compared as absolute magnitudes between panels.',
            'Full/NoBHLR are same-weight functional ablations, NOT independently retrained model ablations.',
            'Per-image max-GT-IoU at diagnostic-conf is a diagnostic, not AP; dataset AP follows engine/evaluator.py.'
        ],
    }
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'per_image.csv',rows)
    save_csv(output/'ranked_iou_changes.csv',sorted(rows,key=lambda r: abs(r['full_minus_off_iou']) if math.isfinite(r['full_minus_off_iou']) else -1,reverse=True))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (output/'report.md').write_text(make_report(summary),encoding='utf-8')
    print('\n'+'='*74)
    print('BHLR INFORMATION UTILIZATION DIAGNOSIS')
    print('Full:    AP50={:.6f}, AP75={:.6f}, AP50:95={:.6f}'.format(metric_on['map50'],metric_on['map75'],metric_on['map5095']))
    print('NoBHLR:  AP50={:.6f}, AP75={:.6f}, AP50:95={:.6f}'.format(metric_off['map50'],metric_off['map75'],metric_off['map5095']))
    print('Delta BHLR: AP50={:+.4f} pp, AP75={:+.4f} pp, AP50:95={:+.4f} pp'.format(
        summary['delta_ap50_pp'],summary['delta_ap75_pp'],summary['delta_ap5095_pp']))
    for field in ('gamma','inject_p3_rms_ratio','fusion_p3_change_rms_ratio','inject_roi_rms_ratio',
                  'inject_bg_rms_ratio','detail_before_pool_roi_bg_energy_ratio',
                  'detail_after_pool_roi_bg_energy_ratio','compression_contrast_relative',
                  'injected_contrast_vs_raw_p3','selector_mean','selector_roi_mean',
                  'selector_bg_mean','support_roi_mean','support_bg_mean','full_minus_off_iou'):
        d0=summary['per_image_statistic_summary'][field]
        print(f'{field:<47}: median={d0.get("median",float("nan")):.6g} p10={d0.get("p10",float("nan")):.6g} p90={d0.get("p90",float("nan")):.6g}')
    print(f'[SAVED] {output / "summary.json"}')
    print(f'[SAVED] {output / "per_image.csv"}')
    print(f'[SAVED] {output / "ranked_iou_changes.csv"}')
    print(f'[SAVED] {output / "report.md"}')
    print(f'[SAVED] {output / "visualizations"} ({count_vis} images)')
    print('[NOTE] Small GT boxes: ROI metrics include padded cells; interpret map contrasts descriptively, not as direct evidence of detail loss.')
    print('='*74)


if __name__=='__main__':
    main()
