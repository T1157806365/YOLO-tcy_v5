#!/usr/bin/env python3
"""End-to-end val/test of frozen global v5 + trained local RGB-T detector.

Evaluates global_only, local_only and safe_replace on the same images.
GT labels are only read by the metric engine / diagnostic counters, never by
ROI selection or inference.
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.runtime import load_config, preprocess_batch, select_device, synchronize
from engine.evaluator import prepare_semantic_ground_truth, process_single_image, postprocess_predictions
from models.gtga_bhlr_factory import load_model_from_checkpoint
from global_local.inference import generate_rois
from global_local.geometry import local_box_to_semantic, local_gt_from_high
from global_local.fusion import merge_local_predictions, safe_replace, pairwise_iou, empty_predictions
from global_local.local_model import load_local_checkpoint


def update_metric(metric, pred, gt, iouv, image_name):
    correct=process_single_image(pred,gt,iouv)
    conf=pred['conf'].detach().float().cpu().numpy()
    cls=pred['cls'].detach().float().cpu().numpy()
    target_cls=gt['cls'].detach().float().cpu().numpy()
    metric.update_stats({
        'tp':correct,'conf':conf,'pred_cls':cls,'target_cls':target_cls,
        'target_img':np.unique(target_cls),'im_name':Path(image_name).name,
    })


def run(cfg, split='val',device=None,limit=0):
    dev=select_device(device or cfg.get('eval',{}).get('device','5'))
    cfg_global=cfg['global']; roi_cfg=cfg['roi']; data=cfg['data'];
    global_model,_,_=load_model_from_checkpoint(cfg_global['weights'],dev,strict=True,verbose=False)
    local_model,_,_=load_local_checkpoint(cfg['eval']['local_weights'],dev)
    global_model.eval();local_model.eval()
    ds=build_dual_resolution_dataset(
        rgb_yaml=data['rgb'],tir_yaml=data['tir'],split=split,
        rgb_high_imgsz=int(data.get('rgb_imgsz',1920)),
        rgb_semantic_imgsz=int(data.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(data.get('tir_imgsz',640)),pair_mode=str(data.get('pair_mode','relative')),
        augment=False,fliplr=0,flipud=0,tir_channels=3,strict_pair=True,
    )
    loader=build_dual_resolution_dataloader(ds,batch_size=1,workers=int(cfg.get('eval',{}).get('workers',2)),shuffle=False,pin_memory=True,drop_last=False)
    metrics={name:DetMetrics(names={0:'UAV'}) for name in ('global_only','local_only','safe_replace')}
    iouv=torch.linspace(0.50,0.95,10,device=dev)
    fcfg=cfg.get('merge',{})
    counts=defaultdict(float);start=time.perf_counter();seen=0;total_gt=0
    for batch in tqdm(loader,total=min(len(loader),limit) if limit else len(loader),desc=f'Global+Local {split}'):
        if limit and seen>=limit:break
        synchronize(dev);t0=time.perf_counter()
        batch=preprocess_batch(batch,dev)
        with torch.inference_mode():
            groups,global_preds=generate_rois(
                global_model,batch,topk=int(roi_cfg.get('topk',2)),
                roi_size=int(roi_cfg.get('size',640)),
                min_score=float(roi_cfg.get('candidate_conf',0.001)),
                min_center_dist=float(roi_cfg.get('min_center_dist',32.0)),
            )
            per_image=[]
            for i,proposals in enumerate(groups):
                if not proposals:
                    per_image.append(empty_predictions(dev));continue
                rgb=torch.stack([x.rgb for x in proposals])
                tir=torch.stack([x.tir for x in proposals])
                raw=local_model(rgb,tir)
                loc_pred=postprocess_predictions(raw,model=local_model,
                     conf_thres=float(cfg.get('eval',{}).get('local_conf',0.001)),
                     iou_thres=float(cfg.get('eval',{}).get('iou',0.70)))
                sem_hw=tuple(batch['rgb_semantic_img'].shape[-2:]);loc=[]
                for prop,pred in zip(proposals,loc_pred):
                    boxes=local_box_to_semantic(pred['bboxes'],prop.geometry,
                        batch['rgb_ratio_pad'][i],batch['rgb_semantic_ratio_pad'][i],sem_hw)
                    loc.append({'bboxes':boxes,'conf':pred['conf'].clone(),'cls':pred['cls'].clone()})
                    counts['roi_tir_valid_sum']+=prop.geometry.tir_valid_fraction
                    counts['rois']+=1
                per_image.append(merge_local_predictions(loc,nms_iou=float(fcfg.get('local_nms_iou',0.5))))
        synchronize(dev);counts['inference_seconds']+=time.perf_counter()-t0
        for i in range(int(batch['rgb_img'].shape[0])):
            seen+=1
            gt=prepare_semantic_ground_truth(batch,i)
            total_gt+=len(gt['bboxes'])
            local=per_image[i];global_pred=global_preds[i]
            combined,changes=safe_replace(global_pred,local,
                min_local_conf=float(fcfg.get('min_local_conf',0.25)),
                min_iou=float(fcfg.get('min_iou',0.12)),
                min_area_ratio=float(fcfg.get('min_area_ratio',0.35)),
                max_area_ratio=float(fcfg.get('max_area_ratio',2.8)),
                max_center_distance=float(fcfg.get('max_center_distance',1.8)),
                unmatched_add_conf=float(fcfg.get('unmatched_add_conf',0.90)),
            )
            for key in ('replaced','added'):counts[key]+=changes[key]
            candidates=groups[i]
            if candidates:
                boxes_high=batch['rgb_bboxes'][batch['rgb_batch_idx']==i]
                if any(len(local_gt_from_high(boxes_high,p.geometry.x0,p.geometry.y0,p.geometry.size,
                          batch['rgb_img'].shape[-1],batch['rgb_img'].shape[-2],min_visible=.95))>0
                       for p in candidates):
                    counts['gt_full_roi_coverage']+=1
            for mode,pred in (('global_only',global_pred),('local_only',local),('safe_replace',combined)):
                update_metric(metrics[mode],pred,gt,iouv,batch['rgb_path'][i])
            if len(gt['bboxes']):
                gt_boxes=gt['bboxes']
                gi=float(pairwise_iou(global_pred['bboxes'],gt_boxes).max().item()) if len(global_pred['bboxes']) else 0.0
                li=float(pairwise_iou(local['bboxes'],gt_boxes).max().item()) if len(local['bboxes']) else 0.0
                counts['global_best_iou_sum']+=gi
                counts['local_best_iou_sum']+=li
                counts['local_iou_improved']+=float(li>gi+1e-6)
    report={'split':split,'images':seen,'instances':total_gt,
            'seconds_model_inference':counts['inference_seconds'],
            'fps_end_to_end_model':seen/max(counts['inference_seconds'],1e-8),
            'roi_coverage_full_gt':counts['gt_full_roi_coverage']/max(seen,1),
            'mean_tir_valid_fraction':counts['roi_tir_valid_sum']/max(counts['rois'],1),
            'mean_best_global_iou':counts['global_best_iou_sum']/max(seen,1),
            'mean_best_local_iou':counts['local_best_iou_sum']/max(seen,1),
            'fraction_local_best_iou_improves':counts['local_iou_improved']/max(seen,1),
            'replaced':int(counts['replaced']),'added':int(counts['added']),
            'modes':{}}
    for name,metric in metrics.items():
        metric.process(plot=False)
        p,r,ap50,ap5095=metric.mean_results()
        report['modes'][name]={'precision':float(p),'recall':float(r),
                              'map50':float(ap50),'map75':float(metric.box.map75),
                              'map5095':float(ap5095)}
    out=Path(cfg.get('eval',{}).get('save_dir','runs/global_local/eval'))
    out.mkdir(parents=True,exist_ok=True)
    file=out/f'{split}_global_local_results.json'
    file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'Saved: {file}')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True);p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--device',default=None);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    run(load_config(args.cfg),args.split,args.device,args.limit)

if __name__=='__main__':main()
