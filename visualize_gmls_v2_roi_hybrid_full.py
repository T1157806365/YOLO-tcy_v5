#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed, label-safe inference visualization of GMLS-v2 ONE-HEAD Hybrid Top-1/Top-2.

Existing v5 source untouched. No GT-guided cropping during visualization.
Actual stages: RGB-sem + TIR -> GMLS/DMLS alignment + FeatureAwareGuide ->
Top-K ROI -> native RGB-high & geometrically sampled TIR -> local RGB-T
backbones / ADD P3/P4/P5 / ONE YOLO26m Detect -> ROI->semantic / NMS.

Training-only auxiliary global shared Detect can be visualized with --aux-global,
clearly labeled as diagnostic, NOT part of inference or final prediction.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from visualize_gmls_v2_roi_hybrid_utils import (
    RGB_GT,TIR_GT,ALIGNED_GT,PRED,ANCHOR,SOURCE_CENTER,
    tensor_bgr,fmap,heat,overlay,xywhn_to_xyxy,transform_source_boxes,
    draw_boxes,draw_point,draw_sigma_ellipse,bbox_iou,stats_text,probability_stats,
    render_probability,render_top_candidates,render_scale_compare,make_overview,
    _put_text_outlined,add_title,
)


def _v(x, i=0):
    """Extract one per-image vector without mutating model tensors."""
    if x is None:
        return None
    if torch.is_tensor(x):
        y=x.detach().float().reshape(x.shape[0],-1)[i].cpu().numpy()
    else:
        y=np.asarray(x,dtype=np.float32).reshape(-1)
    return y.astype(np.float32)


def _map(x, i=0, mode='mean_abs'):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.ndim==4:x=x[i]
        return fmap(x,mode)
    a=np.asarray(x)
    if a.ndim==4:a=a[i]
    if a.ndim==3:return a.mean(0)
    if a.ndim==2:return a
    return None


def _check_meta(meta):
    r=meta['ratio']
    if isinstance(r,(tuple,list)):r=float(r[0])
    return float(r),np.array(meta['pad'],dtype=np.float32)


def _semantic_pixel_to_high(xy, sample, high_hw, sem_hw):
    """Semantic RGB xy pixel centers -> high RGB xy pixel centers (letterbox-aware)."""
    rr,pp=_check_meta(sample['rgb_semantic_ratio_pad'])
    rh,ph=_check_meta(sample['rgb_ratio_pad'])
    v=np.asarray(xy,dtype=np.float32)
    return (v-pp)/rr*rh+ph


def _semantic_xyxy_to_high(boxes, sample, high_hw, sem_hw):
    x=np.asarray(boxes,dtype=np.float32).reshape(-1,2,2)
    return _semantic_pixel_to_high(x,sample,high_hw,sem_hw).reshape(-1,4)


def _normalize_to_sem_pixel(boxes, sem_hw):
    hh,ww=sem_hw
    a=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    return xywhn_to_xyxy(a,ww,hh)


def _aligned_tir_gt(tir_normalized_boxes, scale, trans, sem_hw):
    a=transform_source_boxes(tir_normalized_boxes,scale,trans)
    return _normalize_to_sem_pixel(a,sem_hw)


def _local_gt_from_high(high_boxes, origin, size):
    bb=np.asarray(high_boxes,np.float32).reshape(-1,4)
    org=np.asarray(origin,np.float32)
    lo=np.clip(bb[:,:2]-org,0,size)
    hi=np.clip(bb[:,2:]-org,0,size)
    b=np.concatenate((lo,hi),axis=1)
    orig_area=np.maximum(bb[:,2:]-bb[:,:2],0).prod(1)
    vis=(np.maximum(hi-lo,0).prod(1))/np.maximum(orig_area,1e-6)
    ok=(hi[:,0]>lo[:,0])&(hi[:,1]>lo[:,1]);return b[ok],vis[ok]


def _detection_arr(pred, conf=0.25):
    b=pred['bboxes'].detach().float().cpu().numpy().reshape(-1,4)
    p=pred['conf'].detach().float().cpu().numpy().reshape(-1)
    k=p>=conf
    return b[k],p[k]


def _roi_rectangle(image,origin,size,label,color=(0,200,255),thickness=None):
    out=image.copy();h,w=out.shape[:2]
    x0,y0=map(float,origin)
    a=(int(round(x0)),int(round(y0)))
    b=(int(round(min(x0+size,w-1))),int(round(min(y0+size,h-1))))
    cv2.rectangle(out,a,b,(0,0,0),5 if thickness is None else thickness+2,cv2.LINE_AA)
    cv2.rectangle(out,a,b,color,3 if thickness is None else thickness,cv2.LINE_AA)
    x=max(5,min(w-235,a[0]+8)); y=max(24,min(h-10,a[1]+30))
    _put_text_outlined(out,label,(x,y),0.72,color,2,outline_thickness=4)
    return out


def _make_side_by_side(left,right,subtitle_left='RGB',subtitle_right='TIR'):
    if left.shape[0]!=right.shape[0]:
        right=cv2.resize(right,(right.shape[1],left.shape[0]))
    h,w=left.shape[:2];v=right.shape[1]
    out=np.full((h,w+v+12,3),243,np.uint8)
    out[:,:w]=left;out[:,w+12:]=right
    _put_text_outlined(out,subtitle_left,(12,30),.65,(255,255,255),1,outline_thickness=3)
    _put_text_outlined(out,subtitle_right,(w+24,30),.65,(255,255,255),1,outline_thickness=3)
    return out


def _write_tile(folder,num,slug,img,title,subtitle='',annotation='',canvas=1050):
    """Square tiles match the old GMLS-v2 script; comparison images preserve aspect ratio."""
    folder.mkdir(parents=True,exist_ok=True)
    a=np.asarray(img)
    if a.ndim==2:a=cv2.cvtColor(a.astype('uint8'),cv2.COLOR_GRAY2BGR)
    if a.dtype!=np.uint8:a=np.clip(a,0,255).astype(np.uint8)
    h,w=a.shape[:2]
    if w>h*1.3:
        scale=canvas/w
        a=cv2.resize(a,(canvas,max(64,int(round(h*scale)))),interpolation=cv2.INTER_AREA)
    else:
        a=cv2.resize(a,(canvas,canvas),interpolation=cv2.INTER_LINEAR)
    a=add_title(a,title,subtitle,annotation)
    path=folder/f'{num:02d}_{slug}.jpg'
    if not cv2.imwrite(str(path),a,[cv2.IMWRITE_JPEG_QUALITY,96]):
        raise IOError(f'Cannot save {path}')
    return path


def _plain_bar_graph(entries,title,canvas=1050,suffix=''):
    a=np.full((canvas,canvas,3),248,np.uint8)
    cv2.putText(a,title[:62],(32,50),cv2.FONT_HERSHEY_SIMPLEX,.78,(40,40,40),2,cv2.LINE_AA)
    values=[float(x[1]) for x in entries]
    mx=max([abs(x) for x in values]+[1e-4])
    for i,(name,value) in enumerate(entries):
        y=120+i*max(46,min(94, int((canvas-160)/max(len(entries),1))))
        cv2.putText(a,str(name)[:30],(38,y+26),cv2.FONT_HERSHEY_SIMPLEX,.62,(25,35,45),2,cv2.LINE_AA)
        span=int(500*max(0.,float(value))/mx)
        cv2.rectangle(a,(380,y),(380+span,y+31),(215,137,76),-1)
        cv2.putText(a,f'{value:.5f}{suffix}',(398+span,y+25),cv2.FONT_HERSHEY_SIMPLEX,.51,(25,35,45),1,cv2.LINE_AA)
    return a


def _get_scale_debug(ex,scale=3):
    info=ex['alignment_info']
    raw=info.get('scale_debug',{})
    return raw.get(scale,raw.get(str(scale),{})) if isinstance(raw,dict) else {}


def _select_indices(ds,values,index,first_n):
    if values:
        res=[]
        for x in values:
            for part in str(x).split(','):
                if part.strip():res.append(int(part))
    elif index is not None:res=[index]
    else:res=list(range(min(int(first_n),len(ds))))
    res=list(dict.fromkeys(res))
    for x in res:
        if not 0<=x<len(ds):raise ValueError(f'Invalid sample index {x}, dataset length={len(ds)}')
    return res


@torch.inference_mode()
def visualize_one(idx,net,ds,device,output_root,conf,iou,max_det,canvas,aux_global=False,features=True):
    from engine.runtime import preprocess_batch
    from engine.evaluator import postprocess_predictions
    from gmls_roi_onehead_v2.multi_roi import per_image_merge,crop_box_to_semantic

    sample=ds[idx]
    batch=preprocess_batch(ds.collate_fn([sample]),device)
    net.eval()
    ex=net.encode_and_crop(batch,return_debug=True,train_rois=False)   # NEVER GT crop
    raw=net.detector.predict(ex['rgb_roi'],ex['tir_roi'],return_features=True)
    if not isinstance(raw,dict) or 'pred' not in raw:
        raise RuntimeError('Local detector does not return features in current v5 source')
    loc=postprocess_predictions(raw['pred'],model=net.detector,
                                conf_thres=float(iou if conf is None else min(conf,.001)),
                                iou_thres=float(iou),max_det=int(max_det))
    # Same inference fusion as GMLSV2OneHeadHybrid.predict(): no GT, no global box fusion.
    merged=per_image_merge(loc,ex['roi'],batch,ex['rois_per_image'],
                           conf=float(iou if conf is None else min(conf,.001)),
                           nms_iou=float(net.crop_nms_iou),max_det=int(max_det))
    info=ex['alignment_info'];dbg=_get_scale_debug(ex,net.global_encoder.gtga_reference_scale)
    sem=tensor_bgr(batch['rgb_semantic_img']); high=tensor_bgr(batch['rgb_img']);tir=tensor_bgr(batch['tir_img'])
    sh,sw=sem.shape[:2];hh,hw=high.shape[:2];k=ex['rois_per_image']
    rgb_sem_gt=xywhn_to_xyxy(sample['rgb_semantic_bboxes'],sw,sh)
    rgb_high_gt=xywhn_to_xyxy(sample['rgb_bboxes'],hw,hh)
    tir_gt=xywhn_to_xyxy(sample['tir_bboxes'],tir.shape[1],tir.shape[0])
    sc=_v(info['sampling_scale']);tr=_v(info['sampling_translation'])
    anchor=_v(info['coarse_anchor']);source=_v(info.get('source_center'))
    tir_aligned_boxes_sem=_aligned_tir_gt(sample['tir_bboxes'],sc,tr,(sh,sw))
    braw=_normalize_to_sem_pixel(sample['tir_bboxes'],(sh,sw))
    iou_raw=bbox_iou(rgb_sem_gt[0],braw[0]) if len(rgb_sem_gt)==len(braw)==1 else None
    iou_final=bbox_iou(rgb_sem_gt[0],tir_aligned_boxes_sem[0]) if len(rgb_sem_gt)==len(tir_aligned_boxes_sem)==1 else None

    target=output_root/f'index_{idx:06d}_{Path(sample["rgb_path"]).stem}'
    target.mkdir(parents=True,exist_ok=True)
    # Remove earlier image/preview/JSON outputs for this index only, preserving weights/other images.
    for stale in target.glob('[0-9][0-9]_*.jpg'):stale.unlink()
    for stale in target.glob('99_*.jpg'):stale.unlink()
    paths=[];n=0
    def emit(slug,img,title,subtitle='',annotation=''):
        nonlocal n
        p=_write_tile(target,n,slug,img,title,subtitle,annotation,canvas)
        n+=1;paths.append(p);return p
    emit('rgb_high_gt',draw_boxes(high,rgb_high_gt,RGB_GT,'RGB GT'),'01 / RGB High 1920x1920','Original high-resolution RGB, not interpolated semantic RGB')
    emit('rgb_semantic_gt',draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT'),'02 / RGB Semantic 640x640','Global GMLS-v2 reference coordinates')
    emit('tir_input_gt',draw_boxes(tir,tir_gt,TIR_GT,'TIR GT'),'03 / TIR Global 640x640','Raw thermal image before alignment')
    before=draw_boxes(draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT'),braw,TIR_GT,'TIR GT')
    emit('raw_gt_overlay',before,'04 / Before RGB-T alignment',f'IoU(raw)={iou_raw:.4f}' if iou_raw is not None else 'Green: RGB GT / Orange: TIR GT')

    coarse=_map(dbg.get('coarse_targetness'),mode='mean')
    if coarse is not None:
        emit('rlsfa_coarse_targetness',draw_point(draw_boxes(heat(coarse,(sh,sw),fixed=(0,1)),rgb_sem_gt,RGB_GT,'RGB GT'),anchor,ANCHOR,'GMLS anchor'),
             'RLSFA coarse UAV targetness',stats_text(coarse,sample['rgb_semantic_bboxes']))
    cad=_map(dbg.get('coarse_anchor_distribution'),mode='mean')
    if cad is not None:
        emit('rlsfa_anchor_distribution',draw_point(draw_boxes(heat(cad,(sh,sw)),rgb_sem_gt,RGB_GT,'RGB GT'),anchor,ANCHOR,'c_R'),
             'RLSFA coarse anchor distribution',f'Anchor=[{anchor[0]:.5f}, {anchor[1]:.5f}]')
    for key,slug,label,gtboxes in [
        ('rgb_targetness_raw','rgb_targetness_raw','RGB targetness RAW',rgb_sem_gt),
        ('rgb_local_targetness','rgb_local_targetness','RGB targetness LOCAL',rgb_sem_gt),
        ('tir_targetness_raw','tir_targetness_raw','TIR targetness RAW',braw),
        ('tir_local_targetness','tir_local_targetness','TIR targetness LOCAL',braw),
        ('aligned_tir_local_targetness','aligned_tir_targetness','Aligned TIR local targetness',rgb_sem_gt),
        ('joint_target_support','gtga_joint_support','GTGA joint target support J',rgb_sem_gt)]:
        mm=_map(dbg.get(key),mode='mean')
        if mm is not None:
            emit(slug,draw_boxes(heat(mm,(sh,sw),fixed=(0,1)),gtboxes,RGB_GT if 'tir_targetness' not in slug else TIR_GT,'GT'),
                 label,stats_text(mm,sample['rgb_semantic_bboxes']))

    if 'multi_target_guidance' in info:
        three=info['multi_target_guidance'][0].detach().float()
        for ch,slug,t in [(0,'rgb_support','A_RGB (GMLS)'),(1,'tir_aligned_support','A_TIR aligned (GMLS)'),(2,'joint_support','J (GMLS)')]:
            m=_map(three[ch:ch+1],mode='mean')
            emit(slug,draw_boxes(overlay(sem,m,fixed=(0,1)),rgb_sem_gt,RGB_GT,'RGB GT'),t+' overlay on RGB','Same semantic coordinate system')

    # Stepwise alignment: GMLS and DMLS coarse candidates, selected correction and final warp.
    # Coarse displacements are feature pixels; converting them to normalized q=p+t
    # follows the user's old visualization and keeps the candidate comparison fair.
    raw_reference=dbg.get('rgb_targetness_raw')
    feat_h=int(raw_reference.shape[-2]) if torch.is_tensor(raw_reference) else max(1,sh//8)
    feat_w=int(raw_reference.shape[-1]) if torch.is_tensor(raw_reference) else max(1,sw//8)
    stage_iou=[]
    for key,slug,stage in [
        ('coarse_offset_initial','coarse_only','RLSFA coarse translation only'),
        ('coarse_offset_gmls','gmls_candidate','GMLS candidate translation'),
        ('coarse_offset_dmls','dmls_candidate','DMLS candidate translation'),
        ('coarse_offset_gmls_v2','gmls_v2_correction','GMLS-v2 selected translation'),
        ('refined_offset','refined_translation','Final translation before GTGA scale')]:
        off=dbg.get(key,info.get(key))
        vv=_v(off) if off is not None else None
        if vv is None or len(vv)<2:continue
        shift=vv[:2]/np.array([max(feat_w-1,1),max(feat_h-1,1)],np.float32)
        gobj={'sampling_scale':torch.ones_like(info['sampling_scale']),
              'sampling_translation':info['sampling_translation'].new_tensor(shift).view(1,2)}
        timg=tensor_bgr(net.global_encoder.alignment.warp_tensor(batch['tir_img'],gobj,
                                 output_hw=(sh,sw)))
        box=_aligned_tir_gt(sample['tir_bboxes'],[1,1],shift,(sh,sw))
        st_iou=bbox_iou(rgb_sem_gt[0],box[0]) if len(rgb_sem_gt)==len(box)==1 else None
        stage_iou.append((stage,st_iou))
        pic=draw_boxes(draw_boxes(timg,rgb_sem_gt,RGB_GT,'RGB GT'),box,ALIGNED_GT,'TIR aligned GT')
        emit(slug,pic,stage,f'dx={vv[0]:+.3f},dy={vv[1]:+.3f} P3 feature pixels',
             f'BBox IoU={st_iou:.4f}' if st_iou is not None else 'BBox GT unavailable')

    if dbg.get('coarse_probability') is not None:
        prob=dbg['coarse_probability']
        if torch.is_tensor(prob):
            proba=prob[:1]
            offs=dbg.get('coarse_offset_initial',dbg.get('coarse_offset'))
            if offs is not None:
                try:
                    ps=probability_stats(proba)
                    emit('gmls_coarse_translation_probability',render_probability(proba,offs[:1],canvas),
                         'GMLS coarse translation probability',f'pmax/uniform={ps["max_over_uniform"]:.2f} entropy={ps["entropy_norm"]:.3f}')
                    emit('gmls_top_translation_candidates',render_top_candidates(proba,12,canvas),
                         'Top coarse translation candidates')
                except (ValueError,IndexError,RuntimeError) as er:
                    print(f'[WARN] coarse probability render skipped: {er}')

    offset_metrics=[]
    for key,title in [('coarse_offset_gmls','GMLS candidate'),('coarse_offset_dmls','DMLS candidate'),
                      ('coarse_offset_gmls_v2','Selected GMLS-v2'),('fine_offset','Fine offset'),('refined_offset','Final offset')]:
        vec=_v(info.get(key,dbg.get(key)))
        if vec is not None and vec.size>=2:offset_metrics.extend([(f'{title} dx',float(vec[0])),(f'{title} dy',float(vec[1]))])
    # GMLS candidate weights/evidence expose the multiple-hypothesis search.
    for key,slug,title in [
        ('coarse_gmls_candidate_weights','gmls_candidate_weights','GMLS candidate aggregation weights'),
        ('coarse_gmls_candidate_evidence','gmls_candidate_evidence','GMLS candidate evidence')]:
        vec=_v(info.get(key,dbg.get(key)))
        if vec is not None and vec.size:
            nshow=min(12,len(vec))
            emit(slug,_plain_bar_graph([(f'candidate #{i+1}',float(vec[i])) for i in range(nshow)],title,canvas),title,
                 f'First {nshow} scalar candidate values; source: {key}')
    if offset_metrics:
        emit('gmls_dmls_offset_compare',_plain_bar_graph([(name,abs(x)) for name,x in offset_metrics],'GMLS / DMLS offsets (absolute value)',canvas),
             'GMLS-v2 candidate and selected translation offsets','Absolute values in P3 feature pixels; signs saved in metadata')
    qual=_v(info.get('gmls_v2_quality_scores',dbg.get('gmls_v2_quality_scores')))
    if qual is not None and qual.size>=2:
        emit('gmls_dmls_quality',_plain_bar_graph([('GMLS',qual[0]),('DMLS',qual[1])],'GMLS-v2 predicted candidate quality',canvas),
             'GMLS-v2 candidate quality comparison','Routing scores are NOT bbox IoU')

    # Actual GTGA warp at full semantic-resolution; same S,t as ROI thermal resampling.
    geo={'sampling_scale':info['sampling_scale'],'sampling_translation':info['sampling_translation']}
    g=net.global_encoder
    warped=g.alignment.warp_tensor(batch['tir_img'],geo,output_hw=(sh,sw))
    tir_w=tensor_bgr(warped)
    aligned=draw_boxes(draw_boxes(tir_w,rgb_sem_gt,RGB_GT,'RGB GT'),tir_aligned_boxes_sem,ALIGNED_GT,'TIR aligned GT')
    if stage_iou and iou_final is not None:
        valid_st=[(n,float(v)) for n,v in stage_iou if v is not None]
        emit('alignment_iou_stages',_plain_bar_graph([('Before',iou_raw or 0)]+valid_st+[('Final GTGA',iou_final)],
             'Alignment IoU by stage (GT diagnostic only)',canvas),
             'GTGA / GMLS-v2 / DMLS alignment IoU diagnostic','Positive = larger bbox agreement; no GT used for cropping')
    emit('gtga_aligned_tir',aligned,'GTGA final ONE geometry warp',
         f'S=({sc[0]:.4f},{sc[1]:.4f}); t=({tr[0]:+.4f},{tr[1]:+.4f})',
         f'GT box IoU {iou_raw:.4f} -> {iou_final:.4f}' if iou_final is not None else 'GT bbox overlay')
    emit('rgb_tir_aligned_blend',draw_boxes(cv2.addWeighted(sem,.5,tir_w,.5,0),rgb_sem_gt,RGB_GT,'RGB GT'),
         'RGB and geometrically aligned TIR overlay (50:50)')
    if anchor is not None:
        pivot=draw_point(sem,anchor,ANCHOR,'RGB c_R')
        if source is not None:pivot=draw_point(pivot,source,SOURCE_CENTER,'TIR c_T')
        pivot=draw_boxes(pivot,rgb_sem_gt,RGB_GT,'RGB GT')
        emit('geometry_centers',pivot,'GMLS-v2 target anchor and TIR source center','Sampling q=S*p+t')
    if dbg.get('sampling_scale_raw') is not None:
        rawsc=_v(dbg['sampling_scale_raw'])
        gt_sc=None
        if len(rgb_sem_gt)==len(tir_gt)==1:
            r=sample['rgb_semantic_bboxes'][0].detach().cpu().numpy()
            t=sample['tir_bboxes'][0].detach().cpu().numpy()
            gt_sc=np.maximum(t[2:]/np.maximum(r[2:],1e-8),1e-6)
        emit('gtga_scale_prediction',render_scale_compare(sc,gt_sc,rawsc,canvas),'Local GTGA scale estimate')

    guide_logits=_map(ex['guide_logits'],mode='mean')
    guide_p=_map(ex['guide_map'],mode='mean')
    emit('roi_guide_logits',draw_boxes(heat(guide_logits,(sh,sw)),rgb_sem_gt,RGB_GT,'RGB GT'),
         'ROI Guide raw logits','Learned 4ch alignment support + P3 RGB/TIR feature branch')
    # Show anchor prior separately from trained ROI map to diagnose failure to localize.
    if anchor is not None:
        yy,xx=np.meshgrid(np.linspace(0,1,guide_p.shape[0]),np.linspace(0,1,guide_p.shape[1]),indexing='ij')
        prior=np.exp(-.5*(((xx-anchor[0])/.08)**2+((yy-anchor[1])/.08)**2))
        emit('roi_anchor_prior',draw_boxes(heat(prior,(sh,sw),fixed=(0,1)),rgb_sem_gt,RGB_GT,'RGB GT'),
             'Coarse GMLS anchor Gaussian prior','Model ROI Guide receives this prior as fourth channel')
    emit('roi_guide_distribution',draw_boxes(heat(guide_p,(sh,sw)),rgb_sem_gt,RGB_GT,'RGB GT'),
         'ROI Guide normalized probability',stats_text(guide_p,sample['rgb_semantic_bboxes']))
    emit('roi_guide_heat_overlay',draw_boxes(overlay(sem,guide_p),rgb_sem_gt,RGB_GT,'RGB GT'),
         'ROI Guide probability overlay','GT is visualization ONLY; no GT crop at test time')
    primary=_v(ex['center_pred']);used=ex['center_used'].detach().float().cpu().numpy().reshape(k,2)
    centers=draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT')
    centers=draw_point(centers,primary,ANCHOR,'Primary')
    centers=draw_point(centers,anchor,SOURCE_CENTER,'GMLS anchor')
    for j,c in enumerate(used):
        centers=draw_point(centers,c,[(0,220,255),(255,120,20)][j%2],f'ROI#{j+1}')
    emit('roi_centers_semantic',centers,'ROI Guide primary + GMLS anchor + Top-K centers',
         'Circles are predicted; GREEN BOX is GT for diagnostics only')

    high_vis=draw_boxes(high,rgb_high_gt,RGB_GT,'RGB GT')
    sem_vis=draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT')
    roi_size=int(ex['roi'].size)
    rec=[]
    for j in range(k):
        xy0=_v(ex['roi'].xy0,j)
        col=[(0,215,255),(255,125,0)][j%2]
        high_vis=_roi_rectangle(high_vis,xy0,roi_size,f'ROI#{j+1}',col)
        # Convert actual high-res 640-crop window into RGB-semantic coordinates.
        rc=np.array([[xy0[0],xy0[1],xy0[0]+roi_size,xy0[1]+roi_size]],np.float32).reshape(1,2,2)
        rh,ph=_check_meta(sample['rgb_ratio_pad']);rs,ps=_check_meta(sample['rgb_semantic_ratio_pad'])
        sem_roi=(rc-ph)/rh*rs+ps
        rect=sem_roi.reshape(-1,4).round().astype(int)[0]
        cv2.rectangle(sem_vis,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),col,2,cv2.LINE_AA)
        rec.append({'roi':j,'origin_high_xy':[float(x) for x in xy0],
                    'center_sem_norm':[float(v) for v in used[j]],
                    'tir_valid':float(ex['roi'].tir_valid[j].item())})
    emit('topk_rois_on_rgb_high',high_vis,f'Top-{k} native RGB 640x640 crop windows',
         'RGB High 1920 image; true high-resolution detail kept; green=GT')
    emit('topk_rois_semantic',sem_vis,f'Top-{k} ROIs mapped to semantic RGB 640',
         'LetterBox metadata used, NOT naive x3')

    local_feat=raw
    global_rgb=None;global_aligned=None
    if features or aux_global:
        # This is visualization-only duplicate backbone pass; no GT or inference modification.
        global_rgb=g.forward_rgb_backbone(batch['rgb_semantic_img'])
        global_tir=g.forward_tir_backbone(batch['tir_img'])
        global_aligned=[x for x in global_tir]
        for si,idxlayer in enumerate(g.fusion_indices):
            global_aligned[idxlayer]=g.alignment.warp_tensor(global_tir[idxlayer],geo,output_hw=global_rgb[idxlayer].shape[-2:])
    if features:
        for stage,lid in zip((3,4,5),g.fusion_indices):
            rgbfeat=_map(global_rgb[lid])
            tirfeat=_map(global_aligned[lid])
            emit(f'global_p{stage}_rgb_features',draw_boxes(overlay(sem,rgbfeat),rgb_sem_gt,RGB_GT,'RGB GT'),
                 f'Global RGB Backbone P{stage} features')
            emit(f'global_p{stage}_aligned_tir_features',draw_boxes(overlay(sem,tirfeat),rgb_sem_gt,RGB_GT,'RGB GT'),
                 f'Global GTGA-aligned TIR P{stage} features','Used for GMLS matching and TRAIN-ONLY global auxiliary supervision')
    if aux_global:
        aux_pred=net._shared_head_global_prediction({'global_rgb_features':global_rgb,'global_tir_aligned':global_aligned})
        aux=postprocess_predictions(aux_pred,model=net.detector,
                                    conf_thres=.001,iou_thres=float(iou),max_det=int(max_det))
        bb,ss=_detection_arr(aux[0],conf)
        emit('training_only_global_aux',draw_boxes(draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT'),bb,PRED,'AUX',ss),
             'TRAIN-ONLY shared-head global auxiliary detection',
             'NOT executed during normal Hybrid validation/inference')

    tir_gt_high=_semantic_xyxy_to_high(tir_aligned_boxes_sem,sample,(hh,hw),(sh,sw))
    min_vis=float(net.min_visible)
    roi_views=[]
    per_roi_meta=[]
    for j in range(k):
        folder=target/f'roi_{j+1:02d}'
        folder.mkdir(exist_ok=True)
        for stale in folder.glob('[0-9][0-9]_*.jpg'):stale.unlink()
        xy0=_v(ex['roi'].xy0,j)
        rgbroi=tensor_bgr(ex['rgb_roi'][j]);tirroi=tensor_bgr(ex['tir_roi'][j]);rw=rgbroi.shape[1];rh=rgbroi.shape[0]
        # NATIVE_RGB_TIR_ROI_EXPORT_V1
        # These are the *actual* tensors fed to the local detector, not
        # semantic RGB resized from 1920 and not re-scaled for presentation.
        if rgbroi.shape != tirroi.shape:
            raise RuntimeError(f"ROI RGB/TIR shapes differ: {rgbroi.shape}, {tirroi.shape}")
        def _native_png(filename, pixels):
            dst = folder / filename
            if not cv2.imwrite(str(dst), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise IOError(f"Failed saving native ROI image: {dst}")
        _native_png('90_RGB_native_high_crop_640.png', rgbroi)
        _native_png('91_TIR_GTGA_aligned_crop_640.png', tirroi)
        _native_png('92_RGB_native_vs_TIR_aligned_1280x640.png',
                    np.concatenate((rgbroi, tirroi), axis=1))
        _native_png('93_RGB_TIR_aligned_overlay_640.png',
                    cv2.addWeighted(rgbroi, 0.5, tirroi, 0.5, 0))
        gt_local,vis=_local_gt_from_high(rgb_high_gt,xy0,roi_size)
        tir_local,vis_t=_local_gt_from_high(tir_gt_high,xy0,roi_size)
        pred_local,score_local=_detection_arr(loc[j],conf)
        local_gt_img=draw_boxes(rgbroi,gt_local,RGB_GT,'RGB GT')
        local_tir_img=draw_boxes(tirroi,tir_local,ALIGNED_GT,'TIR aligned GT')
        local_pred_img=draw_boxes(local_gt_img,pred_local,PRED,'DET',score_local)
        _write_tile(folder,0,'rgb_roi_gt',local_gt_img,
                    f'ROI#{j+1} native RGB high-resolution 640 crop',
                    f'High origin x0={xy0[0]:.2f} y0={xy0[1]:.2f}; RGB GT visible='+','.join(f'{v:.2f}' for v in vis),canvas=canvas)
        _write_tile(folder,1,'tir_roi_aligned_gt',local_tir_img,
                    f'ROI#{j+1} geometrically synchronized TIR 640 crop',
                    f'GTGA TIR sample valid={ex["roi"].tir_valid[j].item():.4f}; MAGENTA=aligned TIR GT',canvas=canvas)
        _write_tile(folder,2,'rgb_tir_roi_side_by_side',_make_side_by_side(rgbroi,tirroi,'RGB native crop','GTGA-warped TIR'),
                    f'ROI#{j+1}: true RGB / aligned TIR source crops',canvas=canvas)
        _write_tile(folder,15,'rgb_tir_roi_blend',draw_boxes(cv2.addWeighted(rgbroi,.5,tirroi,.5,0),gt_local,RGB_GT,'RGB GT'),
                    f'ROI#{j+1}: RGB-T geometry consistency blend',canvas=canvas)
        _write_tile(folder,3,'roi_detection_vs_gt',local_pred_img,f'ROI#{j+1}: ONE shared YOLO26m Detect',
                    f'Local predictions conf>={conf:.3f}; num={len(pred_local)}',canvas=canvas)
        # View per-crop detections mapped back to the 640 semantic reference.
        sem_pred=crop_box_to_semantic(loc[j]['bboxes'],ex['roi'],batch,0,j)
        mapped=sem_pred.detach().float().cpu().numpy().reshape(-1,4)
        rawscore=loc[j]['conf'].detach().float().cpu().numpy().reshape(-1)
        validscore=rawscore>=conf
        semloc=draw_boxes(draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT'),mapped[validscore],PRED,f'ROI{j+1}',rawscore[validscore])
        _write_tile(folder,4,'roi_prediction_mapped_to_semantic',semloc,
                    f'ROI#{j+1}: detection mapped back to semantic RGB 640',canvas=canvas)
        if features:
            for si,lid in zip((3,4,5),net.detector.fusion_indices):
                for name,key in [('rgb','rgb_backbone'),('tir','tir_backbone'),('add','fused_backbone')]:
                    if key not in local_feat:continue
                    feat=_map(local_feat[key][lid],j)
                    if feat is None:continue
                    img=overlay(rgbroi,feat)
                    img=draw_boxes(img,gt_local,RGB_GT,'RGB GT')
                    _write_tile(folder,5+(si-3)*3+{'rgb':0,'tir':1,'add':2}[name],f'p{si}_{name}_feature',img,
                                f'ROI#{j+1} P{si} {name.upper()} feature',
                                'ADD uses shared RGB/TIR backbone features' if name=='add' else '',canvas=canvas)
        files=sorted(folder.glob('[0-9][0-9]_*.jpg'))
        make_overview(files,folder/'99_roi_overview.jpg',cols=4,thumb=min(330,max(240,canvas//3)))
        roi_views.append((j,xy0,local_pred_img))
        if len(pred_local) and len(gt_local):
            best_iou=max(bbox_iou(p,g) for p in pred_local for g in gt_local)
        else:best_iou=0.
        per_roi_meta.append({**rec[j],'gt_visible_fractions':vis.tolist(),
                             'gt_covered':bool(np.any(vis>=min_vis)),
                             'tir_gt_visible_fractions':vis_t.tolist(),
                             'detection_count':len(pred_local),'scores':score_local.tolist(),
                             'best_local_pred_gt_iou':float(best_iou)})
        emit(f'roi{j+1}_rgb_crop',local_gt_img,f'ROI#{j+1}: High RGB GT',f'Native {roi_size}x{roi_size} crop; not resized semantic RGB')
        emit(f'roi{j+1}_tir_crop',local_tir_img,f'ROI#{j+1}: synchronized TIR GT')
        emit(f'roi{j+1}_pred_vs_gt',local_pred_img,f'ROI#{j+1}: local prediction versus GT')

    final_boxes,final_scores=_detection_arr(merged[0],conf)
    final_sem=draw_boxes(draw_boxes(sem,rgb_sem_gt,RGB_GT,'RGB GT'),final_boxes,PRED,'DET',final_scores)
    emit('final_detection_semantic',final_sem,'Final Top-K NMS detection in RGB semantic coordinates',
         f'num={len(final_boxes)} conf>={conf:.3f}; no global detection boxes')
    final_high_boxes=_semantic_xyxy_to_high(final_boxes,sample,(hh,hw),(sh,sw))
    final_high=draw_boxes(draw_boxes(high,rgb_high_gt,RGB_GT,'RGB GT'),final_high_boxes,PRED,'DET',final_scores)
    emit('final_detection_high',final_high,'Final detection mapped to RGB High 1920 coordinates',
         'Red=prediction; Green=RGB GT; global-local coordinate transforms restored')

    final_gt_iou=max((bbox_iou(p,g) for p in final_boxes for g in rgb_sem_gt),default=0.)
    gt_centers=sample['rgb_semantic_bboxes'].detach().cpu().numpy().reshape(-1,4)
    gt_main=gt_centers[np.argmax(gt_centers[:,2]*gt_centers[:,3]),:2] if len(gt_centers) else None
    guide_err_px=float(np.linalg.norm((primary-gt_main)*np.array([sw,sh]))) if gt_main is not None else None
    roi_union=any(item['gt_covered'] for item in per_roi_meta)
    meta={'index':idx,'rgb_path':sample['rgb_path'],'tir_path':sample['tir_path'],
          'architecture':'gmls_v2_one_shared_head_global_aux_top2_officialpt_v2',
          'inference_used_gt_crop':False,'inference_used_global_detect':False,
          'topk':k,'image_sizes':{'rgb_high':[hh,hw],'rgb_sem':[sh,sw],'tir':list(tir.shape[:2])},
          'alignment':{'anchor_rgb':anchor.tolist() if anchor is not None else None,
                       'source_center_tir':source.tolist() if source is not None else None,
                       'sampling_scale':sc.tolist(),'sampling_translation':tr.tolist(),
                       'gt_iou_before':iou_raw,'gt_iou_after':iou_final,
                       'candidate_offsets':{n:_v(info.get(n,dbg.get(n))).tolist() if info.get(n,dbg.get(n)) is not None else None
                                            for n in ['coarse_offset_gmls','coarse_offset_dmls','coarse_offset_gmls_v2','fine_offset','refined_offset']},
                       'predicted_qualities':qual.tolist() if qual is not None else None},
          'roi_guide':{'primary_center':primary.tolist(),
                       'candidate_centers':used.tolist(),
                       'center_error_sem_px':guide_err_px,
                       'heat_minmax':[float(guide_p.min()),float(guide_p.max())]},
          'roi_coverage_union':bool(roi_union),'roi':per_roi_meta,
          'final':{'num_predictions_displayed':len(final_boxes),'predictions_xyxy_sem':final_boxes.tolist(),
                   'confidence':final_scores.tolist(), 'best_pred_gt_iou':float(final_gt_iou)}}
    (target/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

    keys=('rgb_high_gt','rgb_semantic_gt','tir_input_gt','raw_gt_overlay',
          'rlsfa_coarse_targetness','rlsfa_anchor_distribution','rgb_targetness_raw','aligned_tir_targetness',
          'gmls_dmls_quality','gtga_aligned_tir','rgb_tir_aligned_blend',
          'roi_guide_distribution','roi_guide_heat_overlay','roi_centers_semantic',
          'topk_rois_on_rgb_high','roi1_rgb_crop','roi1_tir_crop','roi1_pred_vs_gt',
          'roi2_rgb_crop','roi2_tir_crop','roi2_pred_vs_gt','final_detection_semantic','final_detection_high')
    key_paths=[]
    for key in keys:
        found=[p for p in paths if p.stem.split('_',1)[-1]==key]
        if found:key_paths.append(found[0])
    make_overview(key_paths,target/'99_key_overview.jpg',cols=4,thumb=min(390,max(260,canvas//3)))
    print(f'[VIS] index={idx} -> {target}')
    print(f'      GMLS AlignIoU={iou_final if iou_final is not None else float("nan"):.4f}, ROI GT hit={roi_union} guide_err={guide_err_px}')
    print(f'      ROI count={k}, raw per-ROI preds={[int(len(x["conf"])) for x in loc]}, final >= {conf}: {len(final_boxes)}')
    return meta


def main(args):
    # Imports delayed so the module's renderer functions can be tested without ultralytics.
    from engine.runtime import load_config,select_device
    from datasets.rsdt_dataset import build_rsdt_dataset
    from gmls_roi_onehead_v2.model import GMLSV2OneHeadHybrid
    from gmls_roi_onehead_v2.runner import _load_checkpoint
    cfg=load_config(args.cfg)
    weights=Path(args.weights).expanduser().resolve()
    if not weights.is_file():raise FileNotFoundError(f'Missing checkpoint {weights}')
    device=select_device(args.device)
    net=GMLSV2OneHeadHybrid(cfg,verbose=False).to(device)
    ckpt=_load_checkpoint(weights,net,require_opt=False)
    net.eval()
    ck_cfg=ckpt.get('config',{})
    k_ck=int(ck_cfg.get('model',{}).get('roi_single_head',{}).get('topk',net.num_rois))
    if net.num_rois!=k_ck:
        raise ValueError(f'Checkpoint was trained with topk={k_ck}, YAML builds topk={net.num_rois}')
    dc=cfg['data']
    ds=build_rsdt_dataset(rgb_yaml=dc['rgb'],tir_yaml=dc['tir'],split=args.split,
         rgb_high_imgsz=int(dc.get('rgb_imgsz',1920)),
         rgb_semantic_imgsz=int(dc.get('rgb_semantic_imgsz',640)),
         tir_imgsz=int(dc.get('tir_imgsz',640)),pair_mode=str(dc.get('pair_mode','relative')),
         augment=False,tir_channels=3,strict_pair=True)
    indices=_select_indices(ds,args.indices,args.index,args.first_n)
    if not indices:raise ValueError('No visualization indices selected')
    run_dir=weights.parent.parent
    out=Path(args.output).expanduser().resolve() if args.output else run_dir/'visualization_hybrid'/args.split
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    print(f'[GMLS-V2 VIS] checkpoint={weights} split={args.split} samples={len(indices)} device={device}')
    for idx in indices:
        try:
            rec=visualize_one(idx,net,ds,device,out,args.conf,args.iou,args.max_det,args.canvas,
                              args.aux_global,not args.no_features)
            summary.append(rec)
        except Exception as exc:
            if args.skip_errors:
                print(f'[ERROR] index={idx}: {type(exc).__name__}: {exc}')
                continue
            raise
    (out/'summary.json').write_text(json.dumps({
        'checkpoint':str(weights),'config':str(Path(args.cfg).resolve()),
        'split':args.split,'indices':indices,'num_completed':len(summary),
        'samples':summary},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['index','image','topk','align_iou_before','align_iou_after',
                         'guide_error_sem_px','roi_gt_coverage_union','final_n_det','best_pred_gt_iou'])
        for r in summary:
            writer.writerow([r['index'],Path(r['rgb_path']).name,r['topk'],
                r['alignment']['gt_iou_before'],r['alignment']['gt_iou_after'],
                r['roi_guide']['center_error_sem_px'],r['roi_coverage_union'],
                r['final']['num_predictions_displayed'],r['final']['best_pred_gt_iou']])
    print(f'\n[DONE] {len(summary)}/{len(indices)} samples visualized. All files under {out}')
    print('       Existing visualization images and summary are overwritten; checkpoints are untouched.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Complete visualization: GMLS-v2 Hybrid Top-K, one shared head')
    p.add_argument('--cfg',required=True,help='Original GMLS-v2 Hybrid training YAML')
    p.add_argument('--weights',required=True,help='Trained V2 Hybrid checkpoint best.pt/last.pt')
    p.add_argument('--split',choices=('train','val','test'),default='test')
    p.add_argument('--device',default='3')
    p.add_argument('--index',type=int,default=None)
    p.add_argument('--indices',nargs='+',default=None)
    p.add_argument('--first-n',type=int,default=1,help='Default number of samples if no indices specified')
    p.add_argument('--conf',type=float,default=.25,help='Display threshold only; detection always uses conf<=0.001')
    p.add_argument('--iou',type=float,default=.70,help='Local head NMS IoU; cross-ROI NMS from YAML')
    p.add_argument('--max-det',type=int,default=300)
    p.add_argument('--canvas',type=int,default=1050)
    p.add_argument('--output',default=None,help='Default <run-dir>/visualization_hybrid/<split>')
    p.add_argument('--aux-global',action='store_true',help='Optional TRAIN-ONLY auxiliary global detection comparison')
    p.add_argument('--no-features',action='store_true',help='Skip P3/P4/P5 RGB/TIR/ADD feature heatmaps')
    p.add_argument('--skip-errors',action='store_true',help='Skip failing samples instead of stopping')
    a=p.parse_args()
    if a.canvas<256 or not (0<=a.conf<=1) or not (0<a.iou<=1):p.error('canvas>=256, 0<=conf<=1, 0<iou<=1 required')
    main(a)
