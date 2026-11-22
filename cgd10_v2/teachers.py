"""Frozen teacher P3 feature response extraction and geometry-target matching.

Unlike CGD-10 V1, V2 does not manufacture Gaussian heatmaps from teacher boxes.
It uses real frozen YOLO P3 channel energy, localized by training GT in each
native-modality image. Teacher detections only provide reliability gating.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import torch
from .losses import response_from_p3, route_offset_response_kd, final_feature_warp_kd


class FrozenP3Teacher:
    def __init__(self,weights,p3_index,device,microbatch=2):
        from ultralytics import YOLO
        self.path=str(Path(weights).expanduser().resolve())
        if not Path(self.path).is_file():
            raise FileNotFoundError(f'Missing frozen teacher: {self.path}')
        self.net=YOLO(self.path).model.to(device).float().eval()
        self.net.requires_grad_(False)
        self.det=self.net.model[-1]
        if int(getattr(self.det,'nc',-1))!=1:
            raise ValueError(f'Expected 1-class UAV teacher: {self.path}')
        self._feature=None
        idx=int(p3_index)
        if idx<0 or idx>=len(self.net.model):
            raise ValueError(f'Invalid P3 hook index {idx} for teacher {self.path}')
        def hook(_module,_inputs,output):
            self._feature=output
        self._hook=self.net.model[idx].register_forward_hook(hook)
        self.microbatch=max(1,int(microbatch))

    @torch.no_grad()
    def forward(self,images):
        from engine.evaluator import postprocess_predictions
        self.net.eval()
        features=[];predictions=[]
        for chunk in images.split(self.microbatch,dim=0):
            self._feature=None
            pred=self.net(chunk)
            feat=self._feature
            if not torch.is_tensor(feat) or feat.ndim!=4:
                raise RuntimeError(f'P3 teacher hook did not return [B,C,H,W] feature: {self.path}')
            features.append(feat.detach().float())
            meta=SimpleNamespace(nc=1,end2end=bool(
                getattr(self.det,'end2end',getattr(self.net,'end2end',False))))
            predictions.extend(postprocess_predictions(pred,meta,
                                conf_thres=.001,iou_thres=.70,max_det=300))
        if not features:raise RuntimeError('Empty teacher batch')
        return torch.cat(features,0),predictions

    def close(self):
        if getattr(self,'_hook',None) is not None:
            self._hook.remove();self._hook=None


def _xywh_to_xyxy(x):
    return torch.cat((x[...,:2]-x[...,2:]*.5,x[...,:2]+x[...,2:]*.5),dim=-1)


def _single_iou(box,gt):
    lo=torch.maximum(box[:2],gt[:2]);hi=torch.minimum(box[2:],gt[2:])
    inter=(hi-lo).clamp_min(0).prod()
    area1=(box[2:]-box[:2]).clamp_min(0).prod()
    area2=(gt[2:]-gt[:2]).clamp_min(0).prod()
    return inter/(area1+area2-inter).clamp_min(1e-7)


@torch.no_grad()
def choose_teacher_boxes(batch,predictions,modality,min_conf=.3,min_iou=.5):
    """One valid paired GT per image. RGB-sem uses `rgb_batch_idx`, NOT `rgb_semantic_batch_idx`."""
    if modality not in ('rgb','tir'):raise ValueError(modality)
    prefix='rgb_semantic' if modality=='rgb' else 'tir'
    idxkey='rgb_batch_idx' if modality=='rgb' else 'tir_batch_idx'
    clskey='rgb_cls' if modality=='rgb' else 'tir_cls'
    gt=batch[f'{prefix}_bboxes'].detach().float().reshape(-1,4)
    idx=batch[idxkey].long().flatten()
    cls=batch[clskey].long().flatten()
    img=batch[f'{prefix}_img']
    b,h,w=int(img.shape[0]),int(img.shape[-2]),int(img.shape[-1])
    if len(predictions)!=b:raise RuntimeError('Teacher predictions do not match image batch')
    tgt=[];quality=[];valid=[];best_ious=[]
    for i in range(b):
        mask=(idx==i);boxes=gt[mask];classes=cls[mask]
        fallback=(boxes[0] if len(boxes) else gt.new_tensor([.5,.5,.03,.03])).clone()
        fallback=torch.nan_to_num(fallback,nan=.5,posinf=.5,neginf=.5)
        fallback=torch.cat((fallback[:2].clamp(0,1),fallback[2:].clamp(1e-4,1)))
        tgt.append(fallback)
        if len(boxes)!=1 or not bool(torch.isfinite(boxes).all()) or not bool((boxes[0,2:]>0).all()):
            quality.append(0.);valid.append(False);best_ious.append(0.);continue
        p=predictions[i];pb=p['bboxes'].detach().float().reshape(-1,4)
        pc=p['conf'].detach().float().flatten()
        cl=p['cls'].detach().long().flatten()
        keep=(pc>=float(min_conf))&(cl==classes[0])&torch.isfinite(pc)&torch.isfinite(pb).all(dim=1)
        if not bool(keep.any()):
            quality.append(0.);valid.append(False);best_ious.append(0.);continue
        pb=pb[keep];pc=pc[keep]
        target_box=_xywh_to_xyxy(boxes[0])*boxes.new_tensor([w,h,w,h])
        ious=torch.stack([_single_iou(box,target_box) for box in pb])
        scores=ious*pc
        j=int(scores.argmax().item())
        v=float(ious[j].item());c=float(pc[j].item());isvalid=v>=float(min_iou)
        quality.append(c*v if isvalid else 0.)
        valid.append(isvalid);best_ious.append(v)
    dev=gt.device
    return {'gt':torch.stack(tgt).detach(),
            'quality':torch.tensor(quality,device=dev),
            'valid':torch.tensor(valid,device=dev,dtype=torch.bool),
            'mean_iou':torch.tensor(best_ious,device=dev).mean()}


class FeatureGeometryTeachers:
    def __init__(self,cfg,p3_index,device):
        d=cfg['geometry_distillation'];self.d=d
        self.rgb=FrozenP3Teacher(d['rgb_teacher'],p3_index,device,d.get('teacher_microbatch',2))
        self.tir=FrozenP3Teacher(d['tir_teacher'],p3_index,device,d.get('teacher_microbatch',2))
        print('[CGD-10 V2] Frozen P3 RGB/TIR teachers loaded; feature-response geometry KD; no student params added',flush=True)

    @torch.no_grad()
    def targets(self,batch):
        d=self.d
        fr,pr=self.rgb.forward(batch['rgb_semantic_img'])
        ft,pt=self.tir.forward(batch['tir_img'])
        r=choose_teacher_boxes(batch,pr,'rgb',d.get('min_conf',.3),d.get('min_iou',.5))
        t=choose_teacher_boxes(batch,pt,'tir',d.get('min_conf',.3),d.get('min_iou',.5))
        kw={'roi_pad_cells':float(d.get('roi_pad_cells',3.)),
            'window_expand':float(d.get('roi_window_expand',1.25)),
            'response_temperature':float(d.get('response_temperature',.75))}
        rr=response_from_p3(fr,r['gt'],**kw)
        tr=response_from_p3(ft,t['gt'],**kw)
        # Limit teacher geometry perturbations; additionally gate incorrect teacher boxes.
        q=(r['quality']*t['quality']).clamp_min(0).sqrt()
        q=q*(.5+.5*rr['concentration'])*(.5+.5*tr['concentration'])
        # Difference between response center residual and GT center offset.
        delta=(tr['center']-t['gt'][:,:2])-(rr['center']-r['gt'][:,:2])
        discrepancy=delta.norm(dim=1)*640
        sigma=max(float(d.get('pair_consistency_sigma_px',24.)),1.)
        consistency=torch.exp(-.5*(discrepancy/sigma).square())
        q=(q*consistency).detach()
        return r,t,rr,tr,q,discrepancy,consistency

    def __call__(self,batch,info):
        d=self.d
        r,t,rr,tr,q,discrepancy,consistency=self.targets(batch)
        logits=info.get('gmls_v2_quality_logits')
        gmls=info.get('coarse_offset_gmls')
        dmls=info.get('coarse_offset_dmls')
        if not all(torch.is_tensor(v) for v in (logits,gmls,dmls)):
            raise RuntimeError('CGD10 V2 requires actual GMLS/DMLS offsets and router logits')
        shape=info['rgb_target_support'].shape[-2:]
        route,offset,rd=route_offset_response_kd(
            logits,gmls,dmls,r['gt'],t['gt'],rr['center'],tr['center'],q,
            feature_hw=shape,
            route_temperature=float(d.get('route_temperature',.65)),
            route_gt_margin=float(d.get('route_gt_margin',1.5)),
            route_teacher_strength=float(d.get('route_teacher_strength',.3)),
            sigma_features=float(d.get('route_sigma_features',4.)),
            offset_beta_features=float(d.get('offset_beta_features',2.)),
            max_teacher_residual_px=float(d.get('max_teacher_center_residual_px',10.)))
        warp,wd=final_feature_warp_kd(
            info['sampling_scale'],info['sampling_translation'],
            rr,tr,r['gt'],t['gt'],q,
            max_teacher_residual_px=float(d.get('max_teacher_center_residual_px',10.)),
            center_norm_px=float(d.get('warp_center_norm_px',16.)),
            size_weight=float(d.get('warp_size_weight',.15)),
            overlap_weight=float(d.get('warp_overlap_weight',.05)))
        log={'cgd_route_raw':route.detach(),'cgd_offset_raw':offset.detach(),
             'cgd_warp_raw':warp.detach(),
             'cgd_teacher_rgb_valid':r['valid'].float().mean().detach(),
             'cgd_teacher_tir_valid':t['valid'].float().mean().detach(),
             'cgd_pair_valid':(q>0).float().mean().detach(),
             'cgd_pair_quality':q.mean().detach(),
             'cgd_teacher_pair_delta_error_px':discrepancy.mean().detach(),
             'cgd_teacher_geometry_consistency':consistency.mean().detach(),
             'cgd_teacher_rgb_concentration':rr['concentration'].mean().detach(),
             'cgd_teacher_tir_concentration':tr['concentration'].mean().detach(),
             'cgd_teacher_rgb_center_shift_px':(rr['center']-r['gt'][:,:2]).norm(dim=1).mean().detach()*640,
             'cgd_teacher_tir_center_shift_px':(tr['center']-t['gt'][:,:2]).norm(dim=1).mean().detach()*640}
        log.update({f'cgd_{key}':v for key,v in rd.items()})
        log.update({f'cgd_{key}':v for key,v in wd.items()})
        # Keep V1 logging keys available for pre-existing runner patch.
        log['cgd_route_teacher_agreement']=rd['route_agreement']
        log['cgd_warp_soft_overlap']=wd['warp_response_overlap']
        return route,offset,warp,log
