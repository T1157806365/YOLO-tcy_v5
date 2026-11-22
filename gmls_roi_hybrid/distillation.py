"""GPR-MS KD: strictly modality-specific P3/P4/P5 knowledge distillation.

RGB and TIR teachers are independent, frozen Ultralytics YOLO26m detectors.
Teacher predictions are used ONLY to decide which *training* GT boxes are
eligible, never for model output or ROI selection. Student tensors are captured
before RGB/TIR alignment from the original global backbone forward.

No extra student parameters. Teacher modules are intentionally not registered
on the student's nn.Module, so they do not enter optimizer/DDP/state_dict.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

SCALES=(3,4,5)
MODES=('rgb','tir')


def kd_weight(epoch: int, maximum: float=0.01) -> float:
    """Zero-based epoch schedule for exactly 100 epochs.

    1-10: 0; 11-20: linear warmup; 21-60: max;
    61-80: linear decay; 81-100: 0.
    """
    ep=int(epoch)+1
    if not 1 <= ep <= 100:
        raise ValueError(f'KD schedule expects epochs 1..100, got {ep}')
    if ep <= 10 or ep >= 81: return 0.0
    if ep <= 20: return float(maximum)*(ep-10)/10.0
    if ep <= 60: return float(maximum)
    return float(maximum)*(80-ep)/20.0


def _xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    return torch.cat((box[...,:2]-box[...,2:]/2,
                      box[...,:2]+box[...,2:]/2),dim=-1)


def _pair_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU for XYXY boxes. a [N,4], b [M,4]."""
    lt=torch.maximum(a[:,None,:2],b[None,:,:2])
    rb=torch.minimum(a[:,None,2:],b[None,:,2:])
    inter=(rb-lt).clamp_min(0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(area_a[:,None]+area_b[None,:]-inter).clamp_min(1e-7)


def matched_teacher_targets(
    batch: Mapping, predictions: Sequence[Mapping[str,torch.Tensor]],
    modality: str, *, min_conf:float=0.30, min_iou:float=0.50,
) -> Tuple[Sequence[torch.Tensor],torch.Tensor,int,int]:
    """Return eligible native-modality GT boxes, confidence×IoU per image.

    GT box tensors have normalized xywh in modality-specific 640 letterbox
    coordinates. Teacher predictions have pixel XYXY coordinates for the same
    640 input (as returned by postprocess_predictions()).
    """
    if modality=='rgb':
        boxes=batch['rgb_semantic_bboxes']; inds=batch['rgb_batch_idx'];
        classes=batch['rgb_cls']; img=batch['rgb_semantic_img']
    elif modality=='tir':
        boxes=batch['tir_bboxes']; inds=batch['tir_batch_idx'];
        classes=batch['tir_cls']; img=batch['tir_img']
    else:raise ValueError(f'Bad modality: {modality}')
    n=int(img.shape[0]); h,w=img.shape[-2:]
    if len(predictions)!=n: raise ValueError('Teacher prediction count mismatch')
    boxes=boxes.float().reshape(-1,4)
    inds=inds.long().view(-1)
    classes=classes.long().view(-1)
    if len(inds)!=len(boxes) or len(classes)!=len(boxes):
        raise ValueError(f'{modality}: GT labels are mismatched')
    output=[]; confidence=[]; valid_total=0
    for i in range(n):
        mask=inds==i; b=boxes[mask]; cls=classes[mask]
        pred=predictions[i]
        pboxes=pred['bboxes'].float().reshape(-1,4)
        pconf=pred['conf'].float().reshape(-1)
        pcls=pred['cls'].long().reshape(-1)
        if len(pboxes)!=len(pconf) or len(pboxes)!=len(pcls):
            raise ValueError('Teacher detection output malformed')
        keep=pconf>=float(min_conf)
        pboxes,pconf,pcls=pboxes[keep],pconf[keep],pcls[keep]
        if b.numel()==0 or pboxes.numel()==0:
            output.append(b[:0]);confidence.append(boxes.new_zeros(()));continue
        gtxy=_xywh_to_xyxy(b)*b.new_tensor([w,h,w,h])
        ious=_pair_iou(gtxy,pboxes)
        ious=ious.masked_fill(cls[:,None]!=pcls[None,:],0)
        best_iou,best_id=ious.max(dim=1)
        valid=(best_iou>=float(min_iou))&(b[:,2:].min(-1).values>0)
        weights=(best_iou*pconf[best_id]).detach()[valid]
        output.append(b[valid].detach())
        confidence.append(weights.mean() if weights.numel() else boxes.new_zeros(()))
        valid_total+=int(valid.sum().item())
    return output,torch.stack(confidence).detach(),valid_total,int(len(boxes))


def _roi_mask(bboxes:torch.Tensor,h:int,w:int,
              *,min_pixels:int=5,expand:float=1.5,device=None)->torch.Tensor:
    """Bool union of expanded native GT boxes on an h×w feature map."""
    mask=torch.zeros((h,w),dtype=torch.bool,device=device or bboxes.device)
    for xywh in bboxes:
        cx,cy,bw,bh=xywh.unbind()
        rx=torch.maximum(bw*float(expand)*0.5,cx.new_tensor(float(min_pixels)/(2*w)))
        ry=torch.maximum(bh*float(expand)*0.5,cy.new_tensor(float(min_pixels)/(2*h)))
        x0=max(0,min(w-1,int(torch.floor((cx-rx)*w).item())))
        x1=max(x0+1,min(w,int(torch.ceil((cx+rx)*w).item())))
        y0=max(0,min(h-1,int(torch.floor((cy-ry)*h).item())))
        y1=max(y0+1,min(h,int(torch.ceil((cy+ry)*h).item())))
        mask[y0:y1,x0:x1]=True
    return mask


def _attention_logits(feat:torch.Tensor)->torch.Tensor:
    # Normalize per ROI before softmax: avoids degree-of-magnitude mismatch.
    energy=feat.float().square().mean(dim=0)
    return energy


def p3_roi_attention(student:torch.Tensor,teacher:torch.Tensor,
                     target_boxes:Sequence[torch.Tensor],q:torch.Tensor,
                     *,min_pixels:int=5,temperature:float=1.0)->torch.Tensor:
    """Masked KL(teacher || student) in independently matched RGB/TIR ROIs."""
    if temperature<=0:raise ValueError('temperature must be >0')
    losses=[]
    for i,boxes in enumerate(target_boxes):
        if boxes.numel()==0:continue
        a=_attention_logits(student[i]);b=_attention_logits(teacher[i]).detach()
        h,w=a.shape
        mask=_roi_mask(boxes,h,w,min_pixels=min_pixels,device=a.device)
        if int(mask.sum())<2:continue
        sa=a[mask];tb=b[mask]
        # Standardize within the target neighborhood, allowing different
        # RGB/TIR activation scales and separate student/teacher magnitudes.
        sa=(sa-sa.mean())/(sa.std(unbiased=False)+1e-5)
        tb=(tb-tb.mean())/(tb.std(unbiased=False)+1e-5)
        sa=sa/float(temperature);tb=tb/float(temperature)
        kl=F.kl_div(F.log_softmax(sa,dim=0),
                    F.softmax(tb,dim=0),reduction='sum')
        losses.append(kl*q[i])
    if losses: return torch.stack(losses).sum()/max(len(losses),1)
    return student.float().sum()*0.0


def _region_mean(x:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    if not bool(mask.any()):return x.mean(dim=(1,2))
    return x[:,mask].mean(dim=1)


def p4_fg_bg_contrast(student:torch.Tensor,teacher:torch.Tensor,
                      target_boxes:Sequence[torch.Tensor],q:torch.Tensor,
                      *,min_pixels:int=3, beta:float=0.2)->torch.Tensor:
    if beta<=0:raise ValueError('beta must be >0')
    losses=[]
    for i,boxes in enumerate(target_boxes):
        if boxes.numel()==0:continue
        s=student[i].float();t=teacher[i].float().detach()
        c,h,w=s.shape
        fg=_roi_mask(boxes,h,w,min_pixels=min_pixels,expand=1.2,device=s.device)
        outer=_roi_mask(boxes,h,w,min_pixels=min_pixels*3,expand=3.,device=s.device)
        bg=outer&(~fg)
        if not bool(bg.any()): bg=~fg
        if not bool(bg.any()):continue
        ds=_region_mean(s,fg)-_region_mean(s,bg)
        dt=_region_mean(t,fg)-_region_mean(t,bg)
        # sqrt(C) keeps SmoothL1 magnitude comparable across scales.
        ds=F.normalize(ds,dim=0,eps=1e-6)*c**.5
        dt=F.normalize(dt,dim=0,eps=1e-6)*c**.5
        losses.append(F.smooth_l1_loss(ds,dt,beta=beta,reduction='mean')*q[i])
    if losses:return torch.stack(losses).sum()/max(len(losses),1)
    return student.float().sum()*0.0


def p5_semantic_relation(student:torch.Tensor,teacher:torch.Tensor,
                         target_boxes:Sequence[torch.Tensor],q:torch.Tensor)->torch.Tensor:
    s=student.float().mean(dim=(-1,-2))
    t=teacher.float().detach().mean(dim=(-1,-2))
    cos=1-F.cosine_similarity(s,t,dim=-1,eps=1e-6)
    mask=torch.stack([torch.as_tensor(len(b)>0,device=s.device) for b in target_boxes]).bool()
    if bool(mask.any()):return (cos[mask]*q[mask]).mean()
    return student.float().sum()*0.0


@dataclass(frozen=True)
class KDSettings:
    rgb_weights: str
    tir_weights: str
    confidence: float=0.30
    min_iou: float=0.50
    p3: float=0.60
    p4: float=0.30
    p5: float=0.10
    max_weight: float=0.01
    min_roi_pixels: int=5
    temperature: float=1.0
    p4_beta: float=0.2

    @classmethod
    def from_config(cls,cfg: Mapping)->'KDSettings':
        d=cfg.get('distillation',{})
        return cls(
            rgb_weights=str(d['rgb_teacher']),tir_weights=str(d['tir_teacher']),
            confidence=float(d.get('min_conf',.30)),
            min_iou=float(d.get('min_iou',.50)),
            p3=float(d.get('p3_weight',.60)),
            p4=float(d.get('p4_weight',.30)),
            p5=float(d.get('p5_weight',.10)),
            max_weight=float(d.get('max_weight',.01)),
            min_roi_pixels=int(d.get('p3_min_pixels',5)),
            temperature=float(d.get('attention_temperature',1.0)),
            p4_beta=float(d.get('p4_beta',.20)),
        )

    def validate(self):
        if not 0<=self.confidence<=1 or not 0<=self.min_iou<=1:
            raise ValueError('Bad teacher thresholds')
        if self.max_weight<0 or self.min_roi_pixels<2 or self.p4_beta<=0:
            raise ValueError('Invalid KD weight/ROI parameters')
        if abs(self.p3+self.p4+self.p5-1)>1e-6 or min(self.p3,self.p4,self.p5)<0:
            raise ValueError('P3/P4/P5 weights must be nonnegative and sum to 1')
        if self.temperature<=0:raise ValueError('Invalid attention temperature')
        for name in (self.rgb_weights,self.tir_weights):
            if not Path(name).expanduser().is_file():
                raise FileNotFoundError('Teacher checkpoint missing: '+name)


class FrozenYOLOTeacher:
    """Normal Ultralytics YOLO .pt with full P3/P4/P5 backbone and Detect.

    Fails closed on unsupported checkpoint formats; no random/partial
    teacher initialization is ever allowed.
    """
    def __init__(self,weights:str,indices:Mapping[int,int],device:torch.device):
        from ultralytics import YOLO
        from engine.evaluator import postprocess_predictions
        self._postprocess=postprocess_predictions
        self.weights=str(weights)
        try:
            net=YOLO(self.weights).model
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load RGB/TIR teacher {self.weights} as Ultralytics YOLO .pt. '
                'This checkpoint must contain a fully trained YOLO26m single-modal model; '
                'if your teacher was saved as a custom state_dict, export it first.'
            ) from exc
        if not hasattr(net,'model'):
            raise RuntimeError('Teacher has no Ultralytics module graph')
        nclasses=int(getattr(net.model[-1],'nc',-1))
        if nclasses!=1:raise RuntimeError(
            f'Teacher {weights} has nc={nclasses}, expected trained UAV nc=1')
        self.model=net.to(device).eval().float()
        self.indices={int(k):int(v) for k,v in indices.items()}
        self.features={}
        self._handles=[]
        for s,i in self.indices.items():
            if i>=len(self.model.model)-1 or i<0:
                raise RuntimeError(f'Teacher {weights}: invalid P{s} index {i}')
            def _capture(_, __, y, key=s):
                if not torch.is_tensor(y):raise TypeError(f'P{key} teacher feature must be Tensor')
                self.features[key]=y
            self._handles.append(self.model.model[i].register_forward_hook(_capture))
        for p in self.model.parameters():p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self,x:torch.Tensor):
        self.features={}
        preds=self.model(x)
        from types import SimpleNamespace
        det=self.model.model[-1]
        meta=SimpleNamespace(nc=int(det.nc),
                end2end=bool(getattr(det,'end2end',getattr(self.model,'end2end',False))))
        out=self._postprocess(preds,meta,conf_thres=.001,
                              iou_thres=.7,max_det=300)
        if any(k not in self.features for k in self.indices):
            raise RuntimeError('One or more teacher P3/P4/P5 hooks did not fire')
        return {s:t.detach() for s,t in self.features.items()},out

    def clear(self): self.features.clear()


class MultiScaleKDLoss:
    def __init__(self,cfg:Mapping,indices:Mapping[int,int],device:torch.device):
        self.settings=KDSettings.from_config(cfg)
        self.settings.validate()
        self.rgb=FrozenYOLOTeacher(self.settings.rgb_weights,indices,device)
        self.tir=FrozenYOLOTeacher(self.settings.tir_weights,indices,device)
        self.indices=indices

    def __call__(self,batch:Mapping,student_features:Mapping[Tuple[str,int],torch.Tensor]):
        settings=self.settings
        terms=[]; logs={}
        for modality,teacher in [('rgb',self.rgb),('tir',self.tir)]:
            img=batch['rgb_semantic_img'] if modality=='rgb' else batch['tir_img']
            teacher_feat,preds=teacher.forward(img)
            boxes,q,nvalid,ntotal=matched_teacher_targets(
                batch,preds,modality,min_conf=settings.confidence,min_iou=settings.min_iou)
            kd_values={}
            for scale in SCALES:
                student=student_features.get((modality,scale))
                if student is None:raise RuntimeError(f'Missing student {modality.upper()} P{scale} feature')
                target=teacher_feat[scale]
                if student.shape!=target.shape:
                    raise RuntimeError(
                        f'{modality.upper()} P{scale}: student {tuple(student.shape)} '
                        f'!= teacher {tuple(target.shape)}. Use matching YOLO26m teacher weights.')
                if scale==3:
                    term=p3_roi_attention(student,target,boxes,q,
                        min_pixels=settings.min_roi_pixels,temperature=settings.temperature)
                elif scale==4:
                    term=p4_fg_bg_contrast(student,target,boxes,q,
                        beta=settings.p4_beta)
                else:
                    term=p5_semantic_relation(student,target,boxes,q)
                kd_values[scale]=term
                logs[f'kd_{modality}_p{scale}']=term.detach()
            total=(settings.p3*kd_values[3]+settings.p4*kd_values[4]+
                   settings.p5*kd_values[5])
            terms.append(total)
            logs[f'kd_{modality}_valid_rate']=float(nvalid)/max(1,ntotal)
            logs[f'kd_{modality}_quality']=q.mean().detach()
            logs[f'kd_{modality}_raw']=total.detach()
            teacher.clear()
        raw=(terms[0]+terms[1])/2.0
        logs['kd_raw']=raw.detach()
        return raw,logs
