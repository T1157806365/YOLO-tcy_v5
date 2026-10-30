"""Frozen original YOLO26m RGB/TIR teachers, with native coordinate GT filtering."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from engine.evaluator import postprocess_predictions


def xywh2xyxy(x):
    return torch.stack((x[..., 0]-x[..., 2]/2, x[..., 1]-x[..., 3]/2,
                        x[..., 0]+x[..., 2]/2, x[..., 1]+x[..., 3]/2), -1)


def box_iou_single(box, targets):
    if targets.numel()==0: return box.new_zeros((0,))
    a=box.reshape(1,4); b=targets.reshape(-1,4)
    wh=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0)
    inter=wh.prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter).clamp_min(1e-7)


class SingleTeacher:
    def __init__(self, weights, p3_index):
        self.weights=Path(weights).expanduser().resolve()
        if not self.weights.is_file(): raise FileNotFoundError(self.weights)
        self.net=YOLO(str(self.weights)).model.eval()
        self.net.requires_grad_(False)
        if int(getattr(self.net, 'nc',len(getattr(self.net,'names',{}))))!=1:
            raise ValueError(f'Expect nc=1 teacher, got {self.weights}')
        self.feature=None
        def hook(_, __, out): self.feature=out
        self.hook=self.net.model[int(p3_index)].register_forward_hook(hook)

    def to(self, device):
        self.net.to(device).eval()
        return self

    @torch.no_grad()
    def get(self, imgs, boxes, batch_idx, p3_size, conf=0.30, iou=0.50, microbatch=2):
        n=imgs.shape[0]
        h,w=imgs.shape[-2:]
        feats=[]; good=[]; quality=[]; confidences=[]
        self.net.eval()
        for start in range(0,n,max(1,microbatch)):
            end=min(n,start+max(1,microbatch)); x=imgs[start:end]
            self.feature=None
            with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                                enabled=x.device.type=='cuda'):
                out=self.net(x)
            if not torch.is_tensor(self.feature):
                raise RuntimeError('Teacher P3 feature hook failed')
            ft=self.feature.detach().float()
            if ft.shape[-2:]!=tuple(p3_size):
                ft=F.interpolate(ft, size=p3_size, mode='bilinear', align_corners=False)
            feats.append(ft)
            dets=postprocess_predictions(out, self.net,
                conf_thres=max(0.001,conf*0.5),iou_thres=0.70,max_det=100)
            for j,det in enumerate(dets):
                idx=start+j
                gt=boxes[batch_idx==idx]
                if not len(gt):
                    good.append(False);quality.append(0.);confidences.append(0.);continue
                xy=xywh2xyxy(gt.float())*gt.new_tensor([w,h,w,h])
                qbest=0.;cbest=0.
                for box,score,cl in zip(det['bboxes'],det['conf'],det['cls']):
                    if int(cl)!=0 or float(score)<conf:continue
                    q=float(box_iou_single(box.float(),xy).max().item())
                    if q>qbest:qbest=q;cbest=float(score)
                good.append(qbest>=iou);quality.append(qbest);confidences.append(cbest)
        return {'p3':torch.cat(feats,0),
                'good':torch.tensor(good,device=imgs.device,dtype=torch.bool),
                'iou':torch.tensor(quality,device=imgs.device),
                'conf':torch.tensor(confidences,device=imgs.device)}

    def close(self): self.hook.remove()


class DualTeachers:
    def __init__(self,cfg,p3_index,device):
        self.rgb=SingleTeacher(cfg['rgb_weights'],p3_index).to(device)
        self.tir=SingleTeacher(cfg['tir_weights'],p3_index).to(device)
        self.cfg=cfg

    def get(self,batch,p3_size):
        d=self.cfg
        kw=dict(p3_size=p3_size,conf=float(d.get('min_conf',0.30)),
                iou=float(d.get('min_gt_iou',0.50)),microbatch=int(d.get('forward_microbatch',2)))
        return (self.rgb.get(batch['rgb_semantic_img'],batch['rgb_semantic_bboxes'],batch['rgb_batch_idx'],**kw),
                self.tir.get(batch['tir_img'],batch['tir_bboxes'],batch['tir_batch_idx'],**kw))
