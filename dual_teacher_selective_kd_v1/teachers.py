"""Two independent, frozen full YOLO26m teachers. Never registered in the student.

Both teachers operate on native 640x640 inputs and are used ONLY by model.loss().
Teacher predictions are quality checked against native-coordinate GT; teacher
P3 features are captured by a forward hook. No teacher weights are checkpointed.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from engine.evaluator import postprocess_predictions


def xywh_to_xyxy(b):
    return torch.stack((b[...,0]-b[...,2]/2,b[...,1]-b[...,3]/2,
                        b[...,0]+b[...,2]/2,b[...,1]+b[...,3]/2),dim=-1)


def pair_iou(box, others, eps=1e-7):
    a=box.reshape(1,4); b=others.reshape(-1,4)
    if b.numel()==0: return a.new_zeros((0,))
    inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter).clamp_min(eps)


def predicted_response(box, h, w, in_hw, device):
    """Box in native input pixels -> soft UAV localization map at P3."""
    ih,iw=in_hw
    x1,y1,x2,y2=box.detach().float()
    cx=(x1+x2)/(2*iw)*max(w-1,1); cy=(y1+y2)/(2*ih)*max(h-1,1)
    sx=((x2-x1)/iw*w*.25).clamp_min(1.); sy=((y2-y1)/ih*h*.25).clamp_min(1.)
    yy=torch.arange(h,device=device,dtype=torch.float32).view(h,1)
    xx=torch.arange(w,device=device,dtype=torch.float32).view(1,w)
    return torch.exp(-.5*(((xx-cx)/sx)**2+((yy-cy)/sy)**2))


class FrozenTeacher:
    def __init__(self, weights, p3_layer=4):
        weights=Path(weights).expanduser().resolve()
        if not weights.is_file():raise FileNotFoundError(f'Teacher checkpoint not found: {weights}')
        self.weights=weights
        self.net=YOLO(str(weights)).model.eval()
        if int(getattr(self.net,'nc', len(getattr(self.net,'names',{}))))!=1:
            raise ValueError(f'Teacher must be nc=1: {weights}')
        if not hasattr(self.net,'model') or p3_layer>=len(self.net.model):
            raise RuntimeError(f'P3 layer {p3_layer} absent in {weights}')
        for p in self.net.parameters():p.requires_grad_(False)
        self.device=None
        self._activation=None
        def hook(_module,_inputs,output):self._activation=output
        self._hook=self.net.model[int(p3_layer)].register_forward_hook(hook)

    def to(self,device):
        device=torch.device(device)
        if self.device!=device:
            self.net.to(device);self.device=device
        return self

    @torch.no_grad()
    def infer(self, images, gt_boxes, batch_idx, feature_hw, *,
              microbatch=2, conf_min=.15, correct_iou=.5):
        """GT used only to SELECT trustworthy teacher predictions during train."""
        n=int(images.shape[0]); h,w=map(int,feature_hw); inp_hw=tuple(images.shape[-2:])
        feats=[]; scores=[]; heats=[]
        self.net.eval()
        for start in range(0,n,max(1,int(microbatch))):
            end=min(n,start+max(1,int(microbatch)))
            imgs=images[start:end]
            self._activation=None
            with torch.autocast(device_type=imgs.device.type,dtype=torch.float16,
                                enabled=imgs.device.type=='cuda'):
                preds=self.net(imgs)
            f=self._activation
            if not torch.is_tensor(f):raise RuntimeError('Teacher P3 forward hook returned no Tensor')
            if f.shape[-2:]!=(h,w):
                f=F.interpolate(f.float(),size=(h,w),mode='bilinear',align_corners=False)
            feats.append(f.detach().float())
            detections=postprocess_predictions(preds,self.net,conf_thres=max(.01,float(conf_min)*.5),
                                               iou_thres=.7,max_det=30)
            for loc,det in enumerate(detections):
                bi=start+loc
                sample_gt=gt_boxes[(batch_idx==bi)]
                if sample_gt.numel():
                    native=xywh_to_xyxy(sample_gt.float())
                    native=native*sample_gt.new_tensor([inp_hw[1],inp_hw[0],inp_hw[1],inp_hw[0]])
                else:native=sample_gt.new_zeros((0,4))
                chosen=None; chosen_score=0.
                for box,score,cl in zip(det['bboxes'],det['conf'],det['cls']):
                    if int(cl.item())!=0 or float(score)<float(conf_min):continue
                    if native.numel()==0:continue
                    if float(pair_iou(box.float(),native).max())<float(correct_iou):continue
                    if float(score)>chosen_score:
                        chosen,chosen_score=box,float(score)
                scores.append(torch.tensor(chosen_score,device=images.device,dtype=torch.float32))
                heats.append(predicted_response(chosen,h,w,inp_hw,images.device) if chosen is not None
                             else torch.zeros(h,w,device=images.device,dtype=torch.float32))
        return {'p3':torch.cat(feats,0).detach(),
                'score':torch.stack(scores).detach(),
                'response':torch.stack(heats)[:,None].detach()}

    def close(self):self._hook.remove()


class FrozenDualTeachers:
    def __init__(self,rgb_weights,tir_weights,p3_layer=4):
        self.rgb=FrozenTeacher(rgb_weights,p3_layer)
        self.tir=FrozenTeacher(tir_weights,p3_layer)

    def to(self,device):
        self.rgb.to(device);self.tir.to(device);return self
