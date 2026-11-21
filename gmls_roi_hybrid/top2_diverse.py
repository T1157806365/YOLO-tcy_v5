"""GT-free second-ROI selector for GMLS-v2 Hybrid Top-2 inference.

All candidate decisions are made from predicted guide logits, GMLS anchor,
first ROI center, and known letterbox geometry. Ground-truth labels are NEVER
read in this file. No parameter is learned or changed.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F


def _ratio_pad(meta: dict):
    r = meta["ratio"]
    if isinstance(r, (tuple, list)):
        assert len(r) == 2 and abs(float(r[0]) - float(r[1])) < 1e-4, "Nonisotropic letterbox unsupported"
        r = r[0]
    pad = meta["pad"]
    return float(r), float(pad[0]), float(pad[1])


def roi_bounds_high(centers, high_meta, sem_meta, high_hw, sem_hw, crop_size):
    """The SAME center->crop origin mapping as gmls_roi_single_head.geometry.

    centers [N, 2] normalized semantic coords, output [N, 4] high RGB
    bounding rectangles (x0,y0,x1,y1), with border clamping.
    """
    rh, phx, phy = _ratio_pad(high_meta)
    rs, psx, psy = _ratio_pad(sem_meta)
    hh, hw = high_hw
    sh, sw = sem_hw
    if crop_size > min(hh, hw):
        raise ValueError('ROI exceeds RGB high image')
    p = centers.float() * centers.new_tensor([max(sw-1, 1), max(sh-1, 1)]).float()
    cen_high = (p - centers.new_tensor([psx, psy])) / rs * rh + centers.new_tensor([phx, phy])
    lo = (cen_high - crop_size / 2.0).clamp_min(0)
    lim = lo.new_tensor([hw-crop_size, hh-crop_size])
    lo = torch.minimum(lo, lim)
    hi = lo + crop_size
    return torch.cat((lo, hi), -1)


def rect_intersection(boxes1, boxes2):
    lo = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    hi = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    return (hi-lo).clamp_min(0).prod(-1)


def roi_iou(candidate_boxes, first_box, crop_size):
    inter = rect_intersection(candidate_boxes, first_box)
    area = float(crop_size*crop_size)
    return inter / (2*area-inter).clamp_min(1e-8)


def _roi_to_heatmap_bounds(boxes, high_meta, sem_meta, high_hw, sem_hw, heatmap_hw):
    """Compute integer rectangular guide-map bins corresponding to RGB-high crop.

    Approximation for evaluating predicted guide mass, not crop geometry.
    The actual ROI sampling remains the original sample_roi_pair().
    """
    rh, phx, phy = _ratio_pad(high_meta)
    rs, psx, psy = _ratio_pad(sem_meta)
    sh, sw = sem_hw
    h, w = heatmap_hw
    lo_sem = (boxes[:, :2] - boxes.new_tensor([phx, phy])) / rh * rs + boxes.new_tensor([psx, psy])
    hi_sem = (boxes[:, 2:] - boxes.new_tensor([phx, phy])) / rh * rs + boxes.new_tensor([psx, psy])
    # Map semantic-coordinate pixel intervals to [0,W]/[0,H] cell boundaries.
    # The last boundary is clamped so no tensor index is out of range.
    fx = float(w)/float(sw)
    fy = float(h)/float(sh)
    x0=(lo_sem[:,0]*fx).floor().long().clamp(0,w)
    y0=(lo_sem[:,1]*fy).floor().long().clamp(0,h)
    x1=(hi_sem[:,0]*fx).ceil().long().clamp(0,w)
    y1=(hi_sem[:,1]*fy).ceil().long().clamp(0,h)
    return x0,y0,x1,y1


def _heatmap_region_mass(prob, rects, high_meta, sem_meta, high_hw, sem_hw):
    """Sum guide probabilities inside rects [N,4] through integral image."""
    h,w=prob.shape
    p=prob.float().clamp_min(0)
    integ = F.pad(p.cumsum(0).cumsum(1),(1,0,1,0))
    x0,y0,x1,y1 = _roi_to_heatmap_bounds(rects,high_meta,sem_meta,high_hw,sem_hw,(h,w))
    return (integ[y1,x1] - integ[y0,x1] - integ[y1,x0] + integ[y0,x0]).clamp(0,1)


def _rank_heatmap_candidates(logits, primary, anchor, temp, high_meta, sem_meta, high_hw, sem_hw,
                             crop_size, candidate_peaks):
    """GT-free proposals: original anchor + peak points of logits and pooled mass."""
    h,w=logits.shape
    yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=logits.device),
                         torch.linspace(0,1,w,device=logits.device),indexing="ij")
    positions=torch.stack((xx,yy),-1).reshape(-1,2)
    # Predicted probability from the same guide logits used by base model.
    prob=F.softmax(logits.float().reshape(-1)/max(float(temp),.01),dim=0).reshape(h,w)
    # Local maxima avoid choosing a dozen adjacent pixels around the same peak.
    radius=max(2,int(round(min(h,w)*.035)))
    ks=radius*2+1
    locmax = F.max_pool2d(logits.float()[None,None], ks, stride=1, padding=radius)[0,0]
    peaks = (logits >= locmax-1e-7)
    inds=torch.nonzero(peaks.reshape(-1),as_tuple=False).reshape(-1)
    if inds.numel():
        vals=logits.reshape(-1)[inds]
        top=inds[vals.topk(min(int(candidate_peaks), len(vals))).indices]
        p1=positions[top]
    else:
        p1=positions[logits.reshape(-1).argmax():logits.reshape(-1).argmax()+1]
    # Integral / wide average scoring proposes ROIs with high total evidence,
    # including multimodal/hypothesis disagreement that is not a sharp peak.
    # ROI width in semantic guide coordinates, computed from exact high->sem ratio.
    rh,_,_=_ratio_pad(high_meta); rs,_,_=_ratio_pad(sem_meta)
    span_sem = float(crop_size) / rh * rs
    kh=max(1,min(h,int(round(span_sem*float(h)/float(sem_hw[0])))))
    kw=max(1,min(w,int(round(span_sem*float(w)/float(sem_hw[1])))))
    # Odd kernel guarantees same h/w after padding.
    kh=kh if kh%2 else max(1,kh-1)
    kw=kw if kw%2 else max(1,kw-1)
    pooled=F.avg_pool2d(prob[None,None],(kh,kw),stride=1,padding=(kh//2,kw//2))[0,0]
    mass_idx=pooled.reshape(-1).topk(min(int(candidate_peaks)*3,h*w)).indices
    # Thin high-mass candidates with spatial greedy NMS using ~1/8 ROI side.
    mass_points=[]
    sep=float(crop_size)/float(max(high_hw))*0.12
    for idx in mass_idx:
        pp=positions[idx]
        if all(bool(torch.linalg.norm(pp-q)>sep) for q in mass_points):
            mass_points.append(pp)
        if len(mass_points)>=max(4,int(candidate_peaks)//2):
            break
    p2=torch.stack(mass_points) if mass_points else p1[:1]
    return torch.cat((anchor.reshape(1,2), p1, p2),dim=0).clamp(0,1),prob


@dataclass
class SelectionOptions:
    overlap_weight: float = 0.20
    peak_weight: float = 0.04
    switch_margin: float = 0.01
    min_novel_mass: float = 0.02
    candidate_peaks: int = 24


class DiverseTop2Selector:
    """Inference-only hard candidate selection; NEVER uses GT or training data."""
    def __init__(self, options: SelectionOptions, base_fn, roi_size: int):
        self.options=options
        self.base_fn=base_fn
        self.roi_size=int(roi_size)
        self.active_batch=None
        self.changed=0
        self.total=0
        self.novel_mass_sum=0.0
        self.legacy_novel_mass_sum=0.0
        self.peak_score_sum=0.0

    @torch.no_grad()
    def __call__(self, logits, primary, anchor, temperature, min_separation=0.1, *,
                 batch=None, legacy_secondary=None):
        # A direct model call supplies batch explicitly; older V1 ablation code
        # may still use active_batch. Both paths execute the same selection.
        if batch is None:
            batch=self.active_batch
        if batch is None:
            raise RuntimeError('Missing batch for Top-2 diverse selector')
        base=(legacy_secondary if legacy_secondary is not None else
              self.base_fn(logits,primary,anchor,temperature,min_separation))
        if not torch.isfinite(logits).all():
            return base
        B=primary.shape[0]
        assert B==logits.shape[0]
        hh,hw=batch['rgb_img'].shape[-2:]
        sh,sw=batch['rgb_semantic_img'].shape[-2:]
        result=[]
        for i in range(B):
            hi=batch['rgb_ratio_pad'][i]; sem=batch['rgb_semantic_ratio_pad'][i]
            prob_logits=logits[i,0].detach().float()
            candidates,prob = _rank_heatmap_candidates(
                prob_logits,primary[i],anchor[i],temperature,hi,sem,
                (hh,hw),(sh,sw),self.roi_size,self.options.candidate_peaks)
            # When the guide is nearly uniform there is no reliable target
            # evidence: preserve the trained model's original selection.
            if float(prob.max()) * prob.numel() < 1.5:
                result.append(base[i])
                self.total += 1
                continue
            # Candidate 0 is exactly legacy strategy. Always retain as fallback.
            candidates=torch.cat((base[i:i+1].detach().float(),candidates),0)
            boxes=roi_bounds_high(candidates,hi,sem,(hh,hw),(sh,sw),self.roi_size)
            first=roi_bounds_high(primary[i:i+1],hi,sem,(hh,hw),(sh,sw),self.roi_size)
            overlap=roi_iou(boxes,first,self.roi_size)
            mass=_heatmap_region_mass(prob,boxes,hi,sem,(hh,hw),(sh,sw))
            intersection_boxes=torch.cat((torch.maximum(boxes[:,:2],first[:,:2]),
                                          torch.minimum(boxes[:,2:],first[:,2:])),1)
            # zero-sized intersection must map to zero, even under box coordinate reversal.
            valid_inter=(intersection_boxes[:,2:] > intersection_boxes[:,:2]).all(-1)
            inter_mass=_heatmap_region_mass(prob,intersection_boxes,hi,sem,(hh,hw),(sh,sw))
            inter_mass=inter_mass*valid_inter.float()
            novel=(mass-inter_mass).clamp_min(0)
            H,W=prob.shape
            gridxy=torch.stack((candidates[:,0]*(W-1),candidates[:,1]*(H-1)),-1).round().long()
            gridxy[:,0]=gridxy[:,0].clamp(0,W-1)
            gridxy[:,1]=gridxy[:,1].clamp(0,H-1)
            # peak score uses nearby probability density (not raw temperature-scaled logits).
            local_prob=F.avg_pool2d(prob[None,None],5,stride=1,padding=2)[0,0]
            peaks=local_prob[gridxy[:,1],gridxy[:,0]]
            peak_norm=peaks/(prob.max()+1e-9)
            scores=(novel + 0.10*mass + self.options.peak_weight*peak_norm -
                    self.options.overlap_weight*overlap)
            if not bool(torch.isfinite(scores).all()):
                result.append(base[i]);continue
            best=int(scores.argmax().item())
            original_score=scores[0]
            if best!=0 and (float(scores[best]-original_score)<self.options.switch_margin or
                            float(novel[best])<self.options.min_novel_mass):
                best=0
            result.append(candidates[best])
            self.total+=1
            self.changed+=int(best!=0 and float(torch.linalg.norm(candidates[best]-base[i]))>1e-4)
            self.novel_mass_sum+=float(novel[best])
            self.legacy_novel_mass_sum+=float(novel[0])
            self.peak_score_sum+=float(scores[best])
        return torch.stack(result,0).to(device=base.device,dtype=base.dtype).clamp(0,1)


def exact_pair_iou(roi_geometry, k=2):
    """ROI pair overlap based on ACTUAL cropped xy0, rather than guessed centers."""
    if k!=2:
        raise ValueError('Top-2 required')
    xy0=roi_geometry.xy0.reshape(-1,k,2)
    first=xy0[:,0]
    second=xy0[:,1]
    inter=(float(roi_geometry.size)-(first-second).abs()).clamp_min(0).prod(-1)
    area=float(roi_geometry.size**2)
    return inter/(2*area-inter).clamp_min(1e-8)
