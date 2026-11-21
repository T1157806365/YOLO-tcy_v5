"""Differentiable RGB high-resolution ROI and geometry-consistent TIR sampling.

GTGA's q = S*p + t maps RGB-semantic normalized coordinates to TIR normalized
coordinates.  Separate letterbox transforms are respected in BOTH directions.
The 640x640 RGB ROI samples genuine pixels of the 1920 letterboxed image.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn.functional as F


@dataclass
class ROIGeometry:
    xy0: torch.Tensor                 # [B,2] high-image pixel crop origins (float)
    sem_xy: torch.Tensor              # [B,2] normalized RGB-sem crop center
    tir_valid: torch.Tensor           # [B] valid source image sampling fraction
    size: int


def _meta_arrays(metas: list[Mapping], reference: torch.Tensor):
    ratio, pad = [], []
    for m in metas:
        r = m['ratio']
        if isinstance(r, (tuple, list)):
            if len(r) != 2 or abs(float(r[0]) - float(r[1])) > 1e-5:
                raise ValueError('Expected isotropic letterbox scaling')
            r = r[0]
        r = float(r)
        if r <= 0:
            raise ValueError('Invalid letterbox ratio')
        ratio.append(r)
        pad.append([float(x) for x in m['pad']])
    return (reference.new_tensor(ratio).reshape(-1, 1),
            reference.new_tensor(pad))


def semantic_to_high_norm(center_norm, high_metas, sem_metas, high_hw, sem_hw):
    """Normalized semantic RGB center -> high-letterbox pixel center."""
    hs, ws = sem_hw
    rh, ph = _meta_arrays(high_metas, center_norm)
    rs, ps = _meta_arrays(sem_metas, center_norm)
    psxy = center_norm * center_norm.new_tensor([max(ws - 1, 1), max(hs - 1, 1)])
    return (psxy - ps) / rs * rh + ph


def sample_roi_pair(rgb_high: torch.Tensor, tir_full: torch.Tensor,
                    center_norm: torch.Tensor, scale: torch.Tensor,
                    trans: torch.Tensor, high_metas: list[Mapping],
                    sem_metas: list[Mapping], sem_hw: tuple[int, int],
                    size: int = 640, geometry_grad_scale: float = 0.1,
                    center_grad_scale: float = 0.1):
    """Vectorized differentiable sampling with grid_sample(align_corners=True).

    RGB high B,3,H,W and TIR B,3,h,w are normalized to [0,1].
    Predicted centers/geometry may receive controlled detection gradients.
    """
    if rgb_high.ndim != 4 or tir_full.ndim != 4:
        raise ValueError('Image tensors must be B,C,H,W')
    b, _, h, w = rgb_high.shape
    if tir_full.shape[0] != b or center_norm.shape != (b, 2):
        raise ValueError('ROI batch sizes do not match')
    if int(size) > min(h, w) or int(size) < 16:
        raise ValueError(f'Invalid ROI size={size} for high image {(h,w)}')
    if not (0.0 <= center_grad_scale <= 1.0 and 0.0 <= geometry_grad_scale <= 1.0):
        raise ValueError('Gradient scales must be in [0,1]')
    size = int(size)
    def grad_scale(z, ratio):
        return z.detach() + (z - z.detach()) * float(ratio)
    c = grad_scale(center_norm.float(), center_grad_scale)
    ss = grad_scale(scale.float().reshape(b, 2), geometry_grad_scale)
    tt = grad_scale(trans.float().reshape(b, 2), geometry_grad_scale)
    center_h = semantic_to_high_norm(c, high_metas, sem_metas, (h,w), sem_hw)
    xy0 = (center_h - size / 2.0).clamp(min=0)
    xy0 = torch.minimum(xy0, xy0.new_tensor([w-size, h-size]))
    # Pixel-center coordinates. No integer rounding: maintains ROI-center gradients.
    row = torch.arange(size, device=rgb_high.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(row, row, indexing='ij')
    xy_high = torch.stack((xx, yy), dim=-1)[None] + xy0[:, None, None, :]
    rgb_grid = 2.0 * xy_high / xy_high.new_tensor([max(w-1,1),max(h-1,1)]) - 1.0
    rgb_roi = F.grid_sample(rgb_high.float(), rgb_grid, mode='bilinear',
                            align_corners=True, padding_mode='border')
    rh, ph = _meta_arrays(high_metas, center_norm)
    rs, ps = _meta_arrays(sem_metas, center_norm)
    rh = rh[:,None]  # B,1,1,1
    rs = rs[:,None]
    ph = ph[:,None,None,:]
    ps = ps[:,None,None,:]
    xy_sem = (xy_high - ph) / rh.unsqueeze(-1) * rs.unsqueeze(-1) + ps
    hs, ws = sem_hw
    p = xy_sem / xy_sem.new_tensor([max(ws-1,1),max(hs-1,1)])
    q = p * ss[:, None, None, :] + tt[:, None, None, :]
    valid = ((q >= 0) & (q <= 1)).all(-1).float().mean(dim=(1,2)).detach()
    tir_roi = F.grid_sample(tir_full.float(), 2.0*q-1.0,
                            mode='bilinear', align_corners=True, padding_mode='border')
    return rgb_roi.to(dtype=rgb_high.dtype), tir_roi.to(dtype=tir_full.dtype), ROIGeometry(
        xy0=xy0, sem_xy=center_norm, tir_valid=valid, size=size)


def local_labels_from_high(batch: dict, roi: ROIGeometry, min_visible: float = 0.6):
    """Convert existing RGB high labels to ROI normalized xywh (no label leakage into crop)."""
    gt = batch['rgb_bboxes']
    idx = batch['rgb_batch_idx'].long()
    b, _, h, w = batch['rgb_img'].shape
    kept_boxes, kept_cls, kept_indices = [], [], []
    n_positive = 0
    for i in range(b):
        select = idx == i
        box = gt[select]
        if box.numel() == 0:
            continue
        absolute = box * box.new_tensor([w,h,w,h])
        lo = absolute[:,:2] - absolute[:,2:]/2.0
        hi = absolute[:,:2] + absolute[:,2:]/2.0
        xy0 = roi.xy0[i].detach()
        clip_lo = (lo-xy0).clamp(0,roi.size)
        clip_hi = (hi-xy0).clamp(0,roi.size)
        wh = (clip_hi-clip_lo).clamp_min(0)
        visible = wh.prod(-1)/(absolute[:,2:].prod(-1).clamp_min(1e-7))
        keep = (visible >= float(min_visible)) & (wh[:,0]>=1) & (wh[:,1]>=1)
        if bool(keep.any()):
            n_positive += 1
            loc = torch.cat(((clip_lo[keep]+clip_hi[keep])*0.5, wh[keep]), dim=-1) / roi.size
            kept_boxes.append(loc.clamp(0,1))
            kept_cls.append(batch['rgb_cls'][select][keep])
            kept_indices.append(torch.full((len(loc),),i,dtype=torch.long,device=box.device))
    return {
        'rgb_img': None, 'tir_img': None,
        'rgb_cls': torch.cat(kept_cls) if kept_cls else gt.new_zeros((0,1)),
        'rgb_bboxes': torch.cat(kept_boxes) if kept_boxes else gt.new_zeros((0,4)),
        'rgb_batch_idx': torch.cat(kept_indices) if kept_indices else idx.new_zeros((0,)),
    }, n_positive


def boxes_roi_to_semantic(boxes: torch.Tensor, roi: ROIGeometry,
                          batch: dict, sample_idx: int):
    """ROI xyxy absolute pixels -> RGB-semantic xyxy absolute pixels for AP."""
    if boxes.numel() == 0:
        return boxes.reshape(0,4)
    high_metas = batch['rgb_ratio_pad']
    sem_metas = batch['rgb_semantic_ratio_pad']
    rh, ph = _meta_arrays(high_metas, boxes)
    rs, ps = _meta_arrays(sem_metas, boxes)
    p_high = boxes.reshape(-1,2,2) + roi.xy0[sample_idx].reshape(1,1,2)
    p_sem = (p_high - ph[sample_idx].view(1,1,2))/rh[sample_idx] * rs[sample_idx] + ps[sample_idx].view(1,1,2)
    hs, ws = batch['rgb_semantic_img'].shape[-2:]
    p_sem[...,0] = p_sem[...,0].clamp(0,ws)
    p_sem[...,1] = p_sem[...,1].clamp(0,hs)
    return p_sem.reshape(-1,4)


def roi_gt_coverage(batch: dict, roi: ROIGeometry, min_visible=0.6):
    """Post-hoc GT-only validation diagnostic. Never used for choosing an ROI."""
    _, _, h, w = batch['rgb_img'].shape
    labels=batch['rgb_bboxes']; idx=batch['rgb_batch_idx'].long()
    hits=0; total=0
    for i in range(len(roi.xy0)):
        bb=labels[idx==i]
        if len(bb)==0: continue
        a=bb*bb.new_tensor([w,h,w,h]); lo=a[:,:2]-a[:,2:]/2; hi=a[:,:2]+a[:,2:]/2
        origin=roi.xy0[i].detach(); cliplo=torch.maximum(lo,origin); cliphi=torch.minimum(hi,origin+roi.size)
        visible=(cliphi-cliplo).clamp_min(0).prod(-1)/(a[:,2:].prod(-1).clamp_min(1e-7))
        hits += int((visible>=min_visible).sum().item());total+=len(a)
    return hits,total
