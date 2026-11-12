"""YOLO-tcy_v5 adaptation of M²D-style mono- and cross-modality distillation.

M²D's intra-/cross-modal supervision idea is borrowed from ICCV 2025,
Zhao et al., "Rethinking Multi-modal Object Detection from the Perspective of
Mono-Modality Feature Learning". This implementation is ORIGINAL adaptation:
- P3/P4/P5 teacher feature response matching in each native modality;
- teacher spatial attention transported with GTGA RGB->TIR sampling geometry;
- optional low-weight fused P3/P4/P5 target-saliency supervision;
- target ROI masks and teacher GT correctness weights;
- NO LIF, NO teacher parameter copying and NO HR teacher.

The teacher feature maps are detached. Student feature gradients are preserved.
GTGA geometry is detached in cross-modal losses so detection/KD does not
unintentionally backpropagate through the coordinate grid (align loss still
trains GTGA). HR detail is supervised by detector loss via original BHLR.
"""
from __future__ import annotations

from typing import Dict
import torch
from torch.nn import functional as F


def attention_map(feature: torch.Tensor):
    """Differentiable, bounded single-channel spatial teacher/student response."""
    e = feature.float().square().mean(dim=1, keepdim=True)
    mean = e.mean(dim=(-2, -1), keepdim=True)
    std = e.var(dim=(-2, -1), keepdim=True, unbiased=False).add(1e-5).sqrt()
    return torch.sigmoid(((e - mean) / std).clamp(-8, 8))


def roi_mask(boxes, batch_idx, bsz, h, w, device, *, pad=3):
    """Normalized xywh GT -> feature ROI mask; no gradients to GT."""
    mask = torch.zeros(bsz, 1, h, w, device=device, dtype=torch.float32)
    for box, idx in zip(boxes.detach().float().cpu(), batch_idx.detach().long().cpu()):
        b = int(idx)
        if b < 0 or b >= bsz:
            continue
        cx, cy, bw, bh = [float(q) for q in box]
        x0 = max(0, int((cx - bw / 2) * w) - pad)
        x1 = min(w, int((cx + bw / 2) * w) + pad + 1)
        y0 = max(0, int((cy - bh / 2) * h) - pad)
        y1 = min(h, int((cy + bh / 2) * h) + pad + 1)
        if x0 < x1 and y0 < y1:
            mask[b, 0, y0:y1, x0:x1] = 1.0
    return mask


def resize_like(x, reference):
    return x if x.shape[-2:] == reference.shape[-2:] else F.interpolate(
        x, size=reference.shape[-2:], mode='bilinear', align_corners=False)


def _norm_feature(x):
    x = x.float()
    mean = x.mean(dim=(-2,-1),keepdim=True)
    scale = x.var(dim=(-2,-1),keepdim=True,unbiased=False).add(1e-4).sqrt()
    return ((x - mean) / scale).clamp(-6., 6.)


def masked_loss(a, b, mask, *, sample_good=None, feature=False):
    """Per-sample ROI mean avoids tiny masks being diluted by huge backgrounds."""
    if a.shape != b.shape:
        raise ValueError(f'M2D loss shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}')
    if feature:
        delta = F.smooth_l1_loss(_norm_feature(a), _norm_feature(b), reduction='none')
    else:
        delta = (a.float() - b.float()).square()
    delta = delta.mean(dim=1, keepdim=True)
    if mask.shape[-2:] != delta.shape[-2:]:
        mask = resize_like(mask, delta)
    num = (delta * mask).flatten(1).sum(dim=1)
    den = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    out = num / den
    if sample_good is not None:
        # Wrong teacher detections are NOT a hard pseudo-label. Weakly distill
        # them for generic feature statistics, but downweight them strongly.
        sample_w = torch.where(sample_good.bool(), out.new_ones(out.shape),
                               out.new_full(out.shape, 0.1))
        return (sample_w * out).mean()
    return out.mean()


def gtga_warp_teacher(tir_teacher_attention, rgb_reference, alignment_info):
    """Sample a native TIR map onto RGB coordinates using GTGA's affine map.

    geometry_iou in original project verifies:
      RGB bbox = (TIR bbox - sampling_translation)/sampling_scale
    Thus TIR normalized grid = RGB normalized grid * scale + translation.
    This is the RGB->TIR *sampling* transform, NOT its inverse.
    """
    if not isinstance(alignment_info, dict) or not alignment_info.get('enabled', False):
        raise RuntimeError('Cross-modal M2D requires enabled RLSFA-GTGA')
    scale = alignment_info['sampling_scale'].detach().float()
    shift = alignment_info['sampling_translation'].detach().float()
    bsz, _, height, width = rgb_reference.shape
    scale = scale.reshape(bsz, -1)[:, :2]
    shift = shift.reshape(bsz, -1)[:, :2]
    if scale.shape != (bsz,2) or shift.shape != (bsz,2):
        raise RuntimeError('Unexpected GTGA sampling_scale / sampling_translation shape')
    # EXACT GTGA grid convention: linspace(0,1) + align_corners=True.
    # Do not use pixel-center arange()/H here: that subtly shifts small boxes.
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, device=rgb_reference.device, dtype=torch.float32),
        torch.linspace(0, 1, width, device=rgb_reference.device, dtype=torch.float32),
        indexing='ij')
    rx = xx[None].expand(bsz, -1, -1)
    ry = yy[None].expand(bsz, -1, -1)
    tx = scale[:,0,None,None] * rx + shift[:,0,None,None]
    ty = scale[:,1,None,None] * ry + shift[:,1,None,None]
    grid = torch.stack((tx*2.-1., ty*2.-1.), dim=-1)
    native_tir = tir_teacher_attention.detach().float()
    if native_tir.shape[0] != bsz:
        raise ValueError('TIR teacher / RGB student batch mismatch')
    # Grid values are unit normalized. Validity mask rejects extrapolation.
    validity = ((tx >= 0.) & (tx <= 1.) & (ty >= 0.) & (ty <= 1.)).unsqueeze(1).float()
    aligned = F.grid_sample(native_tir, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return aligned, validity


def calculate(features: dict, teachers: dict, batch: dict, cfg: dict, scale_to_index: dict):
    """Return differentiable total + explicit monitor metrics.

    Teacher feature selection uses P3/P4/P5 SAME backbone stages as student.
    Only students are updated. Losses are per-image and dimensionless.
    """
    if not cfg.get('enabled', False):
        raise RuntimeError('M2D calculate called while disabled')
    dev = batch['rgb_semantic_img'].device
    zero = features['rgb_backbone_raw'][scale_to_index[3]].float().sum() * 0.0
    im_rgb = zero; im_tir = zero; cm_rgb = zero; cm_tir = zero; fused = zero
    ratios = []
    n_scales = 0
    valid_rgb = teachers['rgb']['good'].float().mean().detach()
    valid_tir = teachers['tir']['good'].float().mean().detach()
    for scale in (3,4,5):
        idx = scale_to_index[scale]
        sr = features['rgb_backbone_raw'][idx]
        st = features['tir_backbone'][idx]
        sa = features['aligned_tir_backbone'][idx]
        sf = features['fused_backbone'][idx]
        tr = resize_like(teachers['rgb']['features'][scale].detach().to(dev), sr)
        tt = resize_like(teachers['tir']['features'][scale].detach().to(dev), st)
        rm = roi_mask(batch['rgb_semantic_bboxes'],batch['rgb_batch_idx'],
                      sr.shape[0],sr.shape[-2],sr.shape[-1],dev)
        tm = roi_mask(batch['tir_bboxes'],batch['tir_batch_idx'],
                      st.shape[0],st.shape[-2],st.shape[-1],dev)
        im_rgb = im_rgb + masked_loss(sr,tr,rm,feature=True,
                                      sample_good=teachers['rgb']['good'])
        im_tir = im_tir + masked_loss(st,tt,tm,feature=True,
                                      sample_good=teachers['tir']['good'])
        # M2D_BASELINE_INTRA_ONLY_V1: without GTGA, distill native RGB/TIR only.
        # Skip cross-modal warp and fused-teacher supervision entirely.
        if bool(cfg.get('intra_only', False)):
            n_scales += 1
            continue
        att_r = attention_map(tr).detach()
        att_t = attention_map(tt).detach()
        att_aligned, valid = gtga_warp_teacher(att_t, sr, features['alignment_info'])
        # Only reliable locations inside RGB object ROI. GTGA support is a SOFT
        # confidence modulation, but never a mandatory nonzero mask; otherwise
        # early GTGA all-zero masks could shut off all cross KD gradients.
        conf_maps = features['alignment_info'].get('support_by_scale', {})
        joint = conf_maps.get(scale, conf_maps.get(str(scale)))
        if joint is None:
            joint = torch.zeros_like(rm)
        joint = resize_like(joint.detach().float(), rm).clamp(0,1)
        cross_mask = rm * valid * (0.25 + 0.75 * joint)
        ratios.append((cross_mask > 0).float().mean().detach())
        # Each cross term distills target PRESENCE, not cross-modality raw
        # channel features, which would force incompatible feature spaces.
        cm_rgb = cm_rgb + masked_loss(attention_map(sr),att_aligned,cross_mask,
                                     sample_good=teachers['tir']['good'])
        cm_tir = cm_tir + masked_loss(attention_map(sa),att_r,cross_mask,
                                     sample_good=teachers['rgb']['good'])
        # Fused branch receives a low-weight teacher ORACLE response: a target
        # visible in either teacher should remain salient after multimodal fusion.
        desired = torch.maximum(att_r, att_aligned)
        fused = fused + masked_loss(attention_map(sf),desired,rm,
                                    sample_good=(teachers['rgb']['good'] |
                                                 teachers['tir']['good']))
        n_scales += 1
    inv = 1.0 / max(1, n_scales)
    im_rgb *= inv; im_tir *= inv; cm_rgb *= inv; cm_tir *= inv; fused *= inv
    cm_factor = float(cfg.get('lambda_cross', 0.5))
    fusion_factor = float(cfg.get('lambda_fused', 0.15))
    total = 0.5*(im_rgb + im_tir) + cm_factor*0.5*(cm_rgb+cm_tir) + fusion_factor*fused
    return {
        'total': total,
        'im_rgb': im_rgb, 'im_tir': im_tir,
        'cm_rgb': cm_rgb, 'cm_tir': cm_tir,
        'fused': fused,
        'rgb_teacher_correct': valid_rgb,
        'tir_teacher_correct': valid_tir,
        'align_valid_ratio': torch.stack(ratios).mean() if ratios else zero.detach(),
    }
