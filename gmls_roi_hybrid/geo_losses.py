"""Training-only, GT-supervised FINAL alignment center loss for GMLS-v2 Hybrid.

The GTGA final inverse sampling transform is q=S*p+t (p is RGB, q TIR).
No ground truth is ever used for ROI selection or inference.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def final_center_loss(batch, alignment_info, pixel_norm: float = 8.0,
                      beta: float = 0.5):
    """Return (scalar loss, valid_pair_count, pixel_error_mean).

    Only one-to-one RGB/TIR GT instances are used. Both boxes are normalized
    to the corresponding modality's 640 semantic canvas after letterbox.
    Center loss is evaluated on the *actual final* sampling_scale and
    sampling_translation; this ensures that the full composed warp is trained.

    The calculation is FP32 even under AMP; invalid pairs contribute zero.
    """
    if pixel_norm <= 0.0 or beta <= 0.0:
        raise ValueError('pixel_norm and beta must be > 0')
    scale = alignment_info['sampling_scale'].float().reshape(-1, 2).clamp_min(1e-6)
    trans = alignment_info['sampling_translation'].float().reshape(-1, 2)
    rgb = batch['rgb_semantic_bboxes'].float()
    tir = batch['tir_bboxes'].float()
    rgb_idx = batch['rgb_batch_idx'].long().flatten()
    tir_idx = batch['tir_batch_idx'].long().flatten()
    sem = batch['rgb_semantic_img']
    h, w = int(sem.shape[-2]), int(sem.shape[-1])
    pixel_scale = scale.new_tensor([float(w), float(h)])
    losses = []
    errors = []
    for i in range(scale.shape[0]):
        r = rgb[rgb_idx == i]
        t = tir[tir_idx == i]
        if len(r) != 1 or len(t) != 1:
            continue
        # q = S*p + t => RGB p = (q-t)/S.
        center_pred = (t[0, :2] - trans[i]) / scale[i]
        delta_px = (center_pred - r[0, :2]) * pixel_scale
        # Normalize the pixel errors to keep the loss well scaled across
        # heterogeneous images; do NOT detach S, t, or center_pred.
        loss_i = F.smooth_l1_loss(
            delta_px / float(pixel_norm), torch.zeros_like(delta_px),
            beta=float(beta), reduction='mean')
        losses.append(loss_i)
        errors.append(torch.linalg.vector_norm(delta_px.detach(), ord=2))
    if losses:
        return torch.stack(losses).mean(), len(losses), torch.stack(errors).mean()
    # Graph-safe connected zero, not sqrt(0), avoids NaN in empty-GT batches.
    zero = (scale.sum() + trans.sum()) * 0.0
    return zero, 0, zero.detach()


def _iou_giou_xywh(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Differentiable IoU/GIoU for paired normalized XYWH boxes, [N, 4]."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 4:
        raise ValueError(f"Expected equal [N,4] xywh boxes; got {pred.shape}, {target.shape}")
    pwh = pred[:, 2:].clamp_min(eps)
    twh = target[:, 2:].clamp_min(eps)
    plo, phi = pred[:, :2] - pwh * 0.5, pred[:, :2] + pwh * 0.5
    tlo, thi = target[:, :2] - twh * 0.5, target[:, :2] + twh * 0.5
    il = torch.maximum(plo, tlo)
    ih = torch.minimum(phi, thi)
    intersection = (ih - il).clamp_min(0).prod(dim=-1)
    area_p, area_t = pwh.prod(dim=-1), twh.prod(dim=-1)
    union = (area_p + area_t - intersection).clamp_min(eps)
    iou = intersection / union
    cl = torch.minimum(plo, tlo)
    ch = torch.maximum(phi, thi)
    enclosing = (ch - cl).clamp_min(eps).prod(dim=-1)
    giou = iou - (enclosing - union) / enclosing.clamp_min(eps)
    return iou, giou


def final_box_geometry_loss(batch: dict, alignment_info: dict, *,
                            size_beta: float = 0.2):
    """Return size, GIoU, pairs, mean IoU, and center error in RGB semantic px.

    Only images with exactly one valid RGB and TIR box are supervised, matching
    the existing final_center_loss convention. Filtering uses no model output.
    Every tensor involved in the predicted boxes retains differentiability.
    """
    if size_beta <= 0:
        raise ValueError('size_beta must be > 0')
    scale = alignment_info['sampling_scale'].float().reshape(-1, 2).clamp_min(1e-6)
    trans = alignment_info['sampling_translation'].float().reshape(-1, 2)
    rgb, tir = batch['rgb_semantic_bboxes'].float(), batch['tir_bboxes'].float()
    ri = batch['rgb_batch_idx'].long().flatten()
    ti = batch['tir_batch_idx'].long().flatten()
    h, w = batch['rgb_semantic_img'].shape[-2:]
    pixel_scale = scale.new_tensor([float(w), float(h)])
    pred_list, gt_list, center_errors = [], [], []
    for n in range(scale.shape[0]):
        r = rgb[ri == n]
        t = tir[ti == n]
        if r.shape[0] != 1 or t.shape[0] != 1:
            continue
        if not (bool(torch.isfinite(r).all().item()) and
                bool(torch.isfinite(t).all().item()) and
                bool((r[0, 2:] > 0).all().item()) and
                bool((t[0, 2:] > 0).all().item())):
            continue
        # GTGA source-coordinate map q = S*p + t. Inverse maps TIR box to RGB.
        pred_center = (t[0, :2] - trans[n]) / scale[n]
        pred_size = t[0, 2:] / scale[n]
        pred = torch.cat((pred_center, pred_size), dim=0)
        pred_list.append(pred)
        gt_list.append(r[0])
        center_errors.append(((pred_center - r[0, :2]) * pixel_scale).norm().detach())

    if not pred_list:
        # Connected zero so even zero-GT minibatches have a well-defined loss.
        zero = (scale.sum() + trans.sum()) * 0.0
        return zero, zero, 0, zero.detach(), zero.detach()

    pred = torch.stack(pred_list, dim=0)
    target = torch.stack(gt_list, dim=0)
    # Scale error in relative/logarithmic units; appropriate for tiny UAV boxes.
    log_ratio = torch.log(pred[:, 2:].clamp_min(1e-7) / target[:, 2:].clamp_min(1e-7))
    size_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio),
                                 beta=size_beta, reduction='mean')
    iou, giou = _iou_giou_xywh(pred, target)
    giou_loss = (1.0 - giou).mean()
    center_px = torch.stack(center_errors).mean()
    return size_loss, giou_loss, len(pred_list), iou.detach().mean(), center_px
