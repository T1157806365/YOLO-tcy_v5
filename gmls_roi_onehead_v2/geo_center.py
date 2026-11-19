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
