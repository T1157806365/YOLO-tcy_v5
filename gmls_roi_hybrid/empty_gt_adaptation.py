"""Explicit per-image treatment of asymmetric RGB/TIR YOLO labels.

When RGB has no GT but TIR has a target, RGB detection GT is *unknown*.
Such images are excluded from RGB-coordinate detection supervision.
When only RGB has GT, RGB detection is supervised as usual, and TIR
heatmap supervision is masked. Both-empty is retained as true background,
subject to the dataset's annotation policy.

No pseudo labels, no coordinates transferred between modalities.
"""
from __future__ import annotations
import torch


def label_presence(batch: dict, batch_size: int):
    """Return per-image bool tensors (rgb_has_gt, tir_has_gt)."""
    rgb = batch['rgb_batch_idx'].long().flatten()
    tir = batch['tir_batch_idx'].long().flatten()
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if (rgb.numel() and (rgb.min()<0 or rgb.max()>=batch_size)) or (tir.numel() and (tir.min()<0 or tir.max()>=batch_size)):
        raise ValueError('batch indices out of range')
    rgb_has = torch.bincount(rgb, minlength=batch_size) > 0
    tir_has = torch.bincount(tir, minlength=batch_size) > 0
    return rgb_has, tir_has


def rgb_supervision_mask(batch: dict, batch_size: int):
    """Ignore RGB-absent / TIR-present samples; keep both-empty as negatives."""
    rgb_has, tir_has = label_presence(batch, batch_size)
    return rgb_has | (~tir_has)


def select_rgb_local_labels(batch_labels: dict, rgb_roi: torch.Tensor,
                            tir_roi: torch.Tensor, keep_images: torch.Tensor,
                            rois_per_image: int):
    """Reindex RGB labels after filtering B*K crops; preserve gradient of crops."""
    b = int(keep_images.numel())
    k = int(rois_per_image)
    if b <= 0 or k <= 0 or rgb_roi.shape[0] != b*k or tir_roi.shape[0] != b*k:
        raise ValueError('Invalid crop shapes or K')
    crop_keep = keep_images.repeat_interleave(k)
    indices = crop_keep.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return None
    old_ids = batch_labels['rgb_batch_idx'].long().flatten()
    mapping = torch.full((b*k,), -1, dtype=torch.long, device=indices.device)
    mapping[indices] = torch.arange(indices.numel(), device=indices.device)
    gt_keep = crop_keep[old_ids]
    return {
        'rgb_img': rgb_roi[indices], 'tir_img': tir_roi[indices],
        'rgb_cls': batch_labels['rgb_cls'][gt_keep],
        'rgb_bboxes': batch_labels['rgb_bboxes'][gt_keep],
        'rgb_batch_idx': mapping[old_ids[gt_keep]],
    }


def select_global_training_data(ex: dict, batch: dict, keep_images: torch.Tensor):
    """Reindex RGB GT and subset global RGB/TIR backbone features."""
    b = int(keep_images.numel())
    indices = keep_images.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return None, None
    idx = batch['rgb_batch_idx'].long().flatten()
    gt_keep = keep_images[idx]
    remap = torch.full((b,), -1, device=indices.device, dtype=torch.long)
    remap[indices] = torch.arange(indices.numel(), device=indices.device)
    subset = dict(ex)
    subset['global_rgb_features'] = [x[indices] for x in ex['global_rgb_features']]
    subset['global_tir_aligned'] = [x[indices] for x in ex['global_tir_aligned']]
    target = {
        'img': batch['rgb_semantic_img'][indices],
        'cls': batch['rgb_cls'][gt_keep],
        'bboxes': batch['rgb_semantic_bboxes'][gt_keep],
        'batch_idx': remap[idx[gt_keep]],
    }
    return subset, target
