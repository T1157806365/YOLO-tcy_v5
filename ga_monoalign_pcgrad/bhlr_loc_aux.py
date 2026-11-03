"""Training-only P3 detail localization supervision for YOLO-tcy_v5.

The 5-channel prediction tensor has one center logit, two fractional center
coordinates, and two log(w,h) values in P3-cell units. This is NOT an
inference head: it is called only by AlignmentProtectedStudent.loss().

GT uses batch['rgb_semantic_bboxes'] (normalized xywh, RGB coordinate system),
paired with batch['rgb_batch_idx']. It never uses TIR bbox coordinates.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BHLRP3LocalizationHead(nn.Module):
    """Tiny, BN-free localization head on BHLR `usable_detail` at P3 size."""

    def __init__(self, in_channels: int = 48, hidden_channels: int = 24):
        super().__init__()
        c = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), c, kernel_size=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(c, 5, kernel_size=1, bias=True),
        )
        # Initially predict a low probability of a center. This prevents
        # 80x80 background heatmaps from dominating the first iterations.
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = -3.0

    def forward(self, usable_detail: Tensor) -> Tensor:
        if usable_detail.ndim != 4:
            raise ValueError(f"Expected BCHW P3 detail, got {tuple(usable_detail.shape)}")
        return self.net(usable_detail)


def _make_targets(
    pred: Tensor,
    bboxes: Tensor,
    batch_idx: Tensor,
    gaussian_sigma: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    """Targets for normalized RGB semantic xywh GT; no gradient through targets.

    Multiple objects in the same P3 cell cannot be regressed by a one-slot
    center head. Keep the smaller-area object deterministically at that cell.
    GT heatmap still includes all valid centers.
    """
    bsz, _, h, w = pred.shape
    device = pred.device
    heatmap = pred.new_zeros((bsz, 1, h, w), dtype=torch.float32)
    positive = torch.zeros((bsz, h, w), device=device, dtype=torch.bool)
    target_xy = pred.new_zeros((bsz, 2, h, w), dtype=torch.float32)
    target_logwh = pred.new_zeros((bsz, 2, h, w), dtype=torch.float32)
    valid_box_count = 0

    boxes = bboxes.detach().float().reshape(-1, 4)
    indices = batch_idx.detach().long().reshape(-1)
    if boxes.shape[0] != indices.numel():
        raise RuntimeError(
            f"GT bboxes and batch_idx size mismatch: {boxes.shape[0]} vs {indices.numel()}"
        )

    chosen = {}  # (batch, iy, ix): (area, offset_x, offset_y, logw, logh)
    radius = max(1, int(math.ceil(3.0 * float(gaussian_sigma))))
    # One host transfer per batch avoids one GPU synchronization per label.
    rows = boxes.cpu().tolist()
    ids = indices.cpu().tolist()
    with torch.no_grad():
        for row, batch_id in zip(rows, ids):
            b = int(batch_id)
            if b < 0 or b >= bsz:
                raise RuntimeError(f"GT batch_idx={b} outside batch size {bsz}")
            cx, cy, bw, bh = (float(x) for x in row)
            if not all(math.isfinite(x) for x in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                continue
            cx = min(max(cx, 0.0), 1.0 - 1e-7)
            cy = min(max(cy, 0.0), 1.0 - 1e-7)
            gw = max(float(bw) * w, 1e-4)
            gh = max(float(bh) * h, 1e-4)
            u = cx * w
            v = cy * h
            ix = min(w - 1, int(math.floor(u)))
            iy = min(h - 1, int(math.floor(v)))
            valid_box_count += 1

            # Small gaussian radius for the center-target heatmap.
            ya, yb = max(0, iy - radius), min(h, iy + radius + 1)
            xa, xb = max(0, ix - radius), min(w, ix + radius + 1)
            ys = torch.arange(ya, yb, device=device, dtype=torch.float32)
            xs = torch.arange(xa, xb, device=device, dtype=torch.float32)
            square = (ys[:, None] - iy).square() + (xs[None, :] - ix).square()
            gaussian = torch.exp(-square / (2.0 * gaussian_sigma ** 2))
            patch = heatmap[b, 0, ya:yb, xa:xb]
            torch.maximum(patch, gaussian, out=patch)

            key = (b, iy, ix)
            area = gw * gh
            if key not in chosen or area < chosen[key][0]:
                chosen[key] = (area, u - ix, v - iy, math.log(gw), math.log(gh))

        for (b, iy, ix), (_, dx, dy, lw, lh) in chosen.items():
            positive[b, iy, ix] = True
            target_xy[b, :, iy, ix] = torch.tensor([dx, dy], device=device)
            target_logwh[b, :, iy, ix] = torch.tensor([lw, lh], device=device)

    return heatmap, positive, target_xy, target_logwh, indices, valid_box_count


def localization_loss(
    pred: Tensor,
    bboxes: Tensor,
    batch_idx: Tensor,
    *,
    heatmap_weight: float = 1.0,
    offset_weight: float = 1.0,
    size_weight: float = 0.25,
    gaussian_sigma: float = 1.5,
    negative_weight: float = 0.25,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Returns a *per-image* loss and detached/inspectable components.

    Reduction differs deliberately from naive CenterNet sum: the background
    BCE is averaged over all negative P3 cells, preventing the huge number
    of negatives from overwhelming the standard YOLO losses.
    """
    if pred.shape[1] != 5:
        raise ValueError(f"Localization head must output 5 channels, got {pred.shape[1]}")
    if gaussian_sigma <= 0 or negative_weight < 0:
        raise ValueError('gaussian_sigma must be > 0 and negative_weight >= 0')
    heatmap, positive, target_xy, target_logwh, _, nboxes = _make_targets(
        pred, bboxes, batch_idx, gaussian_sigma
    )
    # fp32 for numeric stability under CUDA AMP.
    p = pred.float()
    logits = p[:, 0:1]
    pos4 = positive.unsqueeze(1)

    if bool(positive.any()):
        pos_heat = F.binary_cross_entropy_with_logits(
            logits[pos4], torch.ones_like(logits[pos4]), reduction='mean'
        )
        offset = F.smooth_l1_loss(
            p[:, 1:3].sigmoid().permute(0, 2, 3, 1)[positive],
            target_xy.permute(0, 2, 3, 1)[positive], reduction='mean'
        )
        size = F.smooth_l1_loss(
            p[:, 3:5].permute(0, 2, 3, 1)[positive],
            target_logwh.permute(0, 2, 3, 1)[positive], reduction='mean'
        )
    else:
        zero = p.sum() * 0.0  # Preserve train-time DDP graph on empty GT batches.
        pos_heat, offset, size = zero, zero, zero

    # Gaussian neighbors are treated as 'ignore-ish' negative locations.
    neg_mask = (~pos4).to(logits.dtype)
    neg = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction='none'
    ) * neg_mask * (1.0 - heatmap).pow(4)
    neg_loss = neg.sum() / neg_mask.sum().clamp_min(1.0)
    heat = pos_heat + float(negative_weight) * neg_loss
    total = (float(heatmap_weight) * heat
             + float(offset_weight) * offset
             + float(size_weight) * size)
    stats = {
        'heatmap': heat.detach(),
        'center_offset': offset.detach(),
        'log_wh': size.detach(),
        'valid_boxes': p.new_tensor(float(nboxes)),
        'positive_cells': p.new_tensor(float(positive.sum().item())),
    }
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError('Nonfinite BHLR P3 localization loss')
    return total, stats
