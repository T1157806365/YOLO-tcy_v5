"""AG-HRRefine: RGB-T alignment-conditioned, sparse 1920-RGB box refinement.

All proposals and targets are in RGB semantic-image (e.g. 640x640) xyxy pixels.
This branch NEVER alters class/confidence, detector head, alignment warp, or BHLR.
The last regression layer starts at zero: old boxes are initially unchanged.
"""
from __future__ import annotations

import math
from typing import Dict, List
import torch
from torch import nn
from torch.nn import functional as F


def _grad_scale(x: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Keep forward input unchanged while scaling its backward gradient."""
    k = float(coefficient)
    return x.detach() + (x - x.detach()) * k


def _box_iou_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    wa = (a[..., 2:] - a[..., :2]).clamp(min=0)
    wb = (b[..., 2:] - b[..., :2]).clamp(min=0)
    area_a, area_b = wa[..., 0] * wa[..., 1], wb[..., 0] * wb[..., 1]
    return intersection / (area_a + area_b - intersection).clamp_min(1e-6)


def _cxcywh_to_xyxy(z: torch.Tensor) -> torch.Tensor:
    center = z[..., :2]
    half = z[..., 2:].clamp_min(1e-4) * 0.5
    return torch.cat((center - half, center + half), dim=-1)


class AGHRRefine(nn.Module):
    """Small ROI-local refiner: [1920 RGB - 640 RGB] + aligned TIR P3 + alignment support.

    The high resolution image is never processed by a full-frame CNN: sample
    candidate ROIs with grid_sample *before* convolution to bound memory.
    """

    def __init__(self, tir_channels: int, roi_size: int = 9, roi_expand: float = 2.0,
                 hidden: int = 32, feedback_scale: float = 0.10,
                 max_center_shift: float = 0.50, max_log_scale: float = math.log(1.5),
                 chunk_size: int = 32):
        super().__init__()
        self.roi_size = int(roi_size)
        self.roi_expand = float(roi_expand)
        self.feedback_scale = float(feedback_scale)
        self.max_center_shift = float(max_center_shift)
        self.max_log_scale = float(max_log_scale)
        self.chunk_size = int(chunk_size)
        if self.roi_size < 3 or self.chunk_size < 1:
            raise ValueError('Invalid AGHR ROI size/chunk size')
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tir_encoder = nn.Sequential(
            nn.Conv2d(int(tir_channels) + 1, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regression = nn.Sequential(
            nn.Linear(hidden * 2 + 5, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.regression[-1].weight)
        nn.init.zeros_(self.regression[-1].bias)

    def _roi_grid(self, boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Input boxes: [B,N,4] in semantic-image xyxy *pixel* coordinates.
        # align_corners=False permits grid normalization x -> 2*x/w - 1.
        boxes = boxes.float()
        center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0)
        # At least six SEMANTIC pixels so small UAVs have useful context.
        extent = wh.clamp_min(6.0) * self.roi_expand
        t = (torch.arange(self.roi_size, device=boxes.device, dtype=torch.float32)
             + 0.5) / self.roi_size - 0.5
        yy, xx = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack((xx, yy), dim=-1)
        pixel_coords = center[:, :, None, None, :] + extent[:, :, None, None, :] * xy
        factor = boxes.new_tensor([2.0 / w, 2.0 / h])
        return pixel_coords * factor - 1.0

    def _sample(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        b, n, r, _, _ = grid.shape
        if n == 0:
            return image.new_zeros((b, 0, image.shape[1], r, r))
        # Match sampling dtype to image; keep center/box computation float32.
        result = F.grid_sample(
            image, grid.reshape(b, n * r, r, 2).to(dtype=image.dtype),
            mode='bilinear', padding_mode='border', align_corners=False,
        )
        return result.reshape(b, image.shape[1], n, r, r).permute(0, 2, 1, 3, 4)

    def forward(self, rgb_high: torch.Tensor, rgb_sem: torch.Tensor,
                aligned_tir_p3: torch.Tensor, support_p3: torch.Tensor,
                boxes_xyxy: torch.Tensor, *, use_hr: bool = True,
                use_guidance: bool = True, use_tir: bool = True):
        if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
            raise ValueError('boxes_xyxy must have shape [B,N,4]')
        b, n, _ = boxes_xyxy.shape
        if b != rgb_high.shape[0] or b != aligned_tir_p3.shape[0]:
            raise ValueError('AGHR batch mismatch')
        if n == 0:
            return boxes_xyxy, {'delta_abs_px': boxes_xyxy.new_zeros(()),
                                'support_mean': boxes_xyxy.new_zeros(())}
        h, w = rgb_sem.shape[-2:]
        # Torch AMP works for convolution; grid coordinate construction stays fp32.
        grid = self._roi_grid(boxes_xyxy, int(h), int(w))
        high = self._sample(rgb_high, grid)
        low = self._sample(rgb_sem, grid)
        if not use_hr:
            high = low
        rgb_input = torch.cat((high, high - low, low), dim=2)
        if not use_tir:
            aligned_tir_p3 = torch.zeros_like(aligned_tir_p3)
        tir = self._sample(_grad_scale(aligned_tir_p3, self.feedback_scale), grid)
        if not use_guidance:
            support_p3 = torch.zeros_like(support_p3)
        # The original GTGA consensus is sqrt(RGB_support * TIR_support).
        # Its backward derivative is singular at zero support (0 * inf -> NaN).
        # Keep geometry support as a FORWARD-ONLY conditioning signal.
        # Alignment still receives a controlled gradient through aligned TIR P3.
        support = self._sample(support_p3.detach(), grid)
        r = self.roi_size
        rgb_emb = self.rgb_encoder(rgb_input.reshape(b*n, 9, r, r)).flatten(1)
        tir_emb = self.tir_encoder(torch.cat((tir, support), dim=2)
                                .reshape(b*n, tir.shape[2] + 1, r, r)).flatten(1)
        box = boxes_xyxy.float()
        wh = (box[..., 2:] - box[..., :2]).clamp_min(1.0)
        center = 0.5 * (box[..., :2] + box[..., 2:])
        geom = torch.stack((center[..., 0] / w, center[..., 1] / h,
                            torch.log1p(wh[..., 0]), torch.log1p(wh[..., 1]),
                            support.mean((-1, -2, -3))), dim=-1)
        feat = torch.cat((rgb_emb, tir_emb, geom.reshape(b*n, 5)), dim=-1)
        raw = self.regression(feat).float().reshape(b, n, 4).tanh()
        # Additive deltas guarantee EXACT identity at zero initialization,
        # including initially off-image detector boxes.
        shift = raw[..., :2] * self.max_center_shift * wh.clamp_min(4.0)
        half_growth = 0.5 * wh * torch.expm1(raw[..., 2:] * self.max_log_scale)
        displacement = torch.cat((shift - half_growth, shift + half_growth), dim=-1)
        result = box + displacement
        x = result[..., [0, 2]].clamp(0.0, float(w))
        y = result[..., [1, 3]].clamp(0.0, float(h))
        clipped = torch.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), dim=-1)
        # No clipping at exact zero-init: preserve original NMS boxes even if
        # original detector predicted beyond the image. Once a nonzero
        # refinement is learned, clip the refined box to RGB semantic canvas.
        has_update = raw.detach().abs().amax(dim=-1, keepdim=True) > 0.0
        refined = torch.where(has_update, clipped, result)
        shift = (refined - boxes_xyxy.float()).abs().mean()
        return refined, {'delta_abs_px': shift,
                         'support_mean': support.mean().detach(),
                         'raw_delta_abs': raw.abs().mean().detach()}

    def refine_postprocessed(self, preds: List[Dict], rgb_high: torch.Tensor,
                             rgb_sem: torch.Tensor, tir_p3: torch.Tensor,
                             support_p3: torch.Tensor, mode: str = 'full'):
        if mode not in ('full', 'identity', 'no_hr', 'no_guidance', 'no_tir', 'hr_only'):
            raise ValueError(f'Unknown AGHR mode: {mode}')
        result = []
        delta_sum = rgb_high.new_zeros((), dtype=torch.float32)
        count = 0
        for i, pred in enumerate(preds):
            boxes = pred['bboxes']
            if mode == 'identity' or boxes.shape[0] == 0:
                result.append(pred)
                continue
            pieces = []
            for start in range(0, int(boxes.shape[0]), self.chunk_size):
                sl = boxes[start:start+self.chunk_size]
                refined, diag = self(
                    rgb_high[i:i+1], rgb_sem[i:i+1], tir_p3[i:i+1],
                    support_p3[i:i+1], sl.unsqueeze(0),
                    use_hr=(mode != 'no_hr'),
                    use_guidance=(mode not in ('no_guidance', 'hr_only')),
                    use_tir=(mode not in ('no_tir', 'hr_only')),
                )
                pieces.append(refined[0].to(dtype=boxes.dtype))
                delta_sum = delta_sum + diag['delta_abs_px'].detach() * sl.shape[0]
                count += int(sl.shape[0])
            updated = dict(pred)
            updated['bboxes'] = torch.cat(pieces, dim=0)
            # Confidence and class IDs explicitly untouched.
            result.append(updated)
        self.last_inference = {
            'delta_abs_px': (delta_sum / max(count, 1)).detach(),
            'refined_boxes': int(count), 'mode': mode,
        }
        return result

    def supervised_loss(self, batch: Dict, features: Dict,
                        jitter_count: int = 4, jitter_std: float = 0.20,
                        mode: str = 'full'):
        """Train the branch on GT-proximal noisy boxes (positive RoI supervision).

        This is NOT a claim that noisy-GT samples match the test-time prediction
        distribution; a real, fixed-checkpoint val experiment is required.
        """
        rgb_high = batch['rgb_img']
        rgb_sem = batch['rgb_semantic_img']
        tir_p3 = features['aligned_tir_backbone'][features['scale_to_index'][3]]
        support = features['alignment_info']['support_by_scale'][3]
        gt_xywh = batch['rgb_semantic_bboxes'].float()
        gt_index = batch['rgb_batch_idx'].reshape(-1).long()
        b, _, h, w = rgb_sem.shape
        k = max(1, int(jitter_count))
        slots = []
        targets = []
        valid = []
        for i in range(b):
            bb = gt_xywh[gt_index == i]
            if bb.shape[0] == 0:
                slots.append(gt_xywh.new_zeros((0, 4)))
                targets.append(gt_xywh.new_zeros((0, 4)))
                continue
            # Original labels are normalized xywh in semantic RGB coordinates.
            xywh = bb * bb.new_tensor([w, h, w, h])
            truth = _cxcywh_to_xyxy(xywh)
            base = xywh[:, None, :].expand(-1, k, -1).clone()
            noise = torch.randn_like(base)
            wh = xywh[:, None, 2:].clamp_min(2.0)
            base[..., :2] += noise[..., :2] * wh * float(jitter_std)
            base[..., 2:] *= torch.exp(noise[..., 2:].clamp(-2.0, 2.0) * float(jitter_std))
            base[..., 2:] = base[..., 2:].clamp_min(1.0)
            proposal = _cxcywh_to_xyxy(base.reshape(-1, 4))
            gt_repeated = truth[:, None, :].expand(-1, k, -1).reshape(-1, 4)
            proposal[..., [0, 2]] = proposal[..., [0, 2]].clamp(0.0, float(w))
            proposal[..., [1, 3]] = proposal[..., [1, 3]].clamp(0.0, float(h))
            slots.append(proposal)
            targets.append(gt_repeated)
        maximum = max([v.shape[0] for v in slots] + [0])
        zero = self.regression[-1].weight.sum() * 0.0
        if maximum == 0:
            return zero, {'aghr_loss_raw': zero.detach(),
                          'aghr_gt_iou_before': zero.detach(),
                          'aghr_gt_iou_after': zero.detach(),
                          'aghr_gt_iou_gain': zero.detach(),
                          'aghr_delta_abs_px': zero.detach(),
                          'aghr_support_mean': zero.detach(),
                          'aghr_samples': 0.0}
        padded_proposals = gt_xywh.new_zeros((b, maximum, 4))
        padded_targets = gt_xywh.new_zeros((b, maximum, 4))
        mask = torch.zeros((b, maximum), device=rgb_high.device, dtype=torch.bool)
        for i, (proposal, truth) in enumerate(zip(slots, targets)):
            n = proposal.shape[0]
            if n:
                padded_proposals[i, :n] = proposal
                padded_targets[i, :n] = truth
                mask[i, :n] = True
        refined, diag = self(rgb_high, rgb_sem, tir_p3, support, padded_proposals,
                                  use_hr=(mode != 'no_hr'),
                                  use_guidance=(mode not in ('no_guidance','hr_only')),
                                  use_tir=(mode not in ('no_tir','hr_only')))
        # Normalize coordinate error by *target* size (important for tiny UAVs).
        ref_wh = (padded_targets[..., 2:] - padded_targets[..., :2]).clamp_min(3.0)
        normalized = (refined - padded_targets) / torch.cat((ref_wh, ref_wh), dim=-1)
        loss_box = F.smooth_l1_loss(normalized[mask], torch.zeros_like(normalized[mask]),
                                    reduction='mean', beta=0.20)
        before = _box_iou_pair(padded_proposals, padded_targets)[mask]
        after = _box_iou_pair(refined, padded_targets)[mask]
        loss_iou = (1.0 - after).mean()
        total = loss_box + 0.25 * loss_iou
        stat = {
            'aghr_loss_raw': total.detach(),
            'aghr_gt_iou_before': before.mean().detach(),
            'aghr_gt_iou_after': after.mean().detach(),
            'aghr_gt_iou_gain': (after.mean()-before.mean()).detach(),
            'aghr_delta_abs_px': diag['delta_abs_px'].detach(),
            'aghr_support_mean': diag['support_mean'].detach(),
            'aghr_samples': float(mask.sum().item()),
        }
        return total, stat
