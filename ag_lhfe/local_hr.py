"""AG-LHFE v1: RGB/autonomous + aligned cross-modal proposals -> HR local patches -> P3.

No ground-truth locations enter the proposal pipeline at inference or training.
ROI targets are used ONLY by auxiliary losses. This is a research baseline, not a
validated performance improvement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, stride: int = 1):
        groups = max(1, min(8, c2))
        while c2 % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(c1, c2, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, c2),
            nn.SiLU(inplace=True),
        )


class PatchEncoder(nn.Module):
    def __init__(self, out_dim: int, use_edges: bool = True):
        super().__init__()
        self.use_edges = bool(use_edges)
        self.body = nn.Sequential(
            ConvGNAct(5, 24, stride=2),
            ConvGNAct(24, 32, stride=2),
            ConvGNAct(32, 48),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(48 * 4, out_dim), nn.SiLU())
        sobel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]) / 8.
        sobel_y = sobel_x.T.contiguous()
        self.register_buffer('sobel_x', sobel_x.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer('sobel_y', sobel_y.reshape(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*K, 3, patch, patch] in [0, 1]
        if self.use_edges:
            gray = (x[:, 0:1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114)
            gx = F.conv2d(gray.float(), self.sobel_x.float(), padding=1).to(x.dtype)
            gy = F.conv2d(gray.float(), self.sobel_y.float(), padding=1).to(x.dtype)
        else:
            gx = torch.zeros_like(x[:, :1])
            gy = torch.zeros_like(x[:, :1])
        return self.proj(self.body(torch.cat([x, gx, gy], dim=1)))


class LocalHighResFeatureExtractor(nn.Module):
    """Fixed-compute proposal sampler, two-scale RGB patch encoder, differentiable P3 splat.

    proposal_mode:
      'rgb'       : all K proposals from RGB-only target-support map
      'rgb_align' : half RGB-only, half aligned RGB-T, excluding duplicates
    """
    def __init__(
        self,
        channels: int,
        proposal_mode: str = 'rgb_align',
        topk: int = 16,
        nms_radius: int = 3,
        patch_size: int = 32,
        crop_sizes_high: Tuple[int, int] = (48, 96),
        latent_dim: int = 64,
        use_edges: bool = True,
        strength_init: float = 0.05,
        strength_max: float = 0.20,
        max_offset: float = 2.5,
        splat_radius: int = 2,
        positive_radius: float = 2.5,
        lambda_obj: float = 0.05,
        lambda_loc: float = 0.02,
    ):
        super().__init__()
        if proposal_mode not in ('rgb', 'rgb_align'):
            raise ValueError(f'Invalid proposal_mode={proposal_mode}')
        if not (1 <= topk <= 128):
            raise ValueError('topk must be between 1 and 128')
        if len(crop_sizes_high) != 2 or min(crop_sizes_high) <= 0:
            raise ValueError('crop_sizes_high needs exactly two positive sizes')
        if not (0 < strength_init < strength_max <= 1):
            raise ValueError('strength_init must be in (0, strength_max]')
        self.channels = int(channels)
        self.proposal_mode = proposal_mode
        self.topk = int(topk)
        self.nms_radius = int(nms_radius)
        self.patch_size = int(patch_size)
        self.crop_sizes_high = tuple(float(c) for c in crop_sizes_high)
        self.latent_dim = int(latent_dim)
        self.max_offset = float(max_offset)
        self.splat_radius = int(splat_radius)
        self.positive_radius = float(positive_radius)
        self.lambda_obj = float(lambda_obj)
        self.lambda_loc = float(lambda_loc)
        self.strength_max = float(strength_max)
        # Non-zero initial strength, but constrained to avoid overwhelming P3.
        init_fraction = strength_init / strength_max
        self.strength_logit = nn.Parameter(torch.tensor(math.log(init_fraction / (1 - init_fraction))))
        self.injection_enabled = True  # Runtime probe: disable WITHOUT changing weights

        self.rgb_encoder = PatchEncoder(self.latent_dim, use_edges=use_edges)
        self.large_encoder = PatchEncoder(self.latent_dim, use_edges=use_edges)
        self.context_proj = nn.Sequential(
            nn.Linear(self.channels * 2 + 4, self.latent_dim),
            nn.LayerNorm(self.latent_dim), nn.SiLU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(self.latent_dim * 3, self.latent_dim), nn.LayerNorm(self.latent_dim), nn.SiLU(),
        )
        self.obj_head = nn.Linear(self.latent_dim, 1)
        self.offset_head = nn.Linear(self.latent_dim, 2)
        self.size_head = nn.Linear(self.latent_dim, 2)
        self.to_canvas = nn.Linear(self.latent_dim, self.latent_dim)
        self.canvas_refine = nn.Sequential(
            ConvGNAct(self.latent_dim, self.latent_dim),
            nn.Conv2d(self.latent_dim, self.channels, kernel_size=1, bias=False),
        )
        self.use_edges = bool(use_edges)
        self.register_buffer('_probe_zero', torch.tensor(0.0), persistent=False)

    @staticmethod
    def _resize_map(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if x.shape[-2:] == (h, w):
            return x
        return F.interpolate(x, (h, w), mode='bilinear', align_corners=False)

    def _peak_topk(self, x: torch.Tensor, k: int, forbidden: Optional[torch.Tensor] = None):
        b, _, h, w = x.shape
        score = x.float().clamp(0, 1)
        r = self.nms_radius
        if r:
            peak = score >= F.max_pool2d(score, kernel_size=2*r+1, stride=1, padding=r) - 1e-6
            score = score.masked_fill(~peak, -1e5)
        if forbidden is not None:
            score = score.masked_fill(forbidden, -1e5)
        _, idx = score.flatten(1).topk(min(k, h*w), dim=1)
        return idx

    @torch.no_grad()
    def proposals(self, rgb_support: torch.Tensor, tir_support: torch.Tensor,
                  joint_support: torch.Tensor, geo_quality: torch.Tensor):
        b, _, h, w = rgb_support.shape
        count = min(self.topk, h*w)
        if self.proposal_mode == 'rgb':
            idx = self._peak_topk(rgb_support, count)
            kind = torch.zeros_like(idx)
        else:
            n_rgb = (count + 1) // 2
            n_cross = count - n_rgb
            idx_rgb = self._peak_topk(rgb_support, n_rgb)
            taken = torch.zeros((b, 1, h*w), dtype=rgb_support.dtype, device=rgb_support.device)
            taken.scatter_(2, idx_rgb.unsqueeze(1), 1.0)
            taken = taken.view(b, 1, h, w)
            exclusion = F.max_pool2d(taken, kernel_size=3, stride=1, padding=1) > 0
            # Use aligned TIR independently so cross-modal proposals can recover
            # a UAV weak in RGB. The selected GMLS/DMLS quality estimate
            # controls how much thermal evidence is trusted.
            q = geo_quality[:,None,None,None]
            cross_score = 0.20 * rgb_support + q * (0.55 * tir_support + 0.25 * joint_support)
            idx_cross = self._peak_topk(cross_score, n_cross, forbidden=exclusion) if n_cross else idx_rgb[:, :0]
            idx = torch.cat((idx_rgb, idx_cross), dim=1)
            kind = torch.cat((torch.zeros_like(idx_rgb), torch.ones_like(idx_cross)), dim=1)
        coords = torch.stack(((idx % w).float(), torch.div(idx, w, rounding_mode='floor').float()), dim=-1)
        return coords, idx, kind

    def _crop_rgb(self, rgb_high: torch.Tensor, centers: torch.Tensor, p3_hw: Tuple[int,int], crop_px: float):
        # Only ONE full-res image per batch element is held by grid_sample:
        # grid shape [B,K*patch,patch,2], no expansion to [B*K,3,H,W].
        b, k, _ = centers.shape
        h_high, w_high = rgb_high.shape[-2:]
        h3, w3 = p3_hw
        half = (self.patch_size - 1.) / 2.
        lin = (torch.arange(self.patch_size, device=rgb_high.device, dtype=torch.float32) - half)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
        cell_x = (centers[..., 0] + 0.5) * (w_high / w3) - 0.5
        cell_y = (centers[..., 1] + 0.5) * (h_high / h3) - 0.5
        step = crop_px / self.patch_size
        px = cell_x[:, :, None, None] + grid_x[None, None] * step
        py = cell_y[:, :, None, None] + grid_y[None, None] * step
        gx = (px + 0.5) * 2.0 / w_high - 1.0
        gy = (py + 0.5) * 2.0 / h_high - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        grid = grid.reshape(b, k*self.patch_size, self.patch_size, 2)
        patch = F.grid_sample(rgb_high, grid.to(rgb_high.dtype), mode='bilinear',
                              padding_mode='border', align_corners=False)
        return (patch.view(b, 3, k, self.patch_size, self.patch_size)
                .permute(0, 2, 1, 3, 4).reshape(b*k, 3, self.patch_size, self.patch_size))

    @staticmethod
    def _gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        z = x.flatten(2).gather(2, idx[:, None].expand(b, c, idx.shape[1]))
        return z.transpose(1, 2)

    def _splat(self, emb: torch.Tensor, coords: torch.Tensor,
               offsets: torch.Tensor, scores: torch.Tensor, h: int, w: int):
        # Scatter a 2D Gaussian neighborhood: gradients propagate through
        # embeddings, predicted offsets, and objectness scores.
        b, k, c = emb.shape
        radius = self.splat_radius
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius+1, device=emb.device),
            torch.arange(-radius, radius+1, device=emb.device), indexing='ij')
        shifts = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        centers_int = coords.long()
        grid = centers_int[:, :, None, :] + shifts.view(1, 1, -1, 2)
        valid = ((grid[..., 0] >= 0) & (grid[..., 0] < w) &
                 (grid[..., 1] >= 0) & (grid[..., 1] < h))
        clamped = grid.clone()
        clamped[..., 0].clamp_(0, w-1)
        clamped[..., 1].clamp_(0, h-1)
        idx = clamped[..., 1]*w + clamped[..., 0]
        target = coords + offsets
        delta = (grid.float() - target[:, :, None, :]).square().sum(-1)
        weight = torch.exp(-0.5 * delta / (1.15**2)) * valid.to(delta.dtype)
        weight = weight * scores.squeeze(-1)[:, :, None]
        # Leave absolute strength informative; divide by mass of candidates
        # mapped to same cell, but not by full Gaussian integral.
        value = emb[:, :, :, None] * weight[:, :, None, :]
        canvas = emb.new_zeros((b, c, h*w))
        canvas.scatter_add_(2, idx.reshape(b, 1, -1).expand(b, c, -1), value.reshape(b, c, -1))
        mass = emb.new_zeros((b, 1, h*w))
        mass.scatter_add_(2, idx.reshape(b, 1, -1), weight.reshape(b, 1, -1))
        canvas = canvas / (1.0 + mass)
        return canvas.view(b, c, h, w), mass.view(b, 1, h, w)

    def forward(self, rgb_high: torch.Tensor, rgb_p3: torch.Tensor, tir_p3: torch.Tensor,
                rgb_support: torch.Tensor, tir_support: torch.Tensor, joint_support: torch.Tensor,
                geometry_quality: Optional[torch.Tensor] = None):
        b, c, h3, w3 = rgb_p3.shape
        if c != self.channels:
            raise RuntimeError(f'P3 channels {c} != configured {self.channels}')
        sr = self._resize_map(rgb_support, h3, w3).detach().clamp(0, 1)
        st = self._resize_map(tir_support, h3, w3).detach().clamp(0, 1)
        sj = self._resize_map(joint_support, h3, w3).detach().clamp(0, 1)
        geo_quality = (torch.ones((b,), device=rgb_p3.device, dtype=torch.float32)
                       if geometry_quality is None else geometry_quality.detach().float().view(b).clamp(0,1))
        coords, idx, kind = self.proposals(sr, st, sj, geo_quality)
        k = idx.shape[1]
        with torch.no_grad():
            source_strength = self._gather_points(sr, idx)
            thermal_strength = self._gather_points(st, idx)
            joint_strength = self._gather_points(sj, idx)
            geo_strength = geo_quality[:,None,None].expand(b, k, 1)
        patches_small = self._crop_rgb(rgb_high, coords, (h3,w3), self.crop_sizes_high[0])
        patches_large = self._crop_rgb(rgb_high, coords, (h3,w3), self.crop_sizes_high[1])
        detail1 = self.rgb_encoder(patches_small).view(b,k,self.latent_dim)
        detail2 = self.large_encoder(patches_large).view(b,k,self.latent_dim)
        rgb_sem = self._gather_points(rgb_p3, idx)
        tir_sem = self._gather_points(tir_p3, idx)
        semantic = torch.cat((rgb_sem, tir_sem, source_strength, thermal_strength, joint_strength, geo_strength), dim=-1)
        context = self.context_proj(semantic)
        z = self.merge(torch.cat((detail1, detail2, context), dim=-1))
        obj_logits = self.obj_head(z)
        offsets = self.max_offset * torch.tanh(self.offset_head(z))
        logsize = self.size_head(z)
        obj = torch.sigmoid(obj_logits)
        local_emb = self.to_canvas(z)
        # Limit initial perturbation; objectness is learned with bbox supervision.
        canvas, mass = self._splat(local_emb, coords, offsets, obj, h3, w3)
        residual = self.canvas_refine(canvas)
        strength = self.strength_max * torch.sigmoid(self.strength_logit)
        delta = strength * residual if self.injection_enabled else residual * 0.0
        enhanced = rgb_p3 + delta
        state = {
            'coords': coords, 'kind': kind, 'obj_logits': obj_logits,
            'offsets': offsets, 'logsize': logsize, 'mass': mass,
            'delta': delta, 'rgb_p3': rgb_p3, 'strength': strength,
            'joint_strength': joint_strength, 'geo_quality': geo_quality,
        }
        with torch.no_grad():
            ratio = (delta.float().square().mean().sqrt() /
                     rgb_p3.float().square().mean().sqrt().clamp_min(1e-8))
            state['inject_rms_ratio'] = ratio.detach()
            state['gate_mean'] = obj.detach().mean()
        return enhanced, state

    def auxiliary_loss(self, state: Dict[str, Any], batch: Dict[str, torch.Tensor]):
        centers = state['coords']
        b, k, _ = centers.shape
        h, w = state['rgb_p3'].shape[-2:]
        gt = batch['rgb_semantic_bboxes']
        bids = batch['rgb_batch_idx'].long().view(-1)
        # One or multiple boxes per image; no GT is used to MAKE proposals.
        quality_target = torch.zeros((b,k), device=centers.device, dtype=centers.dtype)
        target_offsets = torch.zeros_like(centers)
        target_logsizes = torch.zeros_like(centers)
        positive = torch.zeros((b,k), device=centers.device, dtype=torch.bool)
        n_gt = 0
        n_covered = 0
        with torch.no_grad():
            for bi in range(b):
                boxes = gt[bids == bi].float()
                if not boxes.numel():
                    continue
                xy = boxes[:, :2] * boxes.new_tensor([w, h]) - 0.5
                wh = boxes[:, 2:] * boxes.new_tensor([w, h])
                dist = torch.cdist(centers[bi].float(), xy)
                mindist, arg = dist.min(1)
                hit = mindist <= self.positive_radius
                quality_target[bi,hit] = 1.
                positive[bi] = hit
                target_offsets[bi] = (xy[arg] - centers[bi]).clamp(-self.max_offset,self.max_offset)
                target_logsizes[bi] = wh[arg].clamp_min(0.125).log()
                covered = (dist.min(0).values <= self.positive_radius).sum().item()
                n_gt += boxes.shape[0]
                n_covered += int(covered)
        obj_loss = F.binary_cross_entropy_with_logits(state['obj_logits'][...,0].float(),quality_target)
        if positive.any():
            off_loss = F.smooth_l1_loss(state['offsets'][positive].float(),target_offsets[positive])
            size_loss = F.smooth_l1_loss(state['logsize'][positive].float(),target_logsizes[positive])
            loc_loss = off_loss + 0.25*size_loss
        else:
            loc_loss = state['offsets'].sum()*0. + state['logsize'].sum()*0.
        weighted = b*(self.lambda_obj*obj_loss + self.lambda_loc*loc_loss)
        metrics = {
            'lhfe_obj':obj_loss.detach(), 'lhfe_loc':loc_loss.detach(),
            'lhfe_weighted': weighted.detach()/max(b,1),
            'lhfe_positive_ratio': positive.float().mean().detach(),
            'lhfe_proposal_gt_recall':float(n_covered/max(1,n_gt)),
            'lhfe_injection_rms':state['inject_rms_ratio'].detach(),
            'lhfe_strength':state['strength'].detach(),
            'lhfe_obj_gate_mean':state['gate_mean'].detach(),
            'lhfe_geometry_quality':state['geo_quality'].detach().mean(),
        }
        return weighted, metrics
