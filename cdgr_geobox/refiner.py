"""Center-decoupled cross-modal geometry refinement.

Composition uses GTGA's exact inverse-sampling convention q = S * p + t,
with p in RGB-semantic normalized coordinates and q in TIR normalized coords.
The baseline GMLS/DMLS and raw GTGA losses are preserved; this module predicts
bounded residuals and applies one *refined* geometry consistently to every
aligned P3/P4/P5 feature, target-support map, and native TIR ROI crop.
No GT labels or teachers are read by forward().
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.modules.alignment.rlsfa_gtga_centered import warp_with_geometry


def _feature_patch(feature: torch.Tensor, center: torch.Tensor,
                   resolution: int, radius: float) -> torch.Tensor:
    """Differentiable square patch about a normalized RGB reference anchor."""
    if feature.ndim != 4 or center.shape != (feature.shape[0], 2):
        raise ValueError('Unexpected feature/anchor geometry')
    b = feature.shape[0]
    lin = torch.linspace(-radius, radius, resolution, device=center.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    offsets = torch.stack((xx, yy), -1)[None]
    grid = ((center.float()[:, None, None, :] + offsets) * 2.0 - 1.0)
    # Grid and conv features both use the reference normalized coordinate system.
    return F.grid_sample(feature.float(), grid, mode='bilinear',
                         padding_mode='border', align_corners=True)


class LocalCenterScaleResidual(nn.Module):
    """Small P3-local RGB/TIR matching network, zero-residual initialization."""
    def __init__(self, channels: int = 512, proj_channels: int = 32,
                 hidden_channels: int = 64, patch_size: int = 11,
                 patch_radius: float = 0.12, max_shift_px: float = 24.0,
                 max_log_scale: float = 0.50):
        super().__init__()
        if channels <= 0 or proj_channels < 8 or hidden_channels < 16:
            raise ValueError('Invalid geometry refinement channels')
        if patch_size < 3 or patch_size % 2 != 1 or not 0 < patch_radius < 0.5:
            raise ValueError('patch_size must be odd >=3, 0<patch_radius<0.5')
        if max_shift_px <= 0 or max_log_scale <= 0:
            raise ValueError('Residual bounds must be positive')
        self.patch_size = int(patch_size)
        self.patch_radius = float(patch_radius)
        self.max_shift_px = float(max_shift_px)
        self.max_log_scale = float(max_log_scale)
        gn_proj = 8 if proj_channels % 8 == 0 else 1
        gn_hidden = 8 if hidden_channels % 8 == 0 else 1
        self.rgb_project = nn.Sequential(
            nn.Conv2d(channels, proj_channels, 1, bias=False),
            nn.GroupNorm(gn_proj, proj_channels), nn.SiLU(inplace=False))
        self.tir_project = nn.Sequential(
            nn.Conv2d(channels, proj_channels, 1, bias=False),
            nn.GroupNorm(gn_proj, proj_channels), nn.SiLU(inplace=False))
        self.match = nn.Sequential(
            nn.Conv2d(4 * proj_channels + 3, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_hidden, hidden_channels), nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_hidden, hidden_channels), nn.SiLU(inplace=False))
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_channels + 6, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, 4))
        # Exactly reproduce initial GTGA geometry at initialization.
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, rgb_p3: torch.Tensor, aligned_tir_p3: torch.Tensor,
                guidance: torch.Tensor, anchor: torch.Tensor,
                base_scale: torch.Tensor, base_translation: torch.Tensor,
                semantic_hw: tuple[int, int], scale_bounds: tuple[float, float]):
        b, _, h, w = rgb_p3.shape
        if aligned_tir_p3.shape != rgb_p3.shape:
            raise ValueError('Expected same-shape RGB/initial-aligned TIR reference feature')
        if guidance.shape != (b, 3, h, w):
            raise ValueError(f'Expected 3-channel target support at P3, got {tuple(guidance.shape)}')
        anchor = anchor.float().reshape(b, 2)
        old_s = base_scale.float().reshape(b, 2).clamp_min(1e-6)
        old_t = base_translation.float().reshape(b, 2)
        r = self.rgb_project(_feature_patch(rgb_p3, anchor, self.patch_size, self.patch_radius))
        t = self.tir_project(_feature_patch(aligned_tir_p3, anchor, self.patch_size, self.patch_radius))
        u = _feature_patch(guidance, anchor, self.patch_size, self.patch_radius)
        matched = self.match(torch.cat((r, t, (r - t).abs(), r * t, u), 1))
        mean_pool = matched.mean((-2, -1))
        max_pool = matched.amax((-2, -1))
        prior = torch.cat((anchor, old_s.log(), old_t), -1)
        raw = self.regressor(torch.cat((mean_pool, max_pool, prior), -1)).float()
        height, width = semantic_hw
        norm = raw.new_tensor([max(int(width) - 1, 1), max(int(height) - 1, 1)])
        shift_px = torch.tanh(raw[:, 0:2]) * self.max_shift_px
        shift = shift_px / norm  # TIR normalized q-coordinates
        delta_log_scale = torch.tanh(raw[:, 2:4]) * self.max_log_scale
        # Smoothly correct the moment scale, without making it the only scale estimator.
        new_scale = (old_s * torch.exp(delta_log_scale)).clamp(*scale_bounds)
        # Pivot around the original target anchor so scale changes cannot induce
        # an artificial target-center shift. q(a) = (S0*a+t0)+delta.
        old_q_at_anchor = old_s * anchor + old_t
        new_translation = old_q_at_anchor + shift - new_scale * anchor
        return new_scale, new_translation, {
            'center_shift_px': shift_px.detach().norm(dim=-1).mean(),
            'scale_log_residual': delta_log_scale.detach().abs().mean(),
            'scale_saturation_rate': ((new_scale <= scale_bounds[0] + 1e-6)
                                      | (new_scale >= scale_bounds[1] - 1e-6)).float().mean().detach(),
        }


class CenterDecoupledAligner(nn.Module):
    """Wrap the preexisting GTGA alignment module, keeping raw diagnostics intact."""
    def __init__(self, base: nn.Module, cfg: Mapping):
        super().__init__()
        self.base = base
        channels = int(cfg.get('p3_channels', 512))
        self.residual = LocalCenterScaleResidual(
            channels=channels,
            proj_channels=int(cfg.get('projection_channels', 32)),
            hidden_channels=int(cfg.get('hidden_channels', 64)),
            patch_size=int(cfg.get('patch_size', 11)),
            patch_radius=float(cfg.get('patch_radius', .12)),
            max_shift_px=float(cfg.get('max_center_shift_semantic_px', 24.0)),
            max_log_scale=float(cfg.get('max_log_scale', .50)))
        self.last_stats: dict[str, torch.Tensor] = {}
        self.semantic_imgsz = int(cfg.get('semantic_imgsz', 640))

    @property
    def unit(self):
        return self.base.unit

    @property
    def reference_scale(self):
        return self.base.reference_scale

    @property
    def apply_scales(self):
        return self.base.apply_scales

    def align_pyramid(self, rgb_features: Sequence[torch.Tensor],
                      tir_features: Sequence[torch.Tensor],
                      scale_to_index: Mapping[int, int],
                      rgb_valid_mask=None, tir_valid_mask=None,
                      return_debug: bool = False):
        original_aligned, info = self.base.align_pyramid(
            rgb_features, tir_features, scale_to_index,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_debug=return_debug)
        idx = int(scale_to_index[self.reference_scale])
        rgb_ref, tir_ref = rgb_features[idx], original_aligned[idx]
        b, _, h, w = rgb_ref.shape
        old_s = info['sampling_scale'].float().reshape(b, 2)
        old_t = info['sampling_translation'].float().reshape(b, 2)
        anchor = info['coarse_anchor'].reshape(b, 2)
        guidance = info['multi_target_guidance']
        refined_s, refined_t, stats = self.residual(
            rgb_ref, tir_ref, guidance, anchor, old_s, old_t,
            semantic_hw=(self.semantic_imgsz, self.semantic_imgsz),
            scale_bounds=(float(self.unit.scale_min), float(self.unit.scale_max)))
        # Important: P3 coordinate normalization for shift is (w-1,h-1),
        # NOT 640 or high-resolution 1920. Use exact feature-reference pixel
        # coordinates so q=S*p+t has the same normalization as the original GTGA.
        # The predicted bounded shift is defined in semantic image pixels, so
        # convert 640-canvas shift to P3 normalized geometry using 640-1.
        # (The spatially normalized q coordinate itself is resolution-invariant.)
        refined_aligned = list(original_aligned)
        for scale in self.apply_scales:
            i = int(scale_to_index[scale])
            refined_aligned[i] = warp_with_geometry(
                tir_features[i], refined_s, refined_t,
                output_hw=rgb_features[i].shape[-2:],
                padding_mode=self.base.padding_mode,
                align_corners=self.base.align_corners)
        # Rewarp the prealigned TIR target-support channel using the *relative*
        # transform between original and corrected GTGA geometry. This avoids
        # training ROI guide on a stale support map.
        relative_s = refined_s / old_s.clamp_min(1e-6)
        relative_t = (refined_t - old_t) / old_s.clamp_min(1e-6)
        tir_support = warp_with_geometry(
            guidance[:, 1:2], relative_s, relative_t,
            output_hw=guidance.shape[-2:], padding_mode='zeros',
            align_corners=self.base.align_corners).clamp(0, 1)
        rgb_support = guidance[:, 0:1].clamp(0, 1)
        # EMPTYGT_CDGR_JOINT_SQRT_FIX_V1
        # Use FP32 and a clamped branch; 1e-8 underflows to zero in FP16.
        product=(rgb_support.float()*tir_support.float()).clamp_min(0.0)
        joint=torch.where(product>0,product.clamp_min(1e-6).sqrt(),
                          torch.zeros_like(product)).clamp(0,1)
        multi = torch.cat((rgb_support, tir_support, joint), 1)
        info = dict(info)
        info['sampling_scale_raw_gtga'] = old_s
        info['sampling_translation_raw_gtga'] = old_t
        info['sampling_scale'] = refined_s
        info['sampling_translation'] = refined_t
        info['source_center'] = refined_s * anchor + refined_t
        info['rgb_target_support'] = rgb_support
        info['aligned_tir_target_support'] = tir_support
        info['joint_target_support'] = joint
        info['multi_target_guidance'] = multi
        info['bhlr_guidance_map'] = joint
        for scale in self.apply_scales:
            i = int(scale_to_index[scale])
            resized = F.interpolate(multi, size=rgb_features[i].shape[-2:], mode='bilinear', align_corners=False).clamp(0, 1)
            info['support_by_scale'][scale] = resized[:, 2:3]
            info['guidance_by_scale'][scale] = {
                'multi_target_guidance': resized,
                'rgb_target_support': resized[:, 0:1],
                'aligned_tir_target_support': resized[:, 1:2],
                'joint_target_support': resized[:, 2:3],
            }
        self.last_stats = {f'cdgr_{name}': value.detach() for name, value in stats.items()}
        return refined_aligned, info

    def forward(self, rgb_features, tir_features, scale_to_index,
                return_debug=False, rgb_valid_mask=None, tir_valid_mask=None):
        return self.align_pyramid(rgb_features, tir_features, scale_to_index,
                                  rgb_valid_mask, tir_valid_mask, return_debug)
