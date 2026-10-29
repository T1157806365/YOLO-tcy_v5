# -*- coding: utf-8 -*-
"""GMLS-v2 + Fine + Scale + BHLR-v2, with alignment-reliability fusion.

Only replaces the final P3/P4/P5 ADD operation.  The historical alignment,
BHLR, loss, YOLO head, dataset, training and validation implementations are
otherwise untouched.

Important: the gate is initialized as an EXACT ADD residual:
    base = rgb + aligned_tir_projected
    new  = base + (multiplier - 1) * aligned_tir_projected
    multiplier = 2 * sigmoid(logit); logit=0 at initialization.
This yields base bit-for-bit at initial inference, before any gate updates.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)


def _get_scale(mapping: Mapping, scale: int):
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(scale, mapping.get(str(scale)))


class SpatialAlignmentReliabilityGate(nn.Module):
    """Spatial modulation estimated from alignment evidence AND feature agreement.

    Inputs (all [B,1,H,W]):
      RGB local support; aligned TIR local support; joint target support;
      cosine(RGB,TIR); normalized absolute discrepancy; relative RGB energy;
      relative TIR energy; normalized energy mismatch; global alignment confidence.

    The 1-channel output is shared across channels per pixel.  A range (0,2)
    intentionally permits both suppression and modest amplification; do not
    interpret 2*sigmoid as a calibrated confidence probability.
    """

    def __init__(self, hidden_channels: int = 16, detach_alignment_support: bool = True):
        super().__init__()
        width = max(4, int(hidden_channels))
        self.detach_alignment_support = bool(detach_alignment_support)
        self.net = nn.Sequential(
            nn.Conv2d(9, width, kernel_size=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(width, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _support(map_value, ref):
        if not torch.is_tensor(map_value):
            raise KeyError('Alignment guidance must contain RGB, TIR and joint support tensors.')
        x = map_value
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x.float(), size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x.float().clamp(0, 1)

    def forward(self, rgb, tir_contribution, support: Mapping[str, torch.Tensor], coarse_confidence):
        if rgb.shape != tir_contribution.shape:
            raise ValueError(f'Fusion tensors must have same shape: {rgb.shape} vs {tir_contribution.shape}')
        # Feature-statistic normalization is done in float32 for safe AMP use.
        rgb32 = rgb.float()
        tir32 = tir_contribution.float()
        rgb_rms = (rgb32.square().mean(1, keepdim=True) + 1.e-6).sqrt()
        tir_rms = (tir32.square().mean(1, keepdim=True) + 1.e-6).sqrt()
        denom = rgb_rms + tir_rms + 1.e-4
        similarity = F.cosine_similarity(rgb32, tir32, dim=1, eps=1.e-6).unsqueeze(1)
        discrepancy = (rgb32 - tir32).abs().mean(1, keepdim=True) / denom
        rgb_energy = rgb_rms / denom
        tir_energy = tir_rms / denom
        energy_mismatch = (rgb_rms - tir_rms).abs() / denom

        rg = self._support(support.get('rgb_target_support'), rgb32)
        tg = self._support(support.get('aligned_tir_target_support'), rgb32)
        jg = self._support(support.get('joint_target_support'), rgb32)
        if self.detach_alignment_support:
            rg, tg, jg = rg.detach(), tg.detach(), jg.detach()

        if torch.is_tensor(coarse_confidence):
            cf = coarse_confidence.float().reshape(rgb.shape[0], -1).mean(1).view(-1, 1, 1, 1)
            cf = cf.expand(-1, 1, rgb.shape[-2], rgb.shape[-1]).clamp(0, 1)
            if self.detach_alignment_support:
                cf = cf.detach()
        else:
            cf = torch.zeros_like(rg)

        stats = torch.cat((rg, tg, jg, similarity, discrepancy, rgb_energy,
                           tir_energy, energy_mismatch, cf), dim=1)
        logit = self.net(stats)
        multiplier = (2.0 * torch.sigmoid(logit)).to(dtype=rgb.dtype)
        return multiplier


class RGBTAlignmentReliabilityFusionDetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    """Compatibility subclass; inherits all original detection/align/BHLR losses."""

    def __init__(self, *args, reliability_fusion_cfg=None, **kwargs):
        # DO NOT create extra modules before parent initialises its own model.
        super().__init__(*args, **kwargs)
        cfg = dict(reliability_fusion_cfg or {})
        if not bool(cfg.get('enabled', True)):
            raise ValueError('The reliability_fusion variant requires reliability_fusion.enabled: true')
        if str(self.fusion_type).lower() != 'add':
            raise ValueError('Reliability fusion requires model.fusion: add for baseline-preserving initialization.')
        if not self.alignment_enabled:
            raise ValueError('Reliability fusion requires alignment.enabled: true')

        self.reliability_fusion_cfg = cfg
        self.reliability_gates = nn.ModuleDict({
            str(scale): SpatialAlignmentReliabilityGate(
                hidden_channels=int(cfg.get('hidden_channels', 16)),
                detach_alignment_support=bool(cfg.get('detach_alignment_support', True)),
            )
            for scale in sorted(self.scale_to_index)
        })
        print('[RELIABILITY FUSION] P3/P4/P5 trainable spatial gates; '
              'initial multiplier=1.000000 (exact original ADD); '
              f'align-support detached={bool(cfg.get("detach_alignment_support", True))}.', flush=True)

    def forward_features(self, rgb, tir, rgb_semantic=None, rgb_valid_mask=None,
                         tir_valid_mask=None, return_bhlr_debug=False,
                         return_alignment_debug=False):
        output = super().forward_features(
            rgb, tir, rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        info = output['alignment_info']
        guidance_by_scale = info.get('guidance_by_scale', {})
        if not guidance_by_scale:
            raise RuntimeError('Alignment guidance_by_scale is missing. Cannot claim reliability-guided fusion.')

        # This override is intentionally after original ADD: it uses exactly
        # the projected/size-aligned TIR contribution from the old ADD fusion,
        # including any learned tir_proj, with no duplicate projection.
        fused = list(output['fused_backbone'])
        fusion_features = dict(output['fusion_features'])
        modulation_maps = {}
        test_mode = ('auto' if self.training else
                     os.environ.get('RGBT_RELIABILITY_EVAL_MODE', 'auto').strip().lower())
        if test_mode not in ('auto', 'force_add'):
            raise ValueError('RGBT_RELIABILITY_EVAL_MODE must be auto or force_add')

        for scale, idx in self.scale_to_index.items():
            rgb_feat = output['rgb_backbone'][idx]    # AFTER BHLR enhancement
            native_fused = fusion_features[idx]       # RGB + projected aligned TIR
            tir_effect = native_fused - rgb_feat      # exact original ADD contribution
            support = _get_scale(guidance_by_scale, scale)
            if support is None:
                raise RuntimeError(f'Missing alignment reliability guidance for P{scale}')
            gate = self.reliability_gates[str(scale)](
                rgb_feat, tir_effect, support, info.get('coarse_confidence')
            )
            disabled_scales = {
                int(s.strip())
                for s in os.environ.get(
                    "RGBT_ARF_DISABLE_SCALES", ""
                ).split(",")
                if s.strip()
            }

            if test_mode == 'force_add' or scale in disabled_scales:
                gate = torch.ones_like(gate)

            # At initial zero-logit, (gate - 1) is exactly 0, so output is
            # numerically IDENTICAL to the old ADD (not approximate 0.95).
            current_fused = native_fused + (gate - 1.0) * tir_effect
            fused[idx] = current_fused
            fusion_features[idx] = current_fused
            modulation_maps[scale] = gate

        output['fused_backbone'] = fused
        output['fusion_features'] = fusion_features
        output['reliability_modulation_maps'] = modulation_maps
        output['reliability_fusion_type'] = 'alignment_reliability_modulated_add'
        return output


__all__ = [
    'SpatialAlignmentReliabilityGate',
    'RGBTAlignmentReliabilityFusionDetectionModel',
]
