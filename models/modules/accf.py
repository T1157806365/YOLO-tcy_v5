"""Alignment-Conditioned Complementary Fusion, P3 additive-residual variant.

Retains the original RGB+aligned TIR Add path; all extra parameters are
zero-output initialized. The forward identity is exact at initialization.

Two matched modes have the same parameter count:
  use_alignment=False: feature-only complementary residual; geometric inputs zero
  use_alignment=True:  the same network with spatial/geometry priors

Inputs derive from predicted features / geometry ONLY, never ground-truth labels.
"""
from __future__ import annotations
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


class ACCFP3(nn.Module):
    """A light residual gate; operates on existing fused P3, BEFORE post-fusion BHLR."""

    def __init__(self, channels: int, hidden: int = 48,
                 use_alignment: bool = True, residual_scale: float = 0.25,
                 alignment_gradient_scale: float = 0.10):
        super().__init__()
        if channels < 1 or hidden < 8:
            raise ValueError('Invalid ACCF channel counts')
        if not (0.0 <= alignment_gradient_scale <= 1.0):
            raise ValueError('alignment_gradient_scale must be within [0,1]')
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.use_alignment = bool(use_alignment)
        self.residual_scale = float(residual_scale)
        self.alignment_gradient_scale = float(alignment_gradient_scale)
        groups = 8 if hidden % 8 == 0 else 1

        def projection():
            return nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                 nn.GroupNorm(groups, hidden), nn.SiLU(inplace=False))

        self.rgb_projection = projection()
        self.tir_projection = projection()
        # 12 conditioning maps: R/T/J/U/disagreement/confidence/validity/
        # cosine_agreement/dx/dy/dscale_x/dscale_y
        self.gate = nn.Sequential(
            nn.Conv2d(hidden * 3 + 12, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden), nn.SiLU(inplace=False),
            nn.Conv2d(hidden, hidden, 1), nn.Sigmoid())
        self.complement = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden), nn.SiLU(inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden), nn.SiLU(inplace=False))
        self.output_projection = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_stats = {}

    @staticmethod
    def _batched_vector(info: Mapping, key: str, batch: int, dim: int,
                        reference: torch.Tensor, fill: float = 0.0):
        x = info.get(key, None)
        if not torch.is_tensor(x):
            return reference.new_full((batch, dim), fill)
        if x.shape[0] != batch:
            raise ValueError(f'Alignment {key} batch mismatch: {tuple(x.shape)}')
        x = x.reshape(batch, -1)
        if x.shape[1] == 1 and dim > 1:
            x = x.expand(-1, dim)
        if x.shape[1] < dim:
            raise ValueError(f'Alignment {key}: expected >= {dim} entries, got {x.shape[1]}')
        return x[:, :dim].to(dtype=reference.dtype)

    def _conditioning(self, rgb: torch.Tensor, tir: torch.Tensor,
                      info: Optional[Mapping], pr: torch.Tensor,
                      pt: torch.Tensor):
        b, _, h, w = rgb.shape
        if not self.use_alignment:
            return rgb.new_zeros((b, 12, h, w))
        if not isinstance(info, Mapping) or not info.get('enabled', False):
            raise RuntimeError('ACCF use_alignment=True requires active alignment_info')
        packets = info.get('multi_target_guidance', None)
        if not torch.is_tensor(packets) or packets.ndim != 4 or packets.shape[1] != 3:
            raise RuntimeError('ACCF requires predicted multi_target_guidance [B,3,H,W]')
        if packets.shape[-2:] != (h, w):
            packets = F.interpolate(packets, size=(h, w), mode='bilinear', align_corners=False)
        packets = packets.clamp(0.0, 1.0).to(dtype=rgb.dtype)
        mr, mt, joint = packets[:, :1], packets[:, 1:2], packets[:, 2:3]
        union = 1.0 - (1.0 - mr) * (1.0 - mt)
        disagreement = (mr - mt).abs()

        conf = self._batched_vector(info, 'coarse_confidence', b, 1, rgb)
        conf = torch.tanh(conf).reshape(b, 1, 1, 1).expand(-1, -1, h, w)
        scale = self._batched_vector(info, 'sampling_scale', b, 2, rgb, fill=1.0)
        trans = self._batched_vector(info, 'sampling_translation', b, 2, rgb)

        # Same coordinate convention as the existing alignment warp:
        # normalized output [x,y] mapped into TIR source [sx*x+tx, sy*y+ty].
        xs = torch.linspace(0.0, 1.0, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, w)
        ys = torch.linspace(0.0, 1.0, h, device=rgb.device, dtype=rgb.dtype).view(1, 1, h, 1)
        x_src = scale[:, 0:1, None, None] * xs + trans[:, 0:1, None, None]
        y_src = scale[:, 1:2, None, None] * ys + trans[:, 1:2, None, None]
        # Smooth source-FOV validity, allowing gradient near boundaries.
        valid = (torch.sigmoid(32.0*x_src)*torch.sigmoid(32.0*(1.0-x_src))
                 *torch.sigmoid(32.0*y_src)*torch.sigmoid(32.0*(1.0-y_src)))
        valid = valid.expand(-1, -1, h, w)
        cos = (F.normalize(pr.float(), dim=1, eps=1e-5)
               * F.normalize(pt.float(), dim=1, eps=1e-5)).sum(dim=1, keepdim=True)
        cos = cos.to(dtype=rgb.dtype)

        def expand_two(x):
            return x.reshape(b, 2, 1, 1).expand(-1, -1, h, w)

        offset = expand_two(torch.tanh(trans * 2.0))
        ds = expand_two(torch.tanh((scale-1.0) * 2.0))
        cond = torch.cat([mr, mt, joint, union, disagreement, conf,
                          valid, cos, offset, ds], dim=1)
        assert cond.shape[1] == 12, cond.shape
        # Controlled detection-feedback strength into predicted aligner features,
        # not into RGB/TIR backbone features through the normal fusion path.
        k = self.alignment_gradient_scale
        return cond.detach() + k * (cond - cond.detach())

    def forward(self, fused: torch.Tensor, rgb: torch.Tensor,
                aligned_tir: torch.Tensor, alignment_info: Optional[Mapping] = None):
        if fused.shape != rgb.shape or fused.shape != aligned_tir.shape:
            raise ValueError('ACCF requires same-shape fused/RGB/aligned-TIR P3 features')
        if fused.shape[1] != self.channels:
            raise ValueError(f'ACCF expected C={self.channels}, got C={fused.shape[1]}')
        pr = self.rgb_projection(rgb)
        pt = self.tir_projection(aligned_tir)
        semantic = torch.cat([pr, pt, (pr-pt).abs()], dim=1)
        cond = self._conditioning(rgb, aligned_tir, alignment_info, pr, pt)
        mask = self.gate(torch.cat([semantic, cond], dim=1))
        residual = self.output_projection(self.complement(semantic)*mask)
        out = fused + self.residual_scale*residual
        if not torch.jit.is_tracing():
            with torch.no_grad():
                self.last_stats = {
                    'residual_abs': float((self.residual_scale*residual.detach()).float().abs().mean()),
                    'fused_abs': float(fused.detach().float().abs().mean()),
                    'gate_mean': float(mask.detach().float().mean()),
                }
        return out
