"""ACCF-v3: geometry-conditioned P3 complement AND BHLR guidance refinement.

Integration contract:
    Add(P3) -> ACCFP3V3 -> pre-existing Post-Fusion BHLR(P3)

BHLR's insertion point and internal implementation remain unchanged. Its
existing external 'joint_target_support' key receives an optional zero-init
residual-logit correction from ACCF-v3. Both branches start as exact identities.
No GT coordinates are read. Alignment predictions are detached from this NEW
condition route by default to protect the geometry training objectives.
"""
from __future__ import annotations
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class ACCFP3V3(nn.Module):
    def __init__(self, channels: int, hidden: int = 48,
                 use_alignment: bool = True, residual_scale: float = .25,
                 alignment_gradient_scale: float = 0.0):
        super().__init__()
        if channels <= 0 or hidden < 8 or hidden % 8:
            raise ValueError('Expected channels>0 and hidden multiple of 8, >=8')
        if not (0.0 <= alignment_gradient_scale <= 1.0):
            raise ValueError('alignment_gradient_scale must lie in [0,1]')
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.use_alignment = bool(use_alignment)
        self.residual_scale = float(residual_scale)
        self.alignment_gradient_scale = float(alignment_gradient_scale)

        def proj():
            return nn.Sequential(
                nn.Conv2d(channels, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden), nn.SiLU())

        self.rgb_projection = proj()
        self.tir_projection = proj()
        self.gate = nn.Sequential(
            nn.Conv2d(3*hidden + 12, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1), nn.Sigmoid())
        self.complement = nn.Sequential(
            nn.Conv2d(3*hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU())
        self.output_projection = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        # A separate pathway lets geometry affect WHICH high-resolution detail
        # BHLR selects, rather than only modulating low-resolution P3 fusion.
        # Its last convolution is zero-initialized, so initial BHLR guidance
        # and original Post-Fusion model behavior are exactly preserved.
        self.detail_guidance_delta = nn.Sequential(
            nn.Conv2d(14, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.detail_guidance_delta[-1].weight)
        nn.init.zeros_(self.detail_guidance_delta[-1].bias)
        self._cache_cond = None
        self._cache_gate = None
        self.last_stats = {}

    @staticmethod
    def _vector(info, name, ref, dims, fill=0):
        x = info.get(name)
        n = ref.shape[0]
        if not torch.is_tensor(x):
            return ref.new_full((n, dims), float(fill))
        x = x.reshape(n, -1)
        if x.shape[1] < dims:
            raise RuntimeError(f'ACCF-v3 {name} shape inconsistent: {tuple(x.shape)}')
        return x[:, :dims].to(ref.dtype)

    def _conditions(self, rgb, pr, pt, info):
        b, _, h, w = rgb.shape
        if not self.use_alignment:
            return rgb.new_zeros((b, 12, h, w))
        if not isinstance(info, Mapping) or not info.get('enabled', False):
            raise RuntimeError('ACCF-v3 requires enabled alignment_info')
        guidance = (info.get('guidance_by_scale') or {}).get(3)
        if guidance is None:
            guidance = (info.get('guidance_by_scale') or {}).get('3')
        if isinstance(guidance, Mapping):
            vals = [guidance.get(k) for k in
                    ('rgb_target_support','aligned_tir_target_support','joint_target_support')]
            if all(torch.is_tensor(v) for v in vals):
                maps = torch.cat(vals, dim=1)
            else:
                maps = info.get('multi_target_guidance')
        else:
            maps = info.get('multi_target_guidance')
        if not torch.is_tensor(maps) or maps.ndim != 4 or maps.shape[1] != 3:
            raise RuntimeError('ACCF-v3: cannot find predicted RGB/TIR/joint support maps')
        if maps.shape[-2:] != (h, w):
            maps = F.interpolate(maps, (h, w), mode='bilinear', align_corners=False)
        maps = maps.to(dtype=rgb.dtype).clamp(0,1)
        mr, mt, joint = maps[:,0:1],maps[:,1:2],maps[:,2:3]
        union = 1 - (1-mr)*(1-mt)
        mismatch = (mr-mt).abs()
        conf = self._vector(info, 'coarse_confidence', rgb, 1)
        conf = conf.clamp(0,1).reshape(b,1,1,1).expand(b,1,h,w)
        scale = self._vector(info, 'sampling_scale', rgb, 2, fill=1)
        shift = self._vector(info, 'sampling_translation', rgb, 2)
        xx = torch.linspace(0,1,w,device=rgb.device,dtype=rgb.dtype).reshape(1,1,1,w)
        yy = torch.linspace(0,1,h,device=rgb.device,dtype=rgb.dtype).reshape(1,1,h,1)
        sx = scale[:,0:1,None,None]*xx+shift[:,0:1,None,None]
        sy = scale[:,1:2,None,None]*yy+shift[:,1:2,None,None]
        valid = (torch.sigmoid(32*sx)*torch.sigmoid(32*(1-sx))*
                 torch.sigmoid(32*sy)*torch.sigmoid(32*(1-sy))).expand(b,1,h,w)
        # This cosine is auxiliary FEATURE AGREEMENT, not geometric correctness.
        cosine = (F.normalize(pr.float(),dim=1,eps=1e-5)*
                  F.normalize(pt.float(),dim=1,eps=1e-5)).sum(1,keepdim=True).to(rgb.dtype)
        norm_shift = (shift*2).tanh().reshape(b,2,1,1).expand(b,2,h,w)
        norm_scale = ((scale-1)*2).tanh().reshape(b,2,1,1).expand(b,2,h,w)
        cond = torch.cat([mr,mt,joint,union,mismatch,conf,valid,cosine,
                          norm_shift,norm_scale],1)
        if cond.shape != (b,12,h,w):
            raise RuntimeError(f'ACCF-v3 unexpected condition shape {tuple(cond.shape)}')
        k = self.alignment_gradient_scale
        # k=0 cuts ONLY this newly introduced condition-gradient path, leaving
        # all existing Warp/Detection/Alignment gradients unchanged.
        if k == 0.0:
            return cond.detach()
        return cond.detach() + k * (cond - cond.detach())

    def forward(self, fused, rgb, aligned_tir, alignment_info=None):
        if fused.shape != rgb.shape or rgb.shape != aligned_tir.shape:
            raise ValueError('ACCF-v3 requires matching RGB/TIR/fused P3 feature shapes')
        if rgb.shape[1] != self.channels:
            raise ValueError(f'ACCF-v3 expected {self.channels} P3 channels')
        pr = self.rgb_projection(rgb)
        pt = self.tir_projection(aligned_tir)
        pair = torch.cat([pr,pt,(pr-pt).abs()],dim=1)
        cond = self._conditions(rgb,pr,pt,alignment_info)
        gate = self.gate(torch.cat([pair,cond],dim=1))
        delta = self.residual_scale*self.output_projection(self.complement(pair)*gate)
        self._cache_cond = cond
        self._cache_gate = gate.mean(dim=1,keepdim=True)
        self.last_stats = {}
        with torch.no_grad():
            self.last_stats.update({
                'residual_abs': float(delta.detach().float().abs().mean().item()),
                'fused_abs': float(fused.detach().float().abs().mean().item()),
                'gate_mean': float(gate.detach().float().mean().item()),
                'gate_std': float(gate.detach().float().std().item()),
                'condition_abs': float(cond.detach().float().abs().mean().item()),
            })
        return fused + delta

    def refine_bhlr_guidance(self, original_guidance):
        """Called at original BHLR P3 guidance creation site, not a new BHLR site."""
        if self._cache_cond is None or self._cache_gate is None:
            raise RuntimeError('ACCF-v3 must run before Post-Fusion BHLR')
        s = original_guidance
        cond = self._cache_cond
        gate = self._cache_gate
        if s.shape[-2:] != cond.shape[-2:]:
            cond = F.interpolate(cond, s.shape[-2:], mode='bilinear',align_corners=False)
            gate = F.interpolate(gate, s.shape[-2:], mode='bilinear',align_corners=False)
        if s.shape[0] != cond.shape[0] or s.shape[1] != 1:
            raise RuntimeError('Unexpected BHLR guidance shape')
        predictor = torch.cat([s,gate,cond],dim=1) # 1 + 1 + 12 = 14
        residual_logit = self.detail_guidance_delta(predictor)
        # Zero residual at init; epsilon-safe logit transform.
        eps = 1e-5
        safe = s.clamp(eps,1-eps)
        base_logit = torch.log(safe)-torch.log1p(-safe)
        adjusted = torch.sigmoid(base_logit + residual_logit)
        # With s exactly 0/1, logit-clamping introduces tiny (~1e-5) mismatch.
        # At zero-init, preserve exact original support including endpoints.
        out = s + (adjusted-torch.sigmoid(base_logit))
        with torch.no_grad():
            self.last_stats['detail_guidance_delta_abs'] = float((out-s).detach().float().abs().mean().item())
            self.last_stats['detail_guidance_mean'] = float(out.detach().float().mean().item())
        return out.clamp(0,1)
