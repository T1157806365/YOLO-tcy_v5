"""P3 three-branch *gain correction* that is exactly identity at initialization.

Base: F0 = RGB + projected(aligned TIR) + original Post-Fusion BHLR residual.
New : F  = F0 + a_R*RGB + a_T*TIR_term + a_H*HR_residual.
Coefficients a_i = tanh(logit_i) * max_change, initially all 0.
The alignment guidance is DETACHED before entering the gate, avoiding
unsafe sqrt(0) gradient paths; existing GTGA gradients are unchanged.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class AlignmentAwareTriGate(nn.Module):
    MODES = ('rgb_tir', 'rgb_tir_hr', 'align_rgb_tir_hr')
    ABLATIONS = ('none', 'off', 'rgb_off', 'tir_off', 'hr_off', 'no_alignment')

    def __init__(self, channels: int, mode: str, hidden: int = 24, max_change: float = 1.0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Invalid tri-gate mode: {mode}')
        if not (0 < float(max_change) <= 1):
            raise ValueError('max_change must be in (0,1]')
        self.mode = mode
        self.ablation = 'none'
        self.max_change = float(max_change)
        hidden = int(hidden)
        if hidden < 4:
            raise ValueError('hidden must be >= 4')
        self.proj_rgb = nn.Conv2d(channels, hidden, 1, bias=True)
        self.proj_tir = nn.Conv2d(channels, hidden, 1, bias=True)
        self.proj_hr = nn.Conv2d(channels, hidden, 1, bias=True)
        self.features = nn.Sequential(
            nn.Conv2d(hidden * 3 + 4, hidden, 3, padding=1, bias=True),
            nn.GroupNorm(4 if hidden % 4 == 0 else 1, hidden),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.SiLU(inplace=False),
        )
        self.logit = nn.Conv2d(hidden, 3, 1)
        # Crucial: new features are *exactly* old features initially.
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)
        self.last_stats = {}

    @staticmethod
    def _scaled(x: torch.Tensor):
        # Stop gradients through global magnitude normalization.
        x32 = x.float()
        denom = x32.detach().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(0.1)
        return x32 / denom

    @staticmethod
    def _map(value, ref):
        if value is None:
            return torch.zeros(ref.shape[0], 1, *ref.shape[-2:], device=ref.device, dtype=ref.dtype)
        if not torch.is_tensor(value):
            raise TypeError('Alignment guidance must be Tensor')
        value = value.detach().float()  # avoid sqrt(0) backward into aligner
        if value.ndim != 4 or value.shape[0] != ref.shape[0] or value.shape[1] != 1:
            raise ValueError(f'Invalid support map shape {tuple(value.shape)}, ref {tuple(ref.shape)}')
        if value.shape[-2:] != ref.shape[-2:]:
            value = F.interpolate(value, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)

    def _align_condition(self, info, ref):
        if self.mode != 'align_rgb_tir_hr' or self.ablation == 'no_alignment':
            return torch.zeros(ref.shape[0], 3, *ref.shape[-2:], device=ref.device, dtype=torch.float32)
        if not isinstance(info, dict) or not info.get('enabled', False):
            raise RuntimeError('Alignment-aware gate requires enabled RLSFA-GTGA')
        dbg = info.get('scale_debug', {})
        ref_dbg = dbg.get(3, dbg.get('3', {}))
        if not isinstance(ref_dbg, dict):
            ref_dbg = {}
        supports = info.get('support_by_scale', {})
        joint = supports.get(3, supports.get('3'))
        rgb = ref_dbg.get('rgb_local_targetness')
        tir = ref_dbg.get('aligned_tir_local_targetness')
        if joint is None or rgb is None or tir is None:
            raise RuntimeError('Missing P3 joint/RGB/TIR support maps; refuse silent zero-guidance')
        return torch.cat([self._map(rgb, ref), self._map(tir, ref), self._map(joint, ref)], dim=1)

    def forward(self, baseline, rgb_term, tir_term, hr_delta, alignment_info=None):
        if self.ablation not in self.ABLATIONS:
            raise ValueError(f'Invalid ablation {self.ablation!r}')
        if not (baseline.shape == rgb_term.shape == tir_term.shape == hr_delta.shape):
            raise ValueError('All P3 terms must have identical shape')
        if self.ablation == 'off':
            self.last_stats = {'w_rgb': 1.0, 'w_tir': 1.0, 'w_hr': 1.0,
                               'delta_abs': 0.0, 'alignment_cond_abs': 0.0}
            return baseline

        rgb_in = self.proj_rgb(self._scaled(rgb_term))
        tir_in = self.proj_tir(self._scaled(tir_term))
        hr_in = self.proj_hr(self._scaled(hr_delta))
        if self.mode == 'rgb_tir':
            hr_in = hr_in * 0.0  # parameter remains in graph, HR gate frozen at 1
        q = self._align_condition(alignment_info, baseline)
        # Relative disagreement can help identify conflict regions.
        diff = (self._scaled(rgb_term) - self._scaled(tir_term)).abs().mean(1, keepdim=True).clamp(0, 4)
        inp = torch.cat([rgb_in, tir_in, hr_in, q.to(rgb_in.dtype), diff.to(rgb_in.dtype)], dim=1)
        logits = self.logit(self.features(inp))
        delta = self.max_change * torch.tanh(logits.float())
        if self.mode == 'rgb_tir':
            delta = torch.cat([delta[:, :2], delta[:, 2:3] * 0.0], dim=1)
        # Fixed-checkpoint counterfactuals, not separately trained networks.
        if self.ablation in ('rgb_off', 'tir_off', 'hr_off'):
            idx = {'rgb_off': 0, 'tir_off': 1, 'hr_off': 2}[self.ablation]
            groups = [delta[:, k:k+1] for k in range(3)]
            groups[idx] = torch.ones_like(groups[idx]) * -1.0
            delta = torch.cat(groups, dim=1)
        changes = (delta[:, 0:1].to(baseline.dtype) * rgb_term
                   + delta[:, 1:2].to(baseline.dtype) * tir_term
                   + delta[:, 2:3].to(baseline.dtype) * hr_delta)
        result = baseline + changes
        if not torch.isfinite(result).all():
            raise FloatingPointError('Non-finite fused P3 in tri-gate')
        with torch.no_grad():
            self.last_stats = {
                'w_rgb': float((1.0 + delta[:, 0]).mean().item()),
                'w_tir': float((1.0 + delta[:, 1]).mean().item()),
                'w_hr': float((1.0 + delta[:, 2]).mean().item()),
                'delta_abs': float(changes.detach().abs().mean().item()),
                'alignment_cond_abs': float(q.detach().abs().mean().item()),
            }
        return result
