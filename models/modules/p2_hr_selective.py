"""Selective high-resolution residual injection into native YOLO26m P2 feature.

Keeps identity at initialization. Reuses existing BHLR detail bank;
no new image encoder, no geometry loss or distillation.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class P2SelectiveHR(nn.Module):
    def __init__(self, p2_channels: int = 128, detail_channels: int = 48,
                 hidden: int = 32, gamma_init: float = 0.0):
        super().__init__()
        self.p2_channels = int(p2_channels)
        self.detail_channels = int(detail_channels)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(self.detail_channels, self.p2_channels, 1, bias=False),
            nn.BatchNorm2d(self.p2_channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(self.p2_channels * 2 + 1, int(hidden), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden), 1, 1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, p2, detail_raw, support=None):
        if p2.ndim != 4 or p2.shape[1] != self.p2_channels:
            raise ValueError(f'P2 expected C={self.p2_channels}, got {tuple(p2.shape)}')
        if detail_raw.ndim != 4 or detail_raw.shape[1] != self.detail_channels:
            raise ValueError(f'BHLR detail expected C={self.detail_channels}, got {tuple(detail_raw.shape)}')
        hw = p2.shape[-2:]
        detail = F.adaptive_avg_pool2d(detail_raw, hw)
        detail = self.detail_proj(detail)
        if support is None:
            support = p2.new_zeros(p2.shape[0], 1, *hw)
        else:
            support = F.interpolate(support.detach(), size=hw, mode='bilinear', align_corners=False)
            support = support.clamp(0, 1).to(dtype=p2.dtype)
        gate = torch.sigmoid(self.gate(torch.cat((p2, detail, support), dim=1)))
        gamma = torch.tanh(self.gamma_raw)
        delta = gamma * gate * detail
        enhanced = p2 + delta
        return enhanced, {
            'p2_hr_gamma': gamma,
            'p2_hr_gate': gate.detach().float().mean(),
            'p2_hr_delta_l1': delta.detach().float().abs().mean(),
            'p2_hr_detail_l1': detail.detach().float().abs().mean(),
        }
