"""P3 high-resolution regression residual, not a new detector.

The HR feature is the existing 320x320 BHLR detail bank (1920->640 S2D->320).
A tiny 1x1 compressor and pixel-unshuffle(4) preserve the spatial ordering
of 4x4 subcells in the final 80x80 P3 representation. Corrections are to
YOLO26 raw P3 LTRB regression channels, before decoding and before the loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AGHLRP3RegAdapter(nn.Module):
    def __init__(self, neck_channels: int, detail_channels: int = 48,
                 hidden_channels: int = 24, pre_s2d_channels: int = 8,
                 subcell_ratio: int = 4, max_offset: float = 0.50):
        super().__init__()
        if subcell_ratio < 1 or hidden_channels < 4 or pre_s2d_channels < 1:
            raise ValueError('Invalid AGHLR channels/ratio')
        if not (0 < max_offset <= 2.0):
            raise ValueError('max_offset must be in (0,2] P3 grid units')
        self.ratio = int(subcell_ratio)
        self.max_offset = float(max_offset)
        c = int(hidden_channels)
        self.pre_s2d = nn.Sequential(
            nn.Conv2d(int(detail_channels), int(pre_s2d_channels), 1, bias=False),
            nn.BatchNorm2d(int(pre_s2d_channels)),
            nn.SiLU(inplace=False),
        )
        self.detail_to_p3 = nn.Sequential(
            nn.Conv2d(int(pre_s2d_channels)*self.ratio**2, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=False),
        )
        self.semantic_to_p3 = nn.Sequential(
            nn.Conv2d(int(neck_channels), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=False),
        )
        # 4x hidden channels + 2 spatial guidance maps (alignment, selector).
        self.fuse = nn.Sequential(
            nn.Conv2d(4*c + 2, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=False),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=False),
        )
        self.regression = nn.Conv2d(c, 4, 1, bias=True)
        # Exact zero correction before the first SGD update, preserving
        # the teacher-initialized regression logits at Epoch 1.
        nn.init.zeros_(self.regression.weight)
        nn.init.zeros_(self.regression.bias)

    def forward(self, neck_p3: torch.Tensor, detail_320: torch.Tensor,
                alignment_support: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        if neck_p3.ndim != 4 or detail_320.ndim != 4:
            raise ValueError('Expect 4-D P3 and BHLR details')
        b, _, h, w = neck_p3.shape
        r = self.ratio
        if detail_320.shape[0] != b or detail_320.shape[-2:] != (h*r, w*r):
            raise RuntimeError(f'AGHLR needs detail {(h*r,w*r)} to match P3 {(h,w)}; '
                               f'got {tuple(detail_320.shape)}')
        if alignment_support.ndim != 4 or selector.ndim != 4 or (
            alignment_support.shape[0] != b or selector.shape[0] != b
        ):
            raise ValueError('AGHLR guidance expected [B,1,H,W]')
        if alignment_support.shape[1] != 1 or selector.shape[1] != 1:
            raise ValueError('Guidance/selector must have one channel')
        a = F.interpolate(alignment_support.float(), size=(h,w), mode='bilinear',
                          align_corners=False).to(neck_p3.dtype)
        s = F.interpolate(selector.float(), size=(h,w), mode='bilinear',
                          align_corners=False).to(neck_p3.dtype)
        x = self.pre_s2d(detail_320)
        x = F.pixel_unshuffle(x, r)
        d = self.detail_to_p3(x)
        n = self.semantic_to_p3(neck_p3)
        joint = torch.cat((n, d, n*d, (n-d).abs(), a, s), dim=1)
        z = self.fuse(joint)
        # delta is in head's native l/t/r/b grid-distance units, not xyxy pixels.
        return self.max_offset * torch.tanh(self.regression(z))
