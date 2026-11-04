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
                 subcell_ratio: int = 4, max_offset: float = 0.50,
                 sampling_mode: str = 'legacy', sampling_radius_p3: int = 2,
                 sampling_temperature: float = 10.0,
                 sampling_distance_penalty: float = 0.75,
                 sampling_max_shift_p3: float = 0.75,
                 sampling_gain: float = 0.75):
        super().__init__()
        if subcell_ratio < 1 or hidden_channels < 4 or pre_s2d_channels < 1:
            raise ValueError('Invalid AGHLR channels/ratio')
        if not (0 < max_offset <= 2.0):
            raise ValueError('max_offset must be in (0,2] P3 grid units')
        self.ratio = int(subcell_ratio)
        self.max_offset = float(max_offset)
        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in ('legacy', 'support_spatial'):
            raise ValueError('sampling_mode must be legacy or support_spatial')
        self.sampling_radius_p3 = int(sampling_radius_p3)
        self.sampling_temperature = float(sampling_temperature)
        self.sampling_distance_penalty = float(sampling_distance_penalty)
        self.sampling_max_shift_p3 = float(sampling_max_shift_p3)
        self.sampling_gain = float(sampling_gain)
        if not 1 <= self.sampling_radius_p3 <= 4:
            raise ValueError('sampling_radius_p3 must be in [1,4]')
        if self.sampling_temperature <= 0 or self.sampling_distance_penalty < 0:
            raise ValueError('Bad sampler softmax parameters')
        if not (0 < self.sampling_max_shift_p3 <= 2):
            raise ValueError('sampling_max_shift_p3 must be in (0,2]')
        if not (0 <= self.sampling_gain <= 2):
            raise ValueError('sampling_gain must be in [0,2]')
        self.last_sampling_stats = {}  # diagnostics, intentionally not in state_dict
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

    def _offset_from_aligned_support(self, aligned_support: torch.Tensor):
        """Find local aligned-support centroids in P3 pixel coordinates.

        This map is a *spatial support map*, not a GMLS/DMLS geometric flow.
        A local soft-argmax makes aligned target evidence control WHERE we
        resample the RGB HR detail bank. Constant/zero support => zero offset.
        """
        b, _, h, w = aligned_support.shape
        r = self.sampling_radius_p3
        k = 2 * r + 1
        s = torch.tanh(aligned_support.float())
        # Replication padding avoids border bias on a constant support map.
        patches = F.unfold(F.pad(s, (r, r, r, r), mode='replicate'),
                           kernel_size=k).reshape(b, k * k, h, w)
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=s.dtype, device=s.device),
            torch.arange(-r, r + 1, dtype=s.dtype, device=s.device), indexing='ij'
        )
        x = x.reshape(1, -1, 1, 1)
        y = y.reshape(1, -1, 1, 1)
        dist = x.square() + y.square()
        logits = self.sampling_temperature * (patches - s) - self.sampling_distance_penalty * dist
        weights = logits.softmax(dim=1)
        dx = (weights * x).sum(dim=1, keepdim=True)
        dy = (weights * y).sum(dim=1, keepdim=True)
        flow = torch.cat((dx, dy), dim=1)
        # Norm clamp preserves the direction of local spatial evidence.
        norm = flow.square().sum(dim=1, keepdim=True).add(1e-12).sqrt()
        flow = flow * (self.sampling_max_shift_p3 / norm).clamp(max=1)
        return flow

    def _sample_hr_by_support(self, hr_feature: torch.Tensor, support_p3: torch.Tensor):
        """Differentiable P3-aligned spatial readout of the 320x320 HR bank.

        No interpolation of the ORIGINAL 1920 RGB image is performed here:
        we sample BHLR's 320x320 high-resolution detail features BEFORE
        pixel_unshuffle(4), so the 4x4 subcell evidence remains available.
        """
        b, _, hh, ww = hr_feature.shape
        _, _, h, w = support_p3.shape
        if hh != h * self.ratio or ww != w * self.ratio:
            raise RuntimeError('Sampling expects HR bank spatial size = ratio * P3')
        flow_p3 = self._offset_from_aligned_support(support_p3)
        flow_hr = F.interpolate(flow_p3, size=(hh, ww),
                                mode='bilinear', align_corners=False)
        yy, xx = torch.meshgrid(
            torch.arange(hh, device=hr_feature.device, dtype=torch.float32),
            torch.arange(ww, device=hr_feature.device, dtype=torch.float32), indexing='ij'
        )
        base_x = (xx + 0.5) * (2.0 / ww) - 1.0
        base_y = (yy + 0.5) * (2.0 / hh) - 1.0
        # One P3 cell = 'ratio' HR cells. Normalized displacement = 2*delta/Wp3.
        grid = torch.stack((
            base_x[None] + 2.0 * flow_hr[:, 0] / w,
            base_y[None] + 2.0 * flow_hr[:, 1] / h,
        ), dim=-1)
        # float32 avoids bf16 CPU grid_sample restrictions and unstable grids.
        sampled = F.grid_sample(hr_feature.float(), grid, mode='bilinear',
                                padding_mode='border', align_corners=False)
        with torch.no_grad():
            self.last_sampling_stats = {
                'mean_shift_p3': float(flow_p3.detach().float().norm(dim=1).mean().cpu()),
                'max_shift_p3': float(flow_p3.detach().float().norm(dim=1).amax().cpu()),
                'mean_support': float(support_p3.detach().float().mean().cpu()),
                'support_std': float(support_p3.detach().float().std(unbiased=False).cpu()),
            }
        return sampled.to(dtype=hr_feature.dtype)

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
        if self.sampling_mode == 'support_spatial':
            # CRITICAL: aligned RGB-T spatial evidence controls HR sampling
            # BEFORE high-resolution features are aggregated to P3 cells.
            x = self._sample_hr_by_support(x, a)
        else:
            self.last_sampling_stats = {}
        x = F.pixel_unshuffle(x, r)
        d = self.detail_to_p3(x)
        if self.sampling_mode == 'support_spatial':
            # Auxiliary spatial readout gate. For zero/constant support it is
            # exactly 1, so 'zero guidance' remains a clean ablation.
            local_context = F.avg_pool2d(a.float(), 5, stride=1, padding=2)
            gate = 1.0 + self.sampling_gain * torch.tanh((a.float() - local_context) * 2.0)
            d = d * gate.to(dtype=d.dtype)
        n = self.semantic_to_p3(neck_p3)
        joint = torch.cat((n, d, n*d, (n-d).abs(), a, s), dim=1)
        z = self.fuse(joint)
        # delta is in head's native l/t/r/b grid-distance units, not xyxy pixels.
        return self.max_offset * torch.tanh(self.regression(z))
