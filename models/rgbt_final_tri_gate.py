"""P3 RGB/TIR/HR residual gate for the original GTGA+BHLR model.

Only this extra model variant uses the gate. Original models and checkpoints
remain structurally untouched. Official YOLO26m weights still init both streams.
"""
from __future__ import annotations

import torch
from torch import nn

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)


class ResidualThreeWayGate(nn.Module):
    """Independent spatial weights, EXACTLY one at initialization.

    F = F_old + (w_R-1)*R + (w_T-1)*T + (w_H-1)*H
    F_old = R + T + H  (in the original AddFusion+BHLR path)
    H = RGB_after_BHLR - RGB_before_BHLR
    T = original_fused - RGB_after_BHLR  (includes any AddFusion projection)
    """

    def __init__(self, channels: int, hidden: int = 64, alpha: float = 0.5):
        super().__init__()
        if int(channels) < 1 or int(hidden) < 1:
            raise ValueError("channels and hidden must be positive")
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must be within (0, 1]")
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Conv2d(3 * int(channels), int(hidden), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden), int(hidden), 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden), 3, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb, tir, hr, base):
        if not (rgb.shape == tir.shape == hr.shape == base.shape):
            raise ValueError(
                "TriGate shape mismatch: "
                + ", ".join(str(x.shape) for x in (rgb, tir, hr, base))
            )
        logits = self.net(torch.cat((rgb, tir, hr), dim=1))
        weights = 1.0 + self.alpha * torch.tanh(logits)
        wr, wt, wh = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        # Preserve original feature sum bit-for-bit at zero initialization.
        fused = base + ((wr - 1.0) * rgb + (wt - 1.0) * tir + (wh - 1.0) * hr)
        return fused, (wr, wt, wh)


class RGBTRLSFAGTGACenteredBHLRV2TriGateModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    """Add a single trainable P3 gate after the unchanged GTGA and BHLR."""

    def __init__(self, *args, tri_gate_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        gate_cfg = dict(tri_gate_cfg or {})
        if not bool(gate_cfg.get("enabled", False)):
            raise ValueError("TriGate subclass requires tri_gate.enabled: true")
        if self.fusion_type != "add":
            raise ValueError("TriGate v1 requires model.fusion: add")
        if not (self.alignment_enabled and self.bhlr_enabled):
            raise ValueError("TriGate v1 requires GTGA and BHLR enabled")
        scale = int(gate_cfg.get("scale", 3))
        if scale != 3 or scale not in self.bhlr_scales:
            raise ValueError("TriGate v1 is P3-only and requires BHLR at P3")
        self.tri_gate_scale = scale
        self.tri_gate_index = int(self.scale_to_index[scale])
        # Uses the existing channel inference, no new backbone/neck.
        c = self._infer_any_layer_channels((self.tri_gate_index,))
        channels = int(c[self.tri_gate_index]["rgb"])
        self.final_tri_gate = ResidualThreeWayGate(
            channels,
            hidden=int(gate_cfg.get("hidden", 64)),
            alpha=float(gate_cfg.get("alpha", 0.5)),
        )
        self.tri_gate_disabled = False
        self._tri_monitor = {}
        self._tri_monitor_n = 0
        print(
            "[FINAL TRI-FUSION] P3 only, R/T/HR; "
            f"channels={channels}, alpha={self.final_tri_gate.alpha:.2f}, "
            "identity init=True"
        )

    def set_tri_gate_disabled(self, disabled: bool):
        self.tri_gate_disabled = bool(disabled)

    def forward_features(self, *args, **kwargs):
        feat = super().forward_features(*args, **kwargs)
        idx = self.tri_gate_index
        original_rgb = feat["rgb_backbone_raw"][idx]
        enhanced_rgb = feat["rgb_backbone"][idx]
        old_fused = feat["fused_backbone"][idx]
        hr_residual = enhanced_rgb - original_rgb
        # Extract the EXACT AddFusion output of aligned TIR from the old sum,
        # including any existing channel projection. Do not re-run its BN.
        tir_add = old_fused - enhanced_rgb
        if self.tri_gate_disabled:
            new_fused = old_fused
            # no need to execute the gate during gate-off validation
        else:
            new_fused, weights = self.final_tri_gate(
                original_rgb, tir_add, hr_residual, old_fused
            )
            if self.training and torch.is_grad_enabled():
                measures = {
                    "w_rgb": weights[0].detach().float().mean(),
                    "w_tir": weights[1].detach().float().mean(),
                    "w_hr": weights[2].detach().float().mean(),
                    "fusion_delta_l1": (new_fused - old_fused).detach().float().abs().mean(),
                    "hr_delta_l1": hr_residual.detach().float().abs().mean(),
                }
                for name, value in measures.items():
                    old = self._tri_monitor.get(name)
                    self._tri_monitor[name] = value if old is None else old + value
                self._tri_monitor_n += 1
        feat["fused_backbone"][idx] = new_fused
        feat["fusion_features"][idx] = new_fused
        return feat

    @torch.no_grad()
    def tri_gate_monitor(self, reset: bool = True):
        count = int(self._tri_monitor_n)
        vals = {
            name: (float(value.item()) / count if count > 0 else float("nan"))
            for name, value in self._tri_monitor.items()
        }
        if reset:
            self._tri_monitor = {}
            self._tri_monitor_n = 0
        return count, vals
