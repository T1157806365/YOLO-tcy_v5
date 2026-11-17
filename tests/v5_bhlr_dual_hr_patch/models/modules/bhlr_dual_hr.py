"""Optional, checkpoint-reconstructable Dual-HR extension for YOLO-tcy_v5.

Original BHLR-v2 is untouched unless model.bhlr_dual_hr.enabled=True.
The extension supports:
  clean  : strict high->low->high reconstruction residual, original detail_stem
  dual   : clean residual bank + learned high-RGB structure bank
  guided : dual + 3-channel RGB/TIR/joint alignment-guided structure selection

This code is designed for the single P3 BHLR branch used by UAV-CB HVAD.
No alternate detector, decoder, neck or GTGA is introduced.
"""
from __future__ import annotations

from types import MethodType
from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.modules.bhlr_v2 import SharedLostDetailEncoder


MODES = {"clean", "dual", "guided"}


class DualHRDetailMixer(nn.Module):
    """Keep the original detail tensor as an identity-initialized baseline.

    For dual/guided modes:
        detail_out = residual_detail + gain * gate * structure_detail
        gain = max_gain * tanh(gamma_raw), gamma_raw starts at 0.
    Shapes: residual/structure/out [B,D,320,320] for RGB 1920 / 640.
    """

    def __init__(
        self,
        mode: str,
        ratio: int,
        detail_channels: int,
        stem_channels: int,
        gate_hidden: int = 24,
        max_gain: float = 0.50,
        detach_guidance: bool = True,
        min_guided_prior: float = 0.15,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in MODES:
            raise ValueError(f"Unsupported mode={mode!r}; expected {sorted(MODES)}")
        self.ratio = int(ratio)
        self.detail_channels = int(detail_channels)
        self.stem_channels = int(stem_channels)
        self.max_gain = float(max_gain)
        self.detach_guidance = bool(detach_guidance)
        self.min_guided_prior = float(min_guided_prior)
        if self.ratio < 2:
            raise ValueError("This extension requires high/semantic ratio >= 2")
        if not (0.0 <= self.min_guided_prior <= 1.0):
            raise ValueError("min_guided_prior must be within [0,1]")
        if self.max_gain <= 0:
            raise ValueError("max_gain must be > 0")

        if self.mode in {"dual", "guided"}:
            self.structure_encoder = SharedLostDetailEncoder(
                ratio=self.ratio,
                detail_channels=self.detail_channels,
                stem_channels=self.stem_channels,
            )
            guide_c = 3 if self.mode == "guided" else 0
            h = int(gate_hidden)
            self.selector = nn.Sequential(
                nn.Conv2d(2 * self.detail_channels + guide_c, h, 1),
                nn.SiLU(inplace=False),
                nn.Conv2d(h, 1, 1),
            )
            # Gate begins at 0.5 everywhere; global gain begins at zero.
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)
            self.gamma_raw = nn.Parameter(torch.zeros(()))
        else:
            # Clean-only has no trainable new parameters.
            self.structure_encoder = None
            self.selector = None
            self.register_parameter("gamma_raw", None)

        self.last_stats: Dict[str, torch.Tensor] = {}

    def encode_structure(self, rgb_high: torch.Tensor) -> Optional[torch.Tensor]:
        if self.structure_encoder is None:
            return None
        if rgb_high.shape[-2] % self.ratio or rgb_high.shape[-1] % self.ratio:
            raise ValueError("High resolution must be divisible by ratio")
        raw_s2d = F.pixel_unshuffle(rgb_high, self.ratio)
        return self.structure_encoder(raw_s2d)

    def _guidance(
        self,
        external_guidance: Optional[Mapping[str, torch.Tensor]],
        reference: torch.Tensor,
    ):
        if self.mode != "guided":
            return None, None
        if not isinstance(external_guidance, Mapping):
            raise RuntimeError("Guided Dual-HR needs GTGA external_guidance")
        names = ("rgb_target_support", "aligned_tir_target_support", "original_joint_target_support")
        out = []
        for key in names:
            x = external_guidance.get(key)
            if not torch.is_tensor(x):
                raise KeyError(f"Missing GTGA guidance map: {key}")
            if x.ndim != 4 or x.shape[1] != 1 or x.shape[0] != reference.shape[0]:
                raise ValueError(f"Guidance {key} must be [B,1,H,W], got {tuple(x.shape)}")
            if self.detach_guidance:
                x = x.detach()
            x = x.to(device=reference.device, dtype=reference.dtype).clamp(0, 1)
            if x.shape[-2:] != reference.shape[-2:]:
                x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
            out.append(x)
        guide = torch.cat(out, dim=1)
        # RGB-only evidence remains eligible even if TIR correspondence fails.
        reliability = torch.maximum(out[0], out[2])
        prior = self.min_guided_prior + (1.0 - self.min_guided_prior) * reliability
        return guide, prior

    def fuse(
        self,
        residual_detail: torch.Tensor,
        structure_detail: Optional[torch.Tensor],
        external_guidance: Optional[Mapping[str, torch.Tensor]],
    ) -> torch.Tensor:
        if self.mode == "clean":
            zero = residual_detail.new_zeros(())
            self.last_stats = {
                "dualhr_gamma": zero.detach(),
                "dualhr_gate": zero.detach(),
                "dualhr_delta_l1": zero.detach(),
                "dualhr_residual_l1": residual_detail.detach().float().abs().mean(),
                "dualhr_structure_l1": zero.detach(),
                "dualhr_prior": zero.detach(),
            }
            return residual_detail
        if structure_detail is None:
            raise RuntimeError("Missing structure detail: call build_lost_detail_bank before enhance_layer")
        if residual_detail.shape != structure_detail.shape:
            raise ValueError(f"Dual-HR shape mismatch: {tuple(residual_detail.shape)} vs {tuple(structure_detail.shape)}")
        guide, prior = self._guidance(external_guidance, residual_detail)
        parts = [residual_detail, structure_detail]
        if guide is not None:
            parts.append(guide)
        gate = torch.sigmoid(self.selector(torch.cat(parts, dim=1)))
        if prior is not None:
            gate = gate * prior
        gain = self.max_gain * torch.tanh(self.gamma_raw)
        delta = gain * gate * structure_detail
        out = residual_detail + delta
        with torch.no_grad():
            self.last_stats = {
                "dualhr_gamma": gain.detach().float(),
                "dualhr_gate": gate.detach().float().mean(),
                "dualhr_delta_l1": delta.detach().float().abs().mean(),
                "dualhr_residual_l1": residual_detail.detach().float().abs().mean(),
                "dualhr_structure_l1": structure_detail.detach().float().abs().mean(),
                "dualhr_prior": (
                    prior.detach().float().mean() if prior is not None else delta.new_ones(()).float()
                ),
            }
        return out


def attach_dual_hr(model: nn.Module, cfg: Mapping, *, verbose: bool = True) -> nn.Module:
    """Attach once after standard factory creation (including optional HVAD).

    Method patches affect this model instance only, not the original classes.
    Extra trainable parameters are stored under bhlr.dual_hr.* in checkpoint.
    """
    cfg = dict(cfg or {})
    if not cfg.get("enabled", False):
        return model
    if not getattr(model, "bhlr_enabled", False) or getattr(model, "bhlr", None) is None:
        raise RuntimeError("Dual-HR needs an enabled BHLR-v2 model")
    bhlr = model.bhlr
    if hasattr(bhlr, "dual_hr"):
        raise RuntimeError("Dual-HR is already attached")
    if len(getattr(bhlr, "scales", ())) != 1:
        raise RuntimeError("Current Dual-HR extension supports one BHLR scale (P3) only")
    if tuple(getattr(model, "bhlr_scales", ())) != (3,):
        raise RuntimeError("Current Dual-HR extension requires BHLR conceptual scale P3 only")
    if not hasattr(model, "alignment") and cfg.get("mode") == "guided":
        raise RuntimeError("Guided mode requires GTGA alignment")
    if cfg.get("mode") == "guided" and getattr(model, "bhlr_guidance_mode", None) != "alignment":
        raise RuntimeError("Guided mode requires model.bhlr.guidance_mode: alignment")

    detail_c = int(getattr(model, "_dualhr_detail_channels", cfg.get("detail_channels", 48)))
    stem_c = int(cfg.get("stem_channels", 16))
    # Validate against original detail_stem construction using known first/last ConvBNAct layers.
    detail_stem = bhlr.detail_stem
    try:
        # YOLO-tcy_v5 ConvBNAct keeps its Conv2d in .block[0].
        first_conv = detail_stem.net[0].block[0]
        last_conv = detail_stem.net[-1].block[0]
        actual_stem = int(first_conv.out_channels)
        actual_out = int(last_conv.out_channels)
    except (AttributeError, IndexError, TypeError):
        # Permit known-compatible wrapper subclasses; shapes are checked in fuse().
        actual_stem, actual_out = stem_c, detail_c
    if actual_out != detail_c or actual_stem != stem_c:
        raise ValueError(
            f"BHLR detail_stem channel mismatch: original=({actual_stem},{actual_out}) "
            f"dualhr=({stem_c},{detail_c})"
        )
    if int(bhlr.ratio) != int(cfg.get("ratio", bhlr.ratio)):
        raise ValueError("Dual-HR ratio must equal existing BHLR ratio")

    extension = DualHRDetailMixer(
        mode=cfg.get("mode", "guided"),
        ratio=bhlr.ratio,
        detail_channels=detail_c,
        stem_channels=stem_c,
        gate_hidden=int(cfg.get("gate_hidden", 24)),
        max_gain=float(cfg.get("max_gain", 0.50)),
        detach_guidance=bool(cfg.get("detach_guidance", True)),
        min_guided_prior=float(cfg.get("min_guided_prior", 0.15)),
    )
    bhlr.dual_hr = extension
    bhlr._dualhr_structure_cache = None

    old_enhance = bhlr.enhance_layer
    old_loss = model.loss
    old_return_debug = bool(cfg.get("return_debug", False))

    def build_with_dual(self, rgb_high, rgb_semantic):
        # All three new ablations use the same strictly paired clean residual.
        low = F.interpolate(rgb_high.float(), size=rgb_semantic.shape[-2:], mode="area")
        recon = F.interpolate(low, size=rgb_high.shape[-2:], mode="bilinear", align_corners=False)
        recon = recon.to(dtype=rgb_high.dtype)
        residual = rgb_high - recon
        residual_s2d = F.pixel_unshuffle(residual, downscale_factor=self.ratio)
        raw = self.detail_stem(residual_s2d)
        self._dualhr_structure_cache = self.dual_hr.encode_structure(rgb_high)
        return raw, recon, residual

    def enhance_with_dual(
        self, layer_id, detail_raw, rgb_feat, tir_feat,
        external_guidance=None, return_debug=False,
    ):
        detail_fused = self.dual_hr.fuse(detail_raw, self._dualhr_structure_cache, external_guidance)
        self._dualhr_structure_cache = None  # release large activation as soon as consumed
        # HVAD's original hook observes residual bank; update it to the final
        # detail bank so its hypothesis verifier sees the same feature used by BHLR.
        if getattr(model, "hvad", None) is not None:
            model.hvad._detail_bank = detail_fused
        result = old_enhance(
            layer_id=layer_id,
            detail_raw=detail_fused,
            rgb_feat=rgb_feat,
            tir_feat=tir_feat,
            external_guidance=external_guidance,
            return_debug=return_debug,
        )
        if return_debug and old_return_debug:
            enh, dbg = result
            dbg["dualhr_stats"] = dict(self.dual_hr.last_stats)
            return enh, dbg
        return result

    def loss_with_monitor(self, batch, preds=None):
        raw, items = old_loss(batch, preds=preds)
        if preds is not None:
            return raw, items
        if isinstance(items, dict) and self.bhlr.dual_hr.last_stats:
            items = dict(items)
            items.update(self.bhlr.dual_hr.last_stats)
        return raw, items

    bhlr.build_lost_detail_bank = MethodType(build_with_dual, bhlr)
    bhlr.enhance_layer = MethodType(enhance_with_dual, bhlr)
    model.loss = MethodType(loss_with_monitor, model)
    if verbose:
        print(
            f"[BHLR DUAL-HR] P3 mode={extension.mode} ratio={extension.ratio} "
            f"detail={detail_c} max_gain={extension.max_gain:.3f} "
            "clean-residual=True; original neck/head/GTGA unchanged",
            flush=True,
        )
    return model
