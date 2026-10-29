# -*- coding: utf-8 -*-
"""Independent P3/P4 reliability fusion variant; P5 stays EXACT baseline ADD.

Preserves the GMLS-v2 / DMLS + Fine + Scale + BHLR-v2 pipeline, losses,
YOLO26m head, and checkpoint conventions. Not an AHR2Net implementation.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from models.rgbt_reliability_fusion_model import (
    SpatialAlignmentReliabilityGate,
    _get_scale,
)


class RGBTAlignmentReliabilityP34FusionDetectionModel(
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel
):
    """P3/P4: learnable spatial gate; P5: unchanged original ADD.

    Only instantiate trainable gates for P3 and P4. This avoids unused P5
    parameters with DistributedDataParallel(find_unused_parameters=False).
    """

    def __init__(self, *args, reliability_fusion_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(reliability_fusion_cfg or {})
        if not bool(cfg.get("enabled", True)):
            raise ValueError("reliability_fusion.enabled must be true")
        if self.fusion_type != "add":
            raise ValueError("P34-ARF requires model.fusion: add")
        if not self.alignment_enabled:
            raise ValueError("P34-ARF requires enabled alignment")
        if not {3, 4, 5}.issubset(set(self.scale_to_index)):
            raise RuntimeError("P34-ARF requires P3/P4/P5 indices")
        scales = tuple(sorted(int(s) for s in cfg.get("scales", [3, 4])))
        if scales != (3, 4):
            raise ValueError("This dedicated variant requires reliability_fusion.scales: [3, 4]")

        self.reliability_fusion_cfg = cfg
        self.gated_scales = scales
        self.reliability_gates = nn.ModuleDict({
            str(scale): SpatialAlignmentReliabilityGate(
                hidden_channels=int(cfg.get("hidden_channels", 16)),
                detach_alignment_support=bool(cfg.get("detach_alignment_support", True)),
            ) for scale in self.gated_scales
        })
        print("[ARF-P34] P3/P4 trainable spatial gates; P5 exact original ADD; "
              "initial gate multiplier 1.000000 (exact ADD); "
              f"align-support detached={bool(cfg.get('detach_alignment_support', True))}",
              flush=True)

    def forward_features(
        self, rgb, tir, rgb_semantic=None, rgb_valid_mask=None,
        tir_valid_mask=None, return_bhlr_debug=False,
        return_alignment_debug=False,
    ):
        output = super().forward_features(
            rgb, tir, rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        info = output["alignment_info"]
        guidance = info.get("guidance_by_scale", {})
        if not guidance:
            raise RuntimeError("Missing alignment guidance_by_scale")

        mode = "auto" if self.training else os.environ.get(
            "RGBT_RELIABILITY_EVAL_MODE", "auto"
        ).strip().lower()
        if mode not in ("auto", "force_add"):
            raise ValueError("RGBT_RELIABILITY_EVAL_MODE must be 'auto' or 'force_add'")
        disabled = set()
        if not self.training:
            disabled = {
                int(t.strip()) for t in os.environ.get("RGBT_ARF_DISABLE_SCALES", "").split(",")
                if t.strip()
            }
            if disabled - {3, 4}:
                raise ValueError("P34 variant supports disabling gates only on P3/P4; P5 is always ADD")

        fused = list(output["fused_backbone"])
        fusion_features = dict(output["fusion_features"])
        modulations = {}
        for scale in self.gated_scales:
            idx = self.scale_to_index[scale]
            rgb_feat = output["rgb_backbone"][idx]  # includes original BHLR enhancement
            native = fusion_features[idx]              # original ADD
            tir_effect = native - rgb_feat              # includes any TIR 1x1 projection
            support = _get_scale(guidance, scale)
            if support is None:
                raise RuntimeError(f"Missing P{scale} alignment guidance")
            gate = self.reliability_gates[str(scale)](
                rgb_feat, tir_effect, support, info.get("coarse_confidence")
            )
            if mode == "force_add" or scale in disabled:
                gate = torch.ones_like(gate)
            # Exact bitwise ADD at initial 0-logit, and for disabled scales.
            updated = native + (gate - 1.0) * tir_effect
            fused[idx] = updated
            fusion_features[idx] = updated
            modulations[scale] = gate

        # P5 is never rewritten by this subclass: original ADD tensor is kept.
        output["fused_backbone"] = fused
        output["fusion_features"] = fusion_features
        output["reliability_modulation_maps"] = modulations
        output["reliability_fusion_type"] = "P3_P4_modulated_ADD__P5_native_ADD"
        return output


__all__ = ["RGBTAlignmentReliabilityP34FusionDetectionModel"]
