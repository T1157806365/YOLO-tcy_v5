"""Global Neck+Detect / Local Neck+Detect independent YOLO branches.

Depends on the user's INSTALLED gmls_roi_onehead_v2 (GeoCenter, Top-2
Diverse and all other patches remain active). Only the YOLO neck and Detect
layers are duplicated, never RGB/TIR backbone and never ROI detector body.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from models.rgbt_model import get_layer_input
from gmls_roi_onehead_v2.model import GMLSV2OneHeadHybrid


class GMLSV2DualHeadHybrid(GMLSV2OneHeadHybrid):
    architecture = 'gmls_v2_two_independent_global_local_neck_detect_geocenter_top2_v1'

    def __init__(self, cfg, *, verbose=True):
        super().__init__(cfg, verbose=verbose)
        n = int(self.detector.backbone_len)
        layers = list(self.detector.model)
        if not 0 < n < len(layers):
            raise RuntimeError(f'Invalid YOLO split: backbone_len={n}, total_layers={len(layers)}')
        # The official YOLO26m pretrained parameters have already been loaded
        # into self.detector by the parent. Clone the FULL Neck + Detect, not
        # just the last Detect layer, for a true independent auxiliary head.
        self.global_neck_detect = nn.ModuleList(deepcopy(layers[n:]))
        self.global_backbone_len = n
        self.global_aux_separate = True
        if verbose:
            local_params = sum(p.numel() for l in layers[n:] for p in l.parameters())
            global_params = sum(p.numel() for p in self.global_neck_detect.parameters())
            assert local_params == global_params
            print('[DUAL HEAD V1] Global Neck+Detect separate from Local Neck+Detect', flush=True)
            print(f'[DUAL HEAD V1] extra trainable global head parameters={global_params:,}', flush=True)
            print('[DUAL HEAD V1] training: global auxiliary + Top-2 local; '
                  'inference: Top-2 local ONLY', flush=True)
            print(f'[DUAL HEAD V1] architecture={self.architecture}', flush=True)

    def _global_head_from_fused_features(self, backbone_outputs):
        """Match native YOLO forward_head() including its skip connections."""
        if len(backbone_outputs) != self.global_backbone_len:
            raise ValueError('Global head input is not the full fused backbone output list')
        outputs = list(backbone_outputs)
        x = outputs[-1]
        for layer in self.global_neck_detect:
            inp = get_layer_input(layer, x, outputs)
            x = layer(inp)
            outputs.append(x)
        return x

    def _shared_head_global_prediction(self, ex):
        """Override the parent global path; the rest of GeoCenter loss is inherited.

        The name is intentionally preserved because the installed GeoCenter
        model's loss() calls this extension point. The copied independent
        global head is trained with the original global_aux supervision.
        """
        rgb = ex['global_rgb_features']
        aligned = ex['global_tir_aligned']
        if rgb is None or aligned is None:
            raise RuntimeError('Global head invoked without training features')
        fused = list(rgb)
        for idx in self.global_encoder.fusion_indices:
            fused[idx] = rgb[idx] + aligned[idx]
        # Unlike the one-head baseline, this head owns its BN statistics;
        # do NOT apply the baseline policy that forces local-head BN to eval.
        return self._global_head_from_fused_features(fused)

    @torch.no_grad()
    def _test_head_separation(self):
        """Runtime invariants checked by --smoke before training starts."""
        n = self.global_backbone_len
        loc = list(self.detector.model)[n:]
        assert len(loc) == len(self.global_neck_detect)
        for a, b in zip(loc, self.global_neck_detect):
            assert a is not b
        lp = {id(p) for layer in loc for p in layer.parameters()}
        gp = {id(p) for p in self.global_neck_detect.parameters()}
        assert not lp.intersection(gp), 'Global and Local head parameters are shared'
        return True
