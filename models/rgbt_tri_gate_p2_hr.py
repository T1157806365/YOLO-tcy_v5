"""P3 Final Tri-Gate + P2 selective BHLR high-resolution detail injection.

Keeps pre-existing GTGA, BHLR, P3 Tri-Gate, YOLO P2 architecture and losses.
Introduces an identity-initialized HR residual after native P2 neck layer 25.
"""
from __future__ import annotations

import torch
from models.rgbt_final_tri_gate import RGBTRLSFAGTGACenteredBHLRV2TriGateModel
from models.rgbt_model import get_layer_input
from models.modules.p2_hr_selective import P2SelectiveHR


class RGBTRLSFAGTGACenteredBHLRV2TriGateP2HRModel(
        RGBTRLSFAGTGACenteredBHLRV2TriGateModel):

    def __init__(self, *args, p2_hr_cfg=None, **kwargs):
        cfg = dict(p2_hr_cfg or {})
        if not cfg.get('enabled', False):
            raise ValueError('P2HR class requires model.p2_hr_refine.enabled: true')
        if str(kwargs.get('model_name', '')).lower() != 'yolo26m-p2':
            raise ValueError('P2HR v1 requires model.name: yolo26m-p2')
        super().__init__(*args, **kwargs)
        if len(self.model) != 27 or getattr(self.model[25], 'i', None) != 25:
            raise ValueError('P2HR v1 requires native YOLO26m-P2 layers 0..26, P2 at index25')
        if not (self.bhlr_enabled and self.alignment_enabled):
            raise ValueError('P2HR v1 needs GTGA and BHLR enabled')
        dch = int(kwargs.get('detail_channels', 48))
        self.p2_hr_refiner = P2SelectiveHR(
            p2_channels=int(cfg.get('p2_channels', 128)),
            detail_channels=dch,
            hidden=int(cfg.get('hidden', 32)),
            gamma_init=0.0,
        )
        self.p2_hr_refine_enabled = True
        self.p2_hr_disabled = False
        self._p2_monitor = {}
        self._p2_monitor_n = 0
        print('[P2-HR] Native P2=layer25, reuse BHLR detail bank; '
              'P3 Tri-Gate unchanged; gamma init=0 (exact identity)')

    def set_p2_hr_disabled(self, disabled: bool):
        self.p2_hr_disabled = bool(disabled)

    def forward_head(self, backbone_outputs, *, p2_detail_raw=None, p2_support=None):
        if p2_detail_raw is None or self.p2_hr_disabled:
            return super().forward_head(backbone_outputs)
        if len(backbone_outputs) != self.backbone_len:
            raise ValueError('incorrect backbone_outputs length')
        outputs = list(backbone_outputs)
        x = outputs[self.backbone_len - 1]
        found_p2 = False
        for layer in list(self.model)[self.backbone_len:]:
            x = layer(get_layer_input(layer, x, outputs))
            if int(layer.i) == 25:
                x, stats = self.p2_hr_refiner(x, p2_detail_raw, p2_support)
                found_p2 = True
                if self.training and torch.is_grad_enabled():
                    for key, value in stats.items():
                        v = value.detach().float().mean()
                        self._p2_monitor[key] = self._p2_monitor.get(key, 0.0) + v
                    self._p2_monitor_n += 1
            outputs.append(x)
        if not found_p2:
            raise RuntimeError('P2 feature layer 25 was not reached')
        return x

    def predict(self, rgb, tir, rgb_semantic=None, rgb_valid_mask=None,
                tir_valid_mask=None, return_features=False,
                return_bhlr_debug=False, return_alignment_debug=False):
        features = self.forward_features(
            rgb, tir, rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=return_bhlr_debug,
            return_alignment_debug=return_alignment_debug,
        )
        info = features.get('alignment_info', {})
        supp = info.get('support_by_scale', {}) or {}
        p2_support = supp.get(3, supp.get('3'))
        pred = self.forward_head(
            features['fused_backbone'],
            p2_detail_raw=features.get('p2_hr_detail_raw'),
            p2_support=p2_support,
        )
        if return_features or return_bhlr_debug or return_alignment_debug:
            return {'pred': pred, **features}
        return pred

    @torch.no_grad()
    def p2_hr_monitor(self, reset=True):
        n = self._p2_monitor_n
        values = {k: float(v.item()) / n for k, v in self._p2_monitor.items()} if n else {}
        if reset:
            self._p2_monitor_n = 0
            self._p2_monitor = {}
        return n, values
