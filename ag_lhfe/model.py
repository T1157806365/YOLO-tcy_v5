"""AG-LHFE model variant integrated into the original GMLS/DMLS detector.

The inherited RGB/TIR backbones, GMLS-v2/DMLS alignment, P4/P5 fusion,
YOLO26 neck/head, alignment auxiliary losses, checkpoints and evaluator are
unchanged. Only the original BHLR P3 injection is replaced.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)
from ag_lhfe.local_hr import LocalHighResFeatureExtractor


class RGBTAGLHFEDetectionModel(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self, *, ag_lhfe_cfg: Optional[Dict] = None, **kwargs):
        cfg = dict(ag_lhfe_cfg or {})
        if not cfg.get('enabled', True):
            raise ValueError('AG-LHFE model variant requires model.ag_lhfe.enabled=true')
        if not dict(kwargs.get('rlsfa_gtga_centered_cfg') or {}).get('enabled', True):
            raise ValueError('AG-LHFE v1 requires GMLS/DMLS alignment enabled')
        kwargs['bhlr_enabled'] = False
        kwargs['bhlr_guidance_mode'] = 'none'
        super().__init__(**kwargs)
        p3_layer = int(self.scale_to_index[3])
        channels = self._infer_any_layer_channels([p3_layer])[p3_layer]['rgb']
        self.ag_lhfe_p3_layer = p3_layer
        self.ag_lhfe = LocalHighResFeatureExtractor(
            channels=int(channels),
            proposal_mode=str(cfg.get('proposal_mode', 'rgb_align')),
            topk=int(cfg.get('topk', 16)),
            nms_radius=int(cfg.get('nms_radius', 3)),
            patch_size=int(cfg.get('patch_size', 32)),
            crop_sizes_high=tuple(cfg.get('crop_sizes_high', [48, 96])),
            latent_dim=int(cfg.get('latent_dim', 64)),
            use_edges=bool(cfg.get('use_edges', True)),
            strength_init=float(cfg.get('strength_init', 0.05)),
            strength_max=float(cfg.get('strength_max', 0.20)),
            max_offset=float(cfg.get('max_offset', 2.5)),
            splat_radius=int(cfg.get('splat_radius', 2)),
            positive_radius=float(cfg.get('positive_radius', 2.5)),
            lambda_obj=float(cfg.get('lambda_obj', 0.05)),
            lambda_loc=float(cfg.get('lambda_loc', 0.02)),
        )
        self._ag_lhfe_state = None
        self.ag_lhfe_cfg = cfg

    def forward_features(self, rgb, tir, rgb_semantic=None,
                         rgb_valid_mask=None, tir_valid_mask=None,
                         return_bhlr_debug=False, return_alignment_debug=False):
        # Force geometry debug to recover RGB and aligned TIR target-support maps.
        features = super().forward_features(
            rgb, tir, rgb_semantic=rgb_semantic,
            rgb_valid_mask=rgb_valid_mask, tir_valid_mask=tir_valid_mask,
            return_bhlr_debug=False, return_alignment_debug=True,
        )
        p3 = self.ag_lhfe_p3_layer
        scale = 3
        info = features['alignment_info']
        scale_debug = info.get('scale_debug', {})
        ref = scale_debug.get(scale, scale_debug.get(str(scale)))
        if ref is None:
            raise RuntimeError('Missing GMLS P3 scale_debug; AG-LHFE needs alignment support maps')
        sr = ref['rgb_local_targetness']
        st = ref['aligned_tir_local_targetness']
        joint = info['support_by_scale'][scale]
        # Current GMLS/DMLS branch exposes two bbox-supervised quality estimates.
        # This is a learned reliability PROXY, not a calibrated alignment IoU.
        qualities = info.get('gmls_v2_quality_scores')
        chosen_dmls = info.get('gmls_v2_choose_dmls')
        geo_quality = None
        if qualities is not None and chosen_dmls is not None:
            if qualities.ndim != 2 or qualities.shape[1] < 2:
                raise RuntimeError('Unexpected GMLS-v2 quality shape')
            q = qualities.detach()
            geo_quality = torch.where(chosen_dmls.detach().view(-1).bool(), q[:,1], q[:,0])
        rgb_before = features['rgb_backbone'][p3]
        tir_aligned = features['aligned_tir_backbone'][p3]
        rgb_after, state = self.ag_lhfe(rgb, rgb_before, tir_aligned, sr, st, joint, geometry_quality=geo_quality)
        rgb_outputs = list(features['rgb_backbone'])
        rgb_outputs[p3] = rgb_after
        features['rgb_backbone'] = rgb_outputs
        fused_outputs = list(features['fused_backbone'])
        fused_p3 = self.fusions[str(p3)](rgb_after, tir_aligned)
        fused_outputs[p3] = fused_p3
        features['fused_backbone'] = fused_outputs
        features['fusion_features'] = dict(features['fusion_features'])
        features['fusion_features'][p3] = fused_p3
        features['ag_lhfe_enabled'] = True
        features['ag_lhfe_p3_layer'] = p3
        if self.training:
            self._ag_lhfe_state = state
        else:
            self._ag_lhfe_state = None
        # For audit, return lightweight scalar statistics, not full patch/feature banks.
        if return_bhlr_debug or return_alignment_debug:
            features['ag_lhfe_debug'] = {
                'centers_p3': state['coords'].detach(),
                'proposal_kind': state['kind'].detach(),
                'strength': state['strength'].detach(),
                'injection_rms_ratio': state['inject_rms_ratio'].detach(),
                'roi_objectness': state['obj_logits'].sigmoid().detach(),
                'local_offsets': state['offsets'].detach(),
            }
        return features

    def loss(self, batch):
        # The base implementation calculates normal detection + original GMLS
        # geometry losses, via our overridden self.predict / forward_features.
        raw, items = super().loss(batch)
        state = self._ag_lhfe_state
        self._ag_lhfe_state = None
        if state is None:
            raise RuntimeError('AG-LHFE state was not produced during training forward')
        roi_loss, metrics = self.ag_lhfe.auxiliary_loss(state, batch)
        raw = torch.cat((raw.reshape(-1), roi_loss.reshape(1)), dim=0)
        if not isinstance(items, dict):
            items = self._loss_items_to_dict(items)
        items = dict(items)
        items.update(metrics)
        return raw, items
