"""Install optional HVAD logic on the original RGB-T model instance.

No edits to GTGA, BHLR, Tri-Fusion, YOLO neck or Detect source are required.
Only the model factory receives a guarded `attach_hvad()` call.
Checkpoint reconstruction uses the config and the same wrapper each time.
"""
from __future__ import annotations

from types import MethodType
import torch
from .module import HVADGeoHR


def attach_hvad(model, hvad_cfg, *, verbose=True):
    cfg = dict(hvad_cfg or {})
    if not cfg.get('enabled', False):
        return model
    if hasattr(model, 'hvad'):
        raise RuntimeError('HVAD already attached; refusing duplicate installation')
    if not hasattr(model, 'bhlr') or model.bhlr is None:
        raise RuntimeError('HVAD requires enabled BHLR with a detail_stem')
    if not hasattr(model.bhlr, 'detail_stem'):
        raise RuntimeError('Cannot find BHLR detail_stem')
    if not hasattr(model, 'scale_to_index') or 3 not in model.scale_to_index:
        raise RuntimeError('HVAD requires centered GTGA with P3 scale_to_index')
    if not getattr(model, 'alignment_enabled', False):
        raise RuntimeError('HVAD requires enabled GTGA alignment')
    p3 = int(model.scale_to_index[3])
    if p3 not in model.fusion_indices:
        raise RuntimeError('P3 must be an RGB-T fusion index')
    # The existing P3 Tri-Fusion is preserved in the bound forward method.
    # Validate channels from BHLR adapter, not a hardcoded YOLO width.
    adapter = model.bhlr.adapters.get(str(p3), None) if hasattr(model.bhlr.adapters, 'get') else None
    if adapter is None:
        if str(p3) not in model.bhlr.adapters:
            raise RuntimeError(f'No BHLR adapter for P3 backbone layer {p3}')
        adapter = model.bhlr.adapters[str(p3)]
    channels = int(adapter.rgb_channels)
    detail_channels = int(cfg.get('detail_channels', 48))
    if hasattr(model.bhlr.detail_stem, 'net'):
        # Explicit runtime check in forward validates actual detail C.
        pass
    module = HVADGeoHR(
        channels=channels,
        detail_channels=detail_channels,
        hidden=int(cfg.get('hidden', 24)),
        roi_size=int(cfg.get('roi_size', 11)),
        roi_radius=float(cfg.get('roi_radius', 6.0)),
        max_gain=float(cfg.get('max_gain', 0.25)),
        prior_strength=float(cfg.get('prior_strength', 0.5)),
        max_gt_offset=float(cfg.get('max_gt_offset', 20.0)),
        lambda_rank=float(cfg.get('lambda_rank', 0.03)),
        use_hr_evidence=bool(cfg.get('use_hr_evidence', True)),
    )
    # V5-GEOHR-SPATIAL-V1: optional modes, defaults preserve existing checkpoints.
    module.spatial_hr_roi = bool(cfg.get('spatial_hr_roi', False))
    module.hr_roi_mix = float(cfg.get('hr_roi_mix', 1.0))
    module.confidence_gate_enabled = bool(cfg.get('confidence_gate_enabled', False))
    module.confidence_gate_threshold = float(cfg.get('confidence_gate_threshold', 0.50))
    if not 0.0 <= module.hr_roi_mix <= 1.0:
        raise ValueError('hvad.hr_roi_mix must be within [0, 1]')
    if not 0.0 <= module.confidence_gate_threshold < 1.0:
        raise ValueError('hvad.confidence_gate_threshold must be in [0, 1)')
    model.hvad = module
    model.bhlr.detail_stem.register_forward_hook(module.capture_detail)
    original_forward_features = model.forward_features
    original_loss = model.loss

    def forward_features_hvad(self, rgb, tir, *args, **kwargs):
        self.hvad._detail_bank = None
        feats = original_forward_features(rgb, tir, *args, **kwargs)
        bank = self.hvad._detail_bank
        self.hvad._detail_bank = None  # do not hold a second full HR activation
        if self.hvad.enabled_runtime:
            feats = self.hvad(feats, bank, p3)
        else:
            self.hvad.last_aux = None
            self.hvad.last_stats = {'active': 0.0}
        return feats

    def loss_hvad(self, batch, preds=None):
        raw, items = original_loss(batch, preds=preds)
        if preds is not None or not self.hvad.enabled_runtime or not self.hvad.rank_enabled:
            return raw, items
        rank, extra = self.hvad.ranking_loss(batch)
        bsize = int(batch['rgb_semantic_img'].shape[0])
        aux = rank * self.hvad.lambda_rank * float(bsize)
        if isinstance(raw, torch.Tensor):
            raw = torch.cat((raw.reshape(-1), aux.reshape(1)), dim=0)
        else:
            raise TypeError('HVAD expects tensor original loss')
        if isinstance(items, dict):
            items = dict(items)
        else:
            items = self._loss_items_to_dict(items)
        items.update(extra)
        items['hvad_rank_weighted'] = (rank * self.hvad.lambda_rank).detach()
        return raw, items

    model.forward_features = MethodType(forward_features_hvad, model)
    model.loss = MethodType(loss_hvad, model)
    if verbose:
        print(f'[HVAD] attached P3 hypothesis verification | '
              f'channels={channels}, K=GMLS candidates, '
              f'ROI={module.roi_size}, lambda_rank={module.lambda_rank}, '
              f'init_gain={module.gamma_raw.item():.6f}, '
              'existing GTGA/BHLR/Tri-Fusion/P2 untouched', flush=True)
    return model
