"""Standalone RGB-Preserve v1 architecture. No modifications to legacy model files."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from models.gtga_bhlr_factory import build_model_from_config as build_original
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)


class RGBPreserveP3Model(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    """RGB full detection path + zero-initialized aligned TIR P3 residual.

    BHLR's existing gamma_raw=0 means RGB detail injection is initially neutral.
    P4/P5 are deliberately *not* fused in the first controlled experiment.
    """

    def __init__(self, *args, **kwargs):
        kwargs['fusion'] = 'add'  # inherited same-channel projection, not used as ADD
        super().__init__(*args, **kwargs)
        self.p3_residual_raw = nn.Parameter(torch.tensor(0.0))
        self.fusion_type = 'rgb_preserve_p3'
        # P4/P5 fusion objects are not used; avoid unused DDP trainable params.
        for scale in (4, 5):
            idx = self.scale_to_index[scale]
            self.fusions[str(idx)].requires_grad_(False)
        self._preserve_phase = 'identity'
        self._rgb_backbone_frozen = False
        self._rgb_head_frozen = False

    def fuse_backbone_features(self, rgb_outputs, tir_outputs):
        outputs = list(rgb_outputs)
        p3_idx = self.scale_to_index[3]
        rgb = rgb_outputs[p3_idx]
        tir = tir_outputs[p3_idx]
        # Identity -> RGB coordinate dimension is authoritative.
        if tir.shape[-2:] != rgb.shape[-2:]:
            tir = F.interpolate(tir, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        projection = self.fusions[str(p3_idx)].tir_proj
        tir = projection(tir)
        fused = rgb + torch.tanh(self.p3_residual_raw) * tir
        outputs[p3_idx] = fused
        return outputs, {p3_idx: fused}

    def set_training_phase(self, phase: str):
        """Configure requires_grad BEFORE wrapping in DDP (re-wrap between phases)."""
        if phase not in ('identity', 'stage1', 'stage2'):
            raise ValueError(phase)
        self._preserve_phase = phase
        self.requires_grad_(True)
        # Permanently unused compatibility modules remain frozen.
        for scale in (4, 5):
            self.fusions[str(self.scale_to_index[scale])].requires_grad_(False)
        if self.bhlr_enabled and self.bhlr_guidance_mode == 'alignment':
            for adapter in self.bhlr.adapters.values():
                adapter.fallback.target_head.requires_grad_(False)
                adapter.external_rgb_adapter.requires_grad_(False)
                adapter.external_tir_adapter.requires_grad_(False)
        # At 640/640 BHLR is not executed by the inherited forward.
        if not self.bhlr_enabled:
            self.alignment_guidance_fusion.requires_grad_(False)
        if phase in ('identity', 'stage1'):
            self.model.requires_grad_(False)  # whole original RGB backbone + neck + head
            self._rgb_backbone_frozen = True
            self._rgb_head_frozen = True
        elif phase == 'stage2':
            for module in list(self.model)[:self.backbone_len]:
                module.requires_grad_(False)
            self._rgb_backbone_frozen = True
            self._rgb_head_frozen = False
        self.train(phase != 'identity')

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if getattr(self, '_rgb_backbone_frozen', False):
                for m in list(self.model)[:self.backbone_len]:
                    m.eval()  # critical: freeze RGB backbone running BN stats too
            if getattr(self, '_rgb_head_frozen', False):
                # Keep Detect head in TRAINING mode for Ultralytics criterion.
                # Only BN running statistics need freezing in Stage1.
                for module in self.model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self


def build_rgb_preserve_model(cfg: Dict, *, init_single_modal: bool = False, verbose: bool = True):
    """Use existing factory config resolution but construct our subclass directly.

    Build through the original factory once is intentionally avoided: no extra model.
    Extract config the same way as the original centered factory branch.
    """
    m = cfg['model']
    d = cfg.get('data', {})
    loss = cfg.get('loss', {})
    align = dict(m.get('rlsfa_gtga_centered', m.get('alignment', {})) or {})
    aloss = dict(loss.get('alignment', loss.get('rlsfa_gtga_centered', {})) or {})
    bhlr = dict(m.get('bhlr', {}) or {})
    if m.get('variant') != 'rgb_preserve_v1':
        raise ValueError('Expected model.variant: rgb_preserve_v1')
    if m.get('fusion', 'add') != 'add':
        raise ValueError('RGB-Preserve requires model.fusion: add in the YAML (internal override).')
    names = {0: 'UAV'}
    model = RGBPreserveP3Model(
        model_name=str(m.get('name', 'yolo26m')),
        nc=int(m.get('nc', 1)),
        pretrained=False,  # both trained single-modality weights loaded below
        fusion='add',
        fusion_indices=m.get('fusion_indices'),
        align_mode=str(m.get('align_mode', 'bilinear')),
        names=names,
        semantic_imgsz=int(d.get('rgb_semantic_imgsz', 640)),
        rlsfa_gtga_centered_cfg=align,
        rlsfa_gtga_centered_loss_cfg=aloss,
        bhlr_enabled=bool(bhlr.get('enabled', True)),
        bhlr_scales=bhlr.get('scales', [3]),
        bhlr_layers=bhlr.get('layers'),
        high_to_semantic_ratio=int(bhlr.get('high_to_semantic_ratio', 3)),
        detail_channels=int(bhlr.get('detail_channels', 48)),
        stem_channels=int(bhlr.get('stem_channels', 16)),
        guide_channels=int(bhlr.get('guide_channels', 24)),
        bhlr_guidance_mode=str(bhlr.get('guidance_mode', 'alignment')),
        bhlr_detach_alignment_guidance=bool(bhlr.get('detach_alignment_guidance', True)),
        verbose=verbose,
    )
    if init_single_modal:
        p = cfg.get('rgb_preserve', {})
        if not p.get('rgb_weights') or not p.get('tir_weights'):
            raise ValueError('rgb_preserve.rgb_weights and tir_weights are required')
        load_dual_single_modal_weights(model, p['rgb_weights'], p['tir_weights'])
    return model


def _load_yolo_modulelist(path: str):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Single modality .pt not found: {path}')
    source = YOLO(str(path)).model
    if not hasattr(source, 'model'):
        raise TypeError(f'{path} does not contain an Ultralytics DetectionModel')
    if not isinstance(source.model, (nn.Sequential, nn.ModuleList)):
        raise TypeError('Expected YOLO.model.model as Sequential/ModuleList')
    return source


def _load_strict(dst: nn.Module, src_state: Dict[str, torch.Tensor], label: str):
    dst_state = dst.state_dict()
    missing = sorted(set(dst_state) - set(src_state))
    excess = sorted(set(src_state) - set(dst_state))
    bad_shapes = [(k, tuple(dst_state[k].shape), tuple(src_state[k].shape))
                  for k in (set(dst_state) & set(src_state))
                  if dst_state[k].shape != src_state[k].shape]
    if missing or excess or bad_shapes:
        raise RuntimeError(f'{label} state_dict incompatible: '
                           f'missing={missing[:8]} extra={excess[:8]} shapes={bad_shapes[:8]}')
    dst.load_state_dict(src_state, strict=True)
    print(f'[RGB-PRESERVE] {label}: strictly transferred {len(dst_state)} tensors.')


def load_dual_single_modal_weights(model: RGBPreserveP3Model, rgb_weights: str, tir_weights: str):
    rgb_src = _load_yolo_modulelist(rgb_weights)
    tir_src = _load_yolo_modulelist(tir_weights)
    for label, src in [('RGB', rgb_src), ('TIR', tir_src)]:
        if int(getattr(src, 'nc', 1)) != model.nc:
            raise ValueError(f'{label} checkpoint nc={src.nc}, expected {model.nc}')
        if len(src.model) != len(model.model):
            raise ValueError(f'{label} YOLO architecture differs from main model')
    _load_strict(model.model, rgb_src.model.state_dict(), 'RGB entire Backbone+Neck+Head')
    tir_layers = nn.ModuleList(list(tir_src.model)[:model.backbone_len])
    _load_strict(model.tir_backbone, tir_layers.state_dict(), 'TIR Backbone')
    model.names = getattr(rgb_src, 'names', {0: 'UAV'})
    del rgb_src, tir_src
    return model


def load_rgb_preserve_checkpoint(checkpoint_path, device, fusion_override='auto', strict=True, verbose=False):
    if fusion_override not in ('auto', 'add', 'rgb_preserve_p3'):
        raise ValueError('Cannot override RGB-Preserve fusion to another architecture')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt or 'config' not in ckpt:
        raise TypeError('Checkpoint is not an RGB-Preserve training checkpoint')
    cfg = ckpt['config']
    if cfg.get('model', {}).get('variant') != 'rgb_preserve_v1':
        raise ValueError('Wrong model variant for RGB-Preserve checkpoint')
    model = build_rgb_preserve_model(cfg, init_single_modal=False, verbose=verbose)
    model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    model.to(device).eval()
    return model, ckpt, cfg
