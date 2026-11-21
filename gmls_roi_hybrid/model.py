"""Global Neck+Detect / Local Neck+Detect independent YOLO branches.

Uses the co-located model_hybrid, GeoCenter V2 and Top-2 implementation.
Only the YOLO Neck + Detect are duplicated, not the RGB/TIR backbones.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from models.rgbt_model import get_layer_input
from .model_hybrid import GMLSV2OneHeadHybrid


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


from .geo_losses import final_box_geometry_loss


class GMLSV2GeoBoxDualHead(GMLSV2DualHeadHybrid):
    architecture = 'gmls_v2_dualhead_geo_finalbox_scratch100_unified_v1'

    def __init__(self, cfg, *, verbose=True):
        super().__init__(cfg, verbose=verbose)
        ac = cfg['loss']['alignment']
        self.geo_size_weight = float(ac.get('lambda_final_size', 0.0))
        self.geo_giou_weight = float(ac.get('lambda_final_giou', 0.0))
        self.geo_size_beta = float(ac.get('final_size_beta', 0.2))
        self.geo_warmup = int(ac.get('final_box_warmup_epochs', 10))
        if self.geo_size_weight < 0 or self.geo_giou_weight < 0:
            raise ValueError('Final geometry weights must be nonnegative')
        if self.geo_warmup < 0 or self.geo_size_beta <= 0:
            raise ValueError('Invalid final box warmup or beta')
        if not cfg['loss']['alignment'].get('enabled', False):
            raise ValueError('GeoBox training requires loss.alignment.enabled=true')
        self._active_alignment_info = None
        if verbose:
            print('[GEOBOX V1] original GeoCenter V2 + FinalSize + FinalGIoU; '
                  f'size={self.geo_size_weight:g}, giou={self.geo_giou_weight:g}, '
                  f'warmup={self.geo_warmup}; NO previous checkpoint', flush=True)

    def encode_and_crop(self, batch, *, return_debug=False, train_rois=False):
        ex = super().encode_and_crop(batch, return_debug=return_debug,
                                     train_rois=train_rois)
        if train_rois:
            self._active_alignment_info = ex['alignment_info']
        return ex

    def loss(self, batch):
        self._active_alignment_info = None
        try:
            original_loss, parts = super().loss(batch)
            info = self._active_alignment_info
            if info is None:
                raise RuntimeError('Parent loss did not produce final alignment_info')
            parts = dict(parts)
            epoch = float(getattr(self, 'epoch', 0))
            ramp = min(1.0, (epoch + 1.0) / self.geo_warmup) if self.geo_warmup else 1.0
            if self.geo_size_weight == 0 and self.geo_giou_weight == 0:
                # Exact 09 control: no extra geometry graph and no loss changes.
                z = original_loss.detach().new_zeros(())
                parts.update({'geo_final_size': z, 'geo_final_giou': z,
                              'geo_final_box_weighted': z,
                              'geo_final_box_pairs': 0.0, 'geo_final_box_ramp': ramp})
                return original_loss, parts
            with torch.autocast(device_type=original_loss.device.type, enabled=False):
                size, giou, count, mean_iou, err_px = final_box_geometry_loss(
                    batch, info, size_beta=self.geo_size_beta)
                # Follow the parent final_center_loss's batch-size convention.
                B = int(batch['rgb_img'].shape[0])
                addition = B * ramp * (self.geo_size_weight * size +
                                        self.geo_giou_weight * giou)
            full_loss = original_loss + addition  # keep FP32 geometry under AMP
            parts.update({'total': full_loss.detach(),
                          'geo_final_size': size.detach(),
                          'geo_final_giou': giou.detach(),
                          'geo_final_box_weighted': addition.detach(),
                          'geo_final_box_pairs': float(count),
                          'geo_final_box_ramp': float(ramp),
                          'geo_final_box_iou': mean_iou,
                          'geo_final_box_center_px': err_px})
            if 'align_weighted' in parts:
                parts['align_weighted'] = parts['align_weighted'] + addition.detach()
            return full_loss, parts
        finally:
            # No graph retention across steps; no state is saved in checkpoint.
            self._active_alignment_info = None

def build_model(cfg, *, verbose=True):
    """One controlled factory: 01 unchanged dual head, 10 adds GeoBox losses."""
    flag=bool(cfg.get('model',{}).get('geo_box',{}).get('enabled',False))
    a=cfg['loss']['alignment']
    enabled=bool(a.get('enabled',False))
    size=float(a.get('lambda_final_size',0.0))
    giou=float(a.get('lambda_final_giou',0.0))
    if not enabled:
        raise ValueError('Both experiments require alignment supervision')
    if flag:
        if size <= 0 or giou <= 0:
            raise ValueError('GeoBox experiment 10 requires positive size and GIoU weights')
        return GMLSV2GeoBoxDualHead(cfg,verbose=verbose)
    if size != 0 or giou != 0:
        raise ValueError('01 must not silently use final-box geometry weights')
    return GMLSV2DualHeadHybrid(cfg,verbose=verbose)
