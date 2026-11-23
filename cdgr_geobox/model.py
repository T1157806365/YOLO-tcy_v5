"""A complete, non-distillation GeoBox detector with trainable geometry correction.

Everything downstream of the geometry interface is inherited: dual independent
Global/Local heads, Top-2 Diverse native-HR crops, original alignment losses,
FinalCenter, FinalSize, FinalGIoU, and optimizer/training stages (single 100e).
"""
from __future__ import annotations
import torch
from gmls_roi_hybrid.model import GMLSV2GeoBoxDualHead
from .refiner import CenterDecoupledAligner


class CDGRGeoBoxDualHead(GMLSV2GeoBoxDualHead):
    architecture = 'gmls_v2_dualhead_center_decoupled_geometry_refinement_geobox_v1'

    def __init__(self, cfg, *, verbose=True):
        rc = cfg.get('model', {}).get('geometry_refinement', {})
        if not bool(rc.get('enabled', False)):
            raise ValueError('CDGR requires model.geometry_refinement.enabled=true')
        if not (cfg['model']['geo_box']['enabled'] and cfg['model']['dual_head']['enabled']):
            raise ValueError('CDGR must preserve original GeoBox and dual head')
        if cfg.get('distillation', {}).get('enabled') or cfg.get('geometry_distillation', {}).get('enabled'):
            raise ValueError('This model forbids teacher/KD distillation')
        if int(cfg['train']['epochs']) != 100 or cfg['train'].get('resume') is not None:
            raise ValueError('CDGR must start from official YOLO26m and train for 100 epochs')
        if cfg['train'].get('stage2', {}).get('enabled', False):
            raise ValueError('CDGR has no second training stage')
        super().__init__(cfg, verbose=verbose)
        refine_cfg = dict(rc)
        refine_cfg['p3_channels'] = int(cfg['model']['roi_single_head'].get('p3_channels', 512))
        refine_cfg['semantic_imgsz'] = int(cfg['data'].get('rgb_semantic_imgsz', 640))
        base_aligner = self.global_encoder.alignment
        self.global_encoder.alignment = CenterDecoupledAligner(base_aligner, refine_cfg)
        if verbose:
            new_pars = sum(p.numel() for p in self.global_encoder.alignment.residual.parameters())
            print(f'[CDGR] Non-distillation: center+log-scale refinement enabled; '
                  f'extra parameters={new_pars:,}', flush=True)
            print('[CDGR] Initial geometry exactly matches old GTGA; refined geometry '
                  'used consistently by P3/P4/P5, ROI Guide, GeoBox, and TIR ROI', flush=True)
            print('[CDGR] Official YOLO26m initialization; 100e single joint training; '
                  'no old checkpoints, no teachers', flush=True)

    def loss(self, batch):
        loss, parts = super().loss(batch)
        parts = dict(parts)
        parts.update(self.global_encoder.alignment.last_stats)
        return loss, parts
