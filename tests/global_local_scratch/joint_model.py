"""GMLS-v2 global + native high-resolution RGB/TIR local detection.

Both networks initialize randomly; train jointly in one optimizer. GT-based
crop augmentation is used only in TRAIN, as ordinary label-supervised ROI
sampling. Validation/inference never reads GT for crop selection.

The ROI selector uses detector proposals at inference and GTGA coarse_anchor
at training. The discrete crop center is non-differentiable; each detector's
regular losses are fully differentiable. No teacher model / pretrained ckpt.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
import torch
from torch import nn
from models.gtga_bhlr_factory import build_model_from_config
from models.rgbt_model import RGBTDetectionModel
from engine.evaluator import postprocess_predictions
from .geometry import (
    CropGeometry, sample_native_hr_and_tir, local_gt_from_high,
    high_to_semantic, semantic_to_high, norm_xywh_to_xyxy,
    local_box_to_semantic,
)
from .inference import generate_rois
from .fusion import merge_local_predictions, safe_replace, empty_predictions


def _det_component_metrics(items, prefix: str) -> dict:
    """Report YOLO display loss components without changing their gradients.

    The original v5 criterion returns (box,cls,dfl), while a custom
    criterion might return a dictionary. Missing keys are omitted, NOT
    displayed as artificial zero measurements.
    """
    if isinstance(items, dict):
        norm = {}
        for key, value in items.items():
            key = str(key).lower().strip()
            if key.endswith('_loss'):
                key = key[:-5]
            if key in {'box', 'cls', 'dfl'}:
                norm[key] = value
    elif torch.is_tensor(items):
        values = items.detach().reshape(-1)
        norm = dict(zip(('box', 'cls', 'dfl'), values)) if values.numel() == 3 else {}
    elif isinstance(items, (list, tuple)) and len(items) == 3:
        norm = dict(zip(('box', 'cls', 'dfl'), items))
    else:
        norm = {}
    out = {}
    for key, value in norm.items():
        out[f'{prefix}_{key}_loss'] = (value.detach() if torch.is_tensor(value)
                                       else float(value))
    return out


class GMLSV2GlobalLocalScratch(nn.Module):
    """Jointly trained two-detector model; transparent base global network."""
    def __init__(self, cfg: dict, *, verbose: bool = True):
        super().__init__()
        g = cfg['model'].get('global_local', {})
        if not g.get('enabled', False):
            raise ValueError('Set model.global_local.enabled: true')
        if cfg['model'].get('pretrained') is not False:
            raise ValueError('Scratch training requires model.pretrained: false')
        if g.get('local_pretrained', False):
            raise ValueError('Scratch training requires local_pretrained: false')
        if cfg['train'].get('resume') not in (None, False, ''):
            raise ValueError('Scratch training requires train.resume: null')
        if cfg['model'].get('hvad', {}).get('enabled', False):
            raise ValueError('Use original GMLS-v2, not HVAD')
        if cfg['model'].get('tri_gate', {}).get('enabled', False):
            raise ValueError('Use original GMLS-v2, not Tri-Gate')
        self.cfg = cfg
        self.gl = g
        self.global_model = build_model_from_config(cfg, pretrained=False, verbose=verbose)
        self.local_model = RGBTDetectionModel(
            model_name=str(g.get('local_model', cfg['model']['name'])), nc=1,
            pretrained=False, fusion='add', fusion_indices=[4, 6, 10],
            align_mode='bilinear', names={0: 'UAV'}, verbose=verbose,
        )
        for mdl in (self.global_model, self.local_model):
            mdl.args.box = float(cfg.get('loss', {}).get('box', 7.5))
            mdl.args.cls = float(cfg.get('loss', {}).get('cls', 0.5))
            mdl.args.dfl = float(cfg.get('loss', {}).get('dfl', 1.5))
            mdl.criterion = None
        self.nc = 1
        self.names = {0: 'UAV'}
        self.roi_size = int(g.get('roi_size', 640))
        self.local_weight = float(g.get('lambda_local', 1.0))
        self.gt_crop_prob = float(g.get('gt_crop_prob', 0.70))
        self.gt_jitter = float(g.get('gt_jitter', 0.12))
        self.topk = int(g.get('topk', 2))
        self.epoch = 0
        if self.topk not in (1, 2):
            raise ValueError('First implementation: topk must be 1 or 2')

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def forward(self, batch: Mapping):
        if not isinstance(batch, Mapping):
            raise TypeError('Joint training uses a batch dictionary')
        return self.loss(batch)

    @staticmethod
    def _gt_center_high(box_norm, high_hw):
        h, w = high_hw
        xyxy = norm_xywh_to_xyxy(box_norm[:1], w, h)
        return (xyxy[0, :2] + xyxy[0, 2:]) / 2

    def _train_rois(self, batch: dict, info: dict):
        """ROI proposals from GTGA anchor (+ occasional TRAIN-only GT augmentation)."""
        high = batch['rgb_img']
        tir = batch['tir_img']
        sem_hw = tuple(batch['rgb_semantic_img'].shape[-2:])
        high_hw = tuple(high.shape[-2:])
        bsz = int(high.shape[0])
        anchor = info['coarse_anchor'].detach().float().reshape(bsz, 2).clamp(0, 1)
        scale = info['sampling_scale']
        trans = info['sampling_translation']
        rgb_rois = []
        tir_rois = []
        box_list, cls_list, idx_list = [], [], []
        n_positive = 0
        n_valid = 0
        n_rois = 0
        n_gt_assisted = 0
        for i in range(bsz):
            labels = batch['rgb_bboxes'][batch['rgb_batch_idx'] == i]
            gt_sem_centers = []
            if len(labels):
                for one in labels:
                    high_center = self._gt_center_high(one[None], high_hw)
                    semcenter = high_to_semantic(
                        high_center, batch['rgb_ratio_pad'][i],
                        batch['rgb_semantic_ratio_pad'][i],
                    )
                    gt_sem_centers.append(semcenter)
            pred_xy = anchor[i] * anchor.new_tensor([sem_hw[1] - 1, sem_hw[0] - 1])
            centers = [pred_xy]
            if self.topk == 2:
                if gt_sem_centers and random.random() < self.gt_crop_prob:
                    one = gt_sem_centers[random.randrange(len(gt_sem_centers))].detach().clone()
                    # Real 640-pixel ROI: default jitter ±0.12*640 high px.
                    jitter_high = self.roi_size * self.gt_jitter
                    p_hi = semantic_to_high(
                        one, batch['rgb_semantic_ratio_pad'][i], batch['rgb_ratio_pad'][i]
                    )
                    off = p_hi.new_tensor([random.uniform(-jitter_high, jitter_high),
                                           random.uniform(-jitter_high, jitter_high)])
                    centers.append(high_to_semantic(
                        p_hi + off, batch['rgb_ratio_pad'][i],
                        batch['rgb_semantic_ratio_pad'][i]
                    ))
                    n_gt_assisted += 1
                else:
                    # GT-free local exploration around GTGA predicted anchor.
                    centers.append(pred_xy + pred_xy.new_tensor([
                        random.uniform(-60., 60.), random.uniform(-60., 60.)
                    ]))
            for center in centers:
                rgb_roi, tir_roi, geom = sample_native_hr_and_tir(
                    high[i], tir[i], center.detach(), sem_hw,
                    batch['rgb_ratio_pad'][i], batch['rgb_semantic_ratio_pad'][i],
                    scale[i], trans[i], roi_size=self.roi_size,
                )
                rgb_rois.append(rgb_roi)
                tir_rois.append(tir_roi)
                crop_boxes = local_gt_from_high(
                    labels, geom.x0, geom.y0, self.roi_size,
                    high_hw[1], high_hw[0],
                    min_visible=float(self.gl.get('min_gt_visible', .60)),
                )
                ni = len(crop_boxes)
                if ni:
                    box_list.append(crop_boxes)
                    cls_list.append(torch.zeros(ni, 1, device=high.device, dtype=torch.float32))
                    idx_list.append(torch.full((ni,), n_rois, dtype=torch.int64, device=high.device))
                    n_positive += 1
                n_valid += geom.tir_valid_fraction
                n_rois += 1
        if not rgb_rois:
            raise RuntimeError('No ROI crops generated')
        rgb_roi = torch.stack(rgb_rois, 0)
        tir_roi = torch.stack(tir_rois, 0)
        local_batch = {
            'rgb_img': rgb_roi,
            'tir_img': tir_roi,
            'rgb_cls': (torch.cat(cls_list) if cls_list else high.new_zeros((0, 1))),
            'rgb_bboxes': (torch.cat(box_list) if box_list else high.new_zeros((0, 4))),
            'rgb_batch_idx': (torch.cat(idx_list) if idx_list else torch.zeros(0, dtype=torch.int64, device=high.device)),
        }
        return local_batch, n_positive, float(n_valid / max(n_rois, 1)), n_gt_assisted

    def loss(self, batch: dict):
        g = self.global_model
        if g.criterion is None:
            g.criterion = g.init_criterion()
        rgb_mask, tir_mask = g.build_alignment_valid_masks_from_batch(batch)
        out = g.predict(
            batch['rgb_img'], batch['tir_img'],
            rgb_semantic=batch['rgb_semantic_img'],
            rgb_valid_mask=rgb_mask, tir_valid_mask=tir_mask,
            return_features=True,
            return_alignment_debug=g.alignment_loss_enabled,
        )
        target = {
            'img': batch['rgb_semantic_img'], 'cls': batch['rgb_cls'],
            'bboxes': batch['rgb_semantic_bboxes'], 'batch_idx': batch['rgb_batch_idx'],
        }
        det_raw, det_items = g.criterion(out['pred'], target)
        global_det = det_raw.sum()
        if g.alignment_loss_enabled:
            st = g._alignment_auxiliary_loss(batch, out['alignment_info'])
            align_loss = g.lambda_alignment * st['total'] * int(batch['rgb_img'].shape[0])
        else:
            align_loss = global_det.new_zeros(())
        local_batch, positive, valid, gt_assisted = self._train_rois(batch, out['alignment_info'])
        loc_raw, loc_items = self.local_model(local_batch)
        # Ultralytics detection-loss values are batch-size normalized already;
        # restore comparable per-image influence of B*K local batch.
        bsz = int(batch['rgb_img'].shape[0])
        nroi = int(local_batch['rgb_img'].shape[0])
        local_det = loc_raw.sum() * (bsz / nroi)
        loss = global_det + align_loss + self.local_weight * local_det
        parts = {
            'global_det_loss': global_det.detach(),
            'align_weighted_loss': align_loss.detach(),
            'local_det_loss': local_det.detach(),
            'local_weighted_loss': (self.local_weight * local_det).detach(),
            'roi_positive': float(positive / max(nroi, 1)),
            'roi_gt_assisted': float(gt_assisted / max(nroi, 1)),
            'roi_tir_valid_fraction': float(valid),
            'train_rois': float(nroi),
        }
        parts.update(_det_component_metrics(det_items, 'global'))
        parts.update(_det_component_metrics(loc_items, 'local'))
        if g.alignment_loss_enabled:
            for metric_key, source_key in (
                ('align_total', 'total'),
                ('align_anchor', 'anchor'),
                ('align_coarse', 'coarse'),
                ('align_scale', 'scale'),
                ('align_targetness', 'targetness'),
                ('align_v2_quality', 'v2_quality'),
                ('align_v2_route', 'v2_route'),
                ('gmls_gt_quality', 'v2_gmls_gt_quality'),
                ('dmls_gt_quality', 'v2_dmls_gt_quality'),
            ):
                if source_key in st:
                    value = st[source_key]
                    parts[metric_key] = value.detach() if torch.is_tensor(value) else float(value)
            parts['align_valid_pairs'] = float(st.get('valid_pairs', 0))
        return loss, parts

    @torch.no_grad()
    def predict_global_local(self, batch: dict, *, conf=0.001, iou=0.70):
        """Return per-image global, local-only and conservative merged boxes.

        No GT labels are read anywhere in this method.
        """
        if self.global_model.training or self.local_model.training:
            raise RuntimeError('Call model.eval() before predict_global_local')
        g_out = self.global_model.predict(
            batch['rgb_img'], batch['tir_img'],
            rgb_semantic=batch['rgb_semantic_img'], return_features=True,
        )
        gp = postprocess_predictions(g_out['pred'], self.global_model,
                                     conf_thres=conf, iou_thres=iou)
        roi_groups, _ = generate_rois(
            self.global_model, batch, topk=self.topk, roi_size=self.roi_size,
            min_score=float(self.gl.get('proposal_min_score', conf)),
            min_center_dist=float(self.gl.get('min_center_dist', 32.)),
            predictions=gp, alignment_info=g_out['alignment_info'],
        )
        rgb_all=[];tir_all=[];geometry=[]
        for i, rois in enumerate(roi_groups):
            for roi in rois:
                rgb_all.append(roi.rgb)
                tir_all.append(roi.tir)
                geometry.append((i, roi.geometry))
        rgb_batch=torch.stack(rgb_all,0)
        tir_batch=torch.stack(tir_all,0)
        lr = self.local_model.predict(rgb_batch,tir_batch)
        lp = postprocess_predictions(lr, self.local_model, conf_thres=conf,
                                      iou_thres=iou)
        transformed=[[] for _ in range(len(gp))]
        sem_hw=tuple(batch['rgb_semantic_img'].shape[-2:])
        for one_pred,(idx,geo) in zip(lp,geometry):
            b=local_box_to_semantic(one_pred['bboxes'],geo,
                  batch['rgb_ratio_pad'][idx],batch['rgb_semantic_ratio_pad'][idx],
                  sem_hw)
            transformed[idx].append({'bboxes':b,'conf':one_pred['conf'], 'cls':one_pred['cls']})
        locals_=[];merged=[];counts=[]
        for i, group in enumerate(transformed):
            lo=merge_local_predictions(group, max_det=300)
            fu,stat=safe_replace(
                gp[i],lo,
                min_local_conf=float(self.gl.get('replace_min_local_conf',.25)),
                min_iou=float(self.gl.get('replace_min_iou',.12)),
                unmatched_add_conf=float(self.gl.get('unmatched_add_conf',.90)),
            )
            locals_.append(lo);merged.append(fu);counts.append(stat)
        return {'global':gp,'local':locals_,'fused':merged,'merge_stats':counts,
                'roi_count':len(geometry),
                # GT-free proposals; evaluator can post-hoc measure GT coverage.
                'roi_geometries': [[one.geometry for one in group] for group in roi_groups],
                'alignment_geometry': {
                    'sampling_scale': g_out['alignment_info']['sampling_scale'].detach(),
                    'sampling_translation': g_out['alignment_info']['sampling_translation'].detach(),
                }}
