"""Online RGB/TIR target-response distillation for YOLO-tcy_v5/main.

No independent single-modality checkpoints are loaded.  Auxiliary RGB/TIR
heads are trained *inside* the joint model on their own native GT boxes.
They are training-only and are never consulted to form inference boxes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from models.rgbt_rlsfa_gtga_centered_bhlr_model import (
    RGBTRLSFAGTGACenteredBHLRV2DetectionModel,
)


def _xywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


def iou_giou(a, b, eps=1e-7):
    """Pairwise corresponding normalized xyxy boxes: returns IoU and GIoU."""
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    intersection = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = area_a + area_b - intersection
    iou = intersection / union.clamp_min(eps)
    outer_lt = torch.minimum(a[..., :2], b[..., :2])
    outer_rb = torch.maximum(a[..., 2:], b[..., 2:])
    outer = (outer_rb - outer_lt).clamp_min(0).prod(-1)
    giou = iou - (outer - union) / outer.clamp_min(eps)
    return iou, giou


class TargetResponseHead(nn.Module):
    """Lightweight training-only P3 detector: response + normalized xywh."""
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 5, 1),
        )

    def forward(self, x):
        return self.net(x)


class OnlineRGBTKD(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    """Minimal invasive subclass preserving main YOLO26m Detect semantics."""

    def __init__(self, *args, kd_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_cfg = dict(kd_cfg or {})
        self.kd_enabled = bool(self.kd_cfg.get('enabled', True))
        p3_id = int(self.scale_to_index[3])
        channels = self._infer_any_layer_channels((p3_id,))[p3_id]
        cr, ct = int(channels['rgb']), int(channels['tir'])
        if cr != ct:
            raise RuntimeError('P3 residual fusion requires equal RGB/TIR channels; got %s/%s' % (cr, ct))
        self.aux_rgb = TargetResponseHead(cr)
        self.aux_tir = TargetResponseHead(ct)
        self.aux_fused = TargetResponseHead(cr)
        self.tir_projection = nn.Conv2d(ct, cr, 1, bias=False)
        nn.init.dirac_(self.tir_projection.weight)
        self.p3_alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.reliability_net = nn.Sequential(nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.zeros_(self.reliability_net[-1].bias)

        # Historical fusion is not used. Avoid unused trainable DDP parameters.
        for module in self.fusions.values():
            module.requires_grad_(False)
        # Auxiliary heads must always contribute to the loss when enabled.
        if not self.kd_enabled:
            for module in (self.aux_rgb, self.aux_tir, self.aux_fused, self.reliability_net):
                module.requires_grad_(False)

    def fuse_backbone_features(self, rgb_outputs, tir_outputs):
        """Suppress historical unconditional P3/P4/P5 ADD fusion."""
        return list(rgb_outputs), {}

    @staticmethod
    def _bi_vec(x, bi, device, default=(0., 0.)):
        if x is None:
            return torch.tensor(default, device=device, dtype=torch.float32)
        y = x[bi].float().reshape(-1)
        return y[:2]

    def _reliability_features(self, info, batch_size):
        rows = []
        device = info['sampling_scale'].device
        for bi in range(batch_size):
            scale = self._bi_vec(info.get('sampling_scale'), bi, device, (1., 1.))
            trans = self._bi_vec(info.get('sampling_translation'), bi, device)
            coarse = self._bi_vec(info.get('coarse_offset'), bi, device) / 20.0
            conf = info.get('coarse_confidence')
            fine_conf = info.get('fine_confidence')
            c = conf[bi].float().reshape(-1)[:1] if conf is not None else scale.new_zeros(1)
            fc = fine_conf[bi].float().reshape(-1)[:1] if fine_conf is not None else scale.new_zeros(1)
            anchor = self._bi_vec(info.get('coarse_anchor'), bi, device, (.5, .5))
            rows.append(torch.cat([scale, trans, coarse, c, fc, anchor]))
        return torch.stack(rows).detach()  # gradients for quality head, not geometry via this branch

    def forward_features(self, *args, **kwargs):
        features = super().forward_features(*args, **kwargs)
        p3_id = int(self.scale_to_index[3])
        rgb_feat = features['rgb_backbone'][p3_id]     # includes BHLR if enabled
        tir_feat = features['aligned_tir_backbone'][p3_id]
        info = features['alignment_info']
        bsz = rgb_feat.shape[0]
        rel_logit = self.reliability_net(self._reliability_features(info, bsz))
        reliability = rel_logit.sigmoid().view(bsz, 1, 1, 1)
        fused_p3 = rgb_feat + torch.tanh(self.p3_alpha_raw) * reliability * self.tir_projection(tir_feat)
        fused_outputs = list(features['rgb_backbone'])
        fused_outputs[p3_id] = fused_p3
        features['fused_backbone'] = fused_outputs
        features['fusion_features'] = {p3_id: fused_p3}
        features['online_reliability_logit'] = rel_logit
        features['online_reliability'] = reliability
        return features

    @staticmethod
    def _heatmap_loss(logits, targets):
        # Dense small-object target: avoid background-only trivial solution.
        weight = 1.0 + 20.0 * targets
        bce = (F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none') * weight).mean()
        pred = logits.float().sigmoid()
        dice = 1.0 - (2 * (pred * targets).sum((-2,-1)) + 1.0) / (
            pred.sum((-2,-1)) + targets.sum((-2,-1)) + 1.0
        )
        return bce + .25 * dice.mean()

    def _aux_detection_loss(self, prediction, target, boxes, batch_idx):
        """Dense Gaussian response + sparse GT-center xywh box supervision."""
        response = prediction[:, :1]
        bboxes = prediction[:, 1:5].float().sigmoid()
        out = self._heatmap_loss(response, target)
        b, _, h, w = response.shape
        reg_losses = []
        for j in range(len(boxes)):
            bi = int(batch_idx[j].item())
            gt = boxes[j].float().clamp(1e-5, 1.0)
            ix = int((gt[0].detach() * w).long().clamp(0, w - 1).item())
            iy = int((gt[1].detach() * h).long().clamp(0, h - 1).item())
            pred_box = bboxes[bi, :, iy, ix]
            _, giou = iou_giou(_xywh_to_xyxy(pred_box), _xywh_to_xyxy(gt))
            reg_losses.append((1. - giou) + F.smooth_l1_loss(pred_box, gt, reduction='mean'))
        # No valid GT: keep box channels on the graph for robust DDP.
        reg = torch.stack(reg_losses).mean() if reg_losses else bboxes.sum() * 0.
        return out + float(self.kd_cfg.get('aux_box_weight', .5)) * reg

    def _final_geometry_loss(self, batch, info):
        """Return center/GIoU/quality BCE. GT is used ONLY in training."""
        bsz = int(batch['rgb_semantic_img'].shape[0])
        scales = info['sampling_scale'].float()
        translations = info['sampling_translation'].float()
        rel_logit = info['online_reliability_logit'].flatten()
        centers, gious, rels, all_ious = [], [], [], []
        for bi in range(bsz):
            rsel = batch['rgb_batch_idx'] == bi
            tsel = batch['tir_batch_idx'] == bi
            if int(rsel.sum()) != 1 or int(tsel.sum()) != 1:
                continue
            r = batch['rgb_semantic_bboxes'][rsel][0].float()
            t = batch['tir_bboxes'][tsel][0].float()
            scale = scales[bi].reshape(-1)[:2].clamp_min(1e-5)
            trans = translations[bi].reshape(-1)[:2]
            mapped = torch.cat(((t[:2] - trans) / scale, t[2:] / scale))
            iou, giou = iou_giou(_xywh_to_xyxy(mapped), _xywh_to_xyxy(r))
            # Center measured in P3 cells so small UAV error remains meaningful.
            h, w = batch['rgb_semantic_img'].shape[-2:]
            d = (mapped[:2] - r[:2]) * r.new_tensor([w / 8., h / 8.])
            centers.append(F.smooth_l1_loss(d, torch.zeros_like(d), beta=1., reduction='mean'))
            gious.append(1. - giou)
            all_ious.append(iou.detach())
            rels.append(F.binary_cross_entropy_with_logits(rel_logit[bi].float(), iou.detach()))
        zero = scales.sum() * 0. + rel_logit.sum() * 0.
        return (
            torch.stack(centers).mean() if centers else zero,
            torch.stack(gious).mean() if gious else zero,
            torch.stack(rels).mean() if rels else zero,
            torch.stack(all_ious).mean() if all_ious else zero.detach(),
        )

    def loss(self, batch, preds=None):
        if preds is not None:
            return super().loss(batch, preds=preds)
        if self.criterion is None:
            self.criterion = self.init_criterion()
        rgb_mask, tir_mask = self.build_alignment_valid_masks_from_batch(batch)
        out = self.predict(
            batch['rgb_img'], batch['tir_img'],
            rgb_semantic=batch['rgb_semantic_img'],
            rgb_valid_mask=rgb_mask, tir_valid_mask=tir_mask,
            return_features=True, return_alignment_debug=True,
        )
        target_batch = {
            'img': batch['rgb_semantic_img'],
            'cls': batch['rgb_cls'], 'bboxes': batch['rgb_semantic_bboxes'],
            'batch_idx': batch['rgb_batch_idx'],
        }
        det_raw, det_items = self.criterion(out['pred'], target_batch)
        loss_terms = [det_raw.reshape(-1)]
        log = self._loss_items_to_dict(det_items)
        bsz = int(batch['rgb_semantic_img'].shape[0])
        info = out['alignment_info']
        if self.alignment_loss_enabled:
            stats = self._alignment_auxiliary_loss(batch, info)
            weighted = self.lambda_alignment * stats['total']
            loss_terms.append((weighted * bsz).reshape(1))
            log['align_loss'] = stats['total'].detach()
            log['align_weighted_loss'] = weighted.detach()

        if self.kd_enabled:
            p3_id = int(self.scale_to_index[3])
            raw_rgb = out['rgb_backbone_raw'][p3_id]
            raw_tir = out['tir_backbone'][p3_id]
            fused = out['fused_backbone'][p3_id]
            pr = self.aux_rgb(raw_rgb).float()
            pt = self.aux_tir(raw_tir).float()
            pf = self.aux_fused(fused).float()
            lr, lt, lf = pr[:, :1], pt[:, :1], pf[:, :1]
            h, w = lr.shape[-2:]
            rg = self._build_gaussian_target(
                batch['rgb_semantic_bboxes'], batch['rgb_batch_idx'], bsz, h, w,
                lr.device, lr.dtype, sigma_factor=.25, sigma_min_px=1.
            )
            tg = self._build_gaussian_target(
                batch['tir_bboxes'], batch['tir_batch_idx'], bsz, h, w,
                lt.device, lt.dtype, sigma_factor=.25, sigma_min_px=1.
            )
            loss_rgb = self._aux_detection_loss(pr, rg, batch['rgb_semantic_bboxes'], batch['rgb_batch_idx'])
            loss_tir = self._aux_detection_loss(pt, tg, batch['tir_bboxes'], batch['tir_batch_idx'])
            loss_fused = self._aux_detection_loss(pf, rg, batch['rgb_semantic_bboxes'], batch['rgb_batch_idx'])

            # Online RGB response teacher (detach teacher; selected only where teacher closer to GT).
            t_rgb = lr.detach().sigmoid()
            s_fused = lf.sigmoid()
            mask_rgb = (
                (rg > float(self.kd_cfg.get('target_threshold', .08)))
                & ((s_fused.detach() - rg).abs() > (t_rgb - rg).abs() + .03)
            ).float()
            rgb_err = F.binary_cross_entropy_with_logits(lf, t_rgb, reduction='none')
            kd_rgb = (rgb_err * mask_rgb).sum() / mask_rgb.sum().clamp_min(1.)

            # TIR teacher stays in native TIR coordinate, then one predicted warp.
            with torch.no_grad():
                t_tir = lt.detach().sigmoid()
                t_tir_warp = self.alignment.warp_tensor(
                    t_tir,
                    {'sampling_scale': info['sampling_scale'].detach(),
                     'sampling_translation': info['sampling_translation'].detach()},
                    output_hw=(h,w), padding_mode='zeros',
                ).clamp(0, 1)
            mask_tir = (t_tir_warp > float(self.kd_cfg.get('tir_threshold', .20))).float()
            # Gate is detached so it cannot trivially collapse to avoid KD.
            reliable = out['online_reliability'].detach()
            te = F.binary_cross_entropy_with_logits(lf, t_tir_warp, reduction='none')
            kd_tir = (te * mask_tir * reliable).sum() / (mask_tir * reliable).sum().clamp_min(1.)

            lc, lg, lrel, liou = self._final_geometry_loss(batch, {**info, 'online_reliability_logit': out['online_reliability_logit']})
            k = self.kd_cfg
            weights = {
                'aux_rgb': (loss_rgb, float(k.get('lambda_aux_rgb', .10))),
                'aux_tir': (loss_tir, float(k.get('lambda_aux_tir', .10))),
                'aux_fused': (loss_fused, float(k.get('lambda_aux_fused', .10))),
                'rgb_keep': (kd_rgb, float(k.get('lambda_rgb_keep', .05))),
                'tir_kd': (kd_tir, float(k.get('lambda_tir_kd', .02))),
                'reliability': (lrel, float(k.get('lambda_reliability', .10))),
                'final_center': (lc, float(k.get('lambda_final_center', .02))),
                'final_giou': (lg, float(k.get('lambda_final_giou', .10))),
            }
            extra = sum(weight * val for val, weight in weights.values())
            loss_terms.append((extra * bsz).reshape(1))
            for name, (value, _) in weights.items():
                log['kd_' + name] = value.detach()
            log['kd_final_iou'] = liou.detach()
            log['kd_p3_alpha'] = self.p3_alpha_raw.detach().tanh()
            log['kd_reliability_mean'] = out['online_reliability'].detach().mean()
            log['kd_weighted_total'] = extra.detach()
        else:
            # Ensure zero initialized P3 residual adapter gets gradient in ablation without KD.
            zero = out['fused_backbone'][int(self.scale_to_index[3])].sum() * 0.
            loss_terms.append(zero.reshape(1))
        return torch.cat(loss_terms), log
