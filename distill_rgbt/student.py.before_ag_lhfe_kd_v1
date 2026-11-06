"""Original-main RGB-T detector + optional training-only Mono, ROI-KD and feature PCGrad.

Never changes the geometry estimator, BHLR, fusion or inference head.
"""
from __future__ import annotations
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.rgbt_model import get_layer_input
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from .teachers import DualTeachers, roi_attention_kd


class NativeDetectionHead(nn.Module):
    """Exact original YOLO neck/detect clone, trained only as an auxiliary head."""
    def __init__(self, core):
        super().__init__()
        self.backbone_len = int(core.backbone_len)
        self.model = nn.ModuleList([deepcopy(x) for x in list(core.model)[self.backbone_len:]])
        self.nc, self.names, self.stride = core.nc, core.names, core.stride
        self.args, self.end2end = deepcopy(core.args), bool(core.end2end)
        self.criterion = None

    def forward(self, backbone_outputs):
        outputs = list(backbone_outputs)
        if len(outputs) != self.backbone_len:
            raise RuntimeError('Mono head received incorrect number of backbone layers')
        x = outputs[-1]
        for layer in self.model:
            x = layer(get_layer_input(layer, x, outputs))
            outputs.append(x)
        return x

    def calc_loss(self, pred, batch):
        if self.criterion is None:
            self.criterion = E2ELoss(self) if self.end2end else v8DetectionLoss(self)
        return self.criterion(pred, batch)


class DistillStudent(nn.Module):
    def __init__(self, core, cfg):
        super().__init__()
        self.core, self.cfg = core, cfg
        self.model_name, self.fusion_type = core.model_name, core.fusion_type
        self.nc, self.names, self.end2end, self.stride = core.nc, core.names, core.end2end, core.stride
        self.mono_enabled = bool(cfg.get('mono', {}).get('enabled', False))
        self.aux_rgb = NativeDetectionHead(core) if self.mono_enabled else None
        self.aux_tir = NativeDetectionHead(core) if self.mono_enabled else None
        # Frozen teachers must NOT be part of Module/optimizer/DDP/state_dict.
        object.__setattr__(self, '_teachers', None)
        self.stage, self.epoch = 'stage1', 1
        self._pcgrad_backward_scale = 1.0

    def set_progress(self, stage, epoch):
        self.stage, self.epoch = str(stage), int(epoch)

    def attach_teachers(self, device):
        if self._teachers is None:
            obj = DualTeachers(self.cfg['distill']['teachers'],
                               int(self.core.scale_to_index[3]), device, nc=self.nc)
            object.__setattr__(self, '_teachers', obj)
            print('[KD] Frozen RGB/TIR teachers loaded (not optimized)', flush=True)
        return self._teachers

    def kd_weight(self):
        cfg = self.cfg['distill']['kd']
        if not bool(cfg.get('enabled', True)):
            return 0.0
        if self.stage == 'stage2':
            return float(cfg.get('stage2_weight', 0.005))
        e = int(self.epoch)
        st, reach, hold, end = [int(cfg[k]) for k in ('start_epoch','reach_epoch','hold_until_epoch','end_epoch')]
        peak = float(cfg.get('peak_weight', 0.01))
        if not (0 < st < reach <= hold < end):
            raise ValueError('distill.kd schedule must satisfy 0 < start < reach <= hold < end')
        if e <= st: return 0.0
        if e <= reach: return peak * (e - st) / (reach - st)
        if e <= hold: return peak
        if e <= end: return peak * (end - e) / (end - hold)
        return 0.0

    def mono_weight(self):
        p = self.cfg.get('mono', {})
        if not bool(p.get('enabled', False)): return 0.0
        if self.stage == 'stage2': return float(p.get('stage2_weight', 0.05))
        w, warm = float(p.get('peak_weight', 0.10)), max(1, int(p.get('warmup_epochs', 30)))
        return w * min(1.0, self.epoch / warm)

    @staticmethod
    def _native(batch, modality):
        if modality == 'rgb':
            return dict(img=batch['rgb_semantic_img'], cls=batch['rgb_cls'],
                        bboxes=batch['rgb_semantic_bboxes'], batch_idx=batch['rgb_batch_idx'])
        return dict(img=batch['tir_img'], cls=batch['tir_cls'],
                    bboxes=batch['tir_bboxes'], batch_idx=batch['tir_batch_idx'])

    @staticmethod
    def pcgrad_hooks(align_loss, aux_loss, rgb_feats, tir_feats, enabled, backward_scale):
        """Same feature-boundary projection as old ga_monoalign_pcgrad/model.py.

        Only auxiliary gradients opposing alignment are projected; detection
        gradients and the original geometry loss are untouched. AMP scale and
        batch/accumulation normalization are applied to the hook correction.
        """
        out = dict(pcgrad_rgb_cosine=0., pcgrad_tir_cosine=0.,
                   pcgrad_rgb_conflict=0., pcgrad_tir_conflict=0.)
        if not enabled or not align_loss.requires_grad or not aux_loss.requires_grad:
            return out
        pending = []
        for name, feats_all in (('rgb', rgb_feats), ('tir', tir_feats)):
            feats = [f for f in feats_all if torch.is_tensor(f) and f.requires_grad]
            if not feats: continue
            ga = torch.autograd.grad(align_loss, feats, retain_graph=True, allow_unused=True)
            gm = torch.autograd.grad(aux_loss, feats, retain_graph=True, allow_unused=True)
            # FP32 vector operations avoid overflow under mixed precision.
            dot = align_loss.new_zeros((), dtype=torch.float32)
            na, nm = dot.clone(), dot.clone()
            for a, m in zip(ga, gm):
                if a is None or m is None: continue
                a, m = a.detach().float(), m.detach().float()
                dot += (a * m).sum()
                na += a.square().sum()
                nm += m.square().sum()
            cosine = dot / (na.sqrt() * nm.sqrt() + 1e-12)
            conflict = bool((dot < 0).item() and (na > 1e-20).item())
            out[f'pcgrad_{name}_cosine'] = cosine.detach()
            out[f'pcgrad_{name}_conflict'] = float(conflict)
            if conflict:
                coeff = (dot / na.clamp_min(1e-12)).detach()
                for f, a in zip(feats, ga):
                    if a is None: continue
                    correction = (-coeff * a.detach().float() * backward_scale).to(f.dtype)
                    pending.append((f, correction))
        for feat, correction in pending:
            feat.register_hook(lambda grad, delta=correction: grad + delta)
        return out

    def forward(self, rgb, tir=None, rgb_semantic=None, **kwargs):
        if isinstance(rgb, dict): return self.loss(rgb)
        return self.core.predict(rgb, tir, rgb_semantic=rgb_semantic, **kwargs)

    def predict(self, rgb, tir, rgb_semantic=None, **kwargs):
        return self.core.predict(rgb, tir, rgb_semantic=rgb_semantic, **kwargs)

    def aux_predict(self, modality, rgb_img=None, tir_img=None):
        if not self.mono_enabled:
            raise RuntimeError('Auxiliary mono heads disabled')
        if modality == 'rgb':
            return self.aux_rgb(self.core.forward_rgb_backbone(rgb_img))
        if modality == 'tir':
            return self.aux_tir(self.core.forward_tir_backbone(tir_img))
        raise ValueError(modality)

    def loss(self, batch):
        core = self.core
        if core.criterion is None:
            core.criterion = core.init_criterion()
        bsz = int(batch['rgb_semantic_img'].shape[0])
        rgb_mask, tir_mask = core.build_alignment_valid_masks_from_batch(batch)
        feats = core.forward_features(batch['rgb_img'], batch['tir_img'],
            rgb_semantic=batch['rgb_semantic_img'],
            rgb_valid_mask=rgb_mask, tir_valid_mask=tir_mask,
            return_alignment_debug=core.alignment_enabled)
        main_pred = core.forward_head(feats['fused_backbone'])
        det_vec, det_items = core.criterion(main_pred, self._native(batch, 'rgb'))
        det = det_vec.sum()
        logs = core._loss_items_to_dict(det_items)
        align = det * 0.0
        if core.alignment_enabled and core.alignment_loss_enabled:
            stat = core._alignment_auxiliary_loss(batch, feats['alignment_info'])
            align = float(core.lambda_alignment) * stat['total'] * bsz
            logs.update({f'align_{k}': v.detach() if torch.is_tensor(v) else float(v)
                         for k, v in stat.items()})
        wmono, wkd = self.mono_weight(), self.kd_weight()
        mono, kd = det * 0.0, det * 0.0
        rmono, tmono = det * 0.0, det * 0.0
        if self.mono_enabled:
            rp = self.aux_rgb(feats['rgb_backbone_raw'])
            tp = self.aux_tir(feats['tir_backbone'])
            lr, _ = self.aux_rgb.calc_loss(rp, self._native(batch, 'rgb'))
            lt, _ = self.aux_tir.calc_loss(tp, self._native(batch, 'tir'))
            rmono, tmono = lr.sum(), lt.sum()
            # Include heads in graph even at warmup weight=0 (DDP unused-param safety).
            mono = wmono * (rmono + tmono)
        kr, kt = det * 0.0, det * 0.0
        rgb_rate, tir_rate = 0., 0.
        if wkd > 0:
            teachers = self._teachers
            if teachers is None:
                raise RuntimeError('KD teachers must be attached at epoch boundary')
            p3 = int(core.scale_to_index[3])
            sr, st = feats['rgb_backbone_raw'][p3], feats['tir_backbone'][p3]
            with torch.no_grad():
                tr, tt = teachers.get(batch, sr.shape[-2:])
            kr = roi_attention_kd(sr, tr['p3'], tr['good'],
                                  batch['rgb_semantic_bboxes'], batch['rgb_batch_idx'])
            kt = roi_attention_kd(st, tt['p3'], tt['good'],
                                  batch['tir_bboxes'], batch['tir_batch_idx'])
            rgb_rate = tr['good'].float().mean()
            tir_rate = tt['good'].float().mean()
            kd = wkd * bsz * (kr + kt)
        p3p4p5 = [int(core.scale_to_index[s]) for s in (3, 4, 5)]
        pc = self.pcgrad_hooks(align, mono + kd,
            [feats['rgb_backbone_raw'][i] for i in p3p4p5],
            [feats['tir_backbone'][i] for i in p3p4p5],
            enabled=(bool(self.cfg.get('pcgrad', {}).get('enabled', False))
                     and core.alignment_enabled and core.alignment_loss_enabled
                     and (wmono > 0 or wkd > 0)),
            backward_scale=float(self._pcgrad_backward_scale))
        logs.update({
            'det_raw_loss': det.detach() / bsz,
            'align_weighted_loss': align.detach() / bsz,
            'mono_weighted_loss': mono.detach() / bsz,
            'kd_weighted_loss': kd.detach() / bsz,
            'rgb_aux_loss': rmono.detach() / bsz,
            'tir_aux_loss': tmono.detach() / bsz,
            'rgb_kd_response': kr.detach(),
            'tir_kd_response': kt.detach(),
            'teacher_rgb_selected': rgb_rate,
            'teacher_tir_selected': tir_rate,
            'weight_mono': wmono,
            'weight_kd': wkd,
            **pc,
        })
        return torch.stack((det, align, mono, kd)), logs
