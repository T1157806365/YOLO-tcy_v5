"""Strict loading, parameter policy, original losses and real-batch checks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from .bridge import GuidanceBridge

DEFAULT_WEIGHTS = ('runs/ga_monoalign_pcgrad_100e/'
    'yolo26m_UAVCB_OfficialPT_MonoKD_PCGrad_100e_Step_seed0_v1_dualTeacherInit/'
    'weights/best.pt')


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError('Expected YAML mapping')
    if cfg.get('format') != 'ga_module_only_v1':
        raise ValueError('Expected ga_module_only_v1 configuration')
    t = cfg['tune']
    if int(t['epochs']) < 1 or int(t['batch']) < 1 or int(t['accumulate']) < 1:
        raise ValueError('epochs, batch, accumulate must be positive')
    if ',' in str(t['device']):
        raise ValueError('This release uses ONE GPU. Set device: "3", not "3,4".')
    if not 0 < float(t['bridge_eta_max']) <= 1 or int(t['bridge_warmup_epochs']) < 0:
        raise ValueError('Invalid bridge settings')
    for key in ('lr_alignment', 'lr_bhlr', 'lr_guidance'):
        if float(t[key]) <= 0:
            raise ValueError(key + ' must be positive')
    if not 0 < float(t['lrf']) <= 1 or float(t['weight_decay']) < 0:
        raise ValueError('Invalid lrf/weight_decay')
    if int(t['workers']) < 0 or int(t['val_batch']) < 1:
        raise ValueError('Invalid workers/val_batch')
    if float(t['grad_clip']) <= 0:
        raise ValueError('grad_clip must be positive')
    if not t.get('freeze_all_bn', True):
        raise ValueError('This module-only release freezes all BN running statistics')
    return cfg


def choose_device(spec):
    if str(spec) == 'cpu':
        return torch.device('cpu')
    value = str(spec).removeprefix('cuda:')
    if not value.isdigit() or not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device unavailable: {spec}')
    index = int(value)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f'GPU {index} unavailable; visible GPU count={torch.cuda.device_count()}')
    torch.cuda.set_device(index)
    return torch.device('cuda', index)


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    # CUDA grid_sample backward has no deterministic implementation in some
    # supported PyTorch versions. Warn rather than failing the user's run.
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def strip_ddp(sd):
    if sd and all(k.startswith('module.') for k in sd):
        return {k[7:]: v for k, v in sd.items()}
    return sd


def validate_source(cfg, state):
    m = cfg.get('model', {})
    if m.get('variant') != 'rlsfa_gtga_centered_bhlr_v2':
        raise ValueError('Source must be the original rlsfa_gtga_centered_bhlr_v2 checkpoint')
    if m.get('fusion') != 'add' or m.get('name') != 'yolo26m':
        raise ValueError('Expected original YOLO26m + Add source')
    if m.get('rlsfa_gtga_centered', {}).get('enabled') is not True:
        raise ValueError('Source checkpoint has alignment disabled; do not silently enable it')
    if m.get('bhlr', {}).get('enabled') is not True:
        raise ValueError('Source checkpoint has BHLR disabled')
    if not cfg.get('loss', {}).get('alignment', {}).get('enabled', True):
        raise ValueError('Original alignment supervision must be enabled')
    if float(cfg['loss']['alignment'].get('lambda_alignment', .2)) <= 0:
        raise ValueError('Original alignment loss weight must be positive')
    for section in ('aghlr', 'agif', 'agif_v1', 'agif_v2', 'aghr', 'aghr_qnet', 'locp3'):
        for node in (cfg, m):
            val = node.get(section, {})
            if isinstance(val, dict) and val.get('enabled', False):
                raise ValueError(f'{section} enabled: select the original 61.16% checkpoint')
    bad = [k for k in state if any(s in k.lower() for s in
           ('aghlr.', 'agif', 'hr_query.', 'locp3.'))]
    if bad:
        raise ValueError('Source contains later experimental modules: ' + str(bad[:5]))
    if not any(k.startswith('core.') for k in state):
        raise ValueError('Expected full GA wrapper checkpoint with core and two aux heads')
    if not all(any(k.startswith(p) for k in state) for p in ('aux_rgb.', 'aux_tir.')):
        raise ValueError('Source is not the complete dual-teacher-initialized GA student')
    d = cfg.get('data', {})
    if (int(d.get('rgb_imgsz', 0)), int(d.get('rgb_semantic_imgsz', 0)),
            int(d.get('tir_imgsz', 0))) != (1920, 640, 640):
        raise ValueError('This release preserves RGB1920 / RGBsem640 / TIR640')


def load_student(weights, device='cpu'):
    """Bypass teacher initialization and version-specific training validators."""
    from models.gtga_bhlr_factory import build_model_from_config
    from ga_monoalign_pcgrad.model import AlignmentProtectedStudent

    path = Path(weights).expanduser().resolve()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or 'config' not in ck or 'model_state_dict' not in ck:
        raise ValueError('Expected complete project checkpoint with config/model_state_dict')
    cfg = copy.deepcopy(ck['config'])
    sd = strip_ddp(ck['model_state_dict'])
    validate_source(cfg, sd)
    core = build_model_from_config(copy.deepcopy(cfg), pretrained=False, verbose=False)
    if type(core).__name__ != 'RGBTRLSFAGTGACenteredBHLRV2DetectionModel':
        raise RuntimeError('Factory constructed a different model: ' + type(core).__name__)
    student = AlignmentProtectedStudent(core, cfg)
    student.load_state_dict(sd, strict=True)
    # Aliases (core.rsdt/core.bhlr etc.) can otherwise overwrite each other
    # silently during load_state_dict. Check ALL keys after the load.
    after = student.state_dict()
    mismatch = [k for k in sd if not torch.equal(after[k].cpu(), sd[k].cpu())]
    if mismatch:
        raise RuntimeError('Checkpoint did not restore exactly: ' + str(mismatch[:8]))
    if not core.alignment_enabled or not core.bhlr_enabled:
        raise RuntimeError('Original alignment/BHLR forward switches are disabled')
    if core.bhlr_guidance_mode != 'alignment':
        raise RuntimeError('Original BHLR must consume alignment guidance')
    if not getattr(core, 'alignment_loss_enabled', False):
        raise RuntimeError('Alignment supervision is disabled in the constructed core')
    if getattr(student, '_teachers', None) is not None:
        raise RuntimeError('Unexpected teacher initialization')
    student.to(device).eval()
    # Retain metadata, not another full CPU copy of student weights.
    meta = {k: v for k, v in ck.items() if k not in
            ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
             'scaler_state_dict', 'rng_state')}
    return student, meta, cfg


class ModulePolicy:
    def __init__(self, student):
        self.student = student
        core = student.core
        self.roots = {'alignment': core.alignment, 'bhlr': core.bhlr,
                      'guidance': core.alignment_guidance_fusion}
        original_flags = {id(p): p.requires_grad for p in student.parameters()}
        student.requires_grad_(False)
        self.groups = {}
        assigned = set()
        for group, module in self.roots.items():
            params = []
            for p in module.parameters():
                if original_flags[id(p)] and id(p) not in assigned:
                    p.requires_grad_(True)
                    params.append(p)
                    assigned.add(id(p))
            if not params:
                raise RuntimeError(f'No trainable parameters in original {group} module')
            self.groups[group] = params
        self.names = {n for n, p in student.named_parameters() if p.requires_grad}
        self.frozen_names = {n for n, p in student.named_parameters() if not p.requires_grad}
        self.train()

    def train(self):
        s = self.student
        s.train()
        s.aux_rgb.eval()
        s.aux_tir.eval()
        # Set the frozen backbone components to eval without setting the final
        # Detect module to eval: YOLO E2E loss requires its training outputs.
        for layer in list(s.core.model)[:s.core.backbone_len]:
            layer.eval()
        s.core.tir_backbone.eval()
        for mod in s.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()
            if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                mod.eval()
        if not s.core.model[-1].training:
            raise RuntimeError('Detect must stay in training mode for original YOLO loss')
        if any(m.training for m in s.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError('BN statistics were not frozen')

    def report(self):
        return {'trainable_parameters': {g: sum(p.numel() for p in ps)
                                        for g, ps in self.groups.items()},
                'trainable_names': sorted(self.names),
                'frozen_parameters': sum(p.numel() for p in self.student.parameters()
                                         if not p.requires_grad),
                'all_bn_statistics_frozen': True}

    def verify_no_frozen_grad(self):
        bad = [n for n, p in self.student.named_parameters()
               if not p.requires_grad and p.grad is not None]
        if bad:
            raise RuntimeError('Frozen parameters received gradients: ' + str(bad[:8]))


def frozen_digest(student):
    """All frozen parameters AND all registered buffers, including BN stats."""
    h = hashlib.sha256()
    items = [('parameter:' + n, p) for n, p in student.named_parameters()
             if not p.requires_grad]
    items.extend(('buffer:' + n, b) for n, b in student.named_buffers())
    for name, value in sorted(items, key=lambda x: x[0]):
        t = value.detach().cpu().contiguous()
        h.update((name + str(t.shape) + str(t.dtype)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def native_rgb(batch):
    return {'img': batch['rgb_semantic_img'], 'cls': batch['rgb_cls'],
            'bboxes': batch['rgb_semantic_bboxes'], 'batch_idx': batch['rgb_batch_idx']}


def original_losses(student, batch):
    """Original GT detection + original paired-box alignment; NO Mono/KD/OHEM."""
    c = student.core
    if c.criterion is None:
        c.criterion = c.init_criterion()
    rm, tm = c.build_alignment_valid_masks_from_batch(batch)
    fs = c.forward_features(batch['rgb_img'], batch['tir_img'],
        rgb_semantic=batch['rgb_semantic_img'], rgb_valid_mask=rm,
        tir_valid_mask=tm, return_alignment_debug=True)
    pred = c.forward_head(fs['fused_backbone'])
    vec, items = c.criterion(pred, native_rgb(batch))
    det = vec.sum()
    stats = c._alignment_auxiliary_loss(batch, fs['alignment_info'])
    bsz = int(batch['rgb_img'].shape[0])
    align = float(c.lambda_alignment) * stats['total'] * bsz
    logs = c._loss_items_to_dict(items)
    logs.update({'det_per_image': det.detach() / bsz,
                 'alignment_per_image': align.detach() / bsz})
    for key, value in stats.items():
        logs['align_' + key] = value.detach() if torch.is_tensor(value) else value
    return det, align, logs


def eta_for_epoch(tune, epoch):
    warm = int(tune['bridge_warmup_epochs'])
    return float(tune['bridge_eta_max']) * (min(1., epoch / warm) if warm else 1.)


def build_optimizer(policy, tune):
    groups = [{'params': ps, 'lr': float(tune['lr_' + name]), 'name': name}
              for name, ps in policy.groups.items()]
    return torch.optim.SGD(groups, momentum=float(tune['momentum']),
                           weight_decay=float(tune['weight_decay']), nesterov=True)


def make_loader(source_cfg, tune, split, augment=False, shuffle=False, batch=None):
    from engine.trainer import _build_data
    cfg = copy.deepcopy(source_cfg)
    cfg.setdefault('train', {})['workers'] = int(tune['workers'])
    return _build_data(cfg, split, augment, shuffle,
        batch_size=int(batch or tune['batch']), distributed=False, rank=0, world_size=1)


def predictions(student, batch):
    masks = student.core.build_alignment_valid_masks_from_batch(batch)
    return student.predict(batch['rgb_img'], batch['tir_img'],
        rgb_semantic=batch['rgb_semantic_img'], rgb_valid_mask=masks[0], tir_valid_mask=masks[1])


def flatten_tensors(value, prefix=''):
    if torch.is_tensor(value):
        return {prefix: value}
    out = {}
    if isinstance(value, dict):
        for key in sorted(value):
            out.update(flatten_tensors(value[key], prefix + '/' + str(key)))
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            out.update(flatten_tensors(child, prefix + '/' + str(i)))
    return out


def grad_norm(grads):
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in grads if g is not None))


def grad_difference(a, b):
    result = []
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None:
            x = torch.zeros_like(y)
        if y is None:
            y = torch.zeros_like(x)
        result.append(x - y)
    return grad_norm(result)


def real_preflight(student, policy, batch, tune):
    """Restored after testing: forward parity, gradient flow, real optimizer step."""
    saved = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    report = policy.report()
    bridge = None
    try:
        student.eval()
        with torch.no_grad():
            before = {k: v.detach().clone() for k, v in flatten_tensors(
                predictions(student, batch)).items()}
        bridge = GuidanceBridge(student.core, float(tune['bridge_eta_max']))
        with torch.no_grad():
            after = flatten_tensors(predictions(student, batch))
        if before.keys() != after.keys() or not before:
            raise RuntimeError('Prediction structure changed after installing bridge')
        errors = [float((before[k].float() - after[k].float()).abs().max())
                  for k in before if before[k].numel()]
        max_diff = max(errors, default=0.)
        if not math.isfinite(max_diff) or max_diff != 0.:
            raise RuntimeError(f'Bridge changed forward predictions: max diff={max_diff}')
        report['forward_max_abs_diff'] = max_diff
        del before, after
        policy.train()
        frozen_before = frozen_digest(student)
        alignment_params = policy.groups['alignment']
        bridge.set_eta(0.)
        det0, align0, _ = original_losses(student, batch)
        ga0 = torch.autograd.grad(det0, alignment_params, allow_unused=True)
        ga0 = tuple(None if g is None else g.detach().clone() for g in ga0)
        det0_value = float(det0.detach())
        del det0, align0
        bridge.set_eta(float(tune['bridge_eta_max']))
        bridge.reset_counts()
        det, align, _ = original_losses(student, batch)
        bridge.assert_used()
        if not torch.isfinite(det + align):
            raise RuntimeError('Nonfinite loss in preflight')
        if not math.isclose(det0_value, float(det.detach()), rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError('Training forward changed across bridge strengths')
        grads = {}
        for name, params in policy.groups.items():
            grads[name] = torch.autograd.grad(det, params, retain_graph=True, allow_unused=True)
        norms = {name: grad_norm(g) for name, g in grads.items()}
        if any(not math.isfinite(v) or v <= 0 for v in norms.values()):
            raise RuntimeError('Detection-loss gradient missing/nonfinite: ' + str(norms))
        delta = grad_difference(grads['alignment'], ga0)
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError('Bridge does not change alignment gradients; inspect hidden detach/zero BHLR')
        ag = torch.autograd.grad(align, alignment_params, retain_graph=True, allow_unused=True)
        anorm = grad_norm(ag)
        if not math.isfinite(anorm) or anorm <= 0:
            raise RuntimeError('No alignment-supervision gradient; use a batch containing paired UAV GT')
        report.update({'detection_gradient_norms': norms,
                       'alignment_supervision_gradient_norm': anorm,
                       'alignment_gradient_delta_eta0_to_max': delta,
                       'support_calls': bridge.support_calls, 'adapter_calls': bridge.adapter_calls})
        del ga0, grads, ag
        optimizer = build_optimizer(policy, tune)
        optimizer.zero_grad(set_to_none=True)
        ((det + align) / int(batch['rgb_img'].shape[0])).backward()
        policy.verify_no_frozen_grad()
        torch.nn.utils.clip_grad_norm_([p for ps in policy.groups.values() for p in ps],
                                     float(tune['grad_clip']), error_if_nonfinite=True)
        optimizer.step()
        if frozen_before != frozen_digest(student):
            raise RuntimeError('A frozen parameter or buffer changed during the preflight step')
        report['frozen_parameters_and_buffers_unchanged'] = True
        report['one_real_optimizer_step_passed'] = True
    finally:
        student.load_state_dict(saved, strict=True)
        student.zero_grad(set_to_none=True)
        if bridge is not None:
            bridge.close()
        student.eval()
    return report


def finite_metrics(metrics):
    ap = float(metrics['map5095'])
    if not math.isfinite(ap):
        raise RuntimeError('Nonfinite validation AP')
    return ap


def json_safe(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)
