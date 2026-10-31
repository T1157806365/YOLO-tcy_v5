"""Strict teacher-to-student initialization (training-time only)."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn


def _teacher_model(path, expected_nc, name):
    from ultralytics import YOLO
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} teacher checkpoint not found: {path}")
    net = YOLO(str(path)).model.cpu().eval()
    nc = int(getattr(net, 'nc', -1))
    if nc != int(expected_nc):
        raise RuntimeError(f"{name} teacher nc={nc}, student nc={expected_nc}")
    if not hasattr(net, 'model') or not isinstance(net.model, (nn.ModuleList, nn.Sequential)):
        raise TypeError(f"{name} teacher has unexpected layer container: {type(getattr(net,'model',None))}")
    return net


def _copy_verified(dst, src, label):
    """Require exact matching model state keys/shapes, copy parameters+BN buffers."""
    source = src.state_dict()
    target = dst.state_dict()
    source_keys = set(source)
    target_keys = set(target)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)[:12]
        extra = sorted(source_keys - target_keys)[:12]
        raise RuntimeError(f"{label} architecture mismatch; missing={missing}; extra={extra}")
    mismatched = [(k, tuple(target[k].shape), tuple(source[k].shape))
                  for k in target if target[k].shape != source[k].shape]
    if mismatched:
        raise RuntimeError(f"{label} shape mismatch: {mismatched[:12]}")
    dst.load_state_dict(source, strict=True)
    # Verification includes BatchNorm running_mean/running_var/num_batches_tracked.
    copied = dst.state_dict()
    bad = [k for k in source if not torch.allclose(
        copied[k].detach().cpu().float(), source[k].detach().cpu().float(),
        rtol=1e-6, atol=1e-7, equal_nan=False)]
    if bad:
        raise RuntimeError(f"{label} weights differ after loading: {bad[:12]}")
    n_params = sum(p.numel() for p in dst.parameters())
    print(f"[TEACHER-INIT PASS] {label}: {len(source)} state keys, {n_params:,} parameters", flush=True)


def initialize_from_dual_teachers(student, cfg):
    """Copy RGB full YOLO, TIR backbone, and their respective aux necks/heads.

    Called only when factory.build(load_pretrained=True). NEVER invoked during
    checkpoint reconstruction, ensuring validation uses the saved student.
    """
    weights = cfg['dual_teachers']
    n = int(student.core.backbone_len)
    core_layers = student.core.model
    rgb_teacher = _teacher_model(weights['rgb_weights'], student.nc, 'RGB')
    if len(rgb_teacher.model) != len(core_layers):
        raise RuntimeError(f"RGB layer count {len(rgb_teacher.model)} != {len(core_layers)}")
    if len(student.aux_rgb.model) != len(core_layers)-n:
        raise RuntimeError('RGB auxiliary neck/head architecture mismatch')
    # Sanity-check .f connectivity; not included in state_dict.
    for i, (a, b) in enumerate(zip(core_layers, rgb_teacher.model)):
        if type(a) is not type(b) or getattr(a, 'f', None) != getattr(b, 'f', None):
            raise RuntimeError(f"RGB teacher layer {i} type/input mismatch: {type(a)} vs {type(b)}")
    _copy_verified(core_layers, rgb_teacher.model, 'RGB full backbone+neck+head')
    _copy_verified(student.aux_rgb.model,
                   nn.ModuleList(list(rgb_teacher.model)[n:]), 'RGB auxiliary neck+head')
    del rgb_teacher

    tir_teacher = _teacher_model(weights['tir_weights'], student.nc, 'TIR')
    if len(tir_teacher.model) != len(core_layers):
        raise RuntimeError(f"TIR layer count {len(tir_teacher.model)} != {len(core_layers)}")
    for i, (a, b) in enumerate(zip(core_layers, tir_teacher.model)):
        if type(a) is not type(b) or getattr(a, 'f', None) != getattr(b, 'f', None):
            raise RuntimeError(f"TIR teacher layer {i} type/input mismatch: {type(a)} vs {type(b)}")
    _copy_verified(student.core.tir_backbone,
                   nn.ModuleList(list(tir_teacher.model)[:n]), 'TIR backbone')
    _copy_verified(student.aux_tir.model,
                   nn.ModuleList(list(tir_teacher.model)[n:]), 'TIR auxiliary neck+head')
    del tir_teacher
    print('[TEACHER-INIT DONE] Student initialized from the two UAV-CB teachers. '
          'GMLS/DMLS, BHLR, fusion remain independently initialized.', flush=True)
    return student
