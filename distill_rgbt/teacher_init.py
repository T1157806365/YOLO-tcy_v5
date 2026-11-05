"""Strict RGB/TIR teacher initialization; original model architecture unchanged."""
from __future__ import annotations
from torch import nn
import torch
from .teachers import load_yolo_teacher


def _copy(dst_layers, src_layers, name):
    dst, src = nn.ModuleList(list(dst_layers)), nn.ModuleList(list(src_layers))
    if len(dst) != len(src):
        raise ValueError(f'{name}: module count differs {len(dst)} vs {len(src)}')
    for i, (d, s) in enumerate(zip(dst, src)):
        if type(d) is not type(s) or getattr(d, 'f', None) != getattr(s, 'f', None):
            raise ValueError(f'{name}: architecture/.f mismatch at index {i}')
    ds, ss = dst.state_dict(), src.state_dict()
    if ds.keys() != ss.keys():
        raise ValueError(f'{name}: teacher/student state keys differ')
    for k, v in ds.items():
        if v.shape != ss[k].shape:
            raise ValueError(f'{name}: state shape mismatch {k}: {tuple(v.shape)} != {tuple(ss[k].shape)}')
    dst.load_state_dict(ss, strict=True)
    if any(not torch.equal(dst.state_dict()[k].cpu(), ss[k].cpu()) for k in ss):
        raise RuntimeError(f'{name}: copy verification failed')
    print(f'[TEACHER-INIT PASS] {name}: {sum(x.numel() for x in dst.parameters()):,} parameters', flush=True)


def initialize_student(student, cfg):
    if not bool(cfg['distill'].get('init_teacher_backbones', True)):
        print('[TEACHER-INIT] disabled: using original main factory initialization', flush=True)
        return
    scope = str(cfg['distill'].get('teacher_init_scope', 'backbone_only')).lower()
    if scope not in ('full', 'backbone_only'):
        raise ValueError(f'Unknown teacher_init_scope={scope!r}')
    core = student.core
    n = int(core.backbone_len)
    weights = cfg['distill']['teachers']
    rgb = load_yolo_teacher(weights['rgb'], nc=core.nc)
    if scope == 'full':
        _copy(list(core.model), list(rgb.model), 'RGB core full Backbone+Neck+Detect')
    else:
        _copy(list(core.model)[:n], list(rgb.model)[:n], 'RGB Backbone')
    if student.mono_enabled:
        _copy(student.aux_rgb.model, list(rgb.model)[n:], 'RGB auxiliary Neck+Detect')
    del rgb
    tir = load_yolo_teacher(weights['tir'], nc=core.nc)
    _copy(core.tir_backbone, list(tir.model)[:n], 'TIR Backbone')
    if student.mono_enabled:
        _copy(student.aux_tir.model, list(tir.model)[n:], 'TIR auxiliary Neck+Detect')
    del tir
    print(f'[TEACHER-INIT DONE] scope={scope}; alignment, BHLR, fusion are initialized independently', flush=True)
