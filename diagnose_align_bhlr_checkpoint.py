#!/usr/bin/env python3
"""Read-only checkpoint/source audit. No model build, training or checkpoint writes.

Run in YOLO-tcy_v5 with its trusted project checkpoint:
  python diagnose_align_bhlr_checkpoint.py --weights runs/.../weights/best.pt
Optional: --out /new/path/report.json (refuses overwrite).
Configuration records express intent, not proof of executed initialization/hooks.
"""
import argparse
import json
import re
from pathlib import Path

import torch


def safe_value(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.numel() == 1 else {'shape': list(x.shape)}
    if isinstance(x, dict):
        return {str(k): safe_value(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [safe_value(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def stats(t):
    t = t.detach().cpu().double()
    return {'shape': list(t.shape), 'finite': bool(torch.isfinite(t).all()),
            'norm': float(t.norm()), 'max_abs': float(t.abs().max()) if t.numel() else None}


def audit(ck, repo):
    if not isinstance(ck, dict):
        raise ValueError('Expected project checkpoint dictionary')
    cfg, sd = ck.get('config'), ck.get('model_state_dict')
    if not isinstance(cfg, dict) or not isinstance(sd, dict):
        raise ValueError('Expected config and model_state_dict; no guessed format conversion')
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    model, tr, mono = (cfg.get(k, {}) for k in ('model', 'train', 'monoalign'))
    fields = {
        'model_pretrained': model.get('pretrained'),
        'alignment': model.get('rlsfa_gtga_centered'),
        'bhlr': model.get('bhlr'),
        'monoalign': mono,
        'from_epoch1_control': cfg.get('from_epoch1_control'),
        'train': {k: tr.get(k) for k in ('epochs', 'device', 'batch', 'accumulate',
                  'optimizer', 'lr0', 'scheduler', 'lr_milestones', 'lr_gamma',
                  'resume', 'init_student_weights', 'stage2')},
        'alignment_loss': cfg.get('loss', {}).get('alignment'),
        'output': cfg.get('output'),
        'teacher_paths': cfg.get('dual_teachers'),
    }
    gamma, selectors = {}, {}
    for name, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name.endswith('gamma_raw') and ('bhlr.' in name or 'rsdt.' in name):
            gamma[name] = {'raw': value.detach().cpu().tolist(),
                           'effective_tanh': value.detach().cpu().double().tanh().tolist()}
        if re.search(r'(?:bhlr|rsdt)\..*selector\.head\.\d+\.weight$', name):
            selectors[name] = stats(value)
    sources = {}
    paths = {
        'trainer': 'ga_monoalign_pcgrad/train.py',
        'student': 'ga_monoalign_pcgrad/model.py',
        'teacher_init': 'ga_monoalign_pcgrad/teacher_init.py',
        'controller': 'ga_epoch1_control/controller.py',
        'bridge': 'ga_epoch1_control/bridge.py',
        'validation': 'ga_monoalign_pcgrad/val.py',
    }
    markers = {
        'trainer': ['bind_controller', 'EPOCH1_CONTROL_V1_20261104'],
        'student': ['single100_step_v1', '_pcgrad_feature_hooks'],
        'teacher_init': ['teacher_init_scope', 'backbone_only'],
        'controller': ['original_schedule()', 'mono_scale * factor', 'kd_scale * factor'],
        'bridge': ["'bhlr_detach_alignment_guidance', False", 'scale_gradient(tir, self.eta)'],
        'validation': ["'student_scratch':True", "'student_scratch': True"],
    }
    for key, rel in paths.items():
        p = repo / rel
        text = p.read_text(encoding='utf-8') if p.is_file() else ''
        sources[key] = {'path': str(p), 'exists': p.is_file(),
                        'markers_present': {m: m in text for m in markers[key]}}
    control = cfg.get('from_epoch1_control') or {}
    conclusions = []
    if not control.get('enabled', False):
        conclusions.append('Checkpoint config does NOT request the Epoch1 controller. Other custom patches are not ruled out.')
    else:
        conclusions.append('Checkpoint requests Epoch1 control. Source markers alone do NOT prove hooks executed during this run.')
    if model.get('bhlr', {}).get('detach_alignment_guidance'):
        conclusions.append('YAML requests guidance detach; an installed runtime controller may override it. Do not diagnose from YAML alone.')
    if mono.get('init_from_teachers'):
        conclusions.append('Teacher initialization is requested. Actual successful loading requires the original startup log.')
    if not gamma:
        conclusions.append('No recognized BHLR gamma_raw keys: inspect architecture/version; do not infer zero injection.')
    conclusions.append('Gamma and selector weight norms cannot measure useful HR contribution or prove a nonzero detection-to-alignment gradient.')
    conclusions.append('This script does NOT run forward/backward or test AP. It never modifies the checkpoint or project source.')
    return {
        'metadata': {k: safe_value(ck[k]) for k in ('epoch', 'stage', 'best', 'best_metric',
                     'best_fitness', 'metrics', 'format', 'global_epoch') if k in ck},
        'requested_configuration': safe_value(fields), 'bhlr_gamma': gamma,
        'selector_output_weights': selectors, 'installed_source_markers': sources,
        'interpretation_limits': conclusions,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--repo', default='.', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.out and a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}')
    # Use only the user's trusted project-generated checkpoint (PyTorch pickle).
    ck = torch.load(a.weights, map_location='cpu', weights_only=False)
    report = audit(ck, a.repo.resolve())
    report['checkpoint'] = str(a.weights.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x', encoding='utf-8') as f:
            f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
