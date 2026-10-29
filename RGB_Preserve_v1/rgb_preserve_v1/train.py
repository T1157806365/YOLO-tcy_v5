#!/usr/bin/env python3
"""RGB-Preserve v1, two-stage training with safe DDP re-wrap on phase transition."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Allow `python rgb_preserve_v1/train.py` from YOLO-tcy_v5 root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import engine.trainer as tr
from engine.runtime import (
    cleanup_distributed, distributed_barrier, is_main_process, load_config,
    parse_device_ids, set_seed, setup_distributed,
)
from rgb_preserve_v1.model import build_rgb_preserve_model


def _launch_if_needed(config_path: str, cfg: dict):
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return False
    devices = parse_device_ids(cfg.get('train', {}).get('device', '0'))
    if len(devices) <= 1:
        return False
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices)}', str(Path(__file__).resolve()),
               '--cfg', str(Path(config_path).resolve())]
    print(f'[RGB-PRESERVE] DDP physical GPUs={devices}', flush=True)
    subprocess.run(command, check=True, env=env)
    return True


def _custom_optimizer(model, cfg, lr0=None):
    """Stage2 RGB head uses a smaller LR; all other parameters use base LR."""
    bare = tr.unwrap_model(model)
    train_cfg = cfg.get('train', {})
    base_lr = float(lr0 if lr0 is not None else train_cfg.get('lr0', .01))
    head_scale = float(train_cfg.get('stage2', {}).get('rgb_head_lr_scale', .10))
    if not (0 < head_scale <= 1):
        raise ValueError('stage2.rgb_head_lr_scale must be in (0,1]')
    groups = {'aux': [], 'rgb_head': []}
    for name, p in bare.named_parameters():
        if not p.requires_grad:
            continue
        key = 'rgb_head' if name.startswith('model.') else 'aux'
        groups[key].append(p)
    specs = []
    if groups['aux']:
        specs.append({'params': groups['aux'], 'lr': base_lr})
    if groups['rgb_head']:
        specs.append({'params': groups['rgb_head'], 'lr': base_lr * head_scale})
    if not specs:
        raise RuntimeError('No trainable parameters in selected phase')
    if is_main_process():
        print(f'[RGB-PRESERVE] optimizer groups: aux={len(groups["aux"])} tensors '
              f'RGB head={len(groups["rgb_head"])} tensors, head lr multiplier={head_scale if groups["rgb_head"] else 0}', flush=True)
    name = str(train_cfg.get('optimizer', 'SGD')).lower()
    wd = float(train_cfg.get('weight_decay', .0005))
    if name == 'sgd':
        return torch.optim.SGD(specs, lr=base_lr, momentum=float(train_cfg.get('momentum', .937)),
                               weight_decay=wd, nesterov=True)
    if name == 'adamw':
        return torch.optim.AdamW(specs, lr=base_lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(specs, lr=base_lr, weight_decay=wd)
    raise ValueError(f'Unknown optimizer={name}')


def run(cfg):
    t = cfg.get('train', {})
    d = cfg.get('data', {})
    if t.get('resume') not in (None, '', 'null'):
        raise NotImplementedError('This first release does not support resume; use original saved checkpoints as new experiments.')
    if cfg.get('model', {}).get('variant') != 'rgb_preserve_v1':
        raise ValueError('Incorrect model.variant')
    if int(d.get('rgb_semantic_imgsz', 640)) != 640 or int(d.get('tir_imgsz', 640)) != 640:
        raise ValueError('This controlled experiment expects 640 semantic RGB and 640 TIR')
    device, rank, local_rank, world = setup_distributed(t.get('device', '0'))
    distributed = world > 1
    seed = int(t.get('seed', 0))
    set_seed(seed, deterministic=bool(t.get('deterministic', True)))
    local_batch, global_batch, mode = tr._resolve_batch_size(cfg, world)
    out = Path(cfg['output']['project']) / cfg['output']['name']
    if is_main_process():
        (out / 'weights').mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    train_ds, train_dl = tr._build_data(cfg, split='train', augment=True,
        shuffle=True, batch_size=local_batch, distributed=distributed, rank=rank, world_size=world)
    if is_main_process():
        val_ds, val_dl = tr._build_data(cfg, split='val', augment=False,
            shuffle=False, batch_size=int(t.get('val_batch', local_batch)),
            distributed=False, rank=0, world_size=1)
    else:
        val_dl = None
    model = build_rgb_preserve_model(cfg, init_single_modal=True,
                                      verbose=is_main_process()).to(device)
    model.set_training_phase('stage1')
    # Do not wrap before freezing, or DDP will retain unwanted reducer hooks.
    if distributed:
        model = DDP(model, device_ids=[int(device.index)], output_device=int(device.index),
                    broadcast_buffers=True, find_unused_parameters=bool(t.get('ddp_find_unused_parameters', True)))
    set_seed(seed + rank, deterministic=bool(t.get('deterministic', True)))
    original_builder = tr.build_optimizer
    tr.build_optimizer = _custom_optimizer
    try:
        if is_main_process():
            print(f'[RGB-PRESERVE] samples={len(train_ds)} global_batch={global_batch} '
                  f'GPUs={world} output={out}', flush=True)
        best = out / 'weights' / 'best.pt'
        first = tr.run_stage(model=model, cfg=cfg, train_loader=train_dl,
            val_loader=val_dl, device=device, out_dir=out, stage='stage1',
            epochs=int(t.get('epochs', 50)), lr0=float(t.get('lr0', .01)),
            lrf=float(t.get('lrf', .01)), patience=int(t.get('patience', 30)),
            min_delta=float(t.get('early_stop_min_delta', 1e-6)),
            amp=bool(t.get('amp', True)), grad_clip=float(t.get('grad_clip', 10)),
            generic_best_path=best)
        stage2 = t.get('stage2', {}) or {}
        second = None
        if bool(stage2.get('enabled', True)):
            distributed_barrier()
            raw = tr.unwrap_model(model)
            # Drop old DDP reducer. Rebuild after changing trainable parameters.
            if distributed:
                del model
            tr.load_weights_strict(first['stage_best_path'], raw, device)
            raw.set_training_phase('stage2')
            if distributed:
                model = DDP(raw, device_ids=[int(device.index)], output_device=int(device.index),
                            broadcast_buffers=True, find_unused_parameters=bool(t.get('ddp_find_unused_parameters', True)))
            else:
                model = raw
            distributed_barrier()
            second = tr.run_stage(model=model, cfg=cfg, train_loader=train_dl,
                val_loader=val_dl, device=device, out_dir=out, stage='stage2',
                epochs=int(stage2.get('epochs', 100)), lr0=float(stage2.get('lr0', .001)),
                lrf=float(stage2.get('lrf', .01)), patience=int(stage2.get('patience', 30)),
                min_delta=float(stage2.get('early_stop_min_delta', 1e-6)),
                amp=bool(t.get('amp', True)), grad_clip=float(t.get('grad_clip', 10)),
                global_epoch_offset=int(first['epochs_ran']),
                global_best_map=float(first['global_best_map']), generic_best_path=best)
        result = {
            'best_checkpoint': str(best.resolve()),
            'stage1_best_val_ap5095': float(first['stage_best_map']),
            'stage2_best_val_ap5095': None if second is None else float(second['stage_best_map']),
            'world_size': world, 'global_batch': global_batch,
            'architecture': 'RGB 640 detection retained, TIR P3 zero-init residual',
        }
        if is_main_process():
            (out / 'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result, indent=2), flush=True)
        distributed_barrier()
        return result
    finally:
        tr.build_optimizer = original_builder
        cleanup_distributed()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    args = p.parse_args()
    cfg = load_config(args.cfg)
    if _launch_if_needed(args.cfg, cfg):
        return
    run(cfg)

if __name__ == '__main__':
    main()
