"""Single-GPU module-only continuation. Run from the YOLO-tcy_v5 root."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from .bridge import GuidanceBridge
from .runtime import (read_yaml, choose_device, seed_all, load_student, ModulePolicy,
    build_optimizer, make_loader, original_losses, real_preflight, frozen_digest,
    eta_for_epoch, finite_metrics, json_safe, write_json)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_checkpoint(path, student, source_cfg, run_cfg, source_meta, epoch,
                    optimizer, scheduler, scaler, best_ap, best_epoch,
                    trained_best, baseline_metrics, history, frozen_hash):
    # Preserve source config verbatim so original inference reconstruction and
    # source scale/geometry/loss definitions remain unchanged.
    payload = {
        'format': 'ga_module_only_checkpoint_v1',
        'model_state_dict': student.state_dict(),
        'config': copy.deepcopy(source_cfg),
        'module_only_config': copy.deepcopy(run_cfg),
        'module_only_epoch': int(epoch),
        'epoch': int(epoch), 'stage': 'module_only',
        'source_epoch': source_meta.get('epoch'),
        'source_global_epoch': source_meta.get('global_epoch'),
        'source_checkpoint_sha256': source_meta.get('source_checkpoint_sha256'),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_map5095': float(best_ap), 'best_module_only_epoch': int(best_epoch),
        'best_trained_map5095': float(trained_best),
        'baseline_val_metrics': baseline_metrics, 'module_only_history': history,
        'frozen_state_sha256': frozen_hash, 'rng_state': rng_state(),
        'model_name': student.model_name, 'fusion': student.fusion_type,
        'nc': student.nc, 'names': student.names,
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def accumulation_step(optimizer, scaler, params, images, grad_clip):
    """Mean over actual images, including short final accumulation windows."""
    if images < 1:
        raise RuntimeError('Empty accumulation window')
    scaler.unscale_(optimizer)
    for p in params:
        if p.grad is not None:
            p.grad.div_(float(images))
    norm = torch.nn.utils.clip_grad_norm_(params, grad_clip,
                                         error_if_nonfinite=not scaler.is_enabled())
    if not torch.isfinite(norm):
        # An AMP overflow must reduce the scale and skip this window, rather
        # than aborting or applying inf/NaN gradients. Explicit update also
        # handles a norm overflow with finite individual gradient elements.
        scaler.update(new_scale=scaler.get_scale() * .5)
        optimizer.zero_grad(set_to_none=True)
        return None
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def run_epoch(student, policy, bridge, loader, optimizer, scaler, device, tune, epoch):
    from engine.runtime import preprocess_batch
    from tqdm import tqdm
    policy.train()
    bridge.set_eta(eta_for_epoch(tune, epoch))
    amp = bool(tune['amp']) and device.type == 'cuda'
    params = [p for ps in policy.groups.values() for p in ps]
    optimizer.zero_grad(set_to_none=True)
    images = 0
    total_images = 0
    sums = {}
    last_norm = 0.
    skipped_windows = 0
    iterator = tqdm(loader, desc=f'module-only {epoch}/{tune["epochs"]}', dynamic_ncols=True)
    for bi, raw in enumerate(iterator):
        batch = preprocess_batch(raw, device)
        bsz = int(batch['rgb_img'].shape[0])
        bridge.reset_counts()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            det, align, logs = original_losses(student, batch)
            loss_sum = det + align
        bridge.assert_used()
        if not torch.isfinite(loss_sum):
            raise RuntimeError(f'Nonfinite loss at epoch={epoch}, batch={bi}; weights not saved')
        scaler.scale(loss_sum).backward()
        policy.verify_no_frozen_grad()
        images += bsz
        total_images += bsz
        for k, value in logs.items():
            number = float(value.detach().mean()) if torch.is_tensor(value) else float(value)
            sums[k] = sums.get(k, 0.) + number * bsz
        if (bi + 1) % int(tune['accumulate']) == 0 or bi + 1 == len(loader):
            norm = accumulation_step(optimizer, scaler, params, images, float(tune['grad_clip']))
            if norm is None:
                skipped_windows += 1
            else:
                last_norm = norm
            images = 0
        iterator.set_postfix(det=f'{float(det.detach()) / bsz:.4f}',
                             align=f'{float(align.detach()) / bsz:.4f}', eta=f'{bridge.eta:.3f}')
    if not total_images:
        raise RuntimeError('Empty training loader')
    return {**{k: v / total_images for k, v in sums.items()},
            'bridge_eta': bridge.eta, 'gradient_norm_last_step': last_norm,
            'amp_skipped_windows': skipped_windows,
            'train_images': total_images, 'mono_weight': 0., 'kd_weight': 0.}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', required=True)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--resume', default=None, help='ONLY this release\'s weights/last.pt')
    args = p.parse_args()
    cfg = read_yaml(args.cfg)
    tune = cfg['tune']
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Single-GPU release: invoke python directly, not torchrun')
    device = choose_device(tune['device'])
    seed_all(int(tune['seed']), bool(tune['deterministic']))
    out = Path(cfg['output']).expanduser().resolve()
    source = Path(cfg['source_weights']).expanduser().resolve()
    resume = None
    if args.resume:
        if args.preflight_only:
            raise ValueError('--preflight-only cannot be combined with --resume')
        resume_path = Path(args.resume).expanduser().resolve()
        if resume_path != out / 'weights' / 'last.pt':
            raise ValueError('--resume must be this output directory\'s weights/last.pt')
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume.get('format') != 'ga_module_only_checkpoint_v1':
            raise ValueError('Cannot resume an old whole-network optimizer in module-only mode')
        if resume['module_only_config'] != cfg:
            raise ValueError('Resume configuration differs from the saved run')
        student, source_meta, source_cfg = load_student(resume_path, device)
    else:
        if out.exists() and any(out.iterdir()) and not args.preflight_only:
            raise FileExistsError(f'Output already contains files: {out}; use --resume or a new output name')
        student, source_meta, source_cfg = load_student(source, device)
        if source_meta.get('epoch') != 17 or source_meta.get('stage') != 'stage1':
            raise ValueError('Expected original stage1 epoch17 checkpoint; source changed or wrong weights')
        source_meta['source_checkpoint_sha256'] = file_hash(source)
    policy = ModulePolicy(student)
    print('[LOADED] Complete original student, strict=True; no teacher/official reinitialization', flush=True)
    print('[TRAINABLE]', policy.report()['trainable_parameters'], flush=True)
    print('[FROZEN] RGB/TIR backbone, Neck/Detect, aux heads, all BN statistics', flush=True)
    print('[OBJECTIVE] original GT detection + original alignment; Mono=KD=0; no hard-target weighting', flush=True)

    # Real paired train image; no training augmentation in a connectivity probe.
    from engine.runtime import preprocess_batch
    from engine.evaluator import validate_metrics
    _, probe_loader = make_loader(source_cfg, {**tune, 'workers': 0},
                                  'train', batch=1)
    probe = preprocess_batch(next(iter(probe_loader)), device)
    report = real_preflight(student, policy, probe, tune)
    del probe, probe_loader
    print('[PREFLIGHT PASS]', json.dumps(json_safe({k: v for k, v in report.items()
          if k != 'trainable_names'}), ensure_ascii=False), flush=True)
    if args.preflight_only:
        print('[DONE] Model restored after the test step. No training checkpoint written.', flush=True)
        return
    # Probe consumed random numbers; actual training always starts with this seed.
    seed_all(int(tune['seed']), bool(tune['deterministic']))
    if not args.resume:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'weights').mkdir(exist_ok=False)
        write_json(out / 'preflight.json', report)
        write_json(out / 'source_metadata.json', source_meta)
        write_json(out / 'trainable_parameters.json', policy.report())
        (out / 'module_only.yaml').write_text(
            __import__('yaml').safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    _, train_loader = make_loader(source_cfg, tune, 'train', augment=True, shuffle=True)
    _, val_loader = make_loader(source_cfg, tune, 'val', batch=int(tune['val_batch']))
    optimizer = build_optimizer(policy, tune)
    epochs = int(tune['epochs'])
    lrf = float(tune['lrf'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda e: lrf + (1. - lrf) * (1. + math.cos(math.pi * min(e, epochs) / epochs)) / 2.)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(tune['amp']) and device.type == 'cuda')
    bridge = GuidanceBridge(student.core, 0.)
    start = 1
    history = []
    baseline = None
    best_epoch = 0
    trained_best = -1.
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        scaler.load_state_dict(resume['scaler_state_dict'])
        start = int(resume['module_only_epoch']) + 1
        history = resume['module_only_history']
        baseline = resume['baseline_val_metrics']
        best_ap = float(resume['best_map5095'])
        best_epoch = int(resume['best_module_only_epoch'])
        trained_best = float(resume['best_trained_map5095'])
        source_meta['source_checkpoint_sha256'] = resume['source_checkpoint_sha256']
        if frozen_digest(student) != resume['frozen_state_sha256']:
            raise RuntimeError('Frozen-state integrity check failed on resume')
        restore_rng(resume['rng_state'])
        del resume
    else:
        student.eval()
        with torch.inference_mode():
            baseline = validate_metrics(student, val_loader, device,
                amp=bool(tune['val_amp']) and device.type == 'cuda',
                conf_thres=.001, iou_thres=.70, max_det=300, preview_num=0)
        best_ap = finite_metrics(baseline)
        print(f'[BASELINE VAL] AP50:95={best_ap:.6f}; this is validation, not the historical 61.16% test score', flush=True)
        write_json(out / 'baseline_val.json', baseline)
    frozen_hash = frozen_digest(student)
    saver = lambda name, e: save_checkpoint(out / 'weights' / name, student, source_cfg,
        cfg, source_meta, e, optimizer, scheduler, scaler, best_ap, best_epoch,
        trained_best, baseline, history, frozen_hash)
    if not args.resume:
        saver('best.pt', 0)
        saver('last.pt', 0)
    print(f'[TRAIN] {device}; batch={tune["batch"]}; accumulate={tune["accumulate"]}; '
          f'full-window effective batch={int(tune["batch"]) * int(tune["accumulate"])}', flush=True)
    try:
        for epoch in range(start, epochs + 1):
            lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
            logs = run_epoch(student, policy, bridge, train_loader, optimizer, scaler, device, tune, epoch)
            if frozen_digest(student) != frozen_hash:
                raise RuntimeError('Frozen parameters/BN buffers changed; stopping before saving')
            student.eval()
            with torch.inference_mode():
                metrics = validate_metrics(student, val_loader, device,
                    amp=bool(tune['val_amp']) and device.type == 'cuda',
                    conf_thres=.001, iou_thres=.70, max_det=300, preview_num=0)
            ap = finite_metrics(metrics)
            if frozen_digest(student) != frozen_hash:
                raise RuntimeError('Validation mutated a frozen parameter/buffer')
            scheduler.step()
            is_best = ap > best_ap
            is_trained_best = ap > trained_best
            if is_best:
                best_ap, best_epoch = ap, epoch
            if is_trained_best:
                trained_best = ap
            history.append({'epoch': epoch, 'lr': lrs, 'train': logs,
                            'val': metrics, 'best_map5095': best_ap,
                            'best_epoch': best_epoch})
            write_json(out / 'history.json', history)
            if is_best:
                saver('best.pt', epoch)
            if is_trained_best:
                saver('best_trained.pt', epoch)
            saver('last.pt', epoch)
            print(f'[EPOCH {epoch}/{epochs}] val_AP50:95={ap:.6f} '
                  f'baseline={float(baseline["map5095"]):.6f} '
                  f'best={best_ap:.6f} best_epoch={best_epoch} '
                  f'eta={bridge.eta:.4f} frozen=PASS', flush=True)
    finally:
        bridge.close()
    summary = {'baseline_val_map5095': float(baseline['map5095']),
               'best_val_map5095': best_ap, 'best_epoch': best_epoch,
               'best_is_original_source': best_epoch == 0,
               'gain_on_val_pp': 100 * (best_ap - float(baseline['map5095'])),
               'test_evaluated_during_training': False,
               'frozen_state_sha256': frozen_hash}
    write_json(out / 'summary.json', summary)
    print('[COMPLETE]', json.dumps(summary, ensure_ascii=False), flush=True)
    if best_epoch == 0:
        print('[NOTE] No validation improvement; best.pt retains the original source weights. '
              'best_trained.pt is the best actually updated model.', flush=True)


if __name__ == '__main__':
    main()
