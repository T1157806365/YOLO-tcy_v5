"""DDP-capable four-arm teacher distillation on the original YOLO-tcy_v5/main.

Capabilities: auto torchrun DDP; accumulation + no_sync; YOLO official/pretrained
and two 640x640 teacher initialization (backbone_only/full); GT alignment; RGB/TIR
mono detection auxiliary heads; reliable ROI P3 KD; feature PCGrad; AMP; 1/2
stages; stage1 resume; per-epoch val, periodic geometry/mono validation; original
main-compatible best.pt (core-only model_state_dict) plus training-only aux state.

No changes are made to original models/, engine/, datasets/ or val code.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from engine import trainer as tr
from engine.evaluator import validate_metrics
from engine.runtime import (load_config, parse_device_ids, select_device, set_seed,
    setup_distributed, cleanup_distributed, distributed_barrier, distributed_mean,
    distributed_mean_dict, preprocess_batch, autocast_context, make_grad_scaler,
    is_main_process)
from models.gtga_bhlr_factory import build_model_from_config
from .teacher_init import initialize_student
from .student import DistillStudent
from .metrics import geometry_iou, mono_metrics


ARMS = {
    'A_baseline': (False, False),
    'B_align_only': (True, False),
    'C_hr_only': (False, True),
    'D_align_hr': (True, True),
}


def validate_cfg(cfg, check_files=True):
    m, t, d = cfg['model'], cfg['train'], cfg['distill']
    # AG-LHFE-KD v1: allow original four arms plus two matched local-HR arms.
    variant = str(m.get('variant', ''))
    is_lhfe = variant == 'rlsfa_gtga_centered_ag_lhfe_v1'
    if variant not in ('rlsfa_gtga_centered_bhlr_v2',
                       'rlsfa_gtga_centered_ag_lhfe_v1'):
        raise ValueError(f'Unsupported distillation variant: {variant}')
    if m.get('pretrained') is not True:
        raise ValueError('model.pretrained must be YAML boolean true')
    if m.get('fusion') != 'add':
        raise ValueError('Use ADD fusion across all ablations')
    arm = str(d.get('arm', ''))
    lhfe_arms = {'E1_local_rgb': 'rgb', 'E2_local_aligned': 'rgb_align'}
    if arm not in ARMS and arm not in lhfe_arms:
        raise ValueError(f'Unknown distill.arm={arm!r}')
    a, b = m['rlsfa_gtga_centered'], m['bhlr']
    align_on, hr_on = bool(a['enabled']), bool(b['enabled'])
    if is_lhfe:
        if arm not in lhfe_arms:
            raise ValueError(f'{variant} requires one of {sorted(lhfe_arms)}')
        if not align_on or hr_on:
            raise ValueError('AG-LHFE requires GMLS/DMLS on and original BHLR off')
        local_cfg = m.get('ag_lhfe', {}) or {}
        if not bool(local_cfg.get('enabled', False)):
            raise ValueError('AG-LHFE must be enabled')
        if str(local_cfg.get('proposal_mode')) != lhfe_arms[arm]:
            raise ValueError(f'{arm} requires proposal_mode={lhfe_arms[arm]}')
        if b.get('guidance_mode') != 'none':
            raise ValueError('AG-LHFE arms must disable BHLR guidance')
    else:
        if arm not in ARMS:
            raise ValueError(f'Original model requires one of {list(ARMS)}')
        if (align_on, hr_on) != ARMS[arm]:
            raise ValueError(f'{arm} expects alignment/HR={ARMS[arm]}, got {(align_on,hr_on)}')
        expected_guidance = 'alignment' if align_on and hr_on else 'none'
        if b.get('guidance_mode') != expected_guidance:
            raise ValueError(f'{arm} requires BHLR guidance_mode={expected_guidance}')
    if a.get('apply_scales') != [3,4,5] or b.get('scales') != [3]:
        raise ValueError('P3/P4/P5 alignment and P3 HR scales must be unchanged')
    if not bool(d.get('kd', {}).get('enabled', False)):
        raise ValueError('This suite REQUIRES distill.kd.enabled=true')
    if not bool(d.get('init_teacher_backbones', False)):
        raise ValueError('This suite REQUIRES teacher-backbone initialization')
    if bool(cfg.get('mono', {}).get('enabled', False)):
        raise ValueError('Matched KD suite uses Mono loss OFF')
    if bool(cfg.get('pcgrad', {}).get('enabled', False)):
        raise ValueError('Matched KD suite uses PCGrad OFF')
    if bool(cfg['loss']['alignment']['enabled']) != align_on:
        raise ValueError('loss.alignment.enabled must follow alignment switch')
    if bool(cfg.get('aghlr', {}).get('enabled', False)) or bool(cfg.get('geometry_hr', {}).get('enabled',False)):
        raise ValueError('Use original GMLS/DMLS+BHLR only, no AGHLR/GeometryHR')
    if len(parse_device_ids(t['device'])) < 1:
        raise ValueError('This experiment needs CUDA; train.device must name GPU IDs')
    if t.get('batch_mode', 'global') != 'global':
        raise ValueError('Use global batch mode for matched DDP/single runs')
    if int(t['batch']) < 1 or int(t.get('accumulate', 1)) < 1:
        raise ValueError('train.batch and train.accumulate must be positive')
    actual_effective = int(t['batch']) * int(t.get('accumulate', 1))
    expected_effective = t.get('expected_effective_batch')
    if expected_effective is not None and actual_effective != int(expected_effective):
        raise ValueError(f'Effective batch mismatch: global microbatch '
                         f'{t["batch"]} * accumulate {t.get("accumulate",1)} '
                         f'= {actual_effective}, expected {expected_effective}. '
                         'Use matching settings for all ablation arms.')
    ndevices = len(parse_device_ids(t['device']))
    if int(t['batch']) % ndevices != 0:
        raise ValueError('Global train.batch must be divisible by number of GPUs')
    if int(t.get('epochs', 0)) < 1:
        raise ValueError('train.epochs must be positive')
    if bool(t.get('stage2', {}).get('enabled', False)) and int(t['stage2'].get('epochs',0)) < 1:
        raise ValueError('stage2 enabled with zero epochs')
    if bool(d.get('init_teacher_backbones', True)) and str(d.get('teacher_init_scope', 'backbone_only')) not in ('full', 'backbone_only'):
        raise ValueError('teacher_init_scope must be full/backbone_only')
    if check_files and (bool(d.get('init_teacher_backbones', True)) or bool(d.get('kd',{}).get('enabled',True))):
        for mod in ('rgb', 'tir'):
            p = Path(d['teachers'][mod]).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f'{mod} teacher checkpoint missing: {p}')
    p = d.get('kd',{})
    if bool(p.get('enabled', True)):
        st,reach,hold,end = (int(p[k]) for k in ('start_epoch','reach_epoch','hold_until_epoch','end_epoch'))
        if not (0 < st < reach <= hold < end):
            raise ValueError('Invalid KD epoch schedule')
    return cfg


def build_student(cfg, device):
    core = build_model_from_config(cfg, pretrained=bool(cfg['model']['pretrained']), verbose=is_main_process())
    model = DistillStudent(core, cfg)
    initialize_student(model, cfg)
    return model.to(device)


def unwrap(net):
    return net.module if isinstance(net, DDP) else net


def save_checkpoint(path, net, cfg, stage, epoch, offset, optimizer, scheduler, scaler, best, patience_counter):
    """Important: original val_rlsfa_gtga_bhlr_v2.py expects ORIGINAL core state keys."""
    if not is_main_process(): return
    student = unwrap(net)
    state = {
        'epoch':int(epoch), 'stage':str(stage), 'stage_epoch':int(epoch),
        'global_epoch':int(offset + epoch),
        'model_state_dict':student.core.state_dict(),  # no core. prefix, no aux parameters
        'auxiliary_state_dict':({
            'rgb':student.aux_rgb.state_dict(), 'tir':student.aux_tir.state_dict()
        } if student.mono_enabled else None),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'best_map5095':float(best), 'early_stop_counter':int(patience_counter),
        'config':cfg, 'model_name':student.model_name,'fusion':student.fusion_type,
        'nc':student.nc,'names':student.names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, student, device, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    student.core.load_state_dict(ck['model_state_dict'], strict=True)
    aux = ck.get('auxiliary_state_dict')
    if student.mono_enabled:
        if not isinstance(aux, dict):
            raise RuntimeError('Training resume requires auxiliary_state_dict for mono-enabled runs')
        student.aux_rgb.load_state_dict(aux['rgb'], strict=True)
        student.aux_tir.load_state_dict(aux['tir'], strict=True)
    if optimizer is not None: optimizer.load_state_dict(ck['optimizer_state_dict'])
    if scheduler is not None: scheduler.load_state_dict(ck['scheduler_state_dict'])
    if scaler is not None: scaler.load_state_dict(ck['scaler_state_dict'])
    return ck


def grad_norm(module):
    if module is None: return 0.0
    vals = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    if not vals: return 0.0
    return float(torch.stack(vals).sum().sqrt().item())



def _global_invalid_flag(local_bad: bool, device: torch.device) -> bool:
    """DDP ranks must agree to either all step or all skip an update."""
    flag = torch.tensor([int(local_bad)], device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _report_nonfinite_gradients(net, stage, epoch, batch_i, limit=16):
    """Called ONLY when the gradient norm is non-finite. Diagnostic, no mutations."""
    rank = int(os.environ.get('RANK', '0'))
    found, shown = 0, 0
    for name, p in unwrap(net).named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        ok = torch.isfinite(g)
        if bool(ok.all().item()):
            continue
        found += 1
        if shown >= limit:
            continue
        nonfinite = int((~ok).sum().item())
        n_nan = int(torch.isnan(g).sum().item())
        n_inf = int(torch.isinf(g).sum().item())
        print(f'[NONFINITE-GRAD] rank={rank} stage={stage} epoch={epoch} '
              f'batch={batch_i + 1} param={name} shape={tuple(g.shape)} '
              f'bad={nonfinite} nan={n_nan} inf={n_inf}', flush=True)
        shown += 1
    if not found:
        print(f'[NONFINITE-NORM] rank={rank} stage={stage} epoch={epoch} '
              f'batch={batch_i + 1}: all tested elements are finite, but '
              f'total grad norm overflowed (or exceeded numeric range).', flush=True)
    elif found > limit:
        print(f'[NONFINITE-GRAD] rank={rank}: {found} parameters contain invalid '
              f'gradients; first {limit} shown.', flush=True)
    return found


def _safe_gradient_update(net, optimizer, scaler, device, cfg,
                          stage, epoch, batch_i, consecutive_bad):
    """After accumulated backward: unscale, finite check, DDP sync, clip, step.

    A first-step AMP overflow is not necessarily a corrupt model. Unlike simply
    changing error_if_nonfinite=False, this NEVER clips or applies bad grads.
    Both DDP ranks skip the same update and receive the same AMP scale reduction.
    """
    clip = float(cfg['train'].get('grad_clip', 10.0))
    max_bad = int(cfg['train'].get('max_consecutive_bad_steps', 12))
    if max_bad < 1:
        raise ValueError('train.max_consecutive_bad_steps must be >= 1')

    # This also registers found_inf in GradScaler's per-optimizer state.
    scaler.unscale_(optimizer)
    norm = None
    bad_local = False
    try:
        # max_norm=inf checks without clipping when grad_clip <= 0.
        norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=clip if clip > 0 else math.inf,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        # Propagate unrelated issues, e.g. unsupported grad types.
        if 'non-finite' not in str(exc):
            raise
        bad_local = True
    bad_any = _global_invalid_flag(bad_local, device)
    if bad_any:
        rank = int(os.environ.get('RANK', '0'))
        if bad_local:
            _report_nonfinite_gradients(net, stage, epoch, batch_i)
        old_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
        # Reduce the scale on *all* ranks, even if only one had overflow.
        # This prevents divergent DDP model/optimizer states on later steps.
        if scaler.is_enabled():
            new_scale = max(1.0, old_scale * 0.5)
            scaler.update(new_scale=new_scale)
        else:
            new_scale = old_scale
        optimizer.zero_grad(set_to_none=True)
        consecutive_bad += 1
        print(f'[GRAD-SKIP] rank={rank} stage={stage} epoch={epoch} '
              f'batch={batch_i + 1} bad_local={int(bad_local)} '
              f'amp_scale={old_scale:g}->{new_scale:g} '
              f'consecutive={consecutive_bad}/{max_bad}; '
              'optimizer.step NOT executed', flush=True)
        if not scaler.is_enabled():
            raise FloatingPointError(
                f'FP32 nonfinite gradients detected at {stage} epoch={epoch} '
                f'batch={batch_i + 1}. See [NONFINITE-GRAD] diagnostics.')
        if consecutive_bad >= max_bad:
            raise FloatingPointError(
                f'{max_bad} consecutive nonfinite gradient windows in {stage} '
                f'epoch={epoch}. Not an isolated AMP overflow; inspect '
                '[NONFINITE-GRAD] and try train.amp: false as a diagnostic.')
        return False, consecutive_bad, float('nan'), {}

    # No rank saw a bad gradient: collect finite post-clip norms, then update.
    raw = unwrap(net)
    group_norms = {
        'grad_norm_align': grad_norm(raw.core.alignment),
        'grad_norm_bhlr': grad_norm(raw.core.bhlr),
        'grad_norm_lhfe': grad_norm(getattr(raw.core,'ag_lhfe',None)),
        'grad_norm_rgb_backbone': grad_norm(raw.core.model),
        'grad_norm_tir_backbone': grad_norm(raw.core.tir_backbone),
        'grad_norm_aux': grad_norm(raw.aux_rgb) + grad_norm(raw.aux_tir),
    }
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True, 0, float(norm.item()) if norm is not None else float('nan'), group_norms


def train_epoch(net, loader, optimizer, scaler, device, cfg, stage, epoch, num_epochs):
    raw = unwrap(net)
    raw.set_progress(stage, epoch)
    if raw.kd_weight() > 0 and raw._teachers is None:
        raw.attach_teachers(device)
    net.train()
    sampler = getattr(loader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'): sampler.set_epoch(epoch)
    amp = bool(cfg['train'].get('amp', True))
    accum = int(cfg['train'].get('accumulate',1))
    clip = float(cfg['train'].get('grad_clip',10.0))
    optimizer.zero_grad(set_to_none=True)
    running, sum_loss, count, steps = {}, 0., 0, 0
    skipped_steps, consecutive_bad = 0, 0
    progress=tqdm(enumerate(loader), total=len(loader), disable=not is_main_process(),
                  desc=f'{stage} {epoch}/{num_epochs}', dynamic_ncols=True)
    gradients = dict(grad_norm_align=0., grad_norm_bhlr=0., grad_norm_lhfe=0., grad_norm_rgb_backbone=0.,
                     grad_norm_tir_backbone=0., grad_norm_aux=0.)
    for bi,batch in progress:
        batch = preprocess_batch(batch,device)
        n = max(1,int(batch['rgb_semantic_img'].shape[0]))
        window_start = (bi // accum) * accum
        window_steps = min(accum, len(loader) - window_start)
        sync_now = ((bi+1)%accum==0 or bi+1==len(loader))
        raw._pcgrad_backward_scale = float(scaler.get_scale()) / (n * window_steps)
        ctx = net.no_sync() if isinstance(net,DDP) and not sync_now else contextlib.nullcontext()
        with ctx:
            with autocast_context(device,amp):
                vec, log = net(batch)
                loss = vec.sum() / (n * window_steps)
            bad_loss_local = not bool(torch.isfinite(loss).item())
            if _global_invalid_flag(bad_loss_local, device):
                rank = int(os.environ.get('RANK', '0'))
                print(f'[NONFINITE-LOSS] rank={rank} stage={stage} epoch={epoch} '
                      f'batch={bi + 1} bad_local={bad_loss_local} '
                      f'loss={float(loss.detach().item())}', flush=True)
                raise FloatingPointError(
                    'Nonfinite FORWARD loss detected before backward; check '
                    'detection/geometry loss construction, not GradScaler.')
            scaler.scale(loss).backward()
        sum_loss += float(vec.detach().sum().item()) / n
        count += 1
        items = tr.normalize_loss_items(log)
        for k,v in items.items(): running[k] = running.get(k,0.) + float(v)
        if sync_now:
            applied, consecutive_bad, preclip_norm, group_norms = _safe_gradient_update(
                net, optimizer, scaler, device, cfg, stage, epoch, bi, consecutive_bad)
            if applied:
                for key, value in group_norms.items():
                    gradients[key] += value
                steps += 1
            else:
                skipped_steps += 1
        if is_main_process() and (bi%8==0 or sync_now):
            progress.set_postfix(loss=f'{sum_loss/count:.4f}',mono=f'{float(items.get("mono_weighted_loss",0)):.4f}',
                                 kd=f'{float(items.get("kd_weighted_loss",0)):.5f}',accum=accum)
    progress.close()
    logs = {k:v/max(1,count) for k,v in running.items()}
    logs.update({k:v/max(1,steps) for k,v in gradients.items()})
    logs['gradient_steps_applied'] = float(steps)
    logs['gradient_steps_skipped'] = float(skipped_steps)
    logs['amp_scale_final'] = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    logs['effective_batch']=float(cfg['train']['batch']*accum)
    logs['accumulate']=float(accum)
    return distributed_mean(sum_loss/max(1,count),device), distributed_mean_dict(logs,device)


def broadcast_any(metrics, device):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        item = [metrics if is_main_process() else None]
        torch.distributed.broadcast_object_list(item, src=0, device=device)
        metrics = item[0]
    return metrics


def save_history(out, stage, rows):
    if not is_main_process(): return
    (out/f'history_{stage}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8')
    keys = []
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with (out/f'history_{stage}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def make_scheduler(opt,cfg,stage,epochs):
    trcfg=cfg['train']
    stcfg=trcfg.get('stage2',{}) if stage=='stage2' else trcfg
    mode=str(stcfg.get('scheduler',trcfg.get('scheduler','cosine'))).lower()
    if mode=='multistep':
        milestones=[int(x) for x in stcfg.get('lr_milestones', [50,80])]
        if any(x<1 or x>=epochs for x in milestones):
            raise ValueError(f'Invalid MultiStepLR milestones={milestones} for {epochs}e')
        gamma=float(stcfg.get('lr_gamma',0.2))
        return torch.optim.lr_scheduler.MultiStepLR(opt,milestones=sorted(milestones),gamma=gamma)
    if mode=='cosine':
        return tr.build_scheduler(opt,epochs,lrf=float(stcfg.get('lrf',0.01)))
    raise ValueError(f'Unknown scheduler {mode!r}')


def run_stage(net,cfg,train_loader,val_loader,device,out,stage,epochs,lr0,
              offset,global_best,reset_from=None,resume_path=None):
    raw=unwrap(net)
    if reset_from:
        load_checkpoint(reset_from,raw,device)
    optimizer=tr.build_optimizer(net,cfg,lr0=lr0)
    scheduler=make_scheduler(optimizer,cfg,stage,epochs)
    scaler=make_grad_scaler(device,enabled=bool(cfg['train'].get('amp',True)))
    best=-1.; bad=0; start=1
    stage_best=out/'weights'/f'{stage}_best.pt'
    rows=[]
    if resume_path:
        ck=load_checkpoint(resume_path,raw,device,optimizer,scheduler,scaler)
        if str(ck['stage']) != stage:
            raise ValueError(f'Resume checkpoint stage={ck["stage"]} != current stage={stage}')
        start=int(ck['stage_epoch'])+1
        best=float(ck['best_map5095']);bad=int(ck.get('early_stop_counter',0))
        global_best=max(global_best,best)  # preserve best.pt on an ordinary resume
        hist=out/f'history_{stage}.json'
        if hist.exists(): rows=json.loads(hist.read_text(encoding='utf-8'))
        print(f'[RESUME] {resume_path} starting epoch={start} stage_best={best}',flush=True)
    patience=int((cfg['train'].get('stage2',{}) if stage=='stage2' else cfg['train']).get('patience',100))
    min_epochs=int(cfg.get('monitor',{}).get('minimum_stage1_epochs',epochs)) if stage=='stage1' else 0
    geom_every=int(cfg.get('monitor',{}).get('geometry_val_every',5))
    mono_every=int(cfg.get('monitor',{}).get('mono_val_every',10))
    last_epoch=start-1
    for epoch in range(start,epochs+1):
        last_epoch=epoch
        train_loss,log=train_epoch(net,train_loader,optimizer,scaler,device,cfg,stage,epoch,epochs)
        scheduler.step()
        distributed_barrier()
        metrics=None
        if is_main_process():
            raw.eval()
            metrics=validate_metrics(raw,val_loader,device,amp=bool(cfg['train'].get('amp',True)),
                conf_thres=.001,iou_thres=.70,max_det=300,preview_num=0)
            if geom_every>0 and raw.core.alignment_enabled and (epoch==1 or epoch%geom_every==0):
                metrics.update(geometry_iou(raw,val_loader,device,
                    max_images=int(cfg.get('monitor',{}).get('alignment_val_images',96))))
            if mono_every>0 and raw.mono_enabled and (epoch==1 or epoch%mono_every==0):
                metrics.update(mono_metrics(raw,val_loader,device,bool(cfg['train'].get('amp',True))))
        metrics=broadcast_any(metrics,device)
        score=float(metrics['map5095'])
        min_delta=float(cfg['train'].get('early_stop_min_delta',1e-6))
        improved=score>best+min_delta
        global_improved=score>global_best+min_delta
        if improved: best=score;bad=0
        else: bad+=1
        if global_improved:global_best=score
        if is_main_process():
            last=out/'weights'/f'{stage}_last.pt'
            save_checkpoint(last,net,cfg,stage,epoch,offset,optimizer,scheduler,scaler,best,bad)
            shutil.copy2(last,out/'weights'/'last.pt')
            if improved: shutil.copy2(last,stage_best)
            if global_improved:shutil.copy2(last,out/'weights'/'best.pt')
            row={'stage':stage,'epoch':epoch,'global_epoch':offset+epoch,
                 'train_loss_per_image':train_loss,'lr':float(optimizer.param_groups[0]['lr']),
                 **log, **{k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))}}
            rows.append(row);save_history(out,stage,rows)
            print(f'[{stage}] epoch={epoch}/{epochs} Val fusion AP50:95={score:.6f} '
                  f'AP75={float(metrics.get("map75",0)):.6f} best={best:.6f} '
                  f'AlignIoU={metrics.get("align_iou","not checked")} '
                  f'mono={log.get("weight_mono",0):.4f} KD={log.get("weight_kd",0):.4f} '
                  f'KD-RGB-eligible={log.get("teacher_rgb_selected",0):.3f} '
                  f'KD-TIR-eligible={log.get("teacher_tir_selected",0):.3f} '
                  f'PCGradRGB={log.get("pcgrad_rgb_conflict",0):.3f} '
                  f'PCGradTIR={log.get("pcgrad_tir_conflict",0):.3f} '
                  f'gradAlign={log.get("grad_norm_align",0):.4g} '
                  f'gradBHLR={log.get("grad_norm_bhlr",0):.4g} '
                  f'grad_skips={log.get("gradient_steps_skipped",0):.0f} '
                  f'amp_scale={log.get("amp_scale_final",1):.0f} '
                  f'no_improve={bad}/{patience}',flush=True)
        distributed_barrier()
        if patience>0 and bad>=patience and epoch>=min_epochs:
            if is_main_process():print(f'[EARLY STOP] {stage} epoch={epoch}',flush=True)
            break
    return dict(stage_best=best,global_best=global_best,epochs_ran=last_epoch,
                best_path=str(stage_best))


def smoke_test(cfg):
    device=select_device(cfg['train']['device'])
    set_seed(int(cfg['train'].get('seed',0)),True)
    _,loader=tr._build_data(cfg,'train',False,False,batch_size=1,
                            distributed=False,rank=0,world_size=1)
    net=build_student(cfg,device)
    net.train()
    net.set_progress('stage1',int(cfg['distill']['kd'].get('reach_epoch',30)))
    if net.kd_weight()>0: net.attach_teachers(device)
    batch=preprocess_batch(next(iter(loader)),device)
    net._pcgrad_backward_scale=1.0
    with autocast_context(device,bool(cfg['train'].get('amp',True))):
        vec,logs=net(batch)
        loss=vec.sum()/max(1,len(batch['rgb_semantic_img']))
    if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite smoke loss')
    loss.backward()
    bad_smoke = []
    for param_name, param in net.named_parameters():
        if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
            bad_smoke.append(param_name)
    if bad_smoke:
        raise FloatingPointError('Nonfinite smoke-test gradients; first parameters: '
                                 + ', '.join(bad_smoke[:12]))
    for label,module,active in (('align',net.core.alignment,net.core.alignment_enabled),
                                ('BHLR',net.core.bhlr,net.core.bhlr_enabled),
                                ('AG-LHFE',getattr(net.core,'ag_lhfe',None),
                                 hasattr(net.core,'ag_lhfe'))):
        g=grad_norm(module)
        print(f'[SMOKE] {label} enabled={active} gradient={g:.6g}',flush=True)
        if active and not g>0: raise RuntimeError(f'{label} enabled but gradient is zero')
        if not active and module is not None: raise RuntimeError(f'{label} disabled but module exists')
    print('[SMOKE PASS] Student forward/backward, real batch, KD, Mono, PCGrad conditional',flush=True)


def run(cfg):
    t=cfg['train']
    device, rank, local_rank, world=setup_distributed(t['device'])
    try:
        set_seed(int(t.get('seed',0)),bool(t.get('deterministic',True)))
        local_batch,global_batch,_=tr._resolve_batch_size(cfg,world)
        _,loader=tr._build_data(cfg,'train',True,True,batch_size=local_batch,
                                distributed=world>1,rank=rank,world_size=world)
        _,val_loader=(tr._build_data(cfg,'val',False,False,
                      batch_size=int(t.get('val_batch',2)),distributed=False,
                      rank=0,world_size=1) if is_main_process() else (None,None))
        raw=build_student(cfg,device)
        out=Path(cfg['output']['project'])/str(cfg['output']['name'])
        if is_main_process(): (out/'weights').mkdir(parents=True,exist_ok=True)
        distributed_barrier()
        net=DDP(raw,device_ids=[int(device.index)],output_device=int(device.index),
                broadcast_buffers=True,find_unused_parameters=True) if world>1 else raw
        set_seed(int(t.get('seed',0))+rank,bool(t.get('deterministic',True)))
        print(f'[TRAIN] arm={cfg["distill"]["arm"]} rank={rank}/{world} '
              f'cuda={device} batch/rank={local_batch} global_microbatch={global_batch} '
              f'accumulate={t.get("accumulate",1)} effective_batch={global_batch*int(t.get("accumulate",1))} '
              f'mono={raw.mono_enabled} pcgrad={cfg.get("pcgrad",{}).get("enabled",False)} '
              f'KD={cfg["distill"]["kd"]["enabled"]}',flush=True)
        one=run_stage(net,cfg,loader,val_loader,device,out,'stage1',int(t['epochs']),
                      float(t['lr0']),0,-1.,
                      resume_path=t.get('resume') or None)
        two=None
        if bool(t.get('stage2',{}).get('enabled',False)):
            st=t['stage2']; distributed_barrier()
            two=run_stage(net,cfg,loader,val_loader,device,out,'stage2',
                          int(st['epochs']),float(st['lr0']),one['epochs_ran'],one['global_best'],
                          reset_from=one['best_path'])
        if is_main_process():
            summ=dict(arm=cfg['distill']['arm'],stage1=one,stage2=two,
                      best_ap5095=two['global_best'] if two else one['global_best'],
                      effective_batch=global_batch*int(t.get('accumulate',1)),
                      checkpoint_main_compatible=True)
            (out/'training_summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False),encoding='utf-8')
            print('[TRAIN COMPLETE]',summ,flush=True)
        distributed_barrier()
    finally:
        cleanup_distributed()


# FLEXIBLE_DISTILL_CLI_V1

def _configure_batch_cli(cfg, args):
    """Resolve a global microbatch, GPU ids, and integer accumulation at runtime.

    The original YAML is not modified. In particular, when it contains an old
    accumulate: 16 with expected_effective_batch: 128, accumulation is recomputed
    rather than rejecting the experiment.
    """
    t = cfg['train']
    if args.device is not None:
        t['device'] = args.device

    devices = parse_device_ids(t['device'])
    if not devices:
        raise ValueError('Choose at least one CUDA GPU, e.g. --device 2 or --device 5,6')

    if args.batch is not None:
        t['batch'] = int(args.batch)
    if args.batch_per_gpu is not None:
        t['batch'] = int(args.batch_per_gpu) * len(devices)
    b = int(t['batch'])
    if b < 1:
        raise ValueError('Global batch must be >= 1')
    if b % len(devices) != 0:
        raise ValueError(
            f'Global batch {b} is not divisible by {len(devices)} GPUs {devices}. '
            f'Choose a multiple of {len(devices)} with --batch, or use --batch-per-gpu.')
    if t.get('batch_mode', 'global') != 'global':
        raise ValueError('This trainer uses global batch mode only')

    original_target = t.get('expected_effective_batch')
    if original_target is None:
        original_target = b * int(t.get('accumulate', 1))
    target = int(args.effective_batch if args.effective_batch is not None else original_target)
    if target < 1:
        raise ValueError('Target effective batch must be >= 1')

    if args.accumulate is not None:
        accum = int(args.accumulate)
    else:
        # Nearest feasible integer accumulation; halves rounded upward.
        accum = max(1, (2 * target + b) // (2 * b))
    if accum < 1:
        raise ValueError('Accumulation must be >= 1')

    actual = b * accum
    if args.strict_effective_batch and actual != target:
        raise ValueError(
            f'Target effective batch {target} cannot be achieved with global '
            f'batch={b}, accumulate={accum}. Actual={actual}. '
            'Choose a divisible batch or a different --effective-batch.')

    # Keep original strict validate_cfg() intact by resolving both fields first.
    t['accumulate'] = accum
    t['expected_effective_batch'] = actual
    if actual != target:
        print(f'[BATCH WARNING] Requested target effective batch={target}, '
              f'but integer accumulation gives actual={actual}. '
              'Use --strict-effective-batch to reject this difference.', flush=True)
    print(f'[BATCH RESOLVED] GPUs={devices} world_size={len(devices)} '
          f'global_bs={b} per_gpu_bs={b // len(devices)} '
          f'accumulate={accum} effective_bs={actual} '
          f'original_target={target}', flush=True)
    return devices


def main():
    parser = argparse.ArgumentParser(description='Distillation training with selectable GPUs and batch sizes')
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--device', type=str, default=None,
                        help='Physical CUDA IDs: 2, 5,6, 0,1,2,3, ...')
    batch_opts = parser.add_mutually_exclusive_group()
    batch_opts.add_argument('--batch', type=int, default=None,
                            help='Global microbatch across all selected GPUs')
    batch_opts.add_argument('--batch-per-gpu', type=int, default=None,
                            help='Local microbatch on each selected GPU')
    parser.add_argument('--accumulate', type=int, default=None,
                        help='Optional manual accumulation. Default: auto from target effective batch.')
    parser.add_argument('--effective-batch', type=int, default=None,
                        help='Desired effective batch (defaults to YAML expected_effective_batch)')
    parser.add_argument('--strict-effective-batch', action='store_true',
                        help='Reject requests that cannot exactly achieve target effective batch')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    devices = _configure_batch_cli(cfg, args)
    cfg = validate_cfg(cfg)
    t = cfg['train']
    effective = int(t['batch']) * int(t['accumulate'])
    print(f'[CHECK] arm={cfg["distill"]["arm"]} physical_devices={devices} '
          f'global_batch={t["batch"]} accumulation={t["accumulate"]} '
          f'effective_batch={effective}', flush=True)

    if args.check:
        print('[CHECK PASS] Single GPU and automatic multi-GPU DDP supported.', flush=True)
        return
    if args.smoke_test:
        smoke_test(cfg)
        return

    if len(devices) > 1 and int(os.environ.get('WORLD_SIZE', '1')) == 1:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices)}', '-m', 'distill_rgbt.train',
               '--cfg', str(Path(args.cfg).expanduser().resolve())]
        # CRITICAL: Torchrun children must receive the SAME overrides as parent.
        for flag, value in (
            ('--device', args.device),
            ('--batch', args.batch),
            ('--batch-per-gpu', args.batch_per_gpu),
            ('--accumulate', args.accumulate),
            ('--effective-batch', args.effective_batch),
        ):
            if value is not None:
                cmd.extend((flag, str(value)))
        if args.strict_effective_batch:
            cmd.append('--strict-effective-batch')
        print('[DDP] Physical GPUs', devices, '=> torchrun', len(devices), 'processes', flush=True)
        subprocess.run(cmd, env=env, check=True)
        return

    out = Path(cfg['output']['project']) / str(cfg['output']['name'])
    if is_main_process() and not t.get('resume') and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}. Change output.name or set train.resume.')
    run(cfg)


if __name__ == '__main__':
    main()

