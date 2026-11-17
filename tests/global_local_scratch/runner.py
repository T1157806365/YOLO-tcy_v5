"""Single-run scratch joint training and fair full-image validation for GMLS-v2.

No offline ROI dataset. No frozen teacher. Builds training ROIs online from
GTGA proposals, with GT-assisted crop augmentation only while TRAINING.
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets.rsdt_dataset import build_rsdt_dataset
from engine.runtime import (
    load_config, set_seed, setup_distributed, preprocess_batch,
    autocast_context,
)
from engine.trainer import build_optimizer, build_scheduler
from engine.evaluator import prepare_semantic_ground_truth, process_single_image
from ultralytics.utils.metrics import DetMetrics
from .joint_model import GMLSV2GlobalLocalScratch
from .geometry import norm_xywh_to_xyxy
from .logging_utils import EpochStats, render_epoch_report, flatten_history_row, write_history_csv


def make_loader(cfg, split: str, *, batch_size: int, workers: int,
                distributed: bool = False, rank: int = 0, world_size: int = 1):
    dc, ac, tc = cfg['data'], cfg.get('augment', {}), cfg['train']
    ds = build_rsdt_dataset(
        rgb_yaml=dc['rgb'], tir_yaml=dc['tir'], split=split,
        rgb_high_imgsz=int(dc.get('rgb_imgsz', 1920)),
        rgb_semantic_imgsz=int(dc.get('rgb_semantic_imgsz', 640)),
        tir_imgsz=int(dc.get('tir_imgsz', 640)),
        pair_mode=str(dc.get('pair_mode', 'relative')),
        augment=(split == 'train'), fliplr=float(ac.get('fliplr', 0.5)),
        flipud=float(ac.get('flipud', 0.0)), tir_channels=3,
        strict_pair=True,
    )
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=split == 'train',
        seed=int(tc.get('seed', 0)), drop_last=False,
    ) if distributed else None
    loader = DataLoader(
        ds, batch_size=int(batch_size),
        shuffle=(split == 'train' and sampler is None), sampler=sampler,
        num_workers=int(workers), pin_memory=True,
        collate_fn=ds.collate_fn, drop_last=False,
        persistent_workers=(int(workers) > 0),
    )
    return loader, sampler


def load_scratch_checkpoint(path, cfg, device, *, verbose=False):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if obj.get('architecture') != 'gmls_v2_global_local_scratch':
        raise ValueError(f'Not the GMLS-v2 scratch global-local checkpoint: {path}')
    net = GMLSV2GlobalLocalScratch(cfg, verbose=verbose)
    raw = obj['model_state_dict']
    if raw and all(k.startswith('module.') for k in raw):
        raw = {k[7:]: v for k, v in raw.items()}
    net.load_state_dict(raw, strict=True)
    net.to(device).eval()
    return net, obj


def save_checkpoint(path: Path, net, opt, sched, epoch, score, cfg, *, scaler=None,
                    best_epoch=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save({
        'architecture': 'gmls_v2_global_local_scratch',
        'model_state_dict': {k: v.detach().cpu().clone()
                             for k,v in net.state_dict().items()},
        'config': cfg,
        'epoch': int(epoch),
        'val_map5095': float(score),
        'best_epoch': int(best_epoch if best_epoch is not None else epoch),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
    }, tmp)
    os.replace(tmp, path)


def _posthoc_roi_coverage(batch, sample_index: int, geos, *, threshold: float) -> tuple[int, int]:
    """GT is used only AFTER prediction to audit ROI coverage, never to select ROIs."""
    mask = batch['rgb_batch_idx'] == int(sample_index)
    gt_norm = batch['rgb_bboxes'][mask]
    n_gt = int(gt_norm.shape[0])
    if n_gt == 0 or not geos:
        return 0, n_gt
    H, W = batch['rgb_img'].shape[-2:]
    boxes = norm_xywh_to_xyxy(gt_norm, int(W), int(H)).detach().float()
    a = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1).clamp_min(1e-7)
    best_visible = torch.zeros(n_gt, device=boxes.device)
    for geo in geos:
        roi_min = boxes.new_tensor([geo.x0, geo.y0])
        roi_max = roi_min + float(geo.size)
        lo = torch.maximum(boxes[:, :2], roi_min)
        hi = torch.minimum(boxes[:, 2:], roi_max)
        inter = (hi - lo).clamp_min(0).prod(-1)
        best_visible = torch.maximum(best_visible, inter / a)
    return int((best_visible >= threshold).sum().item()), n_gt


def _posthoc_alignment_iou(batch, sample_index: int, alignment_geometry):
    """GT-based validation diagnostic for q=S*p+t geometry; never an ROI input."""
    rmask = batch['rgb_batch_idx'] == int(sample_index)
    tmask = batch['tir_batch_idx'] == int(sample_index)
    if int(rmask.sum()) != 1 or int(tmask.sum()) != 1:
        return None, None
    rb = batch['rgb_semantic_bboxes'][rmask][0].detach().float()
    tb = batch['tir_bboxes'][tmask][0].detach().float()
    scale = alignment_geometry['sampling_scale'][sample_index].detach().float().reshape(2)
    trans = alignment_geometry['sampling_translation'][sample_index].detach().float().reshape(2)
    if not torch.isfinite(scale).all() or not torch.isfinite(trans).all():
        return None, None
    scale = scale.clamp_min(1e-6)
    tc = (tb[:2] - trans) / scale
    tw = tb[2:] / scale
    rlo, rhi = rb[:2] - rb[2:]*.5, rb[:2] + rb[2:]*.5
    tlo, thi = tc - tw*.5, tc + tw*.5
    intersect = (torch.minimum(rhi,thi) - torch.maximum(rlo,tlo)).clamp_min(0).prod()
    area_r = rb[2:].clamp_min(0).prod()
    area_t = tw.clamp_min(0).prod()
    iou = (intersect / (area_r + area_t - intersect).clamp_min(1e-7)).clamp(0,1)
    hs,ws = batch['rgb_semantic_img'].shape[-2:]
    center_error = torch.linalg.norm((tc-rb[:2])*rb.new_tensor([ws,hs]))
    return float(iou.item()), float(center_error.item())


@torch.inference_mode()
def evaluate(net, loader, device, cfg, *, split='val', smoke=False):
    net.eval()
    ecfg = cfg.get('eval', {})
    conf = float(ecfg.get('conf', 0.001))
    iou = float(ecfg.get('iou', 0.70))
    amp = bool(cfg.get('train', {}).get('amp', True))
    metrics = {key: DetMetrics(names={0: 'UAV'})
               for key in ('global', 'local', 'fused')}
    iouv = torch.linspace(.50, .95, 10, device=device)
    seen, replaces, adds, num_rois = 0, 0, 0, 0
    roi_valid_total, roi_gt_hit, roi_gt_count = 0.0, 0, 0
    align_iou_sum, align_center_sum, align_n = 0.0, 0.0, 0
    pred_counts = {key: 0 for key in metrics}
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=f'{split} joint validation', ncols=120, leave=False)
    for _, batch in enumerate(pbar):
        batch = preprocess_batch(batch, device)
        with autocast_context(device, amp):
            preds = net.predict_global_local(batch, conf=conf, iou=iou)
        bsz = int(batch['rgb_img'].shape[0])
        seen += bsz
        replaces += sum(int(x['replaced']) for x in preds['merge_stats'])
        adds += sum(int(x['added']) for x in preds['merge_stats'])
        num_rois += int(preds['roi_count'])
        rois_by_image = preds.get('roi_geometries', [[] for _ in range(bsz)])
        for j in range(bsz):
            geos = rois_by_image[j]
            roi_valid_total += sum(float(x.tir_valid_fraction) for x in geos)
            nhit, ngt = _posthoc_roi_coverage(
                batch, j, geos,
                threshold=float(cfg['model']['global_local'].get('min_gt_visible', 0.6)),
            )
            roi_gt_hit += nhit
            roi_gt_count += ngt
            geom = preds.get('alignment_geometry')
            if geom is not None:
                aiou, ace = _posthoc_alignment_iou(batch, j, geom)
                if aiou is not None:
                    align_iou_sum += aiou
                    align_center_sum += ace
                    align_n += 1
            gt = prepare_semantic_ground_truth(batch, j)
            target_cls = gt['cls'].detach().float().cpu().numpy()
            for mode in metrics:
                pred = preds[mode][j]
                pred_counts[mode] += int(pred['bboxes'].shape[0])
                tp = process_single_image(pred, gt, iouv)
                metrics[mode].update_stats({
                    'tp': tp,
                    'conf': pred['conf'].detach().float().cpu().numpy(),
                    'pred_cls': pred['cls'].detach().float().cpu().numpy(),
                    'target_cls': target_cls,
                    'target_img': __import__('numpy').unique(target_cls),
                    'im_name': Path(batch['rgb_path'][j]).name,
                })
        if smoke:
            break
    pbar.close()
    out = {
        'images': seen,
        'roi_count': num_rois,
        'rois_per_image': num_rois / max(seen, 1),
        'roi_gt_coverage': roi_gt_hit / max(roi_gt_count, 1),
        'roi_gt_hit': roi_gt_hit,
        'roi_gt_count': roi_gt_count,
        'align_iou': align_iou_sum / align_n if align_n else None,
        'align_center_error_px': align_center_sum / align_n if align_n else None,
        'align_eval_pairs': align_n,
        'roi_tir_valid_fraction': roi_valid_total / max(num_rois, 1),
        'replaced': replaces,
        'added': adds,
        'elapsed_seconds': time.perf_counter() - t0,
    }
    for key, one in metrics.items():
        one.process(plot=False)
        precision, recall, ap50, ap = one.mean_results()
        out[key] = {
            'precision': float(precision),
            'recall': float(recall),
            'map50': float(ap50),
            'map75': float(one.box.map75),
            'map5095': float(ap),
            'preds_per_image': pred_counts[key] / max(seen, 1),
            'pred_count': pred_counts[key],
        }
    return out


def _load_resume_checkpoint(path, net):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('architecture') != 'gmls_v2_global_local_scratch':
        raise ValueError(f'Checkpoint architecture mismatch: {path}')
    state = ckpt['model_state_dict']
    if state and all(str(k).startswith('module.') for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    if 'optimizer_state_dict' not in ckpt or 'scheduler_state_dict' not in ckpt:
        raise KeyError('Resume checkpoint must contain optimizer and scheduler states')
    return ckpt


def run_train(args):
    cfg = load_config(args.cfg)
    tc = cfg['train']
    if args.smoke:
        cfg['train'] = {**tc, 'workers': 0}
        tc = cfg['train']
    if tc.get('resume') is not None or cfg['model'].get('pretrained') is not False:
        raise ValueError('Scratch initialization requires model.pretrained: false, train.resume: null;'
                         ' use --resume last.pt to continue an EXISTING scratch run')
    if int(os.getenv('WORLD_SIZE', '1')) > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://',
                                timeout=timedelta(hours=3))
    device, rank, _, world = setup_distributed(args.device or tc.get('device', '5'))
    set_seed(int(tc.get('seed', 0)) + rank, bool(tc.get('deterministic', True)))
    batch = int(tc.get('batch', 4))
    if batch % world != 0 or batch < world:
        raise ValueError('Global batch must be divisible by world size')
    train_loader, sampler = make_loader(
        cfg, 'train', batch_size=batch // world,
        workers=int(tc.get('workers', 4)), distributed=world > 1,
        rank=rank, world_size=world,
    )
    val_loader = None
    if rank == 0 and not args.smoke:
        val_loader, _ = make_loader(cfg, 'val', batch_size=int(tc.get('val_batch', 1)),
                                    workers=int(tc.get('val_workers', 2)))
    net = GMLSV2GlobalLocalScratch(cfg, verbose=rank == 0).to(device)

    project = Path(cfg['output']['project']) / cfg['output']['name']
    resume_path = getattr(args, 'resume', None)
    ckpt = None
    if resume_path:
        ckpt = _load_resume_checkpoint(resume_path, net)
    elif not args.smoke and (project / 'weights' / 'last.pt').exists():
        raise FileExistsError(
            f'Existing run checkpoint: {project / "weights" / "last.pt"}. '
            'Use --resume <last.pt> to continue, or change output.name. '
            'Existing best.pt will NOT be silently overwritten.'
        )

    if world > 1:
        model = DDP(net, device_ids=[device.index], output_device=device.index,
                    find_unused_parameters=True, broadcast_buffers=True)
    else:
        model = net
    opt = build_optimizer(net, cfg)
    epochs = int(tc.get('epochs', 100))
    sched = build_scheduler(opt, epochs, lrf=float(tc.get('lrf', .01)))
    amp = bool(tc.get('amp', True)) and device.type == 'cuda'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    accumulate = int(tc.get('accumulate', 8))
    if accumulate < 1:
        raise ValueError('train.accumulate must be >=1')
    if rank == 0:
        (project / 'weights').mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best = -1.0
    best_epoch = 0
    if ckpt is not None:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        if ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = int(ckpt['epoch'])
        best = float(ckpt.get('val_map5095', -1.0))
        best_epoch = int(ckpt.get('best_epoch', start_epoch))
        if rank == 0:
            print(f'[RESUME] from={resume_path} completed_epochs={start_epoch} '
                  f'best_fused_AP50:95={best:.5f}', flush=True)
    if rank == 0:
        print(f'[GMLS-V2 SCRATCH JOINT] world={world} microbatch/rank={batch // world} '
              f'accumulate={accumulate}, effective_batch≈{batch * accumulate}; '
              f'global pretrained=False, local pretrained=False; '
              f'start_epoch={start_epoch} (resume={bool(ckpt)})', flush=True)
    if world > 1:
        dist.barrier()

    history_path = project / 'history.json'
    if rank == 0 and history_path.exists() and ckpt is not None:
        history = json.loads(history_path.read_text(encoding='utf-8'))
        history = [item for item in history if int(item['epoch']) <= start_epoch]
    else:
        history = []
    # Old checkpoints did not store best_epoch. Recover it from historical
    # validation scores when possible, instead of pretending it was last epoch.
    if rank == 0 and ckpt is not None and 'best_epoch' not in ckpt:
        matching = [int(rec['epoch']) for rec in history
                    if rec.get('val') is not None and
                    abs(float(rec['val']['fused']['map5095']) - best) <= 1e-8]
        if matching:
            best_epoch = matching[0]
    csv_path = project / 'epoch_metrics.csv'
    if rank == 0 and csv_path.exists() and ckpt is not None:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            history_csv = [r for r in csv.DictReader(f)
                           if int(r.get('epoch') or 0) <= start_epoch]
    else:
        history_csv = []

    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.perf_counter()
        net.set_epoch(epoch)
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        stats = EpochStats()
        lr = float(opt.param_groups[0]['lr'])
        iterator = tqdm(train_loader, disable=rank != 0, ncols=151,
                        desc=f'joint scratch {epoch+1}/{epochs}')
        postfix_interval = max(1, int(cfg.get('logging', {}).get('progress_postfix_interval', 10)))
        for step, batch_data in enumerate(iterator):
            batch_data = preprocess_batch(batch_data, device)
            is_last = step + 1 == len(train_loader)
            sync = (step + 1) % accumulate == 0 or is_last
            ctx = model.no_sync() if world > 1 and not sync else contextlib.nullcontext()
            with ctx:
                with autocast_context(device, amp):
                    loss, parts = model(batch_data)
                    group_start = (step // accumulate) * accumulate
                    denom = min(accumulate, len(train_loader) - group_start)
                    scaled_loss = loss / denom
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(f'nonfinite joint loss epoch={epoch+1} step={step}')
                scaler.scale(scaled_loss).backward()
            if sync:
                scaler.unscale_(opt)
                clip = float(tc.get('grad_clip', 10.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            stats.update(loss, parts)
            if rank == 0 and ((step + 1) % postfix_interval == 0 or is_last):
                iterator.set_postfix({
                    'total': f'{stats.mean("train_total"):.3f}',
                    'G': f'{stats.mean("global_det_loss"):.3f}',
                    'L': f'{stats.mean("local_det_loss"):.3f}',
                    'Align': f'{stats.mean("align_weighted_loss"):.3f}',
                    'ROI+': f'{stats.mean("roi_positive"):.2f}',
                })
            if args.smoke:
                break
        iterator.close()
        if args.smoke:
            if rank == 0:
                print('[SMOKE PASS] scratch joint forward+backward+step', flush=True)
            break

        train_stats = stats.global_means(device)  # all ranks participate
        train_duration = time.perf_counter() - epoch_t0
        if world > 1:
            dist.barrier()
        interval = max(1, int(tc.get('val_interval', 1)))
        do_val = (epoch == 0) or ((epoch + 1) % interval == 0) or (epoch + 1 == epochs)
        if rank == 0:
            report = evaluate(net, val_loader, device, cfg, split='val') if do_val else None
            score = float(report['fused']['map5095']) if report else best
            improved = bool(report is not None and score > best)
            if improved:
                best = score
                best_epoch = epoch + 1
            gpu_reserved = (torch.cuda.memory_reserved(device) / (1024 ** 3)
                            if device.type == 'cuda' else None)
            print(render_epoch_report(epoch + 1, epochs, train_stats, report,
                                      lr=lr, duration_s=train_duration, best=best,
                                      best_epoch=best_epoch, improved=improved,
                                      gpu_memory_gb=gpu_reserved), flush=True)
        # A rank0-only long validation is guarded by the 3h NCCL timeout above.
        if world > 1:
            dist.barrier()
        sched.step()   # step all ranks, after validation; keep checkpoints aligned
        if rank == 0:
            if improved:
                save_checkpoint(project / 'weights' / 'best.pt', net, opt, sched,
                                epoch + 1, best, cfg, scaler=scaler, best_epoch=best_epoch)
            save_checkpoint(project / 'weights' / 'last.pt', net, opt, sched,
                            epoch + 1, best, cfg, scaler=scaler, best_epoch=best_epoch)
            history.append({'epoch': epoch + 1,
                            'train_loss': train_stats.get('train_total'),
                            'train_stats': train_stats,
                            'best': best, 'best_epoch': best_epoch, 'val': report})
            history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                                    encoding='utf-8')
            if bool(cfg.get('logging', {}).get('write_csv', True)):
                history_csv.append(flatten_history_row(
                    epoch + 1, train_stats, report, lr, best, best_epoch, train_duration))
                write_history_csv(csv_path, history_csv)
                print(f'[SAVED] CSV: {csv_path} | best: {project / "weights" / "best.pt"}',
                      flush=True)
        if world > 1:
            dist.barrier()
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()

def run_val(args):
    cfg=load_config(args.cfg)
    device,_,_,_=setup_distributed(args.device or '5')
    set_seed(int(cfg['train'].get('seed',0)))
    ckpt=args.weights
    if not ckpt:
        ckpt=str(Path(cfg['output']['project'])/cfg['output']['name']/'weights'/'best.pt')
    net, obj=load_scratch_checkpoint(ckpt,cfg,device,verbose=False)
    loader,_=make_loader(cfg,args.split,batch_size=int(args.batch),workers=int(args.workers))
    result=evaluate(net,loader,device,cfg,split=args.split,smoke=args.smoke)
    print(f'[CHECKPOINT] {ckpt} | epoch={obj["epoch"]} | split={args.split}')
    for k in ('global','local','fused'):
        row=result[k]
        print(f'{k:>7s} P={row["precision"]:.5f} R={row["recall"]:.5f} '
              f'AP50={row["map50"]:.5f} AP75={row["map75"]:.5f} '
              f'AP50:95={row["map5095"]:.5f}')
    print(f'ROIs={result["roi_count"]}, replaced={result["replaced"]}, '
          f'added={result["added"]}, images={result["images"]}')
    out=Path(args.save_dir or 'runs/gmls_v2_global_local_evaluation')
    out.mkdir(parents=True,exist_ok=True)
    f=out/f'{args.split}_metrics.json'
    f.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[SAVED] {f}')
