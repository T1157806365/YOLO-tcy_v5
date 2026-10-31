"""Rich, explicit GA-MonoAlign training diagnostics; reporting only, no gradients/weight changes."""
from __future__ import annotations
import math
import torch


def _f(d, k, precision=4):
    try:
        v = d.get(k)
        if v is None:
            return 'N/A'
        v = float(v)
        if not math.isfinite(v):
            return 'N/A'
        return f'{v:.{precision}f}'
    except (ValueError, TypeError, OverflowError):
        return 'N/A'


def _mem(device):
    try:
        if device.type == 'cuda':
            return f'{torch.cuda.memory_reserved(device) / 1024**3:.2f}G'
    except (RuntimeError, AttributeError):
        pass
    return 'CPU'


def progress_description(stage, epoch, nepoch, device, logitems, instances, high_imgsz, tir_imgsz):
    """Match the old epoch GPU_mem / box / cls / dfl / instances / size table."""
    return (f'{stage} {epoch:>3}/{nepoch:<3} '
            f'GPU_mem={_mem(device)} '
            f'box={_f(logitems, "box_loss")} '
            f'cls={_f(logitems, "cls_loss")} '
            f'dfl={_f(logitems, "dfl_loss")} '
            f'Instances={int(instances)} Size={high_imgsz}/{tir_imgsz}')


def stage_banner(stage, epochs, lr0, lrf, patience, cfg):
    m = cfg.get('model', {})
    data = cfg.get('data', {})
    tr = cfg.get('train', {})
    g = m.get('rlsfa_gtga_centered', {}) or {}
    return ('\n' + '='*94 + '\n'
            f'{stage} training | GA-MonoAlign-PCGrad (GMLS-v2 preserved)\n'
            + '='*94 + '\n'
            f'epochs: {epochs} | lr0: {lr0:g} | lrf: {lrf:g} | patience: {patience}\n'
            f'RGB high/semantic/TIR: {data.get("rgb_imgsz", "?")}/'
            f'{data.get("rgb_semantic_imgsz", "?")}/{data.get("tir_imgsz", "?")}\n'
            f'GMLS={g.get("coarse_gmls_enabled", "?")} | GMLS-v2 routing={g.get("gmls_v2_correction_enabled", "?")} '
            f'| Fine={g.get("fine_translation_enabled", "?")} '
            f'| BHLR={m.get("bhlr", {}).get("enabled", "?")}\n'
            f'Official YOLO pretrained={m.get("pretrained", False)} '
            f'| student warm start={tr.get("init_student_weights", None)} '
            f'| student resume={tr.get("resume", None)}\n'
            'Train: box / cls / dfl + original alignment + auxiliary detection + teacher KD\n'
            'Val: fusion AP every epoch; RGB/TIR auxiliary AP periodically; geometry on subset\n'
            + '='*94)


def _aux_line(label, prefix, metric, epoch, mono_interval, wm, latest):
    key = prefix + '_map5095'
    if key in metric:
        suffix = ' [head not optimized yet]' if wm <= 0 else ' [current validation]'
        return (f'{label}: P={_f(metric, prefix+"_precision")} '
                f'R={_f(metric, prefix+"_recall")} '
                f'AP50={_f(metric, prefix+"_map50")} '
                f'AP75={_f(metric, prefix+"_map75")} '
                f'AP50:95={_f(metric, key)}{suffix}')
    if latest and key in latest:
        return (f'{label}: NOT EVALUATED this epoch; last@epoch={latest.get("_epoch", "?")} '
                f'AP50:95={_f(latest, key)} (cached, NOT current)')
    return f'{label}: NOT EVALUATED this epoch (validation interval={mono_interval})'


def epoch_report(*, stage, epoch, nepoch, trainloss, lr, logitems, metric,
                 best, global_best, joint_best, no_improve, patience,
                 mono_interval, latest_mono=None):
    wm = float(logitems.get('weight_mono', 0))
    wk = float(logitems.get('weight_kd', 0))
    n = int(metric.get('align_pairs', 0))
    z = int(metric.get('align_zero', 0))
    zero = f'{z}/{n} ({100*z/n:.2f}%)' if n else 'N/A'
    pc_state = 'OFF (mono supervision not active)' if wm<=0 else 'ON'
    kd_state = 'OFF (KD warmup)' if wk<=0 else 'ON'
    logs = [
        '\n' + '='*94,
        f'[{stage}] epoch {epoch}/{nepoch} | lr={lr:.8g} '
        f'| train_loss/image={trainloss:.6f} | best metric=AP50:95',
        '-'*94,
        'Train detection: '
        f'box={_f(logitems,"box_loss",6)} '
        f'cls={_f(logitems,"cls_loss",6)} '
        f'dfl={_f(logitems,"dfl_loss",6)} '
        f'det={_f(logitems,"det_raw_loss",6)}',
        'Train weighted: '
        f'align={_f(logitems,"align_weighted_loss",6)} '
        f'mono={_f(logitems,"mono_weighted_loss",6)} '
        f'KD={_f(logitems,"kd_weighted_loss",6)} '
        f'| lambda_mono={wm:.4f} lambda_KD={wk:.4f}',
        'Val fusion: '
        f'P={_f(metric,"precision",6)} R={_f(metric,"recall",6)} '
        f'AP50={_f(metric,"map50",6)} '
        f'AP75={_f(metric,"map75",6)} '
        f'AP50:95={_f(metric,"map5095",6)}',
        'Val geometry (QUICK SUBSET, not full split): '
        f'AlignIoU={_f(metric,"align_iou",6)} '
        f'zero={zero} scanned_images={int(metric.get("align_images_scanned",0))}',
        _aux_line('RGB-only', 'rgb_only', metric, epoch, mono_interval, wm, latest_mono),
        _aux_line('TIR-only', 'tir_only', metric, epoch, mono_interval, wm, latest_mono),
        'Teacher: '
        f'{kd_state} | RGB reliable={_f(logitems,"teacher_rgb_selected",4)} '
        f'TIR reliable={_f(logitems,"teacher_tir_selected",4)} '
        f'| response RGB={_f(logitems,"rgb_kd_response",6)} '
        f'TIR={_f(logitems,"tir_kd_response",6)}',
        'PCGrad: '
        f'{pc_state} | RGB conflict={_f(logitems,"pcgrad_rgb_conflict",4)} '
        f'cos={_f(logitems,"pcgrad_rgb_cosine",4)} '
        f'| TIR conflict={_f(logitems,"pcgrad_tir_conflict",4)} '
        f'cos={_f(logitems,"pcgrad_tir_cosine",4)}',
    ]
    # Existing GMLS auxiliary terms are emitted rather than silently omitted.
    alignment_terms = [k for k in sorted(logitems) if k.startswith('align_') and
                       k not in ('align_weighted_loss','align_iou','align_zero','align_pairs','align_images_scanned')]
    if alignment_terms:
        logs += ['Alignment components: ' + '  '.join(
            f'{k}={_f(logitems,k,5)}' for k in alignment_terms)]
    logs += [
        f'Best: stage_best={best:.6f} | global_best={global_best:.6f} '
        f'| stage_joint_best={joint_best:.6f} | no_improve={no_improve}/{patience}',
        '='*94
    ]
    return '\n'.join(logs)


def write_history_csv(path, history):
    """Save all scalar per-epoch metrics, blank if a metric was not validated."""
    import csv
    from pathlib import Path
    if not history:
        return
    keys = list(dict.fromkeys(k for row in history for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader()
        w.writerows(history)
