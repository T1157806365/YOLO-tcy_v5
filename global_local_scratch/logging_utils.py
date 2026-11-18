"""Per-epoch logging for GMLS-v2 global+local scratch runs.

All logged values are measurements (not fabricated zeros). DDP rank means are
reduced before writing logs, and full-validation metrics remain rank0-only.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist


TRAIN_KEYS = (
    'train_total', 'global_det_loss', 'global_box_loss', 'global_cls_loss',
    'global_dfl_loss', 'local_det_loss', 'local_weighted_loss',
    'local_box_loss', 'local_cls_loss', 'local_dfl_loss',
    'align_weighted_loss', 'align_total', 'align_anchor', 'align_coarse',
    'align_scale', 'align_targetness', 'align_v2_quality', 'align_v2_route',
    'align_valid_pairs', 'gmls_gt_quality', 'dmls_gt_quality',
    'roi_positive', 'roi_gt_assisted', 'roi_tir_valid_fraction', 'train_rois',
)


def scalar_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError('Expected scalar metric, got ' + str(tuple(value.shape)))
        value = value.detach().item()
    value = float(value)
    return value if math.isfinite(value) else None


class EpochStats:
    """Stable key order makes distributed reductions safe across ranks."""
    def __init__(self):
        self.total = {k: 0.0 for k in TRAIN_KEYS}
        self.count = {k: 0.0 for k in TRAIN_KEYS}

    def update(self, loss, parts):
        values = {'train_total': loss, **parts}
        for key in TRAIN_KEYS:
            if key not in values:
                continue
            v = scalar_float(values[key])
            if v is None:
                continue
            self.total[key] += v
            self.count[key] += 1.0

    def mean(self, key):
        count = self.count[key]
        return (self.total[key] / count) if count else None

    def global_means(self, device):
        data = []
        for k in TRAIN_KEYS:
            data.extend((self.total[k], self.count[k]))
        tensor = torch.tensor(data, dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {
            key: (float(tensor[2 * j].item() / tensor[2 * j + 1].item())
                  if tensor[2 * j + 1].item() > 0 else None)
            for j, key in enumerate(TRAIN_KEYS)
        }


def f(value, digits=5):
    return 'N/A' if value is None else f'{float(value):.{digits}f}'


def render_epoch_report(epoch, epochs, train, val, *, lr, duration_s, best, best_epoch, improved,
                        gpu_memory_gb=None):
    def item(k):
        return f(train.get(k))
    lines = [
        '', '=' * 94,
        f'[EPOCH {epoch:03d}/{epochs}] lr={lr:.8f} | train_time={duration_s:.1f}s '
        f'| rank0_gpu_reserved={f(gpu_memory_gb, 2)}GB',
        '-' * 94,
        f'[TRAIN TOTAL]   total={item("train_total")} | global={item("global_det_loss")} '
        f'| local={item("local_det_loss")} | local*lambda={item("local_weighted_loss")} '
        f'| align*lambda={item("align_weighted_loss")}',
        f'[TRAIN GLOBAL]  box={item("global_box_loss")} cls={item("global_cls_loss")} '
        f'dfl={item("global_dfl_loss")}',
        f'[TRAIN LOCAL ]  box={item("local_box_loss")} cls={item("local_cls_loss")} '
        f'dfl={item("local_dfl_loss")}',
        f'[ALIGN LOSSES] total={item("align_total")} anchor={item("align_anchor")} '
        f'coarse={item("align_coarse")} scale={item("align_scale")} '
        f'targetness={item("align_targetness")}',
        f'[GMLS-v2]      quality_loss={item("align_v2_quality")} route_loss={item("align_v2_route")} '
        f'GMLS_GT_quality={item("gmls_gt_quality")} DMLS_GT_quality={item("dmls_gt_quality")}',
        f'[TRAIN ROI]    positive_ratio={item("roi_positive")} '
        f'GT_assisted_ratio={item("roi_gt_assisted")} TIR_valid={item("roi_tir_valid_fraction")} '
        f'ROI_per_microbatch={item("train_rois")} '
        f'valid_pairs/microbatch={item("align_valid_pairs")}',
    ]
    if val is None:
        lines.append('[VAL] skipped by val_interval')
    else:
        lines.append('-' * 94)
        for key, label in [('global','GLOBAL'), ('local','LOCAL '), ('fused','FUSED ')]:
            row = val[key]
            lines.append(
                f'[VAL {label}]  P={f(row["precision"])} R={f(row["recall"])} '
                f'AP50={f(row["map50"])} AP75={f(row["map75"])} '
                f'AP50:95={f(row["map5095"])} '
                f'preds/img={f(row.get("preds_per_image"), 2)}'
            )
        lines.append(
            f'[VAL ALIGN]    AlignIoU={f(val.get("align_iou"))} '
            f'center_error={f(val.get("align_center_error_px"), 2)}px '
            f'valid_pairs={val.get("align_eval_pairs", 0)}'
        )
        lines.append(
            f'[VAL ROI]      images={val["images"]} ROIs={val["roi_count"]} '
            f'ROI/image={f(val.get("rois_per_image"), 2)} '
            f'GT_coverage={f(val.get("roi_gt_coverage"))} '
            f'TIR_valid={f(val.get("roi_tir_valid_fraction"))} '
            f'replaced={val["replaced"]} added={val["added"]} '
            f'val_time={f(val.get("elapsed_seconds"), 1)}s'
        )
    lines.append(f'[BEST] fused_AP50:95={f(best)} at epoch {best_epoch} '
                 + ('[NEW BEST]' if improved else ''))
    lines.append('=' * 94)
    return '\n'.join(lines)


def flatten_history_row(epoch, train, val, lr, best, best_epoch, duration_s):
    row = {'epoch': epoch, 'lr': lr, 'train_seconds': duration_s,
           'best_fused_map5095': best, 'best_epoch': best_epoch}
    for key in TRAIN_KEYS:
        row[f'train_{key}'] = train.get(key)
    if val is not None:
        for branch in ('global', 'local', 'fused'):
            for key in ('precision', 'recall', 'map50', 'map75', 'map5095', 'preds_per_image'):
                row[f'val_{branch}_{key}'] = val[branch].get(key)
        for key in ('images','roi_count','rois_per_image','roi_gt_coverage',
                    'roi_tir_valid_fraction','align_iou','align_center_error_px',
                    'align_eval_pairs','replaced','added','elapsed_seconds'):
            row[f'val_{key}'] = val.get(key)
    return row


def write_history_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(flatten_history_row(0, {k:None for k in TRAIN_KEYS},
        {branch:{} for branch in ('global', 'local', 'fused')}, 0, 0, 0, 0).keys())
    # Retain any previous custom columns without dropping historical values.
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
