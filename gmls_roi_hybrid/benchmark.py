"""Standalone test benchmark for GMLS-v2 OneHead Hybrid.

Compatible with the original v5 gmls_roi_onehead_v2.runner.evaluate().
Does NOT modify training, network weights, original dataset or runner.

FLOPs = 2 * MACs for nn.Conv{,Transpose}2d and nn.Linear observed during
one REAL whole-pipeline (global + two local ROIs + postprocessing) inference.
It is an ESTIMATE, not a full-op exact FLOPs count (grid_sample / softmax /
custom correlation / NMS / other elementwise ops are excluded).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def model_parameters(model: nn.Module) -> dict[str, float]:
    # named_parameters() removes duplicate tensors owned by a shared module.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": total / 1e6, "trainable_m": trainable / 1e6}


class InferenceMeter:
    def __init__(self, model: nn.Module, device: torch.device, warmup: int = 3,
                 count_flops: bool = True):
        self.model = model
        self.device = device
        self.warmup = max(int(warmup), 0)
        self.count_flops = bool(count_flops)
        self.batch_count = 0
        self.image_count = 0
        self.inference_seconds = 0.0
        self.first_batch_flops = 0.0
        self.first_batch_images = 0
        self._counting = False
        self._acc_flops = 0.0
        self._handles = []
        self.original_predict = None

    def _flop_hook(self, mod, inputs, result):
        if not self._counting:
            return
        out = result if torch.is_tensor(result) else (result[0] if isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]) else None)
        if out is None:
            return
        if isinstance(mod, nn.Conv2d):
            # conv.weight: [C_out, C_in/groups, kH, kW]; MACs = output elements * kernel volume
            kernel = mod.weight.shape[1] * mod.weight.shape[2] * mod.weight.shape[3]
            self._acc_flops += 2.0 * out.numel() * kernel
            if mod.bias is not None:
                self._acc_flops += out.numel()
        elif isinstance(mod, nn.ConvTranspose2d):
            # transpose conv MACs = input elements * output-per-group * kernel volume.
            x = inputs[0]
            kernel = mod.weight.shape[1] * mod.weight.shape[2] * mod.weight.shape[3]
            self._acc_flops += 2.0 * x.numel() * kernel
            if mod.bias is not None:
                self._acc_flops += out.numel()
        elif isinstance(mod, nn.Linear):
            self._acc_flops += 2.0 * out.numel() * mod.in_features
            if mod.bias is not None:
                self._acc_flops += out.numel()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        if self.original_predict is not None:
            raise RuntimeError("InferenceMeter is already active")
        self.original_predict = self.model.predict
        if self.count_flops:
            for mod in self.model.modules():
                if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    self._handles.append(mod.register_forward_hook(self._flop_hook))

        def measured_predict(batch, *args, **kwargs):
            b = int(batch["rgb_img"].shape[0])
            if self.batch_count == 0:
                # Warmup never uses GT or changes model weights; net.eval() enforced in caller.
                for _ in range(self.warmup):
                    self.original_predict(batch, *args, **kwargs)
                self._synchronize()
            self._counting = bool(self.count_flops and self.batch_count == 0)
            self._acc_flops = 0.0
            self._synchronize()
            t0 = time.perf_counter()
            try:
                out = self.original_predict(batch, *args, **kwargs)
                self._synchronize()
            finally:
                self._counting = False
            self.inference_seconds += time.perf_counter() - t0
            if self.batch_count == 0 and self.count_flops:
                self.first_batch_flops = self._acc_flops
                self.first_batch_images = b
            self.batch_count += 1
            self.image_count += b
            return out

        self.model.predict = measured_predict
        return self

    def stop(self):
        if self.original_predict is not None:
            # Restore original class method (do not leave a wrapper on model).
            try:
                del self.model.predict
            except AttributeError:
                self.model.predict = self.original_predict
            self.original_predict = None
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def fps(self):
        return self.image_count / self.inference_seconds if self.inference_seconds > 0 else float("nan")

    @property
    def latency_ms(self):
        return 1000 * self.inference_seconds / self.image_count if self.image_count else float("nan")

    @property
    def gflops_per_image(self):
        return self.first_batch_flops / self.first_batch_images / 1e9 if self.first_batch_images else float("nan")


def report_paths(run_dir: Path, split: str) -> dict[str, Path]:
    target = Path(run_dir) / "evaluation" / str(split)
    return {"directory": target, "plots": target / "plots",
            "json": target / "metrics.json", "csv": target / "metrics.csv",
            "text": target / "metrics.txt"}


def make_row(params: dict[str, float], meter: InferenceMeter, metrics: dict[str, float]) -> dict[str, float]:
    return {
        "Params (M)": params["total_m"],
        "FLOPs (G)": meter.gflops_per_image,
        "FPS": meter.fps,
        "P (%)": metrics["P"] * 100,
        "R (%)": metrics["R"] * 100,
        "AP50 (%)": metrics["AP50"] * 100,
        "AP75 (%)": metrics["AP75"] * 100,
        "AP50:95 (%)": metrics["AP50:95"] * 100,
    }


def _sanitize(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return str(obj)


def save_results(paths, summary, row):
    folder = paths["directory"]
    folder.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(_sanitize(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Excel / Origin friendly one-row summary; overwrites on every call.
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader()
        wr.writerow(row)
    lines = ["=" * 65, "GMLS-v2 UNIFIED DUAL HEAD HYBRID TEST BENCHMARK", "=" * 65]
    for k, v in row.items():
        lines.append(f"{k:<20s}: {v:12.4f}" if isinstance(v, (float, int)) else f"{k:<20s}: {v}")
    lines.extend(["-" * 65,
                  f"checkpoint: {summary['checkpoint']}",
                  f"split: {summary['split']}; images: {summary['image_count']}; batch: {summary['batch']}",
                  f"ROI coverage: {summary['evaluation']['roi_gt_coverage']:.6f}",
                  f"AlignIoU: {summary['evaluation']['align_iou']:.6f}",
                  f"FLOPs definition: {summary['flops_definition']}",
                  f"FPS definition: {summary['fps_definition']}",
                  "=" * 65])
    paths["text"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _write_previews(batch, preds, ex, dest: Path, limit: int, counters: dict):
    if counters["images"] >= limit:
        return
    import cv2
    import numpy as np
    B = int(batch["rgb_img"].shape[0])
    K = int(ex["rois_per_image"])
    dest.mkdir(parents=True, exist_ok=True)
    for bi in range(B):
        if counters["images"] >= limit:
            break
        for j in range(K):
            idx = bi * K + j
            rgb = (ex["rgb_roi"][idx].detach().cpu().float().clamp(0, 1).permute(1,2,0).numpy() * 255).astype(np.uint8)
            tir = (ex["tir_roi"][idx].detach().cpu().float().clamp(0, 1).permute(1,2,0).numpy() * 255).astype(np.uint8)
            guide = ex["guide_map"][bi, 0].detach().cpu().float().numpy()
            guide = cv2.resize(guide, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            guide = np.uint8(255 * (guide - float(guide.min())) / (float(guide.max()) - float(guide.min()) + 1e-8))
            guide = cv2.applyColorMap(guide, cv2.COLORMAP_JET)
            tile = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(tir, cv2.COLOR_RGB2BGR), guide], axis=1)
            cv2.imwrite(str(dest / f"roi_{counters['images']:04d}_k{j}.jpg"), tile)
        counters["images"] += 1


def run_benchmark(args):
    # Deferred dependencies: original runner is NOT modified by this patch.
    from engine.runtime import load_config, select_device
    from .runner import build_loader, evaluate, _load_checkpoint
    from .model import build_model as GMLSV2OneHeadHybrid

    cfg = load_config(args.cfg)
    device = select_device(args.device)
    checkpoint = Path(args.weights or (Path(cfg['output']['project']) / cfg['output']['name'] / 'weights' / 'best.pt')).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.parent.name == 'weights':
        run_dir = checkpoint.parent.parent
    else:
        run_dir = (Path(cfg['output']['project']) / cfg['output']['name']).resolve()
    # TOP2_DIVERSE_INTEGRATED_V1: keep old evaluation/test/metrics.* untouched.
    rc = cfg.get('model', {}).get('roi_single_head', {})
    use_diverse = bool(rc.get('top2_diverse_enabled', False)) and int(rc.get('topk', 2)) == 2
    report_split = f'top2_diverse_v1/{args.split}' if use_diverse else args.split
    paths = report_paths(run_dir, report_split)
    if use_diverse:
        print(f'[TOP2 DIVERSE V1] ENABLED; results: {paths["directory"]}', flush=True)
    if args.plots is not None and str(args.plots).lower() not in ('none', 'false', '0'):
        print(f"[PLOTS] Reusing --plots request; plots will be saved INSIDE training folder: {paths['plots']}", flush=True)
        # Old preview files exclusively managed by this benchmark; do not touch checkpoints.
        paths['plots'].mkdir(parents=True, exist_ok=True)
        for p in paths['plots'].glob('roi_*.jpg'):
            if p.is_file():
                p.unlink()
        make_plots = True
    else:
        make_plots = False
    if getattr(args, 'save_dir', None):
        print('[SAVE DIR] Ignoring --save-dir to keep reports inside the checkpoint training folder.', flush=True)
    net = GMLSV2OneHeadHybrid(cfg, verbose=True).to(device)
    ckpt = _load_checkpoint(checkpoint, net)
    net.eval()
    print(f"[LOADED] {checkpoint} | epoch={ckpt.get('epoch')} | architecture={ckpt.get('architecture')}", flush=True)
    params = model_parameters(net)
    loader, _ = build_loader(cfg, args.split, int(args.batch), int(args.workers), rank=0, world=1)
    meter = InferenceMeter(net, device, warmup=int(args.warmup), count_flops=(not args.no_flops)).start()
    # Reuse exact original evaluate() AP calculation. A decorated predict() is
    # solely for benchmarking and optional correct Top-K visualization.
    if make_plots:
        base_predict = net.predict  # currently InferenceMeter's measured_predict
        counters = {'images': 0}
        def with_previews(batch, *a, **kw):
            preds, ex = base_predict(batch, *a, **kw)
            if counters['images'] < int(args.preview_num):
                _write_previews(batch, preds, ex, paths['plots'], int(args.preview_num), counters)
            return preds, ex
        net.predict = with_previews
    try:
        report = evaluate(net, loader, device, cfg, rank=0, world=1,
                          smoke=bool(args.smoke), preview_dir=None)
    finally:
        meter.stop()
    row = make_row(params, meter, report['metrics'])
    info = {
        'checkpoint': str(checkpoint), 'epoch': ckpt.get('epoch'),
        'split': args.split, 'image_count': int(report['images']), 'batch': int(args.batch),
        'topk': net.num_rois, 'warmup': int(args.warmup),
        'parameters': params,
        'flops': {'gflops_per_image': meter.gflops_per_image,
                  'first_batch_images': meter.first_batch_images,
                  'method': 'forward_hooks_conv_linear_2mac'},
        'inference': {'fps': meter.fps, 'ms_per_image': meter.latency_ms,
                      'timed_images': meter.image_count,
                      'model_seconds': meter.inference_seconds},
        'flops_definition': 'Estimated 2*MACs of Conv2d/ConvTranspose2d/Linear during FULL global+ROI+local inference. Excludes grid_sample, correlation, pointwise, softmax, NMS.',
        'fps_definition': 'GPU-synchronized model.predict including global alignment, K ROI crops, local RGB-T detect, coordinate restoration, NMS; excludes data loading and metrics.',
        'evaluation': report,
        'summary': row,
    }
    text_lines = save_results(paths, info, row)
    print('\n'.join(text_lines), flush=True)
    print(f"[SAVED] {paths['json']}\n[SAVED] {paths['csv']}\n[SAVED] {paths['text']}", flush=True)
    if make_plots:
        print(f"[SAVED] {paths['plots']} (RGB ROI | TIR ROI | Guide heatmap)", flush=True)
    if args.smoke:
        print('[SMOKE] Only the first validation batch was processed; metrics/FPS are NOT full-split results.', flush=True)
    return info
