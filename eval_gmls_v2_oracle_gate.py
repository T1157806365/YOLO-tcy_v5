#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-image GMLS-v2 vs DMLS quality/oracle/router audit for YOLO-tcy_v5.

Run from the YOLO-tcy_v5 repository root. Uses the model's own
_gmls_v2_candidate_quality() exactly (translation-only bbox supervision),
and the same data/geometry path as eval_alignment_iou_displacement_GMLS_v2.py.

One RGB bbox and one TIR bbox are required, matching the training objective;
all other samples are explicitly skipped and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import (
    build_dual_resolution_dataset,
    build_dual_resolution_dataloader,
)
from engine.runtime import autocast_context, preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint
from eval_alignment_iou_displacement_GMLS_v2 import get_ref_debug


def parse_args():
    p = argparse.ArgumentParser(description="Oracle and learned gate audit: GMLS v2 vs DMLS")
    p.add_argument("--weights", required=True, help="Path to exact GMLS-v2 checkpoint")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--rgb", default=None, help="RGB data YAML (default from checkpoint)")
    p.add_argument("--tir", default=None, help="TIR data YAML (default from checkpoint)")
    p.add_argument("--rgb-imgsz", type=int, default=None)
    p.add_argument("--rgb-semantic-imgsz", type=int, default=None)
    p.add_argument("--tir-imgsz", type=int, default=None)
    p.add_argument("--pair-mode", default=None, choices=("relative", "stem", "filename"))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--fusion", default="auto", choices=("auto", "concat", "add", "weighted"))
    p.add_argument("--save-dir", default=None)
    p.add_argument("--max-images", type=int, default=0, help="0=entire split; >0 smoke test")
    p.add_argument("--tie-eps", type=float, default=1e-6)
    p.add_argument("--amp", action="store_true", help="Optional CUDA autocast; FP32 default")
    return p.parse_args()


def box_iou_xywh(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU of two normalized xywh boxes (same coordinate system)."""
    a = box_a.float().reshape(4)
    b = box_b.float().reshape(4)
    alo = a[:2] - a[2:] * 0.5
    ahi = a[:2] + a[2:] * 0.5
    blo = b[:2] - b[2:] * 0.5
    bhi = b[:2] + b[2:] * 0.5
    inter = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod()
    aa = a[2:].clamp_min(0).prod()
    bb = b[2:].clamp_min(0).prod()
    return (inter / (aa + bb - inter).clamp_min(1e-7)).clamp(0, 1)


def candidate_iou(rbox, tbox, offset, feat_h, feat_w):
    """GT IoU after coarse translation only, exactly q=p+d geometry."""
    norm = offset.new_tensor([max(feat_w - 1, 1), max(feat_h - 1, 1)])
    aligned_tir = tbox.float().clone()
    aligned_tir[:2] -= offset.reshape(2).float() / norm
    return float(box_iou_xywh(rbox, aligned_tir).item())


def mean(values):
    return float(np.mean(values)) if values else None


def qstats(rows, key):
    vals = [float(row[key]) for row in rows]
    return {
        "mean": mean(vals),
        "median": float(np.median(vals)) if vals else None,
        "min": float(min(vals)) if vals else None,
        "max": float(max(vals)) if vals else None,
    }


def format_number(v, decimals=6):
    return "N/A" if v is None else f"{v:.{decimals}f}"


@torch.inference_mode()
def evaluate(args):
    if args.tie_eps < 0:
        raise ValueError("--tie-eps must be nonnegative")
    # Avoid counterfactual environment overrides: measure the actual learned gate.
    os.environ["GMLS_V2_EVAL_MODE"] = "auto"
    device = select_device(args.device)
    model, _, cfg = load_model_from_checkpoint(
        args.weights, device=device, fusion_override=args.fusion,
        strict=True, verbose=False,
    )
    model.eval()
    if not hasattr(model, "_gmls_v2_candidate_quality"):
        raise RuntimeError(
            "Loaded model has no _gmls_v2_candidate_quality method. "
            "Check you are in the correct YOLO-tcy_v5 version."
        )

    dc = cfg.get("data", {}) or {}
    rgb_yaml = args.rgb or dc.get("rgb")
    tir_yaml = args.tir or dc.get("tir")
    if not rgb_yaml or not tir_yaml:
        raise ValueError("Cannot locate RGB/TIR dataset YAML; pass --rgb and --tir")
    rgb_imgsz = int(args.rgb_imgsz or dc.get("rgb_imgsz", 1920))
    rgb_sem_imgsz = int(args.rgb_semantic_imgsz or dc.get("rgb_semantic_imgsz", 640))
    tir_imgsz = int(args.tir_imgsz or dc.get("tir_imgsz", 640))
    pair_mode = args.pair_mode or dc.get("pair_mode", "relative")
    dataset = build_dual_resolution_dataset(
        rgb_yaml=rgb_yaml,
        tir_yaml=tir_yaml,
        split=args.split,
        rgb_high_imgsz=rgb_imgsz,
        rgb_semantic_imgsz=rgb_sem_imgsz,
        tir_imgsz=tir_imgsz,
        pair_mode=pair_mode,
        augment=False,
        tir_channels=3,
        strict_pair=True,
    )
    loader = build_dual_resolution_dataloader(
        dataset, batch_size=args.batch, workers=args.workers,
        shuffle=False, pin_memory=(device.type == "cuda"),
    )
    checkpoint = Path(args.weights).expanduser().resolve()
    savedir = (
        Path(args.save_dir).expanduser().resolve()
        if args.save_dir else checkpoint.parent.parent / f"gmls_v2_oracle_{args.split}"
    )
    savedir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = {"no_box": 0, "multi_box": 0}
    scanned = 0
    meta = {"split": args.split, "weights": str(checkpoint), "rgb_yaml": str(rgb_yaml),
            "tir_yaml": str(tir_yaml), "rgb_imgsz": rgb_imgsz,
            "rgb_semantic_imgsz": rgb_sem_imgsz, "tir_imgsz": tir_imgsz,
            "pair_mode": pair_mode, "tie_eps": args.tie_eps}

    pbar = tqdm(loader, desc=f"Oracle / gating on {args.split}")
    for raw_batch in pbar:
        batch = preprocess_batch(raw_batch, device)
        if hasattr(model, "build_alignment_valid_masks_from_batch"):
            rgb_mask, tir_mask = model.build_alignment_valid_masks_from_batch(batch)
        else:
            rgb_mask = tir_mask = None
        with autocast_context(device, enabled=bool(args.amp)):
            outputs = model.predict(
                batch["rgb_img"], batch["tir_img"],
                rgb_semantic=batch["rgb_semantic_img"],
                rgb_valid_mask=rgb_mask, tir_valid_mask=tir_mask,
                return_features=True, return_alignment_debug=True,
            )
        info = outputs["alignment_info"]
        dbg = get_ref_debug(info, model)
        needed = ("coarse_offset_gmls", "coarse_offset_dmls",
                  "gmls_v2_choose_dmls", "gmls_v2_quality_scores", "gmls_v2_route_logit",
                  "rgb_targetness_raw")
        absent = [k for k in needed if dbg.get(k) is None]
        if absent:
            raise KeyError(
                f"Missing GMLS-V2 debug fields: {absent}. "
                "Confirm gmls_v2_correction_enabled=true and checkpoint matches code."
            )
        oh, ow = dbg["rgb_targetness_raw"].shape[-2:]
        gmls_all = dbg["coarse_offset_gmls"]
        dmls_all = dbg["coarse_offset_dmls"]
        gate_all = dbg["gmls_v2_choose_dmls"]
        predicted_q_all = dbg["gmls_v2_quality_scores"]
        route_logit_all = dbg["gmls_v2_route_logit"]

        bsz = int(batch["rgb_semantic_img"].shape[0])
        for bi in range(bsz):
            if args.max_images and scanned >= args.max_images:
                break
            scanned += 1
            rgb_boxes = batch["rgb_semantic_bboxes"][batch["rgb_batch_idx"] == bi]
            tir_boxes = batch["tir_bboxes"][batch["tir_batch_idx"] == bi]
            if rgb_boxes.shape[0] == 0 or tir_boxes.shape[0] == 0:
                skipped["no_box"] += 1
                continue
            if rgb_boxes.shape[0] != 1 or tir_boxes.shape[0] != 1:
                skipped["multi_box"] += 1
                continue
            rbox = rgb_boxes[0].detach().float()
            tbox = tir_boxes[0].detach().float()
            g_offset = gmls_all[bi].detach().float().reshape(2)
            d_offset = dmls_all[bi].detach().float().reshape(2)
            offsets = torch.stack((g_offset, d_offset), dim=0)

            q = model._gmls_v2_candidate_quality(
                rbox=rbox, tbox=tbox, offsets=offsets,
                feat_h=oh, feat_w=ow,
                iou_weight=model.v2_quality_iou_weight,
            )
            qg, qd = [float(v.item()) for v in q]
            iou_g = candidate_iou(rbox, tbox, g_offset, oh, ow)
            iou_d = candidate_iou(rbox, tbox, d_offset, oh, ow)

            chose_dmls = bool(gate_all[bi].detach().item())
            gate_choice = "DMLS" if chose_dmls else "GMLS"
            gap = qd - qg
            gt_preference = "DMLS" if gap > args.tie_eps else "GMLS" if gap < -args.tie_eps else "TIE"
            oracle_choice = "DMLS" if qd > qg else "GMLS"
            predicted_q = predicted_q_all[bi].detach().float().reshape(2)
            selected_quality = qd if chose_dmls else qg
            selected_iou = iou_d if chose_dmls else iou_g
            oracle_quality = max(qg, qd)
            if not all(math.isfinite(v) for v in (qg, qd, iou_g, iou_d, selected_quality)):
                raise ValueError(f"Non-finite quality/IoU at image {scanned}")
            pair_key = batch.get("pair_key", [""] * bsz)[bi]
            rgb_path = batch.get("rgb_path", [""] * bsz)[bi]
            tir_path = batch.get("tir_path", [""] * bsz)[bi]
            rows.append({
                "index": scanned,
                "pair_key": str(pair_key),
                "rgb_path": str(rgb_path),
                "tir_path": str(tir_path),
                "q_gmls": qg, "q_dmls": qd,
                "q_delta_d_minus_g": gap,
                "q_oracle": oracle_quality,
                "q_gate": selected_quality,
                "oracle_choice": oracle_choice,
                "gt_preference_eps": gt_preference,
                "gate_choice": gate_choice,
                "gate_correct_non_tie": "" if gt_preference == "TIE" else int(gate_choice == gt_preference),
                "gate_regret": oracle_quality - selected_quality,
                "predicted_quality_gmls": float(predicted_q[0].item()),
                "predicted_quality_dmls": float(predicted_q[1].item()),
                "route_logit": float(route_logit_all[bi].detach().item()),
                "iou_coarse_gmls": iou_g,
                "iou_coarse_dmls": iou_d,
                "iou_coarse_gate": selected_iou,
                "iou_coarse_oracle_quality_choice": iou_d if oracle_choice == "DMLS" else iou_g,
                "iou_coarse_oracle_iou": max(iou_g, iou_d),
                "coarse_offset_gmls_x": float(g_offset[0].item()),
                "coarse_offset_gmls_y": float(g_offset[1].item()),
                "coarse_offset_dmls_x": float(d_offset[0].item()),
                "coarse_offset_dmls_y": float(d_offset[1].item()),
            })
        pbar.set_postfix(valid=len(rows), scanned=scanned)
        if args.max_images and scanned >= args.max_images:
            break

    if not rows:
        raise RuntimeError("No single-box RGB-T pairs found; check --split and dataset YAMLs")
    csvpath = savedir / "per_image_oracle_gate.csv"
    with csvpath.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    ng = sum(r["gt_preference_eps"] == "GMLS" for r in rows)
    nd = sum(r["gt_preference_eps"] == "DMLS" for r in rows)
    ntie = count - ng - nd
    nc = ng + nd
    correct = sum(r["gate_correct_non_tie"] == 1 for r in rows)
    choose_d = sum(r["gate_choice"] == "DMLS" for r in rows)
    best_fixed_name = "DMLS" if qstats(rows, "q_dmls")["mean"] >= qstats(rows, "q_gmls")["mean"] else "GMLS"
    best_fixed_q = qstats(rows, "q_dmls")["mean"] if best_fixed_name == "DMLS" else qstats(rows, "q_gmls")["mean"]
    q_oracle = qstats(rows, "q_oracle")["mean"]
    q_gate = qstats(rows, "q_gate")["mean"]
    summary = {
        "inputs": meta,
        "images_scanned": scanned,
        "valid_single_box_pairs": count,
        "skipped": skipped,
        "quality_means": {
            "gmls": qstats(rows, "q_gmls")["mean"],
            "dmls": qstats(rows, "q_dmls")["mean"],
            "oracle": q_oracle,
            "learned_gate": q_gate,
            "best_fixed": best_fixed_q,
            "best_fixed_name": best_fixed_name,
        },
        "oracle_gain_over_best_fixed": q_oracle - best_fixed_q,
        "gate_gain_over_best_fixed": q_gate - best_fixed_q,
        "gate_regret_to_oracle": q_oracle - q_gate,
        "gt_preference": {"gmls_wins": ng, "dmls_wins": nd, "ties": ntie,
                          "gmls_win_ratio": ng / count, "dmls_win_ratio": nd / count,
                          "tie_ratio": ntie / count},
        "gate": {"dmls_selected_count": choose_d, "gmls_selected_count": count - choose_d,
                 "dmls_selected_ratio": choose_d / count,
                 "accuracy_on_non_ties": correct / nc if nc else None,
                 "correct_on_non_ties": correct, "non_tie_count": nc},
        "coarse_iou_means": {k: qstats(rows, k)["mean"] for k in (
            "iou_coarse_gmls", "iou_coarse_dmls", "iou_coarse_gate",
            "iou_coarse_oracle_quality_choice", "iou_coarse_oracle_iou")},
        "coarse_iou_zero_counts": {k: sum(r[k] <= 1e-12 for r in rows)
                                   for k in ("iou_coarse_gmls", "iou_coarse_dmls", "iou_coarse_gate")},
        "note": ("GT Quality is bbox-based coarse-translation target: IoU/center blend. "
                 "Oracle is sample-wise maximum of two candidates, NOT an inference-achievable "
                 "method and NOT final Fine/Scale AlignIoU or detection mAP. "
                 "Only exactly one RGB and one TIR bbox per image are scored."),
    }
    jsonpath = savedir / "oracle_gate_summary.json"
    jsonpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"GMLS V2 ORACLE / SAMPLE-WISE GATE: {args.split}")
    print("=" * 72)
    print(f"scanned={scanned} valid_1to1={count} skipped={skipped}")
    print("GT QUALITY (coarse translation only)")
    for label, key in (("GMLS", "gmls"), ("DMLS", "dmls"), ("Learned gate", "learned_gate"),
                       ("Oracle", "oracle")):
        print(f"  {label:15s} {summary['quality_means'][key]:.6f}")
    print(f"  Best fixed       {best_fixed_name} ({best_fixed_q:.6f})")
    print(f"Oracle - best fixed   : {q_oracle - best_fixed_q:+.6f}")
    print(f"Gate   - best fixed   : {q_gate - best_fixed_q:+.6f}")
    print(f"Oracle - gate regret : {q_oracle - q_gate:+.6f}")
    print(f"GT wins: GMLS {ng}/{count}, DMLS {nd}/{count}, TIE {ntie}/{count}")
    print(f"Gate picks DMLS: {choose_d}/{count} ({choose_d/count:.2%})")
    print(f"Gate accuracy on non-ties: {correct}/{nc} ({correct/nc:.2%})" if nc else
          "Gate accuracy on non-ties: N/A")
    print("COARSE translation IoU (not final AlignIoU)")
    for k, v in summary["coarse_iou_means"].items():
        print(f"  {k}: {v:.6f}")
    print(f"Saved: {csvpath}")
    print(f"Saved: {jsonpath}")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
