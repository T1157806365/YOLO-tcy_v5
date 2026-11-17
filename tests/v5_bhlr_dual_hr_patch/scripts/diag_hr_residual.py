#!/usr/bin/env python3
"""Inspect GT-region vs background residual magnitude without changing training.

No model weights needed. Measures reconstruction effects and target-vs-background
signal magnitude, NOT learned feature utility or detection improvement.
"""
import argparse
import csv
from pathlib import Path
import torch
from torch.nn import functional as F
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/experiments/hvad_v1/uavcb_yolo26m_p2_gtga_bhlr_tri_gate_hvad_100e.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="runs/bhlr_dual_hr/residual_diagnostics.csv")
    args = ap.parse_args()
    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset
    c = yaml.safe_load(Path(args.cfg).read_text())
    d = c["data"]
    ds = build_dual_resolution_dataset(
        rgb_yaml=d["rgb"], tir_yaml=d["tir"], split=args.split,
        rgb_high_imgsz=d["rgb_imgsz"], rgb_semantic_imgsz=d["rgb_semantic_imgsz"],
        tir_imgsz=d["tir_imgsz"], pair_mode=d.get("pair_mode", "relative"),
        augment=False,
    )
    rows = []
    for i in range(min(len(ds), args.limit)):
        x = ds[i]
        high = x["rgb_img"].unsqueeze(0).float() / 255
        sem = x["rgb_semantic_img"].unsqueeze(0).float() / 255
        # Original independent-LetterBox reference.
        old_rec = F.interpolate(sem, high.shape[-2:], mode="bilinear", align_corners=False)
        # Clean reconstruction: same 1920->640 geometry and interpolation scheme.
        small = F.interpolate(high, size=sem.shape[-2:], mode="area")
        new_rec = F.interpolate(small, high.shape[-2:], mode="bilinear", align_corners=False)
        old_abs = (high - old_rec).abs().mean(1).squeeze(0)
        new_abs = (high - new_rec).abs().mean(1).squeeze(0)
        h, w = old_abs.shape
        mask = torch.zeros((h, w), dtype=torch.bool)
        boxes = x["rgb_bboxes"]
        for box in boxes:
            xc, yc, bw, bh = [float(v) for v in box]
            l = max(0, int((xc - bw / 2) * w)); r = min(w, int((xc + bw / 2) * w + 0.999))
            t = max(0, int((yc - bh / 2) * h)); b = min(h, int((yc + bh / 2) * h + 0.999))
            if r > l and b > t:
                mask[t:b, l:r] = True
        if not mask.any() or mask.all():
            continue
        rows.append({
            "idx": i, "gt_pixels": int(mask.sum()),
            "old_gt_abs": float(old_abs[mask].mean()),
            "old_bg_abs": float(old_abs[~mask].mean()),
            "clean_gt_abs": float(new_abs[mask].mean()),
            "clean_bg_abs": float(new_abs[~mask].mean()),
            "reconstruction_mean_abs_delta": float((new_rec - old_rec).abs().mean()),
        })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
        print("[OK]", len(rows), "images, saved", out)
        for name in ("old_gt_abs", "old_bg_abs", "clean_gt_abs", "clean_bg_abs", "reconstruction_mean_abs_delta"):
            print(f"{name}: {sum(row[name] for row in rows)/len(rows):.8f}")
    else:
        print("[WARN] no usable GT boxes")


if __name__ == "__main__":
    main()
