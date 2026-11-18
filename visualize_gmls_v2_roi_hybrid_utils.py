# Source-preserving helper functions adapted from the user-provided old GMLS-v2 visualization.
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

RGB_GT = (0, 220, 0)
TIR_GT = (0, 165, 255)
ALIGNED_GT = (255, 0, 255)
PRED = (0, 0, 255)
ANCHOR = (255, 255, 0)
SOURCE_CENTER = (255, 120, 255)
TEXT = (245, 245, 245)

def tensor_bgr(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if float(x.max()) > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def fmap(x: torch.Tensor, mode: str = "mean_abs") -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        return x.numpy().astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"Unsupported map shape: {tuple(x.shape)}")
    if mode == "mean":
        y = x.mean(0)
    elif mode == "max":
        y = x.amax(0)
    else:
        y = x.abs().mean(0)
    return y.numpy().astype(np.float32)


def robust_range(x: np.ndarray, low=1.0, high=99.0) -> Tuple[float, float]:
    x = np.nan_to_num(np.asarray(x, np.float32))
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def heat(x: np.ndarray, hw: Tuple[int, int], fixed=None, symmetric=False) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, np.float32))
    if fixed is not None:
        lo, hi = map(float, fixed)
    elif symmetric:
        q = max(float(np.percentile(np.abs(x), 99)), 1e-6)
        lo, hi = -q, q
    else:
        lo, hi = robust_range(x)
    z = np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)
    g = (z * 255).round().astype(np.uint8)
    h, w = map(int, hw)
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def overlay(base: np.ndarray, x: np.ndarray, alpha=0.48, fixed=None, symmetric=False) -> np.ndarray:
    hm = heat(x, base.shape[:2], fixed=fixed, symmetric=symmetric)
    return cv2.addWeighted(base, 1 - alpha, hm, alpha, 0)


def _put_text_outlined(
    image,
    text,
    org,
    font_scale,
    color,
    thickness=2,
    outline_color=(0, 0, 0),
    outline_thickness=4,
):
    """High-contrast OpenCV text: dark outline + foreground."""
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, outline_color, outline_thickness, cv2.LINE_AA
    )
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, cv2.LINE_AA
    )


def add_title(image, title, subtitle="", annotation=""):
    image = image.copy()
    h, w = image.shape[:2]
    lines = 1 + int(bool(subtitle)) + int(bool(annotation))

    # Larger header and larger fonts for paper/zoom inspection.
    bh = min(h - 1, 48 + (lines - 1) * 32)
    cv2.rectangle(image, (0, 0), (w - 1, bh), (14, 14, 14), -1)

    _put_text_outlined(
        image, title[:135], (14, 32),
        font_scale=0.72, color=(255, 255, 255),
        thickness=2, outline_thickness=4,
    )

    y = 64
    for text in (subtitle, annotation):
        if text:
            _put_text_outlined(
                image, text[:165], (14, y),
                font_scale=0.52, color=(225, 225, 225),
                thickness=1, outline_thickness=3,
            )
            y += 30
    return image


def save(out_dir: Path, number: int, slug: str, image: np.ndarray, title: str, subtitle="", annotation="", canvas=960):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(image, (canvas, canvas), interpolation=cv2.INTER_LINEAR)
    img = add_title(img, title, subtitle, annotation)
    path = out_dir / f"{number:02d}_{slug}.jpg"
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"Failed to write {path}")
    return path


def xywhn_to_xyxy(boxes, w, h):
    if torch.is_tensor(boxes):
        boxes = boxes.detach().float().cpu().numpy()
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    out = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def transform_source_boxes(boxes_xywhn, scale, translation):
    """q=S*p+t -> p=(q-t)/S."""
    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy()
    else:
        b = np.asarray(boxes_xywhn, np.float32)
    b = b.reshape(-1, 4).copy()
    if len(b) == 0:
        return b
    sx, sy = max(float(scale[0]), 1e-6), max(float(scale[1]), 1e-6)
    tx, ty = float(translation[0]), float(translation[1])
    b[:, 0] = (b[:, 0] - tx) / sx
    b[:, 1] = (b[:, 1] - ty) / sy
    b[:, 2] = b[:, 2] / sx
    b[:, 3] = b[:, 3] / sy
    return b


def draw_boxes(image, boxes, color, label, scores=None):
    """Draw slimmer boxes and small outlined labels without any text background box."""
    out = image.copy()
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    h, w = out.shape[:2]

    box_thickness = max(1, int(round(max(h, w) / 420.0)))
    font_scale = max(0.42, min(0.58, max(h, w) / 1500.0))
    text_thickness = 1
    outline_thickness = 2

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))

        # Slim high-contrast box with only a light outline.
        cv2.rectangle(
            out, (x1, y1), (x2, y2), (12, 12, 12),
            box_thickness + 1, cv2.LINE_AA
        )
        cv2.rectangle(
            out, (x1, y1), (x2, y2), color,
            box_thickness, cv2.LINE_AA
        )

        txt = label if len(boxes) == 1 else f"{label}#{i}"
        if scores is not None and i < len(scores):
            txt += f" {float(scores[i]):.2f}"

        (tw, th), baseline = cv2.getTextSize(
            txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )

        # Prefer label above bbox; if not enough room, place just inside the top-left.
        text_x = max(2, min(w - tw - 2, x1))
        if y1 - baseline - 6 >= th:
            text_y = y1 - 5
        else:
            text_y = min(h - baseline - 2, y1 + th + 4)

        _put_text_outlined(
            out, txt, (text_x, text_y),
            font_scale=font_scale, color=color,
            thickness=text_thickness, outline_thickness=outline_thickness,
        )

    return out


def draw_point(image, center_norm, color, label, size=18):
    """Draw a minimal hollow center marker. No cross is drawn inside the UAV box."""
    out = image.copy()
    h, w = out.shape[:2]
    x = int(round(float(center_norm[0]) * (w - 1)))
    y = int(round(float(center_norm[1]) * (h - 1)))

    radius = max(4, int(round(size * 0.28)))
    cv2.circle(out, (x, y), radius + 1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.circle(out, (x, y), radius, color, 2, cv2.LINE_AA)

    # Label is offset from the center so it does not cover the target itself.
    tx = min(max(4, x + radius + 8), max(4, w - 190))
    ty = max(24, y - radius - 8)
    _put_text_outlined(
        out, label, (tx, ty),
        font_scale=0.46, color=color,
        thickness=1, outline_thickness=2,
    )
    return out


def draw_sigma_ellipse(image, center_norm, sigma_norm, color, label):
    """Draw Gaussian extent without a center cross/marker inside the target."""
    out = image.copy()
    h, w = out.shape[:2]
    cx = int(round(float(center_norm[0]) * (w - 1)))
    cy = int(round(float(center_norm[1]) * (h - 1)))
    axes = (
        max(2, int(round(2 * float(sigma_norm[0]) * w))),
        max(2, int(round(2 * float(sigma_norm[1]) * h))),
    )
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, color, 1, cv2.LINE_AA)

    # Put the label near the ellipse edge, not at its center.
    tx = min(max(4, cx + axes[0] + 8), max(4, w - 210))
    ty = max(24, cy - axes[1] - 8)
    _put_text_outlined(
        out, label, (tx, ty),
        font_scale=0.44, color=color, thickness=1, outline_thickness=2,
    )
    return out


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


def spatial_entropy(x: np.ndarray):
    a = np.maximum(np.asarray(x, np.float64), 0)
    p = a.reshape(-1)
    s = p.sum()
    if s <= 1e-12:
        return 1.0
    p = p / s
    h = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(h / max(math.log(len(p)), 1e-12))


def inside_outside_stats(x: np.ndarray, boxes_xywhn):
    a = np.asarray(x, np.float32)
    h, w = a.shape
    mask = np.zeros((h, w), np.uint8)
    if torch.is_tensor(boxes_xywhn):
        b = boxes_xywhn.detach().float().cpu().numpy().reshape(-1, 4)
    else:
        b = np.asarray(boxes_xywhn, np.float32).reshape(-1, 4)
    for cx, cy, bw, bh in b:
        x1 = max(0, min(w - 1, int(math.floor((cx - bw / 2) * w))))
        y1 = max(0, min(h - 1, int(math.floor((cy - bh / 2) * h))))
        x2 = max(x1 + 1, min(w, int(math.ceil((cx + bw / 2) * w))))
        y2 = max(y1 + 1, min(h, int(math.ceil((cy + bh / 2) * h))))
        mask[y1:y2, x1:x2] = 1
    inside = float(a[mask > 0].mean()) if np.any(mask > 0) else float('nan')
    outside = float(a[mask == 0].mean()) if np.any(mask == 0) else float('nan')
    ratio = inside / max(outside, 1e-8) if np.isfinite(inside) and np.isfinite(outside) else float('nan')
    return inside, outside, ratio


def stats_text(x: np.ndarray, boxes=None):
    x = np.asarray(x, np.float32)
    parts = [
        f"min/mean/max={x.min():.3f}/{x.mean():.3f}/{x.max():.3f}",
        f"std={x.std():.3f}",
        f"H={spatial_entropy(x):.3f}",
        f"q99={np.quantile(x, .99):.3f}",
    ]
    if boxes is not None:
        i, o, r = inside_outside_stats(x, boxes)
        parts += [f"inside={i:.3f}", f"outside={o:.3f}", f"in/out={r:.2f}"]
    return '; '.join(parts)


def probability_stats(prob: torch.Tensor):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    k = len(p)
    uniform = 1.0 / k
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(k)
    order = np.argsort(-p)
    return {
        'pmax': float(p[order[0]]),
        'p2': float(p[order[1]]) if k > 1 else float(p[order[0]]),
        'max_over_uniform': float(p[order[0]] / uniform),
        'top1_over_top2': float(p[order[0]] / max(p[order[1]], 1e-12)) if k > 1 else 1.0,
        'entropy_norm': float(ent),
        'effective_candidates': float(math.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
    }


def render_probability(prob, offset, canvas=960):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    side = int(round(math.sqrt(len(p))))
    radius = (side - 1) // 2
    grid = p.reshape(side, side)
    img = cv2.applyColorMap((grid / max(float(grid.max()), 1e-12) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img = cv2.resize(img, (canvas, canvas), interpolation=cv2.INTER_NEAREST)
    cell = canvas / side
    best = int(np.argmax(p))
    br, bc = divmod(best, side)
    cv2.rectangle(img, (int(bc * cell), int(br * cell)), (int((bc + 1) * cell) - 1, int((br + 1) * cell) - 1), (0, 0, 0), 5)
    cv2.rectangle(img, (int(bc * cell), int(br * cell)), (int((bc + 1) * cell) - 1, int((br + 1) * cell) - 1), (255, 255, 255), 3)
    dx = float(offset[0, 0, 0, 0].detach().cpu())
    dy = float(offset[0, 1, 0, 0].detach().cpu())
    px = int(round((dx + radius + 0.5) * cell))
    py = int(round((dy + radius + 0.5) * cell))
    cv2.drawMarker(img, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 22, 2)
    return img


def render_top_candidates(prob, topn=12, canvas=960):
    p = prob[0].detach().float().cpu().numpy().reshape(-1)
    side = int(round(math.sqrt(len(p))))
    radius = (side - 1) // 2
    order = np.argsort(-p)[:topn]
    img = np.full((canvas, canvas, 3), 245, np.uint8)
    y = 95
    cv2.putText(img, "Top coarse translation candidates", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    for rank, idx in enumerate(order, 1):
        row, col = divmod(int(idx), side)
        dx, dy = col - radius, row - radius
        txt = f"#{rank:02d}   dx={dx:+3d}   dy={dy:+3d}   p={p[idx]:.7f}"
        cv2.putText(img, txt, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 1, cv2.LINE_AA)
        y += 60
    return img


def render_scale_compare(pred_scale, gt_scale, raw_scale, canvas=960):
    img = np.full((canvas, canvas, 3), 248, np.uint8)
    cv2.putText(img, "GTGA centered scale: prediction vs bbox GT", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    labels = ['sx', 'sy']
    maxv = max(2.0, *(list(pred_scale) + list(raw_scale) + (list(gt_scale) if gt_scale is not None else [])))
    base_y = 320
    for j, lab in enumerate(labels):
        x0 = 100 + j * 410
        cv2.putText(img, lab, (x0 + 110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
        vals = [('raw', raw_scale[j], (120, 120, 120)), ('pred', pred_scale[j], (60, 160, 60))]
        if gt_scale is not None:
            vals.append(('GT', gt_scale[j], (40, 80, 220)))
        for k, (name, val, color) in enumerate(vals):
            y = 190 + k * 180
            width = int(min(300, max(0, float(val) / maxv * 300)))
            cv2.rectangle(img, (x0, y), (x0 + width, y + 55), color, -1)
            cv2.putText(img, f"{name}: {float(val):.4f}", (x0, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (35, 35, 35), 1, cv2.LINE_AA)
    if gt_scale is not None:
        ex = (float(pred_scale[0]) - float(gt_scale[0])) / max(float(gt_scale[0]), 1e-8) * 100
        ey = (float(pred_scale[1]) - float(gt_scale[1])) / max(float(gt_scale[1]), 1e-8) * 100
        cv2.putText(img, f"relative error: sx={ex:+.1f}%   sy={ey:+.1f}%", (110, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def make_overview(paths: List[Path], out_path: Path, cols=4, thumb=360):
    imgs = []
    for p in paths:
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    rows = math.ceil(len(imgs) / cols)
    canvas = np.full((rows * thumb, cols * thumb, 3), 240, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = im
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])


