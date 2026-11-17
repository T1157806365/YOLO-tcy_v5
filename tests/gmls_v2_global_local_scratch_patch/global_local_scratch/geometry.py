"""V5 global→native-HR local RGB/TIR geometry.

All image coordinates in this module are *letterboxed image pixel* coordinates,
not original camera pixels. GTGA maps normalized RGB-semantic coordinates p to
TIR-source coordinates q = scale*p + translation (see v5
models/modules/alignment/rlsfa_gtga_centered.py).

The RGB original and RGB semantic branches have separate letterbox transforms;
never assume an exact x3 mapping between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropGeometry:
    x0: int
    y0: int
    size: int
    center_high_xy: tuple[float, float]
    tir_valid_fraction: float
    center_sem_xy: tuple[float, float]


def _ratio_pad(meta: Mapping) -> tuple[float, float, float]:
    r = meta['ratio']
    if isinstance(r, (list, tuple)):
        if len(r) != 2 or abs(float(r[0]) - float(r[1])) > 1e-5:
            raise ValueError('This dataset requires isotropic letterbox scaling')
        r = r[0]
    px, py = meta['pad']
    if float(r) <= 0:
        raise ValueError(f'Invalid letterbox ratio: {r}')
    return float(r), float(px), float(py)


def semantic_to_high(xy: torch.Tensor, sem_meta: Mapping, high_meta: Mapping) -> torch.Tensor:
    """RGB semantic letterbox pixel xy -> RGB high letterbox pixel xy."""
    rs, sx, sy = _ratio_pad(sem_meta)
    rh, hx, hy = _ratio_pad(high_meta)
    a = xy.new_tensor([sx, sy])
    b = xy.new_tensor([hx, hy])
    return (xy - a) / rs * rh + b


def high_to_semantic(xy: torch.Tensor, high_meta: Mapping, sem_meta: Mapping) -> torch.Tensor:
    """RGB high letterbox pixel xy -> RGB semantic letterbox pixel xy."""
    rh, hx, hy = _ratio_pad(high_meta)
    rs, sx, sy = _ratio_pad(sem_meta)
    a = xy.new_tensor([hx, hy])
    b = xy.new_tensor([sx, sy])
    return (xy - a) / rh * rs + b


def norm_xywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('boxes must be [N,4]')
    s = boxes.new_tensor([width, height, width, height])
    b = boxes * s
    xy1 = b[:, :2] - b[:, 2:] * 0.5
    xy2 = b[:, :2] + b[:, 2:] * 0.5
    return torch.cat((xy1, xy2), dim=-1)


def xyxy_to_norm_xywh(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    b = boxes.clone()
    xywh = torch.cat(((b[:, :2] + b[:, 2:]) * 0.5, (b[:, 2:] - b[:, :2]).clamp_min(0)), dim=-1)
    return xywh / boxes.new_tensor([width, height, width, height])


def local_gt_from_high(
    boxes_high_norm: torch.Tensor, x0: int, y0: int, size: int,
    high_width: int, high_height: int, min_visible: float = 0.60,
) -> torch.Tensor:
    """GT label normalized xywh within a crop; retains partially visible boxes
    only if the fraction of original box area visible >= min_visible.
    Empty crop -> [0,4] (negative ROI). No GT is used for ROI centers.
    """
    if boxes_high_norm.numel() == 0:
        return boxes_high_norm.new_zeros((0, 4))
    b = norm_xywh_to_xyxy(boxes_high_norm, high_width, high_height)
    p1 = b[:, :2]
    p2 = b[:, 2:]
    a1 = (p1 - b.new_tensor([x0, y0])).clamp(min=0.0, max=float(size))
    a2 = (p2 - b.new_tensor([x0, y0])).clamp(min=0.0, max=float(size))
    visible_size = (a2 - a1).clamp_min(0)
    visible = visible_size.prod(-1)
    area = (p2 - p1).clamp_min(0).prod(-1).clamp_min(1e-6)
    ok = (visible / area >= min_visible) & (visible_size[:, 0] >= 1) & (visible_size[:, 1] >= 1)
    if not bool(ok.any()):
        return b.new_zeros((0, 4))
    crop_box = torch.cat((a1[ok], a2[ok]), -1)
    return xyxy_to_norm_xywh(crop_box, size, size).clamp(0, 1)


def high_box_to_semantic_xyxy(boxes: torch.Tensor, high_meta: Mapping, sem_meta: Mapping,
                               sem_width: int, sem_height: int) -> torch.Tensor:
    """Crop-local xyxy must first be shifted by (x0,y0) before this function."""
    res = torch.cat((high_to_semantic(boxes[..., :2], high_meta, sem_meta),
                     high_to_semantic(boxes[..., 2:], high_meta, sem_meta)), -1)
    res[..., 0::2] = res[..., 0::2].clamp(0, float(sem_width))
    res[..., 1::2] = res[..., 1::2].clamp(0, float(sem_height))
    return res


def sample_native_hr_and_tir(
    rgb_high: torch.Tensor,
    tir_full: torch.Tensor,
    center_sem_xy: Sequence[float] | torch.Tensor,
    sem_hw: tuple[int, int],
    high_meta: Mapping,
    sem_meta: Mapping,
    sampling_scale: torch.Tensor,
    sampling_translation: torch.Tensor,
    roi_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor, CropGeometry]:
    """One 1920/640 pair -> native 640 RGB crop + geometrically matched 640 TIR crop.

    Inputs RGB/TIR [C,H,W], values preferably [0,1], same device. `scale` and
    `translation` are GTGA per-image tensors [2] (or any shape with 2 values).
    GPU float32 grid prevents FP16 coordinate errors; output follows tir dtype.
    """
    if rgb_high.ndim != 3 or tir_full.ndim != 3:
        raise ValueError('rgb_high and tir_full must be [C,H,W]')
    if rgb_high.device != tir_full.device:
        raise ValueError('RGB and TIR tensors must be on same device')
    _, H, W = rgb_high.shape
    hs, ws = map(int, sem_hw)
    size = int(roi_size)
    if size < 32 or size > min(H, W):
        raise ValueError(f'ROI size {size} invalid for image {(H,W)}')

    center_sem = torch.as_tensor(center_sem_xy, device=rgb_high.device, dtype=torch.float32).reshape(2)
    center_high = semantic_to_high(center_sem, sem_meta, high_meta)
    cx, cy = [float(x) for x in center_high.detach().cpu().tolist()]
    x0 = int(round(max(0.0, min(cx - size / 2, W - size))))
    y0 = int(round(max(0.0, min(cy - size / 2, H - size))))
    rgb_crop = rgb_high[:, y0:y0 + size, x0:x0 + size].contiguous()

    dev = rgb_high.device
    ys = torch.arange(size, dtype=torch.float32, device=dev) + float(y0)
    xs = torch.arange(size, dtype=torch.float32, device=dev) + float(x0)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    pix_high = torch.stack((xx, yy), dim=-1)
    pix_sem = high_to_semantic(pix_high, high_meta, sem_meta)
    p = pix_sem / pix_sem.new_tensor([max(ws - 1, 1), max(hs - 1, 1)])
    scale = sampling_scale.detach().float().reshape(2).to(device=dev)
    translation = sampling_translation.detach().float().reshape(2).to(device=dev)
    q = p * scale + translation
    valid_fraction = float(((q >= 0) & (q <= 1)).all(-1).float().mean().item())
    grid = (q * 2.0 - 1.0).unsqueeze(0)
    tir_local = F.grid_sample(
        tir_full.float().unsqueeze(0), grid, mode='bilinear',
        padding_mode='border', align_corners=True,
    )[0].to(dtype=tir_full.dtype).contiguous()
    geom = CropGeometry(x0, y0, size, (cx, cy), valid_fraction, tuple(float(x) for x in center_sem.detach().cpu().tolist()))
    return rgb_crop, tir_local, geom


def local_box_to_semantic(local_boxes: torch.Tensor, geom: CropGeometry,
                          high_meta: Mapping, sem_meta: Mapping,
                          sem_hw: tuple[int, int]) -> torch.Tensor:
    if local_boxes.ndim != 2 or local_boxes.shape[-1] != 4:
        raise ValueError('local_boxes must be [N,4]')
    offset = local_boxes.new_tensor([geom.x0, geom.y0, geom.x0, geom.y0])
    hs, ws = sem_hw
    return high_box_to_semantic_xyxy(local_boxes + offset, high_meta, sem_meta, ws, hs)
