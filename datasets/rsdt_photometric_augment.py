"""Anti-UAV modality-specific image-only augmentation.

Call once on raw RGB/TIR images BEFORE all letterbox operations. No spatial
transform is performed, so bounding boxes, ratio_pad and geometry are unchanged.
"""
from __future__ import annotations

import math
import random
from typing import Mapping

import cv2
import numpy as np


def _prob(cfg: Mapping, name: str, fallback: float = 0.0) -> float:
    p = float(cfg.get(name, fallback))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {p}")
    return p


def _nonnegative(cfg: Mapping, name: str, fallback: float = 0.0) -> float:
    x = float(cfg.get(name, fallback))
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


def _photo(image: np.ndarray, cfg: Mapping, prefix: str) -> np.ndarray:
    if random.random() >= _prob(cfg, f"{prefix}_photo_p"):
        return image
    br = _nonnegative(cfg, f"{prefix}_brightness") if prefix == 'rgb' else 0.0
    ct = _nonnegative(cfg, f"{prefix}_contrast")
    gm = _nonnegative(cfg, f"{prefix}_gamma")
    blur = _prob(cfg, f"{prefix}_blur_p")
    noise = _prob(cfg, f"{prefix}_noise_p")
    std = _nonnegative(cfg, f"{prefix}_noise_std")

    # Small uint8 LUT: avoids creating several 1920x1920x3 float32 arrays.
    if br or ct or gm:
        b = random.uniform(-br, br) if br else 0.0
        c = random.uniform(1.0-ct, 1.0+ct) if ct else 1.0
        g = math.exp(random.uniform(-gm, gm)) if gm else 1.0
        v = np.arange(256, dtype=np.float32)/255.0
        v = np.clip((v - 0.5)*c + 0.5 + b, 0.0, 1.0)
        lut = np.clip(np.power(v, g)*255.0, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, lut)

    if random.random() < blur:
        image = cv2.GaussianBlur(image, (3,3), sigmaX=0.6)

    if std and random.random() < noise:
        # TIR is grayscale replicated to three channels. Preserve that property.
        shape = image.shape if prefix != 'tir' or image.ndim != 3 else image.shape[:2] + (1,)
        jitter = np.random.normal(0.0, std*255.0, size=shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def augment_rgb_tir(rgb: np.ndarray, tir: np.ndarray, cfg: Mapping):
    """Return image-only augmented (RGB,TIR), same H/W and uint8, no label changes."""
    if rgb.dtype != np.uint8 or tir.dtype != np.uint8:
        raise TypeError('Expected uint8 originals before letterbox')
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f'Unexpected RGB shape: {rgb.shape}')
    r = _photo(rgb, cfg, 'rgb')
    t = _photo(tir, cfg, 'tir')
    assert r.shape == rgb.shape and t.shape == tir.shape
    return r, t
