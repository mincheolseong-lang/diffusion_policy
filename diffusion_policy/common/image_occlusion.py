"""Offline rectangular occlusion on RGB uint8 images (training or tooling).

alpha=1.0 is fully opaque (training Lift config uses this for the fixed agentview block).
"""
from __future__ import annotations

import numpy as np


def apply_rectangle_occlusion_hwc_uint8(
    hwc: np.ndarray,
    rect_norm: tuple[float, float, float, float],
    color: tuple[int, int, int] = (55, 55, 60),
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Args:
        hwc: (H, W, 3) uint8 RGB
        rect_norm: (x0, y0, x1, y1) in [0, 1], axis-aligned box
    """
    if hwc.dtype != np.uint8:
        raise TypeError(f"expected uint8 HWC, got {hwc.dtype}")
    H, W = hwc.shape[:2]
    x0, y0, x1, y1 = rect_norm
    ix0 = int(np.clip(x0, 0.0, 1.0) * W)
    ix1 = int(np.clip(x1, 0.0, 1.0) * W)
    iy0 = int(np.clip(y0, 0.0, 1.0) * H)
    iy1 = int(np.clip(y1, 0.0, 1.0) * H)
    if ix1 < ix0:
        ix0, ix1 = ix1, ix0
    if iy1 < iy0:
        iy0, iy1 = iy1, iy0
    ix0 = max(0, min(W, ix0))
    ix1 = max(0, min(W, ix1))
    iy0 = max(0, min(H, iy0))
    iy1 = max(0, min(H, iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return hwc

    out = hwc.copy()
    patch = np.array(color, dtype=np.float32)
    region = out[iy0:iy1, ix0:ix1].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    blended = a * patch + (1.0 - a) * region
    out[iy0:iy1, ix0:ix1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def apply_rectangle_occlusion_thwc_uint8(
    thwc: np.ndarray,
    rect_norm: tuple[float, float, float, float],
    color: tuple[int, int, int] = (55, 55, 60),
    alpha: float = 1.0,
) -> np.ndarray:
    """(T, H, W, 3) uint8 — same box on every frame."""
    if thwc.dtype != np.uint8:
        raise TypeError(f"expected uint8 THWC, got {thwc.dtype}")
    out = np.empty_like(thwc)
    for t in range(thwc.shape[0]):
        out[t] = apply_rectangle_occlusion_hwc_uint8(
            thwc[t], rect_norm, color=color, alpha=alpha
        )
    return out


def _find_red_target_center_norm(hwc: np.ndarray):
    """Return normalized (cx, cy) of red target centroid, or None."""
    r = hwc[..., 0].astype(np.int16)
    g = hwc[..., 1].astype(np.int16)
    b = hwc[..., 2].astype(np.int16)
    mask = (r > 90) & (r - g > 25) & (r - b > 25)
    ys, xs = np.where(mask)
    if xs.size < 8:
        return None
    h, w = hwc.shape[:2]
    cx = float(xs.mean()) / max(1, w - 1)
    cy = float(ys.mean()) / max(1, h - 1)
    return (float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)))


def apply_tracking_occlusion_thwc_uint8(
    thwc: np.ndarray,
    rect_size_norm: tuple[float, float] = (0.4, 0.4),
    color: tuple[int, int, int] = (55, 55, 60),
    alpha: float = 1.0,
) -> np.ndarray:
    """
    (T, H, W, 3) uint8 — rectangle center tracks red target each frame.
    If target is not detected in a frame, reuses previous center (or image center).
    """
    if thwc.dtype != np.uint8:
        raise TypeError(f"expected uint8 THWC, got {thwc.dtype}")
    bw = float(np.clip(rect_size_norm[0], 0.0, 1.0))
    bh = float(np.clip(rect_size_norm[1], 0.0, 1.0))
    prev = (0.5, 0.5)
    out = np.empty_like(thwc)
    for t in range(thwc.shape[0]):
        c = _find_red_target_center_norm(thwc[t])
        if c is not None:
            prev = c
        cx, cy = prev
        x0 = np.clip(cx - bw * 0.5, 0.0, 1.0)
        y0 = np.clip(cy - bh * 0.5, 0.0, 1.0)
        x1 = np.clip(x0 + bw, 0.0, 1.0)
        y1 = np.clip(y0 + bh, 0.0, 1.0)
        out[t] = apply_rectangle_occlusion_hwc_uint8(
            thwc[t], (x0, y0, x1, y1), color=color, alpha=alpha
        )
    return out
