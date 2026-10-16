"""Stage 4 render — composite the final output image from ink mask and source."""

import cv2
import numpy as np


def _add_recovery_layer(
    ink_alpha: np.ndarray, smooth_rgb: np.ndarray,
    strength: float, floor: float,
) -> np.ndarray:
    gray = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (min(gray.shape) // 12) | 1)
    local_paper = cv2.GaussianBlur(gray, (k, k), 0)
    darkness = np.clip((local_paper - gray - floor) / 50.0, 0.0, 1.0)
    return np.clip(ink_alpha + strength * darkness, 0.0, 1.0)


def _render_soft_typewriter(
    smooth_rgb: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Tonal renderer: local darkness below paper level, so fold shadows stay gray not solid black."""
    gray = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(31, (min(gray.shape) // 12) | 1)
    local_paper = cv2.GaussianBlur(gray, (k, k), 0)

    floor = float(params.get("soft_floor", 18.0))
    scale = max(float(params.get("soft_scale", 40.0)), 1.0)
    ink_alpha = np.clip((local_paper - gray - floor) / scale, 0.0, 1.0)

    ink_layer = _darken_ink_gray(
        gray.astype(np.uint8),
        float(params.get("soft_gamma", 0.42)),
        float(params.get("soft_ink_floor", 0.02)),
    ).astype(np.float32)
    out_gray = ink_layer * ink_alpha + 255.0 * (1.0 - ink_alpha)
    out_gray = np.clip(out_gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB), ink_alpha


def _darken_ink_rgb(
    img_rgb: np.ndarray, gamma: float, floor: float,
) -> np.ndarray:
    if gamma >= 1.0 and floor <= 0.0:
        return img_rgb
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, 1.0 / gamma) if gamma < 1.0 else x
    if floor > 0.0:
        y = np.where(y < floor, 0.0, y)
    lut = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    lut[255] = 255
    return cv2.LUT(img_rgb, lut)


def _darken_ink_gray(gray: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    if gamma >= 1.0 and floor <= 0.0:
        return gray
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, 1.0 / gamma) if gamma < 1.0 else x
    if floor > 0.0:
        y = np.where(y < floor, 0.0, y)
    lut = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    lut[255] = 255
    return cv2.LUT(gray, lut)


def _unsharp(img_rgb: np.ndarray, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=0.8)
    sharp = cv2.addWeighted(img_rgb, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
