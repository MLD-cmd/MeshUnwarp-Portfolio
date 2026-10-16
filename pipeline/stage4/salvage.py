"""Salvage rendering for badly damaged fragments.

This path is different from normal Clean Scan. It tries to isolate visible paper
fragments from the photographed scene, put them on a plain white background, and
preserve dark text/detail that may still be readable.
"""

import cv2
import numpy as np
from PIL import Image


def salvage_scan(image_rgb: np.ndarray) -> Image.Image:
    """Return a white-background scan-style view for torn/damaged fragments."""
    img = np.asarray(image_rgb).astype(np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"salvage_scan expects HxWx3 RGB input, got {img.shape}")

    mask = _fragment_mask(img)
    scan = _render_salvage_scan(img, mask)
    return Image.fromarray(scan)


def _fragment_mask(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    short = min(h, w)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)

    blue_cloth = (hue >= 85.0) & (hue <= 132.0) & (sat >= 30.0) & (val >= 45.0)
    white_cloth = (val >= 165.0) & (sat <= 80.0)
    very_dark_bg = (val <= 24.0) & (sat <= 120.0)

    warm_paper = (
        (hue >= 5.0) & (hue <= 38.0)
        & (sat >= 20.0)
        & (val >= 28.0) & (val <= 210.0)
    )
    strong_paper = (
        (hue >= 8.0) & (hue <= 32.0)
        & (sat >= 30.0)
        & (val >= 35.0) & (val <= 185.0)
    )

    grab_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(20, short // 40)
    grab_mask[:border, :] = cv2.GC_BGD
    grab_mask[-border:, :] = cv2.GC_BGD
    grab_mask[:, :border] = cv2.GC_BGD
    grab_mask[:, -border:] = cv2.GC_BGD
    grab_mask[blue_cloth | white_cloth | very_dark_bg] = cv2.GC_BGD
    grab_mask[warm_paper] = cv2.GC_PR_FGD
    grab_mask[strong_paper] = cv2.GC_FGD

    grow = max(21, (short // 60) | 1)
    paper_nearby = cv2.dilate(
        strong_paper.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
    ) > 0
    probable = paper_nearby & ~blue_cloth & ~white_cloth
    grab_mask[probable] = np.maximum(grab_mask[probable], cv2.GC_PR_FGD)

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(img_rgb, grab_mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        mask = (
            (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)
        ).astype(np.uint8) * 255
    except cv2.error:
        mask = warm_paper.astype(np.uint8) * 255

    open_k = max(5, (short // 260) | 1)
    close_k = max(31, (short // 45) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )

    return _keep_plausible_fragments(mask)


def _keep_plausible_fragments(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    image_area = float(h * w)
    center = np.array([w / 2.0, h / 2.0])

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8,
    )
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < image_area * 0.004:
            continue
        cx, cy = centroids[i]
        dist = np.linalg.norm(np.array([cx, cy]) - center) / max(h, w)
        if area >= image_area * 0.015 or dist < 0.42:
            clean[labels == i] = 255

    short = min(h, w)
    grow = max(7, (short // 180) | 1)
    return cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
    )


def _render_salvage_scan(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    short = min(h, w)

    local_bg = cv2.GaussianBlur(gray, (0, 0), max(35.0, short * 0.055))
    fine_bg = cv2.GaussianBlur(gray, (0, 0), max(1.5, short * 0.002))

    broad_dark = np.clip((local_bg - gray - 6.0) * 5.5, 0.0, 185.0)
    fine_dark = np.clip((fine_bg - gray - 2.5) * 8.0, 0.0, 150.0)
    dark_detail = np.maximum(broad_dark, fine_dark)

    scan = 255.0 - dark_detail
    alpha = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0,
        (0, 0),
        max(1.5, short * 0.003),
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    out = 255.0 * (1.0 - alpha) + scan * alpha
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_u8, cv2.COLOR_GRAY2RGB)
