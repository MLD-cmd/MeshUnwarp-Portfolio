"""Stage 4 detectors — classify document type to select the right postprocessing path."""

import cv2
import numpy as np


def _document_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Classical CV page detector; falls back to full image so failure never crops content."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), max(9.0, min(h, w) * 0.018))

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    white_frac = float((thresh > 0).mean())
    if white_frac < 0.10 or white_frac > 0.95:
        return np.full((h, w), 255, dtype=np.uint8)

    k = max(9, (min(h, w) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return np.full((h, w), 255, dtype=np.uint8)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    image_area = float(h * w)
    if area < image_area * 0.12:
        return np.full((h, w), 255, dtype=np.uint8)

    x, y, bb_w, bb_h = cv2.boundingRect(contour)
    touches_frame = sum((
        x <= 2,
        y <= 2,
        x + bb_w >= w - 2,
        y + bb_h >= h - 2,
    ))
    if touches_frame >= 3:
        return np.full((h, w), 255, dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Convex hull swallows a wedge of background on tall receipts; approxPolyDP is safer there.
    bbox_fill = area / max(1.0, float(bb_w * bb_h))
    portrait_aspect = float(bb_h) / max(float(bb_w), 1.0)
    if portrait_aspect > 2.0 and bbox_fill > 0.55:
        perimeter = cv2.arcLength(contour, True)
        contour_to_draw = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    else:
        contour_to_draw = cv2.convexHull(contour)
    cv2.drawContours(mask, [contour_to_draw], -1, 255, thickness=-1)

    pad = max(3, int(min(h, w) * 0.006))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1),
    )
    return cv2.dilate(mask, dilate_kernel)


def _resolve_doc_mask(document_mask: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if document_mask is None:
        return np.ones((h, w), dtype=bool)
    return document_mask > 0


def _looks_lined_handwritten_page(
    img_rgb: np.ndarray,
    document_mask: np.ndarray | None,
) -> bool:
    """Detect ruled notebook pages; conservative so receipts and flyers don't trigger it."""
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0 or (w / float(h)) < 0.60:
        return False

    doc = _resolve_doc_mask(document_mask, h, w)
    doc_area = int(np.count_nonzero(doc))
    if doc_area < h * w * 0.20:
        return False

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    if float(gray[doc].mean()) < 145.0:
        return False

    blue_line = (
        (hue > 85) & (hue < 135)
        & (sat > 18)
        & (gray < 225)
        & doc
    )
    dark_saturated = (sat > 55) & (gray < 150) & doc

    blue_frac = float(np.count_nonzero(blue_line)) / float(doc_area)
    dark_sat_frac = float(np.count_nonzero(dark_saturated)) / float(doc_area)
    if not (0.04 <= blue_frac <= 0.25) or dark_sat_frac > 0.10:
        return False

    line_mask = (blue_line.astype(np.uint8)) * 255
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, int(w * 0.06)), 1),
    )
    long_lines = cv2.morphologyEx(
        line_mask, cv2.MORPH_OPEN, horizontal_kernel,
    ) > 0

    row_width = np.maximum(np.count_nonzero(doc, axis=1), 1)
    row_frac = np.count_nonzero(long_lines, axis=1).astype(np.float32) / row_width
    smooth_rows = cv2.blur(row_frac[None, :], (1, 9))[0]
    active_rows = np.flatnonzero(smooth_rows > 0.020)
    if active_rows.size == 0:
        return False

    groups = np.split(
        active_rows,
        np.where(np.diff(active_rows) > 2)[0] + 1,
    )
    line_groups = sum(1 for group in groups if group.size >= 2)
    return line_groups >= 8


def _looks_faint_typewritten_page(
    img_rgb: np.ndarray,
    document_mask: np.ndarray | None,
) -> bool:
    """Detect aged/low-contrast typewritten pages by saturation, ink density, and tiny stroke count."""
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0 or (w / float(h)) < 0.60:
        return False

    doc = _resolve_doc_mask(document_mask, h, w)
    doc_area = int(np.count_nonzero(doc))
    if doc_area < h * w * 0.20:
        return False

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    dark_saturated = (sat > 55) & (gray < 170) & doc
    dark_sat_frac = float(np.count_nonzero(dark_saturated)) / float(doc_area)
    if dark_sat_frac > 0.08:
        return False

    mean_lum = float(gray[doc].mean())
    # Upper bound 215 (was 175) — shading flatten lifts cream paper to 185–210.
    if not (80.0 <= mean_lum <= 215.0):
        return False

    ink = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            49, 8,
        ),
    )
    ink = cv2.bitwise_and(ink, doc.astype(np.uint8) * 255)
    ink_frac = float(np.count_nonzero(ink)) / float(doc_area)
    # Upper bound 0.145 (was 0.120) — closely-spaced typed body text pushes above 0.12.
    if not (0.060 <= ink_frac <= 0.145):
        return False

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8,
    )
    if n_labels <= 1:
        return False

    small_components = 0
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        bb_w = float(stats[i, cv2.CC_STAT_WIDTH])
        bb_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if 2.0 <= area <= 80.0 and bb_w <= 42.0 and bb_h <= 28.0:
            small_components += 1

    components_per_mp = small_components / max(doc_area / 1_000_000.0, 0.001)
    return components_per_mp >= 1300.0


def _looks_dense_print_page(
    img_rgb: np.ndarray,
    document_mask: np.ndarray | None,
) -> bool:
    """Detect dense-print pages; serif fragments can be only dozens of pixels after warping."""
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0 or (w / float(h)) < 0.65:
        return False

    doc = _resolve_doc_mask(document_mask, h, w)
    doc_area = int(np.count_nonzero(doc))
    if doc_area < h * w * 0.25:
        return False

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    dark_saturated = (sat > 70) & (gray < 170) & doc
    dark_sat_frac = float(np.count_nonzero(dark_saturated)) / float(doc_area)
    if dark_sat_frac > 0.08:
        return False

    ink = cv2.bitwise_not(
        cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            49, 8,
        ),
    )
    ink = cv2.bitwise_and(ink, doc.astype(np.uint8) * 255)
    ink_frac = float(np.count_nonzero(ink)) / float(doc_area)
    if not (0.05 <= ink_frac <= 0.18):
        return False

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8,
    )
    if n_labels <= 1:
        return False

    short = float(min(h, w))
    small_components = 0
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        bb_w = float(stats[i, cv2.CC_STAT_WIDTH])
        bb_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if (
            3.0 <= area <= 350.0
            and bb_w <= short * 0.055
            and bb_h <= short * 0.035
        ):
            small_components += 1

    components_per_mp = small_components / max(doc_area / 1_000_000.0, 0.001)
    return components_per_mp >= 650.0


def _looks_mixed_graphic_page(
    img_rgb: np.ndarray,
    document_mask: np.ndarray | None,
) -> bool:
    """Detect flyer-like pages where colored panels/text need a gentler text rescue."""
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return False

    doc = _resolve_doc_mask(document_mask, h, w)
    doc_area = int(np.count_nonzero(doc))
    if doc_area < h * w * 0.20:
        return False

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    colored_regions = ((sat > 45) & (gray < 190) & doc).astype(np.uint8)
    colored_frac = float(np.count_nonzero(colored_regions)) / float(doc_area)
    if colored_frac < 0.30:
        return False

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        colored_regions * 255, connectivity=8,
    )
    largest_frac = 0.0
    if n_labels > 1:
        largest_frac = max(
            float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)
        ) / float(doc_area)

    return largest_frac >= 0.10 or colored_frac >= 0.45
