"""Two pixel streams from one image: raw → Stage 3 (DocTr), enhanced → quality_gate."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.constants import MIN_DIM, MAX_DIM


@dataclass
class PreprocessResult:
    """Two parallel pixel streams + metadata from one preprocess pass."""
    raw: np.ndarray       # RGB uint8, DocTr-ready
    enhanced: np.ndarray  # RGB uint8, quality-gate-ready
    meta: dict = field(default_factory=dict)


def preprocess(image_path: str) -> PreprocessResult:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = raw.shape[:2]
    scale = 1.0
    
    if min(orig_h, orig_w) < MIN_DIM:
        scale = MIN_DIM / min(orig_h, orig_w)
    elif max(orig_h, orig_w) > MAX_DIM:
        scale = MAX_DIM / max(orig_h, orig_w)

    if scale != 1.0:
        raw = cv2.resize(
            raw,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )

    # Must run before WB correction — per-channel imbalance disappears after scaling.
    lighting = _diagnose_lighting(raw)

    raw = _gray_world_white_balance(raw)

    lab = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    meta = {
        "orig_size": (orig_w, orig_h),
        "upscale":   scale,  # backward-compatible name; can be <1 for downscale
        "scale":     scale,
        "final_size": (raw.shape[1], raw.shape[0]),
        "lighting":  lighting,
    }
    return PreprocessResult(raw=raw, enhanced=enhanced, meta=meta)

def _gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """Clamped to [0.85, 1.15] so it can't push a colored sheet into a wild cast."""
    img_f = img_rgb.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)  # R, G, B
    target = float(means.mean())
    if target < 1.0:
        return img_rgb
    scales = np.clip(target / np.maximum(means, 1.0), 0.85, 1.15)
    out = img_f * scales
    return np.clip(out, 0, 255).astype(np.uint8)


def _diagnose_lighting(img_rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    pixels = float(gray.size)

    glare_frac = float((gray >= 250).sum()) / pixels
    crushed_frac = float((gray <= 10).sum()) / pixels
    mean_lum = float(gray.mean())

    means = img_rgb.astype(np.float32).reshape(-1, 3).mean(axis=0)
    cast_ratio = float(means.max() / max(means.min(), 1.0))
    tone_range, tone_ratio = _paper_tone_variation(img_rgb)

    issues = []
    if glare_frac > 0.02:
        issues.append(f"glare ({glare_frac:.0%})")
    if mean_lum < 70:
        issues.append("underexposed")
    if crushed_frac > 0.05:
        issues.append(f"deep shadows ({crushed_frac:.0%})")
    if cast_ratio > 1.4:
        issues.append("strong color cast")
    if (
        (tone_range >= 58.0 and tone_ratio >= 0.30)
        or (mean_lum < 150.0 and tone_range >= 45.0 and tone_ratio >= 0.34)
    ):
        issues.append("uneven paper tone")

    return {
        "glare_frac":   glare_frac,
        "crushed_frac": crushed_frac,
        "mean_lum":     mean_lum,
        "cast_ratio":   cast_ratio,
        "paper_tone_range": tone_range,
        "paper_tone_ratio": tone_ratio,
        "issues":       issues,
        "warning":      "; ".join(issues) if issues else "",
    }


def _paper_tone_variation(img_rgb: np.ndarray) -> tuple[float, float]:
    """Heavy blur before measuring so high-freq text strokes don't inflate the score."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    h, w = gray.shape
    doc = _document_region_mask(gray)

    sigma = max(15.0, min(h, w) * 0.055)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    paper = doc & (gray > 45) & (gray < 245) & (sat < 110)

    if int(np.count_nonzero(paper)) < gray.size * 0.15:
        paper = doc
    if int(np.count_nonzero(paper)) < gray.size * 0.10:
        paper = np.ones_like(gray, dtype=bool)

    p10, p50, p90 = np.percentile(background[paper], [10, 50, 90])
    tone_range = float(p90 - p10)
    tone_ratio = float(tone_range / max(float(p50), 1.0))

    return tone_range, tone_ratio


def _document_region_mask(gray: np.ndarray) -> np.ndarray:
    """Falls back to central frame when the page can't be isolated from the background."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), max(9.0, min(h, w) * 0.018))
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    white_frac = float((thresh > 0).mean())
    if white_frac < 0.10 or white_frac > 0.95:
        return _central_mask(h, w)

    k = max(9, (min(h, w) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return _central_mask(h, w)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(h * w) * 0.12:
        return _central_mask(h, w)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, thickness=-1)
    return mask > 0


def _central_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = int(h * 0.04), int(h * 0.96)
    x0, x1 = int(w * 0.04), int(w * 0.96)
    mask[y0:y1, x0:x1] = True
    return mask
