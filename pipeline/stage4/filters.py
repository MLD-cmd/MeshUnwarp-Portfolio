"""Stage 4 filters — build and clean the ink mask."""

import cv2
import numpy as np


def _suppress_stains(img_rgb: np.ndarray, strength: float = 0.80) -> np.ndarray:
    """Divisive-normalization stain lift: text stays dark because it's darker than local background."""
    if strength <= 0.0:
        return img_rgb

    h_img, w_img = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch, s_ch = hsv[..., 0], hsv[..., 1]

    paper_white = float(np.percentile(gray, 92))
    if paper_white < 1.0:
        return img_rgb

    warm = (h_ch >= 4.0) & (h_ch <= 38.0) & (s_ch >= 8.0)
    darker = gray < paper_white * 0.93
    not_ink = gray > paper_white * 0.22
    seed = (warm & darker & not_ink).astype(np.uint8) * 255

    short = min(h_img, w_img)
    close_k = max(5, int(short * 0.06) | 1)
    open_k = max(3, int(short * 0.02) | 1)
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed, connectivity=8,
    )
    region = np.zeros_like(seed)
    min_region = float(h_img * w_img) * 0.004
    for i in range(1, n_labels):
        if float(stats[i, cv2.CC_STAT_AREA]) >= min_region:
            region[labels == i] = 255
    if not np.any(region):
        return img_rgb

    sigma_bg = max(20.0, short * 0.07)
    local_bg = cv2.GaussianBlur(gray, (0, 0), sigma_bg)
    # Cap 6.0 is safe: at text pixels the large blur reflects the background level
    # (thin strokes barely dent it), so gain there is the modest background gain.
    gain = np.clip(paper_white / np.maximum(local_bg, 1.0), 1.0, 6.0)

    img_f = img_rgb.astype(np.float32)
    lifted = np.clip(img_f * gain[..., None], 0.0, 255.0)

    lifted_gray = cv2.cvtColor(
        lifted.astype(np.uint8), cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    white_push = np.clip(
        (lifted_gray - paper_white * 0.72) / max(paper_white * 0.20, 1.0),
        0.0, 1.0,
    )
    # Ink is near-achromatic (s_ch ≈ 0–25); stains are chromatic (s_ch ≥ 30).
    # Without this guard, gain-lifted neutral ink would snap to paper white and vanish.
    chroma_mask = (s_ch >= 30).astype(np.float32)
    white_push = (white_push * chroma_mask)[..., None]
    lifted = lifted * (1.0 - white_push) + paper_white * white_push

    grow = max(3, int(short * 0.02) | 1)
    region = cv2.dilate(
        region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
    )

    alpha = cv2.GaussianBlur(
        region.astype(np.float32) / 255.0, (0, 0), max(2.0, short * 0.008),
    )
    alpha = np.clip(alpha * strength, 0.0, 1.0)[..., None]
    out = img_f * (1.0 - alpha) + lifted * alpha

    # Text ink is neutral (not saturated) so it never enters this mask.
    chroma_stain = (
        (h_ch >= 4.0) & (h_ch <= 40.0)
        & (s_ch >= 50.0)
        & (gray >= paper_white * 0.16)
    ).astype(np.float32)
    chroma_alpha = cv2.GaussianBlur(
        chroma_stain, (0, 0), max(1.5, short * 0.004),
    )
    chroma_alpha = np.clip(chroma_alpha * strength, 0.0, 1.0)[..., None]
    out = out * (1.0 - chroma_alpha) + paper_white * chroma_alpha

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _shading_flatten(img_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Divide by large-radius blur to lift fold/crumple shadows; text stays dark (same local level)."""
    if strength <= 0:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    # sigma=0.10×min_dim covers fold-shadow gradients (150–300 px) without smearing into text.
    sigma = max(40.0, min(h, w) * 0.10)
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    paper_level = float(np.percentile(bg, 95))
    if paper_level < 1.0:
        return img_rgb
    # Cap 2.2 (raised from 1.6) so deep shadows (local bg 40–50% below paper) are fully lifted.
    gain = np.clip(paper_level / np.maximum(bg, 1.0), 1.0, 2.2)

    img_f = img_rgb.astype(np.float32)
    flattened = np.clip(img_f * gain[..., None], 0, 255)

    if strength >= 1.0:
        return flattened.astype(np.uint8)
    blended = img_f * (1.0 - strength) + flattened * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _normalize_panel_polarity(
    img_rgb: np.ndarray,
    document_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Locally invert dark-background panels (white text on a dark fill) so the whole
    page reads dark-text-on-light, letting the normal ink pipeline run unchanged.

    Targets flyers with a colored panel beside a white panel. A *uniformly* dim photo of
    light paper is underexposure, not a dark panel, so its local background never drops
    this low and the function is a no-op there. Returns (image, applied).
    """
    h, w = img_rgb.shape[:2]
    short = float(min(h, w))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ch = hsv[..., 0].astype(np.float32)
    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)

    if document_mask is not None:
        doc = document_mask > 0
    else:
        doc = np.ones((h, w), dtype=bool)
    doc_area = int(np.count_nonzero(doc)) or 1

    # A genuine dark panel is dark AND *blue* (navy/slate). Hue is the key
    # discriminator: warm dim paper / wood table are orange-hued and would pass a
    # saturation-only test, but they are not in the blue band, so they are excluded
    # and white/cream backgrounds are never inverted.
    bluish = (
        (h_ch >= 95.0) & (h_ch <= 135.0)
        & (s_ch >= 40.0) & (v_ch <= 155.0)
    ).astype(np.float32)
    # Large blur fills thin text holes so the panel reads as one solid region; on
    # warm or white paper this stays near zero.
    sigma = max(30.0, short * 0.05)
    bluish_density = cv2.GaussianBlur(bluish, (0, 0), sigma)
    panel_seed = ((bluish_density > 0.35) & doc).astype(np.uint8)
    if not np.any(panel_seed):
        return img_rgb, False

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        panel_seed * 255, connectivity=8,
    )
    # A design panel is a wide, solid band spanning the page. Incidental bluish
    # crumple shadows are narrow, scattered, and low-solidity — these shape gates
    # accept the former and reject the latter, so dim/crumpled white-paper photos
    # (which have no real panel) are left untouched.
    min_panel = doc_area * 0.06
    panel_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_panel:
            continue
        bb_w = float(stats[i, cv2.CC_STAT_WIDTH])
        bb_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        spans_width = bb_w >= w * 0.55
        is_solid = area / max(bb_w * bb_h, 1.0) >= 0.62
        if not (spans_width and is_solid):
            continue

        # Final guard against a bluish color-cast on crumpled white paper, whose
        # color/shape stats mimic a navy panel. A real navy panel is either a very
        # dark fill, or a dark fill carrying genuinely bright (white) text. A
        # tinted paper region is mid-toned with only dark ink — no bright text —
        # so it fails both and is left alone (no destructive inversion).
        region_gray = gray[labels == i]
        median_g = float(np.median(region_gray))
        p90_g = float(np.percentile(region_gray, 90))
        dark_frac = float((region_gray < 90).mean())
        is_dark_fill = median_g < 45.0
        is_dark_with_bright_text = p90_g > 140.0 and dark_frac > 0.30
        if is_dark_fill or is_dark_with_bright_text:
            panel_mask[labels == i] = 255

    panel_frac = float(np.count_nonzero(panel_mask)) / float(doc_area)
    # A real panel beside another panel — not the whole page (underexposure).
    if not (0.06 <= panel_frac <= 0.92):
        return img_rgb, False

    # Close fills any residual interior gaps; a modest blur only softens the seam so
    # the inversion follows the panel boundary tightly instead of bleeding outward.
    close_k = max(3, int(short * 0.01) | 1)
    panel_mask = cv2.morphologyEx(
        panel_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )
    alpha = cv2.GaussianBlur(
        panel_mask.astype(np.float32) / 255.0, (0, 0), max(4.0, short * 0.006),
    )
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    inverted = 255.0 - img_rgb.astype(np.float32)

    # White-on-dark has low contrast; once inverted it is low-dynamic-range, so the
    # hard Clean Scan threshold shatters the thin strokes. CLAHE on the inverted L
    # channel restores text/background separation before downstream binarization.
    inv_u8 = np.clip(inverted, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(inv_u8, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    inverted = cv2.cvtColor(
        cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2RGB,
    ).astype(np.float32)

    out = img_rgb.astype(np.float32) * (1.0 - alpha) + inverted * alpha
    return np.clip(out, 0, 255).astype(np.uint8), True


def _bilateral(img_rgb: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return img_rgb
    return cv2.bilateralFilter(
        img_rgb, d=9, sigmaColor=strength, sigmaSpace=strength,
    )


def _adaptive_ink_mask(img_rgb: np.ndarray, block_size: int, C: int) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bs = max(3, block_size | 1)
    bin_paper = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        bs, C,
    )  # 255 = paper, 0 = ink
    return cv2.bitwise_not(bin_paper)  # invert: 255 = ink


def _morph(mask: np.ndarray, op: int, k_size: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.morphologyEx(mask, op, k)


def _recover_masked_edge_text(
    raw_ink_mask: np.ndarray,
    document_mask: np.ndarray,
    smooth_rgb: np.ndarray,
) -> np.ndarray:
    """Rescue text clipped at a warped/shadowed page corner; OR'd into the final mask."""
    h, w = raw_ink_mask.shape
    short = float(min(h, w))
    empty = np.zeros((h, w), dtype=np.uint8)

    margin = max(6, int(short * 0.06))
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
    )
    band = cv2.subtract(cv2.dilate(document_mask, k), document_mask)
    candidate = cv2.bitwise_and(raw_ink_mask, band)
    if not np.any(candidate):
        return empty

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8,
    )
    if n_labels <= 1:
        return empty

    gray = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2GRAY)
    rescue = empty.copy()
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 6.0:
            continue
        bb_w = float(stats[i, cv2.CC_STAT_WIDTH])
        bb_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        comp = labels == i
        if float(gray[comp].mean()) > 165.0:
            continue
        fill = area / max(bb_w * bb_h, 1.0)
        aspect = max(bb_w, bb_h) / max(min(bb_w, bb_h), 1.0)
        long_side = max(bb_w, bb_h)
        if aspect > 8.0:
            continue
        if fill > 0.80 and area > short * short * 0.004:
            continue
        if long_side > short * 0.85 or area > short * short * 0.05:
            continue
        rescue[comp] = 255
    return rescue


def _light_text_recovery_region(
    source_rgb: np.ndarray,
    document_mask: np.ndarray | None,
) -> np.ndarray:
    """Soft mask for rescuing faint text on light paper while avoiding colored panels."""
    h, w = source_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.float32)

    if document_mask is None:
        doc = np.ones((h, w), dtype=bool)
    else:
        doc = document_mask > 0

    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    short = float(min(h, w))

    local_paper = cv2.GaussianBlur(
        gray, (0, 0), max(15.0, short * 0.035),
    )
    light_region = (local_paper > 112.0) & doc

    panel_seed = ((sat > 45.0) & (gray < 185.0) & doc).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        panel_seed, connectivity=8,
    )
    panels = np.zeros((h, w), dtype=np.uint8)
    min_panel_area = float(h * w) * 0.015
    for i in range(1, n_labels):
        if float(stats[i, cv2.CC_STAT_AREA]) >= min_panel_area:
            panels[labels == i] = 255

    if np.any(panels):
        grow = max(5, int(short * 0.02) | 1)
        panels = cv2.dilate(
            panels,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
        )

    region = (light_region & (panels == 0)).astype(np.float32)
    region = cv2.GaussianBlur(
        region, (0, 0), max(2.0, short * 0.006),
    )
    return np.clip(region, 0.0, 1.0)


def _remove_boundary_artifacts(
    ink_mask: np.ndarray,
    document_mask: np.ndarray,
    margin_frac: float,
    preserve_text: bool = False,
) -> np.ndarray:
    h, w = ink_mask.shape
    margin = max(3, int(min(h, w) * margin_frac))
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
    )
    inner = cv2.erode(document_mask, k)
    boundary = cv2.subtract(document_mask, inner)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    if n_labels <= 1:
        return ink_mask

    short = float(min(h, w))
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    bb_w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bb_h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    bbox_area = np.maximum(bb_w * bb_h, 1.0)
    fill = area / bbox_area

    boundary_hits = np.bincount(
        labels.ravel(),
        weights=(boundary > 0).ravel().astype(np.float32),
        minlength=n_labels,
    )
    boundary_overlap = boundary_hits / np.maximum(area, 1.0)
    elongated = np.maximum(bb_w, bb_h) > short * 0.045
    remove = (area >= 24.0) & (boundary_overlap > 0.20) & elongated

    if preserve_text:
        edge_text_like = (
            (area <= short * short * 0.0035)
            & (bb_h <= short * 0.055)
            & (bb_w <= w * 0.55)
            & (fill <= 0.75)
        )
        remove &= ~edge_text_like

    remove[0] = False
    if not np.any(remove):
        return ink_mask
    out = ink_mask.copy()
    out[remove[labels]] = 0
    return out


def _remove_dense_side_bands(ink_mask: np.ndarray) -> np.ndarray:
    """Remove high-density side columns — catches edge shadow/grain split into many small components."""
    h, w = ink_mask.shape
    ink = ink_mask > 0
    density = ink.mean(axis=0).astype(np.float32)
    win = max(9, (w // 90) | 1)
    smooth = cv2.blur(density[None, :], (win, 1))[0]

    out = ink_mask.copy()
    for start, end in ((0, int(w * 0.35)), (int(w * 0.65), w)):
        if end <= start:
            continue
        side = smooth[start:end] > 0.30
        if not np.any(side):
            continue
        idx = np.flatnonzero(side)
        bands = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for band in bands:
            if band.size < 3:
                continue
            b0 = max(start + int(band[0]) - 2, 0)
            b1 = min(start + int(band[-1]) + 3, w)
            if (b1 - b0) <= w * 0.09:
                out[:, b0:b1] = 0
    return out


def _remove_frame_artifacts(
    ink_mask: np.ndarray,
    preserve_text: bool = False,
) -> np.ndarray:
    h, w = ink_mask.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    if n_labels <= 1:
        return ink_mask

    out = ink_mask.copy()
    x = stats[:, cv2.CC_STAT_LEFT].astype(np.float32)
    y = stats[:, cv2.CC_STAT_TOP].astype(np.float32)
    bb_w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bb_h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)

    near_side = (x < w * 0.28) | ((x + bb_w) > w * 0.72)
    tall_strip = (bb_h > h * 0.18) & (bb_w < w * 0.13)
    top_or_bottom_rule = (
        (bb_w > w * 0.35)
        & (bb_h < h * 0.035)
        & ((y < h * 0.08) | ((y + bb_h) > h * 0.92))
    )
    remove = (area >= 24.0) & ((near_side & tall_strip) | top_or_bottom_rule)

    if preserve_text:
        fill = area / np.maximum(bb_w * bb_h, 1.0)
        remove &= ~(top_or_bottom_rule & (fill < 0.45))

    remove[0] = False
    if np.any(remove):
        out[remove[labels]] = 0
    out = _remove_dense_side_bands(out)
    return out


def _remove_speckle(ink_mask: np.ndarray, ratio: float) -> np.ndarray:
    """Erase tiny ink components; threshold scales with resolution (short²/1000) so it works at any DPI."""
    h, w = ink_mask.shape
    short = float(min(h, w))
    min_keep_px = max(2.0, ratio * (short * short) / 1000.0)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    if n_labels <= 1:
        return ink_mask
    remove = stats[:, cv2.CC_STAT_AREA].astype(np.float32) < min_keep_px
    remove[0] = False
    if not np.any(remove):
        return ink_mask
    out = ink_mask.copy()
    out[remove[labels]] = 0
    return out


def _remove_outer_residuals(
    ink_mask: np.ndarray,
    document_mask: np.ndarray | None = None,
    pad_frac: float = 0.075,
) -> np.ndarray:
    """Remove IllTr patch/border residue outside the dominant printed text area."""
    h, w = ink_mask.shape
    ink = ink_mask > 0
    if not np.any(ink):
        return ink_mask

    if document_mask is not None:
        doc = document_mask > 0
        margin = max(5, int(min(h, w) * 0.04))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
        )
        seed_area = cv2.erode(doc.astype(np.uint8), kernel) > 0
        seed = ink & seed_area
    else:
        seed = ink

    if int(np.count_nonzero(seed)) < max(128, int(h * w * 0.0005)):
        return ink_mask

    ys, xs = np.nonzero(seed)
    x0, x1 = np.percentile(xs, [5, 95])
    y0, y1 = np.percentile(ys, [2, 95])
    pad = max(12, int(min(h, w) * pad_frac))
    x0 = max(0, int(x0) - pad)
    x1 = min(w - 1, int(x1) + pad)
    y0 = max(0, int(y0) - pad)
    y1 = min(h - 1, int(y1) + pad)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    if n_labels <= 1:
        return ink_mask

    inside = (
        (centroids[:, 0] >= x0)
        & (centroids[:, 0] <= x1)
        & (centroids[:, 1] >= y0)
        & (centroids[:, 1] <= y1)
    )
    remove = ~inside
    remove[0] = False
    if not np.any(remove):
        return ink_mask
    out = ink_mask.copy()
    out[remove[labels]] = 0
    return out


def _recover_partial_visible_text(
    preclean_ink_mask: np.ndarray,
    cleaned_ink_mask: np.ndarray,
    document_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Re-add visible text fragments that Clean Scan cleanup removed on PARTIAL pages.

    The candidate pool is only ink that already existed before speckle/frame cleanup.
    Components must be text-sized and close to text that survived cleanup, which keeps
    broad shadows and isolated noise from being promoted into output text.
    """
    h, w = cleaned_ink_mask.shape
    short = float(min(h, w))

    candidate = cv2.bitwise_and(
        preclean_ink_mask,
        cv2.bitwise_not(cleaned_ink_mask),
    )
    if document_mask is not None:
        candidate = cv2.bitwise_and(candidate, document_mask)
    if not np.any(candidate):
        return cleaned_ink_mask

    support_w = max(65, int(short * 0.033) | 1)
    support_h = max(11, int(short * 0.006) | 1)
    support = cv2.dilate(
        cleaned_ink_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (support_w, support_h)),
    ) > 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8), connectivity=8,
    )
    if n_labels <= 1:
        return cleaned_ink_mask

    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    bb_w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bb_h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    fill = area / np.maximum(bb_w * bb_h, 1.0)

    support_hits = np.bincount(
        labels.ravel(),
        weights=support.ravel().astype(np.float32),
        minlength=n_labels,
    )
    support_ratio = support_hits / np.maximum(area, 1.0)

    keep = (
        (area >= 2.0)
        & (area <= short * short * 0.0012)
        & (bb_w <= w * 0.10)
        & (bb_h <= h * 0.035)
        & (fill <= 0.85)
        & (support_ratio > 0.0)
    )
    keep[0] = False
    if not np.any(keep):
        return cleaned_ink_mask

    out = cleaned_ink_mask.copy()
    out[keep[labels]] = 255
    return out


def _remove_large_blobs(ink_mask: np.ndarray, area_ratio: float) -> np.ndarray:
    """Erase large solid blobs (stains/shadows); fill > 0.55 guard keeps text lines safe."""
    h, w = ink_mask.shape
    max_area = float(h * w) * area_ratio
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8,
    )
    if n_labels <= 1:
        return ink_mask
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    bb_w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bb_h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    fill = area / np.maximum(bb_w * bb_h, 1.0)
    remove = (area > max_area) & (fill > 0.55)
    remove[0] = False
    if not np.any(remove):
        return ink_mask
    out = ink_mask.copy()
    out[remove[labels]] = 0
    return out
