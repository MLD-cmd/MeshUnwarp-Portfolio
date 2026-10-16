"""Stage 3 — GeoTr geometric de-warping + conditional IllTr illumination correction."""

import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

_DOCTR_DIR = os.path.join(os.path.dirname(__file__), '..', 'DocTr')
sys.path.insert(0, _DOCTR_DIR)

from GeoTr import GeoTr  # noqa: E402
from IllTr import IllTr  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_MODEL_DIR = os.path.join(_DOCTR_DIR, 'model_pretrained')

from pipeline.constants import (
    DOCTR_INPUT_SIZE   as _DOCTR_INPUT_SIZE,
    GRID_FLOW_DENOM    as _GRID_FLOW_DENOM,
    GRID_RANGE_SHRINK  as _GRID_RANGE_SHRINK,
    ILLTR_BATCH        as _ILLTR_BATCH,
    WARP_STRENGTH      as _WARP_STRENGTH,
    EDGE_MARGIN        as _EDGE_MARGIN,
    EDGE_WARP_STRENGTH as _EDGE_WARP_STRENGTH,
    CHAOS_THRESHOLD    as _CHAOS_CHAOTIC_THRESHOLD,
)

_APPLY_DEWARP = True
_APPLY_ILLTR = True

# GeoTr produces smooth fields for curved pages (~0.0005) but noisy fields on heavy
# crumple (~0.005+) that over-correct. We measure gradient std; above CHAOTIC the
# warp is mostly bypassed so output stays near-identity instead of bending text lines.
_CHAOS_SMOOTH_THRESHOLD  = 0.0015
_CHAOS_FLOOR             = 0.05    # multiplier when warp field is fully chaotic

_geotr = None
_illtr = None


def _lighting_needs_illtr(lighting: dict | None) -> bool:
    # "uneven paper tone" alone is often old staining/texture; IllTr can make it patchy.
    # Only run IllTr for harder issues: glare, underexposure, deep shadows, color cast.
    if lighting is None or "issues" not in lighting:
        return True
    issues = lighting["issues"]
    if not issues:
        return False
    return any(issue != "uneven paper tone" for issue in issues)


def _illtr_skip_reason(lighting: dict | None) -> str:
    if not _APPLY_ILLTR:
        return "disabled"
    issues = (lighting or {}).get("issues", [])
    if issues and all(issue == "uneven paper tone" for issue in issues):
        return "tone_only"
    return "lighting_clean"


def _looks_graphic_heavy(
    image_rgb: np.ndarray,
    lighting: dict | None = None,
) -> bool:
    """IllTr treats printed color as illumination on graphic-heavy docs, causing patch artifacts."""
    doc = _graphic_document_mask(image_rgb)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    strict_graphic = (sat > 70) & (val < 150) & doc
    very_graphic = (sat > 90) & (val < 145) & doc
    strict_frac, strict_largest = _largest_component_fraction(strict_graphic, doc)
    very_frac, very_largest = _largest_component_fraction(very_graphic, doc)

    has_large_panel = (
        (very_frac > 0.18 and very_largest > 0.08)
        or (strict_frac > 0.35 and strict_largest > 0.18)
    )
    if not has_large_panel:
        return False

    issues = set((lighting or {}).get("issues", []))
    has_hard_lighting = any(
        "underexposed" in issue or "deep shadows" in issue
        for issue in issues
    )
    if has_hard_lighting and very_largest < 0.20:
        # Dark captures can make gray shadows look saturated — give IllTr a chance.
        return False
    return True


def _graphic_document_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Approximate the paper region so table/background color does not look like page graphics."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), max(9.0, min(h, w) * 0.018))
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    white_frac = float((thresh > 0).mean())
    if white_frac < 0.10 or white_frac > 0.95:
        return np.ones((h, w), dtype=bool)

    k = max(9, (min(h, w) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return np.ones((h, w), dtype=bool)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(h * w) * 0.12:
        return np.ones((h, w), dtype=bool)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, thickness=-1)
    return mask > 0


def _largest_component_fraction(
    mask: np.ndarray,
    area_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    mask_u8 = mask.astype(np.uint8)
    denom = float(np.count_nonzero(area_mask)) if area_mask is not None else float(mask.size)
    if denom <= 0.0:
        return 0.0, 0.0
    frac = float(np.count_nonzero(mask_u8)) / denom
    if frac <= 0.0:
        return 0.0, 0.0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8 * 255, connectivity=8,
    )
    if n_labels <= 1:
        return frac, 0.0
    largest = max(float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels))
    return frac, largest / denom


def _load_models() -> None:
    global _geotr, _illtr
    if _geotr is not None:
        return

    _geotr = GeoTr(num_attn_layers=6)
    state = torch.load(os.path.join(_MODEL_DIR, 'geotr.pth'),
                       map_location=device)
    geotr_dict = _geotr.state_dict()
    # Strip DataParallel's 'module.' prefix selectively — hard-slicing all keys
    # silently drops keys that don't carry it, leaving the model partially uninitialized.
    state = {k.replace('module.', ''): v for k, v in state.items()
             if k.replace('module.', '') in geotr_dict}
    geotr_dict.update(state)
    _geotr.load_state_dict(geotr_dict)
    _geotr = _geotr.to(device).eval()

    _illtr = IllTr()
    state = torch.load(os.path.join(_MODEL_DIR, 'illtr.pth'),
                       map_location=device)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    _illtr.load_state_dict(state)
    _illtr = _illtr.to(device).eval()
    log.info("GeoTr + IllTr loaded onto %s", device)


def dewarp(image: np.ndarray) -> np.ndarray:
    out, _ = dewarp_with_meta(image)
    return out


def dewarp_with_meta(
    image: np.ndarray,
    lighting: dict | None = None,
    source_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"dewarp expects HxWx3 RGB uint8 input, got {image.shape}")
    _load_models()
    bm, grid_meta = _geotr_forward(image, source_name=source_name)
    src = f"[{source_name}] " if source_name else ""
    log.info(
        "%sGeoTr max_displacement=%.4f → apply warp",
        src,
        grid_meta["max_displacement"],
    )

    out = _apply_grid(image, bm) if _APPLY_DEWARP else image

    illum_meta: dict
    needs_illtr = _APPLY_ILLTR and _lighting_needs_illtr(lighting)
    if needs_illtr and _looks_graphic_heavy(out, lighting):
        illum_meta = {"applied": False, "reason": "dark_design_panel"}
        log.info("%silltr skipped: dark design panel", src)
    elif needs_illtr:
        out = _illtr_correct(out)
        illum_meta = {"applied": True, "mode": "illtr_patch"}
    else:
        illum_meta = {
            "applied": False,
            "reason": _illtr_skip_reason(lighting),
        }
        if _APPLY_ILLTR:
            log.info("%silltr skipped: %s", src, illum_meta["reason"])

    meta = {
        "geotr": grid_meta,
        "illumination": illum_meta,
    }
    return out, meta


def _geotr_forward(
    image_rgb: np.ndarray,
    source_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    h, w = image_rgb.shape[:2]
    im_ori = image_rgb.astype(np.float32) / 255.0
    im = cv2.resize(im_ori, (_DOCTR_INPUT_SIZE, _DOCTR_INPUT_SIZE),
                    interpolation=cv2.INTER_LINEAR)
    im_t = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        bm = _geotr(im_t)
        bm = (2.0 * (bm / _GRID_FLOW_DENOM) - 1.0) * _GRID_RANGE_SHRINK
        bm_cpu = bm.cpu()

    bm0 = cv2.resize(bm_cpu[0, 0].numpy(), (w, h))
    bm1 = cv2.resize(bm_cpu[0, 1].numpy(), (w, h))
    grid = np.stack([bm0, bm1], axis=2)

    identity = _identity_grid(h, w)
    chaos = _measure_warp_chaos(grid, identity)
    chaos_mult = _chaos_attenuation(chaos)

    weight = _edge_weight_map(h, w) * chaos_mult
    grid_blended = weight * grid + (1.0 - weight) * identity

    src = f"[{source_name}] " if source_name else ""
    log.info(
        "%swarp chaos=%.4f → mult=%.2f  raw_max=%.4f  (smooth<%.4f, chaotic>%.4f)",
        src,
        chaos, chaos_mult,
        float(np.max(np.abs(grid - identity))),
        _CHAOS_SMOOTH_THRESHOLD, _CHAOS_CHAOTIC_THRESHOLD,
    )

    max_displacement = float(np.max(np.abs(grid_blended - identity)))

    meta = {
        "applied": _APPLY_DEWARP,
        "warp_strength": _WARP_STRENGTH,
        "max_shift_raw": float(np.max(np.abs(grid - identity))),
        "max_displacement": max_displacement,
        "input_size": _DOCTR_INPUT_SIZE,
        "chaos": chaos,
        "chaos_mult": chaos_mult,
    }
    return grid_blended, meta


def _identity_grid(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x, grid_y], axis=2)


def _measure_warp_chaos(grid: np.ndarray, identity: np.ndarray) -> float:
    """Std of the displacement field's spatial gradients — low for curved pages, high for crumple."""
    displacement = grid - identity
    dx = np.diff(displacement, axis=1)
    dy = np.diff(displacement, axis=0)
    return float(np.sqrt(np.var(dx) + np.var(dy)))


def _chaos_attenuation(chaos: float) -> float:
    if chaos <= _CHAOS_SMOOTH_THRESHOLD:
        return 1.0
    if chaos >= _CHAOS_CHAOTIC_THRESHOLD:
        return _CHAOS_FLOOR
    span = _CHAOS_CHAOTIC_THRESHOLD - _CHAOS_SMOOTH_THRESHOLD
    t = (chaos - _CHAOS_SMOOTH_THRESHOLD) / span
    return 1.0 - (1.0 - _CHAOS_FLOOR) * t



def _edge_weight_map(h: int, w: int) -> np.ndarray:
    """Cosine-tapered weight map: _EDGE_WARP_STRENGTH at borders → _WARP_STRENGTH at centre."""
    margin_y = int(h * _EDGE_MARGIN)
    margin_x = int(w * _EDGE_MARGIN)

    ramp_y = np.ones(h, dtype=np.float32)
    if margin_y > 0:
        fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, margin_y, dtype=np.float32))
        ramp_y[:margin_y]  = fade
        ramp_y[-margin_y:] = fade[::-1]

    ramp_x = np.ones(w, dtype=np.float32)
    if margin_x > 0:
        fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, margin_x, dtype=np.float32))
        ramp_x[:margin_x]  = fade
        ramp_x[-margin_x:] = fade[::-1]

    weight_2d = np.minimum(ramp_y[:, None], ramp_x[None, :])
    lo, hi = _EDGE_WARP_STRENGTH, _WARP_STRENGTH
    weight_2d = lo + (hi - lo) * weight_2d
    return weight_2d[:, :, None]


def _apply_grid(image_rgb: np.ndarray, grid: np.ndarray) -> np.ndarray:
    im_f = image_rgb.astype(np.float32) / 255.0
    # .to(device) is required — without it grid_sample silently falls back to CPU on MPS.
    src = torch.from_numpy(im_f.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    lbl = torch.from_numpy(grid).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = F.grid_sample(src, lbl, align_corners=True, padding_mode="border")
    out_np = out[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(out_np * 255.0, 0, 255).astype(np.uint8)


def _illtr_correct(image_rgb: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    patches, padH, padW = _pad_crop(img_bgr)
    results = _ill_correction(_illtr, patches)
    composed = _compose_patch(results, padH, padW, img_bgr)
    composed = np.clip(composed * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)


def _pad_crop(img: np.ndarray):
    
    H, W = img.shape[:2]
    patchRes = 128
    ovlp = int(patchRes * 0.125)

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1)
            * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1)
            * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)
    ynum = int((padImg.shape[0] - patchRes) / (patchRes - ovlp)) + 1
    xnum = int((padImg.shape[1] - patchRes) / (patchRes - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)
    for j in range(ynum):
        for i in range(xnum):
            x = int(i * (patchRes - ovlp))
            y = int(j * (patchRes - ovlp))
            if j == (ynum - 1) and i == (xnum - 1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum - 1):
                totalPatch[j, i] = img[-patchRes:, x:x + patchRes]
            elif i == (xnum - 1):
                totalPatch[j, i] = img[y:y + patchRes, -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:y + patchRes, x:x + patchRes]
    return totalPatch, padH, padW


def _ill_correction(net: IllTr, totalPatch: np.ndarray) -> np.ndarray:
    # Batched instead of per-patch: a per-patch loop paid ~390 CPU↔MPS round-trips
    # on a 1600×2848 page (~40s). Chunked batches cut that by ~64×.
    patches = totalPatch.astype(np.float32) / 255.0
    ynum, xnum = patches.shape[:2]
    n = ynum * xnum

    flat = (
        torch.from_numpy(patches.reshape(n, 128, 128, 3))
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    out_chunks = []
    with torch.no_grad():
        for start in range(0, n, _ILLTR_BATCH):
            chunk = flat[start:start + _ILLTR_BATCH].to(device)
            out_chunks.append(net(chunk).cpu())

    out_all = torch.cat(out_chunks, dim=0).permute(0, 2, 3, 1).numpy()
    return np.clip(out_all, 0.0, 1.0).reshape(ynum, xnum, 128, 128, 3)


def _compose_patch(totalResults: np.ndarray, padH: int, padW: int,
                   img: np.ndarray) -> np.ndarray:
    ynum, xnum = totalResults.shape[:2]
    patchRes = totalResults.shape[2]
    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros(img.shape, dtype=np.float32)
    for j in range(ynum):
        for i in range(xnum):
            sy = int(j * step)
            sx = int(i * step)
            patch = totalResults[j, i].astype(np.float32)
            if j == 0 and i != (xnum - 1):
                resImg[sy:sy + patchRes, sx:sx + patchRes] = patch
            elif i == 0 and j != (ynum - 1):
                resImg[sy + 10:sy + patchRes, sx:sx + patchRes] = patch[10:]
            elif j == (ynum - 1) and i == (xnum - 1):
                resImg[-patchRes + 10:, -patchRes + 10:] = patch[10:, 10:]
            elif j == (ynum - 1) and i == 0:
                resImg[-patchRes + 10:, sx:sx + patchRes] = patch[10:]
            elif j == (ynum - 1) and i != 0:
                resImg[-patchRes + 10:, sx + 10:sx + patchRes] = patch[10:, 10:]
            elif i == (xnum - 1) and j == 0:
                resImg[sy:sy + patchRes, -patchRes + 10:] = patch[:, 10:]
            elif i == (xnum - 1) and j != 0:
                resImg[sy + 10:sy + patchRes, -patchRes + 10:] = patch[10:, 10:]
            else:
                resImg[sy + 10:sy + patchRes,
                       sx + 10:sx + patchRes] = patch[10:, 10:]
    resImg[0, :, :] = 1.0
    return resImg
