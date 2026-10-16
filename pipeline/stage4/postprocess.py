"""Stage 4 — postprocessing. Clean Scan = binary white/black scan. Ink Recovery = tonal faint-stroke rescue."""

import logging

import cv2
import numpy as np
from PIL import Image

from .detectors import (
    _document_mask,
    _looks_lined_handwritten_page,
    _looks_faint_typewritten_page,
    _looks_dense_print_page,
    _looks_mixed_graphic_page,
)
from .filters import (
    _normalize_panel_polarity,
    _suppress_stains,
    _shading_flatten,
    _bilateral,
    _adaptive_ink_mask,
    _morph,
    _recover_masked_edge_text,
    _light_text_recovery_region,
    _remove_boundary_artifacts,
    _remove_frame_artifacts,
    _recover_partial_visible_text,
    _remove_large_blobs,
    _remove_outer_residuals,
    _remove_speckle,
)
from .render import (
    _add_recovery_layer,
    _render_soft_typewriter,
    _darken_ink_rgb,
    _darken_ink_gray,
    _unsharp,
)

log = logging.getLogger(__name__)


_MODE_PARAMS: dict[str, dict] = {
    "Clean Scan": dict(
        shading_flatten=1.0,          # divisive shadow flatten strength (0=off, 1=full)
        bilateral_strength=20,        # edge-preserving denoise before threshold
        block_size=49, threshold_C=10,
        close_kernel=1, open_kernel=1,
        remove_blobs_above=0.008,     # erase solid blobs > this fraction of frame
        speckle_below_ratio=0.025,    # erase tiny components (per-mille of short²/1000)
        mask_blur=0.25,
        recovery=0.0, recovery_floor=0.0,
        ink_gamma=0.65, ink_floor=0.16,
        keep_color=False,
        sharp_amt=1.0,
        document_mask=True,
        boundary_clean=0.018,
        frame_clean=True,
        binary_output=True,
        binary_alpha_threshold=None,
        stain_suppress=0.85,
    ),
    "Ink Recovery": dict(
        # shading_flatten raised from 0.50 — at 50% fold shadows are only half-removed,
        # still enough to confuse the adaptive threshold into treating them as ink.
        shading_flatten=0.72,
        bilateral_strength=16,
        block_size=65, threshold_C=6,  # permissive — captures faint bleed-through text
        close_kernel=2, open_kernel=1,
        remove_blobs_above=0.002,
        speckle_below_ratio=0.008,    # gentle — protects small fragments of faint letters
        mask_blur=0.6,
        recovery=0.65, recovery_floor=8.0,
        ink_gamma=0.45, ink_floor=0.03,
        keep_color=False,
        sharp_amt=0.7,
        document_mask=True,
        boundary_clean=0.006,
        frame_clean=False,
        binary_output=False,
        binary_alpha_threshold=None,
        stain_suppress=0.72,
    ),
}
_MODE_ALIASES = {
    "Standard": "Clean Scan",
    "Heavy Damage": "Clean Scan",
}
_DEFAULT_MODE = "Clean Scan"
_DEFAULT_PARAMS = _MODE_PARAMS[_DEFAULT_MODE]


def _has_only_uneven_tone(lighting: dict | None) -> bool:
    issues = (lighting or {}).get("issues", [])
    return bool(issues) and all(issue == "uneven paper tone" for issue in issues)


def _has_hard_lighting(lighting: dict | None) -> bool:
    issues = (lighting or {}).get("issues", [])
    return any(
        "underexposed" in issue
        or "deep shadows" in issue
        or "glare" in issue
        for issue in issues
    )


def _illtr_was_applied(stage3_meta: dict | None) -> bool:
    illumination = (stage3_meta or {}).get("illumination", {})
    return bool(illumination.get("applied"))


def _is_partial_quality_gate(qg_details: dict | None) -> bool:
    partial = (qg_details or {}).get("partial", {})
    return bool(partial.get("is_partial"))


def postprocess(
    image: np.ndarray,
    mode: str = _DEFAULT_MODE,
    lighting: dict | None = None,
    stage3_meta: dict | None = None,
    qg_details: dict | None = None,
) -> Image.Image:
    requested_mode = mode
    mode = _MODE_ALIASES.get(mode, mode)
    p = _MODE_PARAMS.get(mode, _DEFAULT_PARAMS).copy()
    in_h, in_w = image.shape[:2]
    img = image.astype(np.uint8)
    document_mask = _document_mask(img) if p.get("document_mask") else None

    # Flip dark panels (white-on-navy) to dark-on-light before anything else so the
    # whole document is polarity-consistent; otherwise the ink mask reads the dark
    # panel as a solid black blob.
    img, polarity_flipped = _normalize_panel_polarity(img, document_mask)
    if polarity_flipped and p.get("document_mask"):
        # After a navy/graphic panel is inverted, Otsu can lock onto only that
        # newly bright panel and crop away the normal white half of the flyer.
        # Use the full frame for this rare path so both polarities survive the
        # same ink cleanup pass.
        document_mask = np.full((in_h, in_w), 255, dtype=np.uint8)

    partial_clean_scan = (
        requested_mode == "Clean Scan" and _is_partial_quality_gate(qg_details)
    )
    mixed_graphic = _looks_mixed_graphic_page(img, document_mask)
    mixed_tone_only = mixed_graphic and _has_only_uneven_tone(lighting)
    faint_typewritten = _looks_faint_typewritten_page(img, document_mask)
    lined_handwritten = _looks_lined_handwritten_page(img, document_mask)
    dense_print = _looks_dense_print_page(img, document_mask)
    illtr_text_cleanup = (
        mode == "Clean Scan"
        and _illtr_was_applied(stage3_meta)
        and _has_hard_lighting(lighting)
        and dense_print
        and not mixed_graphic
    )

    if illtr_text_cleanup:
        p.update(
            # IllTr already handled the broad lighting, so keep Stage 4 focused on
            # residual cleanup while preserving weak printed strokes.
            shading_flatten=0.25,
            stain_suppress=0.45,
            bilateral_strength=20,
            block_size=49,
            threshold_C=8,
            remove_blobs_above=0.012,
            speckle_below_ratio=0.006,
            mask_blur=0.0,
            recovery=0.0,
            recovery_floor=0.0,
            sharp_amt=0.6,
            boundary_clean=0.050,
            frame_clean=True,
            outer_residual_clean=True,
            preserve_edge_text=False,
            binary_output=True,
        )
        mode = "Clean Scan+IllTrText"
    elif mode == "Clean Scan" and faint_typewritten:
        # Strict binary scan drops typewriter letters; full Ink Recovery keeps fold texture.
        # Small recovery + snap-to-black keeps letters without promoting crease shadows.
        p.update(
            threshold_C=7,
            speckle_below_ratio=0.006,
            remove_blobs_above=0.012,
            boundary_clean=0.008,
            recovery=0.14,
            recovery_floor=12.0,
            soft_typewriter_output=True,
            soft_floor=18.0,
            soft_scale=40.0,
            soft_gamma=0.42,
            soft_ink_floor=0.02,
            binary_output=False,
        )
        mode = "Clean Scan+FaintText"
    elif mode == "Clean Scan" and lined_handwritten:
        p.update(
            threshold_C=7,
            speckle_below_ratio=0.008,
            recovery=0.22,
            recovery_floor=10.0,
            boundary_clean=0.008,
            binary_alpha_threshold=0.14,
        )
        mode = "Clean Scan+Handwriting"
    elif mode == "Clean Scan" and dense_print:
        p.update(
            threshold_C=8,
            speckle_below_ratio=0.006,
            remove_blobs_above=0.012,
            boundary_clean=0.006,
            preserve_edge_text=True,
        )
        mode = "Clean Scan+DenseText"
    elif mode == "Clean Scan" and mixed_graphic:
        p.update(
            block_size=65,
            threshold_C=16 if mixed_tone_only else 8,
            remove_blobs_above=0.012,
            speckle_below_ratio=0.040 if mixed_tone_only else 0.006,
            mask_blur=0.0 if mixed_tone_only else 0.35,
            recovery=0.0 if mixed_tone_only else 0.20,
            recovery_floor=14.0 if mixed_tone_only else 9.0,
            ink_gamma=0.55,
            ink_floor=0.02,
            sharp_amt=0.7,
            boundary_clean=0.006,
            preserve_edge_text=True,
            light_text_recovery=True,
            binary_output=mixed_tone_only,
        )
        mode = "Clean Scan+MixedTone" if mixed_tone_only else "Clean Scan+MixedText"
    elif mode == "Ink Recovery" and faint_typewritten:
        p.update(
            shading_flatten=1.0,
            soft_typewriter_output=True,
            soft_floor=6.0,
            soft_scale=45.0,
            soft_gamma=0.42,
            soft_ink_floor=0.02,
            binary_output=False,
        )
        mode = "Ink Recovery+FaintText"
    elif mode == "Ink Recovery" and mixed_graphic:
        p.update(
            block_size=65,
            threshold_C=16 if mixed_tone_only else 8,
            remove_blobs_above=0.012,
            speckle_below_ratio=0.040 if mixed_tone_only else 0.006,
            mask_blur=0.0 if mixed_tone_only else 0.45,
            recovery=0.0 if mixed_tone_only else 0.20,
            recovery_floor=14.0 if mixed_tone_only else 9.0,
            ink_gamma=0.55,
            ink_floor=0.02,
            sharp_amt=0.7,
            boundary_clean=0.006,
            preserve_edge_text=True,
            light_text_recovery=True,
            binary_output=mixed_tone_only,
        )
        mode = "Ink Recovery+MixedTone" if mixed_tone_only else "Ink Recovery+MixedText"

    if partial_clean_scan and mode in {"Clean Scan", "Clean Scan+DenseText"}:
        p.update(threshold_C=7)
        mode = (
            "Clean Scan+PartialText"
            if mode == "Clean Scan" else f"{mode}+PartialText"
        )

    source_img = img.copy() if p.get("light_text_recovery", False) else None

    if p.get("stain_suppress", 0.0) > 0.0:
        img = _suppress_stains(img, p["stain_suppress"])

    if p.get("shading_flatten", 0.0) > 0.0:
        img = _shading_flatten(img, p["shading_flatten"])

    smooth = _bilateral(img, p["bilateral_strength"])

    if p.get("soft_typewriter_output", False):
        out, ink_alpha = _render_soft_typewriter(smooth, p)
        if p["sharp_amt"] > 0:
            out = _unsharp(out, p["sharp_amt"])
        log.info(
            "postprocess mode=%s requested=%s ink_frac=%.3f binary=%s",
            mode, requested_mode, float((ink_alpha > 0.5).mean()),
            p.get("binary_output", False),
        )
        if out.shape[0] != in_h or out.shape[1] != in_w:
            out = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(out)

    raw_ink_mask = _adaptive_ink_mask(
        smooth, block_size=p["block_size"], C=p["threshold_C"],
    )
    edge_text_rescue = None
    if document_mask is not None and p.get("preserve_edge_text", False):
        edge_text_rescue = _recover_masked_edge_text(
            raw_ink_mask, document_mask, smooth,
        )
    ink_mask = raw_ink_mask

    if document_mask is not None:
        ink_mask = cv2.bitwise_and(ink_mask, document_mask)
    if p["close_kernel"] > 0:
        ink_mask = _morph(ink_mask, cv2.MORPH_CLOSE, p["close_kernel"])
    if p["open_kernel"] > 0:
        ink_mask = _morph(ink_mask, cv2.MORPH_OPEN, p["open_kernel"])
    if p["remove_blobs_above"] > 0.0:
        ink_mask = _remove_large_blobs(ink_mask, p["remove_blobs_above"])
    preclean_ink_mask = ink_mask.copy()
    if p.get("speckle_below_ratio", 0.0) > 0.0:
        ink_mask = _remove_speckle(ink_mask, p["speckle_below_ratio"])
    if document_mask is not None and p.get("boundary_clean", 0.0) > 0.0:
        ink_mask = _remove_boundary_artifacts(
            ink_mask, document_mask, p["boundary_clean"],
            preserve_text=p.get("preserve_edge_text", False),
        )
    if p.get("frame_clean", False):
        ink_mask = _remove_frame_artifacts(
            ink_mask,
            preserve_text=p.get("preserve_edge_text", False),
        )
    if p.get("outer_residual_clean", False):
        ink_mask = _remove_outer_residuals(ink_mask, document_mask)
    if requested_mode == "Clean Scan" and _is_partial_quality_gate(qg_details):
        ink_mask = _recover_partial_visible_text(
            preclean_ink_mask, ink_mask, document_mask,
        )
    # Done last so boundary/frame cleaners can't re-delete rescued corner text.
    if edge_text_rescue is not None:
        ink_mask = cv2.bitwise_or(ink_mask, edge_text_rescue)

    ink_alpha = ink_mask.astype(np.float32) / 255.0
    if p["recovery"] > 0.0:
        recovered_alpha = _add_recovery_layer(
            ink_alpha, smooth,
            strength=p["recovery"], floor=p["recovery_floor"],
        )
        if p.get("light_text_recovery", False):
            recovery_region = _light_text_recovery_region(
                source_img, document_mask,
            )
            extra_alpha = np.clip(recovered_alpha - ink_alpha, 0.0, 1.0)
            ink_alpha = np.clip(
                ink_alpha + extra_alpha * recovery_region, 0.0, 1.0,
            )
        else:
            ink_alpha = recovered_alpha

    if p["mask_blur"] > 0:
        k = max(3, int(round(p["mask_blur"] * 4)) | 1)
        ink_alpha = cv2.GaussianBlur(ink_alpha, (k, k), p["mask_blur"])

    hard_threshold = p.get("binary_alpha_threshold")
    if hard_threshold is not None:
        ink_alpha = np.where(ink_alpha >= hard_threshold, 1.0, 0.0).astype(
            np.float32,
        )

    alpha = ink_alpha[..., None]
    if p.get("binary_output"):
        out_gray = np.clip(255.0 * (1.0 - ink_alpha), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)
    else:
        if p["keep_color"]:
            ink_layer = _darken_ink_rgb(img, p["ink_gamma"], p["ink_floor"])
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ink_layer = cv2.cvtColor(
                _darken_ink_gray(gray, p["ink_gamma"], p["ink_floor"]),
                cv2.COLOR_GRAY2RGB,
            )

        out = ink_layer.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        out = np.clip(out, 0, 255).astype(np.uint8)

    if p["sharp_amt"] > 0:
        out = _unsharp(out, p["sharp_amt"])

    log.info(
        "postprocess mode=%s requested=%s ink_frac=%.3f binary=%s",
        mode, requested_mode, float((ink_alpha > 0.5).mean()),
        p.get("binary_output", False),
    )

    if out.shape[0] != in_h or out.shape[1] != in_w:
        out = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(out)
