"""Stage 5 Apple Vision engine — primary OCR on M3 Neural Engine."""

import logging
from threading import Lock

import cv2
import numpy as np

from pipeline.constants import (
    VISION_MIN_CONF          as _VISION_MIN_CONF,
    VISION_TARGET_SHORT_DIM  as _VISION_TARGET_SHORT_DIM,
    VISION_MAX_SHORT_DIM     as _VISION_MAX_SHORT_DIM,
    VISION_FAST_ACCEPT_CHARS as _VISION_FAST_ACCEPT_CHARS,
)
from ._format import _normalize_lang, _format_paragraphs, _enhance_for_ocr, _clean_for_ocr, _scale_for_short_dim

log = logging.getLogger(__name__)

_vision_warmed = False
_vision_warm_lock = Lock()
_vision_runtime_ok: bool | None = None


def _vision_can_perform() -> bool:
    global _vision_runtime_ok
    if _vision_runtime_ok is not None:
        return _vision_runtime_ok
    try:
        import Vision
        probe = np.full((96, 240, 3), 255, dtype=np.uint8)
        cv2.putText(
            probe, "TEST", (16, 62), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cg = _np_to_cgimage(probe)
        req = _vision_make_request("en")
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg, None,
        )
        ok, _err = handler.performRequests_error_([req], None)
        _vision_runtime_ok = bool(ok)
    except Exception as exc:
        log.warning("ocr: Apple Vision preflight raised: %s", exc)
        _vision_runtime_ok = False
    return _vision_runtime_ok


def _ocr_vision(images: list[np.ndarray], lang: str) -> str:
    best = {"text": "", "score": -1.0, "label": "", "obs": 0, "mean_conf": 0.0}

    for source_idx, source in enumerate(images):
        source_label = "primary" if source_idx == 0 else f"extra{source_idx}"
        raw = _scale_for_vision(source)
        text, obs, mean_conf = _ocr_vision_once(raw, lang)
        score = _ocr_text_score(text, obs, mean_conf)
        if score > best["score"]:
            best = {
                "text": text,
                "score": score,
                "label": f"{source_label}:raw",
                "obs": obs,
                "mean_conf": mean_conf,
            }

    if len(best["text"].replace(" ", "").replace("\n", "")) >= _VISION_FAST_ACCEPT_CHARS:
        log.info(
            "ocr: vision selected %s (%d chars, %d obs, mean_conf=%.2f)",
            best["label"], len(best["text"]), best["obs"], best["mean_conf"],
        )
        return best["text"]

    for source_idx, source in enumerate(images):
        source_label = "primary" if source_idx == 0 else f"extra{source_idx}"
        variants = (
            ("clean", _clean_for_ocr(source)),
            ("enhanced", _enhance_for_ocr(source)),
        )
        for variant_label, variant in variants:
            text, obs, mean_conf = _ocr_vision_once(
                _scale_for_vision(variant), lang,
            )
            score = _ocr_text_score(text, obs, mean_conf)
            if score > best["score"]:
                best = {
                    "text": text,
                    "score": score,
                    "label": f"{source_label}:{variant_label}",
                    "obs": obs,
                    "mean_conf": mean_conf,
                }

    log.info(
        "ocr: vision selected %s (%d chars, %d obs, mean_conf=%.2f)",
        best["label"], len(best["text"]), best["obs"], best["mean_conf"],
    )
    return best["text"]


def _ocr_vision_once(arr: np.ndarray, lang: str) -> tuple[str, int, float]:
    import Vision

    h, w = arr.shape[:2]
    cg = _np_to_cgimage(arr)

    req = _vision_make_request(lang)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        log.warning("ocr: vision performRequests failed: %s", err)
        return "", 0, 0.0

    observations = req.results() or []

    items: list[dict] = []
    for obs in observations:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        cand = cands[0]
        text = (cand.string() or "").replace("\n", " ").strip()
        if not text:
            continue
        conf = float(cand.confidence())

        # Vision bboxes: normalized [0,1], origin at bottom-left → convert to pixel xyxy top-left.
        bb = obs.boundingBox()
        x1 = bb.origin.x * w
        x2 = (bb.origin.x + bb.size.width) * w
        y1 = (1.0 - (bb.origin.y + bb.size.height)) * h
        y2 = (1.0 - bb.origin.y) * h
        items.append({"xyxy": (x1, y1, x2, y2), "text": text, "conf": conf})

    text = _format_paragraphs(items, min_conf=_VISION_MIN_CONF)
    mean_conf = float(np.mean([b["conf"] for b in items])) if items else 0.0
    log.info(
        "ocr: vision extracted %d chars (%d obs, mean_conf=%.2f) from %dx%d image",
        len(text), len(items), mean_conf, w, h,
    )
    return text, len(items), mean_conf


def _vision_make_request(lang: str):
    import Vision
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    code = "en-US" if _normalize_lang(lang) == "en" else lang
    req.setRecognitionLanguages_([code])
    if hasattr(req, "setMinimumTextHeight_"):
        req.setMinimumTextHeight_(0.01)
    return req


def _vision_prewarm() -> None:
    global _vision_warmed
    if _vision_warmed:
        return
    with _vision_warm_lock:
        if _vision_warmed:
            return
        import Vision
        log.info("ocr: prewarming Apple Vision")
        tiny = np.full((32, 32, 3), 255, dtype=np.uint8)
        cg = _np_to_cgimage(tiny)
        req = _vision_make_request("en")
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        handler.performRequests_error_([req], None)
        _vision_warmed = True


def _np_to_cgimage(arr: np.ndarray):
    import Quartz
    from Foundation import NSData

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = np.ascontiguousarray(arr)

    h, w = arr.shape[:2]
    data = NSData.dataWithBytes_length_(arr.tobytes(), arr.nbytes)
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
    return Quartz.CGImageCreate(
        w, h,
        8,      # bits per component
        24,     # bits per pixel (3 × 8)
        w * 3,  # bytes per row
        colorspace,
        Quartz.kCGImageAlphaNone | Quartz.kCGBitmapByteOrderDefault,
        provider,
        None, False,
        Quartz.kCGRenderingIntentDefault,
    )


def _scale_for_vision(arr: np.ndarray) -> np.ndarray:
    return _scale_for_short_dim(arr, _VISION_TARGET_SHORT_DIM, _VISION_MAX_SHORT_DIM)


def _ocr_text_score(text: str, obs_count: int, mean_conf: float) -> float:
    chars = len("".join(text.split()))
    return chars * max(mean_conf, 0.25) + obs_count * 3.0
