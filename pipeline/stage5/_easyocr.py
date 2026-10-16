"""Stage 5 EasyOCR engine — fallback when Apple Vision is unavailable."""

import logging
from threading import Lock

import numpy as np

from pipeline.constants import (
    EASY_TARGET_SHORT_DIM as _EASY_TARGET_SHORT_DIM,
)
from ._format import _normalize_lang, _format_paragraphs, _enhance_for_ocr, _clean_for_ocr, _scale_for_short_dim

log = logging.getLogger(__name__)

_EASY_MAX_SHORT_DIM = 2400
_EASY_MIN_CONF = 0.3  # below this, EasyOCR detections are almost always texture/noise

_easy_reader = None
_easy_reader_lock = Lock()


def _ocr_easyocr_best(images: list[np.ndarray], lang: str) -> str:
    best_text = ""
    best_score = -1.0
    for arr in images:
        text = _ocr_easyocr(arr, lang)
        score = len("".join(text.split()))
        if score > best_score:
            best_text = text
            best_score = score
    return best_text


def _ocr_easyocr(arr: np.ndarray, lang: str) -> str:
    reader = _easyocr_get_reader(_normalize_lang(lang))
    if reader is None:
        return ""

    arr = _scale_for_ocr(arr)
    enhanced = _enhance_for_ocr(arr)
    cleaned = _clean_for_ocr(arr)

    base_kwargs = dict(
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        text_threshold=0.45,
        low_text=0.3,
        link_threshold=0.4,
        add_margin=0.15,
        canvas_size=2560,
    )
    passes = [
        ("clean-1.5", cleaned,  {**base_kwargs, "mag_ratio": 1.5}),
        ("enh-1.5",   enhanced, {**base_kwargs, "mag_ratio": 1.5}),
    ]

    merged: list[dict] = []
    pass_stats = []
    for label, img, kwargs in passes:
        try:
            boxes = reader.readtext(img, **kwargs)
        except Exception as exc:
            log.warning("ocr: easyocr call failed on %s image: %s", label, exc)
            boxes = []
        mean_conf = float(np.mean([float(c) for _, _, c in boxes])) if boxes else 0.0
        pass_stats.append((label, len(boxes), round(mean_conf, 2)))
        _absorb_boxes(merged, boxes)

    text = _format_paragraphs(merged, min_conf=_EASY_MIN_CONF)
    log.info(
        "ocr: easyocr extracted %d chars (passes=%s, merged=%d) from %dx%d image",
        len(text), pass_stats, len(merged), arr.shape[1], arr.shape[0],
    )
    return text


def _easyocr_get_reader(lang_code: str):
    global _easy_reader
    if _easy_reader is not None:
        return _easy_reader
    with _easy_reader_lock:
        if _easy_reader is not None:
            return _easy_reader
        try:
            import easyocr
        except ImportError:
            log.warning("ocr: easyocr fallback unavailable — pip install easyocr")
            return None
        try:
            log.info("ocr: initializing easyocr reader (lang=%s)", lang_code)
            _easy_reader = easyocr.Reader([lang_code], gpu=False, verbose=False)
        except Exception as exc:
            log.warning("ocr: easyocr init failed: %s", exc)
            return None
    return _easy_reader


def _absorb_boxes(merged: list[dict], new_boxes, iou_thresh: float = 0.4) -> None:
    """Merge detections; spatial duplicates (IoU >= threshold) keep higher-confidence result."""
    for bbox, text, conf in new_boxes:
        xyxy = _box_xyxy(bbox)
        conf_f = float(conf)
        match_idx = -1
        best_iou = 0.0
        for i, o in enumerate(merged):
            iou = _iou(xyxy, o["xyxy"])
            if iou > best_iou:
                best_iou = iou
                match_idx = i
        if best_iou >= iou_thresh and match_idx >= 0:
            if conf_f > merged[match_idx]["conf"]:
                merged[match_idx] = {"xyxy": xyxy, "text": text, "conf": conf_f}
        else:
            merged.append({"xyxy": xyxy, "text": text, "conf": conf_f})


def _box_xyxy(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _scale_for_ocr(arr: np.ndarray) -> np.ndarray:
    return _scale_for_short_dim(arr, _EASY_TARGET_SHORT_DIM, _EASY_MAX_SHORT_DIM)
