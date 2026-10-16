"""Stage 5 — OCR. Apple Vision primary (M3 Neural Engine); EasyOCR fallback."""

import logging
from threading import Lock

import numpy as np
from PIL import Image

from ._format import _normalize_lang
from ._vision import _ocr_vision, _vision_prewarm, _vision_can_perform
from ._easyocr import _ocr_easyocr_best, _easyocr_get_reader

log = logging.getLogger(__name__)

_engine: str | None = None  # "vision" | "easyocr" | None; resolved on first use
_engine_lock = Lock()


def _select_engine() -> str:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            import Vision   # noqa: F401
            import Quartz   # noqa: F401
            from Foundation import NSData  # noqa: F401
            if _vision_can_perform():
                _engine = "vision"
                log.info("ocr: using Apple Vision engine")
            else:
                log.warning(
                    "ocr: Apple Vision imports but request execution fails; "
                    "falling back to EasyOCR"
                )
                _engine = "easyocr"
        except ImportError as exc:
            log.warning(
                "ocr: pyobjc Vision unavailable (%s); falling back to EasyOCR",
                exc,
            )
            _engine = "easyocr"
        return _engine


def ocr(
    image: Image.Image | np.ndarray,
    lang: str = "en",
    extra_images: (
        list[Image.Image | np.ndarray]
        | tuple[Image.Image | np.ndarray, ...]
        | None
    ) = None,
) -> str:
    text, _meta = ocr_with_meta(image, lang=lang, extra_images=extra_images)
    return text


def ocr_with_meta(
    image: Image.Image | np.ndarray,
    lang: str = "en",
    extra_images: (
        list[Image.Image | np.ndarray]
        | tuple[Image.Image | np.ndarray, ...]
        | None
    ) = None,
) -> tuple[str, dict]:
    arr = _image_to_rgb_array(image)
    candidates = [arr]
    for extra in extra_images or ():
        candidates.append(_image_to_rgb_array(extra))

    engine = _select_engine()
    meta = {
        "engine": engine,
        "engine_label": "Apple Vision" if engine == "vision" else "EasyOCR",
        "chars": 0,
        "fallback_checked": False,
        "fallback_used": False,
    }
    try:
        if engine == "vision":
            text = _ocr_vision(candidates, lang)
            meta["chars"] = len(text)
            meta["selected"] = "vision"
            if len("".join(text.split())) >= 40:
                return text, meta
            fallback = _ocr_easyocr_best(candidates, lang)
            meta["fallback_checked"] = True
            meta["vision_chars"] = len(text)
            meta["easyocr_chars"] = len(fallback)
            if len(fallback) > len(text):
                meta.update({
                    "engine": "easyocr",
                    "engine_label": "EasyOCR",
                    "chars": len(fallback),
                    "selected": "easyocr",
                    "fallback_used": True,
                    "reason": "vision_sparse",
                })
                return fallback, meta
            meta["reason"] = "vision_sparse_but_kept"
            return text, meta
        text = _ocr_easyocr_best(candidates, lang)
        meta.update({"chars": len(text), "selected": "easyocr"})
        return text, meta
    except Exception as exc:
        log.warning("ocr: %s engine failed: %s", engine, exc)
        meta.update({"chars": 0, "error": str(exc)})
        return "", meta


def _image_to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def prewarm(lang: str = "en") -> None:
    engine = _select_engine()
    try:
        if engine == "vision":
            _vision_prewarm()
        else:
            _easyocr_get_reader(_normalize_lang(lang))
    except Exception as exc:
        log.warning("ocr: prewarm failed (%s): %s", engine, exc)
