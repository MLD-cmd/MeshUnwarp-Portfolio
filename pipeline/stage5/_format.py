"""Stage 5 formatting — text layout reconstruction and shared image preprocessing."""

import cv2
import numpy as np

_LINE_CLUSTER_FRAC = 0.4  # tight enough to keep adjacent lines separate, loose enough for mild tilt
_PARAGRAPH_GAP_FRAC = 1.35
_COLUMN_GAP_FRAC = 2.4
_MAX_INSERTED_SPACES = 10


def _normalize_lang(lang: str) -> str:
    return "en" if lang in ("eng", "en") else lang


def _scale_for_short_dim(arr: np.ndarray, target: int, maximum: int) -> np.ndarray:
    h, w = arr.shape[:2]
    short = min(h, w)
    if short < target:
        scale = target / short
    elif short > maximum:
        scale = maximum / short
    else:
        return arr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def _enhance_for_ocr(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)


def _clean_for_ocr(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    gray_f = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(15.0, min(h, w) * 0.04)
    bg = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    paper_level = float(np.percentile(bg, 95))
    if paper_level >= 1.0:
        gain = np.clip(paper_level / np.maximum(bg, 1.0), 1.0, 1.6)
        flat_rgb = np.clip(arr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    else:
        flat_rgb = arr

    denoised = cv2.bilateralFilter(flat_rgb, d=7, sigmaColor=25, sigmaSpace=25)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    bg2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    paper2 = float(np.percentile(bg2, 95))
    threshold = max(0.0, paper2 - 6.0)
    out = gray.astype(np.float32)
    span = max(1.0, paper2 - threshold)
    lift = np.clip((out - threshold) / span, 0.0, 1.0)
    out = out * (1.0 - lift) + 255.0 * lift
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def _format_paragraphs(boxes, min_conf: float):
    items = [
        b for b in boxes
        if b["text"] and b["text"].strip() and b["conf"] >= min_conf
    ]
    if not items:
        return ""

    items.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    median_h = float(np.median([b["xyxy"][3] - b["xyxy"][1] for b in items])) or 1.0

    lines: list[list[dict]] = []
    line_thresh = _LINE_CLUSTER_FRAC * median_h
    for b in items:
        placed = False
        for line in lines:
            if _same_text_line(line, b, line_thresh):
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    line_records = []
    for line in lines:
        line.sort(key=lambda b: b["xyxy"][0])
        text = _join_line_boxes(line)
        if not text:
            continue
        top = min(b["xyxy"][1] for b in line)
        bottom = max(b["xyxy"][3] for b in line)
        line_records.append({"text": text, "top": top, "bottom": bottom})

    line_records.sort(key=lambda r: r["top"])

    sections: list[list[str]] = []
    para_gap = _PARAGRAPH_GAP_FRAC * median_h
    for i, rec in enumerate(line_records):
        if i == 0:
            sections.append([rec["text"]])
            continue
        prev = line_records[i - 1]
        if (rec["top"] - prev["bottom"]) > para_gap:
            sections.append([rec["text"]])
        else:
            sections[-1].append(rec["text"])

    return "\n\n".join("\n".join(section) for section in sections).strip()


def _same_text_line(line: list[dict], box: dict, line_thresh: float) -> bool:
    y1, y2 = box["xyxy"][1], box["xyxy"][3]
    cy = (y1 + y2) / 2.0
    line_y1 = min(o["xyxy"][1] for o in line)
    line_y2 = max(o["xyxy"][3] for o in line)
    line_cy = (line_y1 + line_y2) / 2.0

    overlap = max(0.0, min(y2, line_y2) - max(y1, line_y1))
    min_h = max(1.0, min(y2 - y1, line_y2 - line_y1))
    return (overlap / min_h) >= 0.45 or abs(cy - line_cy) <= line_thresh


def _join_line_boxes(line: list[dict]) -> str:
    pieces: list[str] = []
    char_widths = []
    for b in line:
        text = _clean_ocr_box_text(b["text"])
        if not text:
            continue
        x1, _, x2, _ = b["xyxy"]
        char_widths.append(max(1.0, (x2 - x1) / max(len(text), 1)))

    char_w = float(np.median(char_widths)) if char_widths else 8.0
    prev_box = None
    prev_text = ""
    for b in line:
        text = _clean_ocr_box_text(b["text"])
        if not text:
            continue
        if prev_box is None:
            pieces.append(text)
        else:
            gap = b["xyxy"][0] - prev_box["xyxy"][2]
            pieces.append(_gap_to_spaces(gap, char_w, prev_text, text))
            pieces.append(text)
        prev_box = b
        prev_text = text
    return "".join(pieces).strip()


def _clean_ocr_box_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def _gap_to_spaces(gap: float, char_w: float, left: str, right: str) -> str:
    if not left or not right:
        return ""
    if right[0] in ".,:;!?)]}%" or left[-1] in "([{":
        return ""
    if gap <= char_w * _COLUMN_GAP_FRAC:
        return " "
    spaces = int(round(gap / max(char_w, 1.0)))
    return " " * max(2, min(spaces, _MAX_INSERTED_SPACES))
