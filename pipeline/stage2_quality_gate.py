"""Stage 2 quality gate: brightness floor → damage precondition → content confidence → HIGH/MEDIUM/LOW/PARTIAL/NONE."""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

THRESHOLDS = {
    # brightness_floor: 40 lets shadowed folds pass; near-black destroyed pages fail.
    "brightness_floor":     40,

    # damage_reject: product of normalized noise × wide-dark; BOTH must elevate to reject,
    # so crumpled-but-clean photos (modest on each axis) still pass.
    "noise_normalizer":     20.0,   # Otsu bright-class σ; clean ≈ 8–18, moldy ≈ 25+
    "widefrac_normalizer":  0.30,   # wide-dark fraction; clean ≈ 0.10–0.30, moldy ≈ 0.35+
    "damage_reject":        2.50,

    "confidence_high":      0.70,
    "confidence_medium":    0.45,
    "confidence_low":       0.25,

    # Calibrated from datasets/user_papers + run.log OCR outcomes.
    "edge_density_full":    0.10,   # Measures edge/details in an image
    "text_area_full":       0.13, # Measures how much of the image looks like text-shaped ink
    "component_count_full": 8000.0, # Counts small connected components, like letters, words, punctations, strokes
    "contrast_floor":       18.0, # Measure grayscale contrast (using gray.std())
    "contrast_span":        42.0, # Measure grayscale conttrast also (using gray.std())

    "partial_dark_max":      110, # Pixels darker than or equal to 110 are considered dark candidates for a large obstruction.
    "partial_region_min":    0.35,# If one broad dark region covers at least 35% of the image, it may be marked partial.


    "partial_region_std_max": 16.0, # Checks if that dark region is flat/smooth. A smooth dark mass is likely obstruction, black paper, shadow, or covered area.
}


def quality_gate(image: np.ndarray) -> tuple[str, float, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    
    mean_brightness = float(gray.mean())
    if mean_brightness < THRESHOLDS["brightness_floor"]:
        log.info("rejected: brightness %.1f < %s",
                 mean_brightness, THRESHOLDS["brightness_floor"])
        return "NONE", 0.0, {
            "mean_brightness": mean_brightness,
            "reject_reason": "brightness",
        }

    features = _content_features(gray)
    qg_details = {
        "mean_brightness": mean_brightness,
        "features": features,
    }

    damage = _damage_score(gray)
    qg_details["damage"] = damage
    if damage >= THRESHOLDS["damage_reject"]:
        if _is_salvage_text_candidate(features):
            confidence = min(_content_confidence(features), 0.55)
            qg_details["confidence"] = confidence
            qg_details["salvage_text"] = {
                "enabled": True,
                "reason": "damage_with_text_like_fragments",
            }
            log.info(
                "partial salvage: damage score %.2f >= %s, text fragments present",
                damage, THRESHOLDS["damage_reject"],
            )
            return "PARTIAL", confidence, qg_details

        log.info("rejected: damage score %.2f >= %s (mold / staining)",
                 damage, THRESHOLDS["damage_reject"])
        return "NONE", 0.0, {
            **qg_details,
            "reject_reason": "damage",
        }

    if not features["has_edges"]:
        qg_details["reject_reason"] = "no_edges"
        return "NONE", 0.0, qg_details

    confidence = _content_confidence(features)
    qg_details["confidence"] = confidence

    if confidence < THRESHOLDS["confidence_low"]:
        log.info("rejected: content confidence %.2f < %s",
                 confidence, THRESHOLDS["confidence_low"])
        qg_details["reject_reason"] = "low_confidence"
        return "NONE", 0.0, qg_details

    partial = _partial_obscuration_features(gray)
    qg_details["partial"] = partial

    if partial["is_partial"]:
        log.info(
            "partial: broad dark region %.0f%%, interior std %.1f",
            partial["region_fraction"] * 100.0,
            partial["interior_std"],
        )
        return "PARTIAL", confidence, qg_details

    if confidence >= THRESHOLDS["confidence_high"]:
        return "HIGH", confidence, qg_details
    if confidence >= THRESHOLDS["confidence_medium"]:
        return "MEDIUM", confidence, qg_details
    return "LOW", confidence, qg_details


def _content_confidence(features: dict[str, float | bool]) -> float:
    confidence = (
        0.10
        + 0.30 * float(features["edge_score"])
        + 0.25 * float(features["text_area_score"])
        + 0.20 * float(features["component_score"])
        + 0.10 * float(features["contrast_score"])
    )
    return min(float(confidence), 1.0)


def _is_salvage_text_candidate(features: dict[str, float | bool]) -> bool:
    """Damaged pages can still contain OCR-worthy fragments; no edges/text stays NONE."""
    return (
        bool(features["has_edges"])
        and float(features["edge_score"]) >= 0.20
        and (
            float(features["text_area_score"]) >= 0.08
            or float(features["component_score"]) >= 0.35
        )
    )

#This function checks if the paper half is covered, a big shadow blocks part of the page, part of the image is black/dark, a document is only partially visible
def _partial_obscuration_features(gray: np.ndarray) -> dict[str, float | bool]:
    """Detect a single large, flat dark mass — accepted as PARTIAL rather than rejected."""
    h, w = gray.shape
    image_area = float(h * w)

    dark = (gray <= THRESHOLDS["partial_dark_max"]).astype(np.uint8) * 255
    kernel_size = max(31, (min(h, w) // 20) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
    )
    broad_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (broad_dark > 0).astype(np.uint8), connectivity=8,
    )
    if n_labels <= 1:
        return {
            "is_partial": False,
            "region_fraction": 0.0,
            "interior_std": 0.0,
        }

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region_fraction = float(stats[largest_idx, cv2.CC_STAT_AREA]) / image_area

    region = labels == largest_idx
    erode_size = max(15, (min(h, w) // 80) | 1)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size, erode_size),
    )
    interior = cv2.erode(region.astype(np.uint8), erode_kernel) > 0
    if int(np.count_nonzero(interior)) < image_area * 0.01:
        interior = region

    interior_std = (
        float(gray[interior].std()) if int(np.count_nonzero(interior)) else 0.0
    )
    is_partial = (
        region_fraction >= THRESHOLDS["partial_region_min"]
        and interior_std <= THRESHOLDS["partial_region_std_max"]
    )
    return {
        "is_partial": bool(is_partial),
        "region_fraction": region_fraction,
        "interior_std": interior_std,
    }


def _content_features(gray: np.ndarray) -> dict[str, float | bool]:
    """Count text-shaped adaptive-threshold components; largest-contour metrics underestimate fragmented text."""
    h, w = gray.shape
    image_area = float(h * w)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / image_area

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    has_edges = bool(contours)

    ink = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        49, 8,
    )
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8,
    )

    component_count = 0
    text_area = 0.0
    max_component_area = image_area * 0.01
    max_component_width = w * 0.35
    max_component_height = h * 0.08
    
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        bb_w = float(stats[i, cv2.CC_STAT_WIDTH])
        bb_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if (
            2.0 <= area <= max_component_area
            and bb_w <= max_component_width
            and bb_h <= max_component_height
        ):
            component_count += 1
            text_area += area

    text_area_ratio = text_area / image_area
    contrast = float(gray.std())

    edge_score = np.clip(
        edge_density / THRESHOLDS["edge_density_full"], 0.0, 1.0,
    )
    text_area_score = np.clip(
        text_area_ratio / THRESHOLDS["text_area_full"], 0.0, 1.0,
    )
    component_score = np.clip(
        np.log1p(component_count) / np.log1p(THRESHOLDS["component_count_full"]),
        0.0, 1.0,
    )
    contrast_score = np.clip(
        (contrast - THRESHOLDS["contrast_floor"])
        / THRESHOLDS["contrast_span"],
        0.0, 1.0,
    )

    return {
        "has_edges": has_edges,
        "edge_score": float(edge_score),
        "text_area_score": float(text_area_score),
        "component_score": float(component_score),
        "contrast_score": float(contrast_score),
    }


def _damage_score(gray: np.ndarray) -> float:
    h, w = gray.shape

    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light = gray[gray >= t]
    paper_noise = float(light.std()) if light.size else 0.0

    big = max(51, (min(h, w) // 6) | 1) 
    wide_dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, big, 10,
    )
    wide_frac = float((wide_dark > 0).mean())

    return (paper_noise / THRESHOLDS["noise_normalizer"]) * (wide_frac / THRESHOLDS["widefrac_normalizer"])
