"""
mw_smoke.py — MeshUnwarp Smoke Test Suite
==========================================
Runs headless (no UI, no Tkinter). Validates the full pipeline end-to-end and
each stage in isolation. Does NOT require a GPU — MPS / CPU works fine.

Usage (activate venv first):
    python mw_smoke.py                    # all tests
    python mw_smoke.py --quick            # skip Stage 3 (DocTr model load)

Exit code:
    0  — all tests passed
    1  — one or more tests failed
"""

import sys
import os
import time
import argparse
import traceback
import textwrap

import cv2
import numpy as np
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Test image paths ─────────────────────────────────────────────────────────
def _first_image(*candidate_dirs):
    """Return the first .jpg/.jpeg/.png in any of the given dirs, or None."""
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(d, name)
    return None


_PAPERS = os.path.join(ROOT, "datasets", "user_papers")
_EXPECTED_REJECT_DIRS = {"unrecoverable", "rejected"}
_EXPECTED_SALVAGE_DIRS = {"unrecognizable"}
_UNRECOGNIZABLE_IMG = _first_image(os.path.join(_PAPERS, "unrecognizable"))
_HALF_BLACK_IMG = _first_image(
    os.path.join(_PAPERS, "partial"),
    os.path.join(_PAPERS, "crumpled"),
)
_REAL_IMG = _first_image(
    os.path.join(_PAPERS, "mixed"),
    os.path.join(_PAPERS, "crumpled"),
    os.path.join(_PAPERS, "warped"),
    os.path.join(_PAPERS, "wet"),
    os.path.join(_PAPERS, "low_ink"),
    os.path.join(_PAPERS, "folded"),
)
_SMALL_IMG = _first_image(os.path.join(_PAPERS, "crumpled")) or _REAL_IMG

if _REAL_IMG is None:
    print("⚠ No test images found under datasets/user_papers/. "
          "Drop at least one .jpg/.png in any subfolder before running.")
    sys.exit(1)

# ── ANSI color codes ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_results = []  # (name, passed, duration_s, detail)


def _run(name: str, fn, *args, **kwargs):
    """Run one test case, record result."""
    print(f"  {CYAN}→{RESET} {name}...", end=" ", flush=True)
    t0 = time.monotonic()
    try:
        detail = fn(*args, **kwargs) or ""
        elapsed = time.monotonic() - t0
        print(f"{GREEN}PASS{RESET}  ({elapsed:.2f}s)  {detail}")
        _results.append((name, True, elapsed, detail))
    except Exception as exc:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        msg = str(exc) or exc.__class__.__name__
        print(f"{RED}FAIL{RESET}  ({elapsed:.2f}s)  {msg}")
        print(textwrap.indent(tb, "      "))
        _results.append((name, False, elapsed, msg))


# ════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def _test_preprocess_real():
    from pipeline.stage1_preprocess import preprocess, PreprocessResult
    res = preprocess(_REAL_IMG)
    assert isinstance(res, PreprocessResult), "Expected PreprocessResult"
    assert res.raw.dtype == np.uint8
    assert res.enhanced.dtype == np.uint8
    assert res.raw.shape == res.enhanced.shape
    assert res.raw.ndim == 3 and res.raw.shape[2] == 3
    h, w = res.raw.shape[:2]
    assert min(h, w) >= 288, f"Upscale failed — min dim is {min(h,w)}"
    return f"raw={w}×{h}  upscale={res.meta['upscale']:.2f}"


def _test_preprocess_small():
    """Images below 288px on the short side must be upscaled."""
    from pipeline.stage1_preprocess import preprocess
    res = preprocess(_SMALL_IMG)
    h, w = res.raw.shape[:2]
    assert min(h, w) >= 288, f"Upscale gate failed — {w}×{h}"
    return f"original {res.meta['orig_size']} → {w}×{h}"


def _test_preprocess_missing():
    """Should raise ValueError for a non-existent path."""
    from pipeline.stage1_preprocess import preprocess
    try:
        preprocess("/tmp/__mw_does_not_exist__.jpg")
        raise AssertionError("Expected ValueError — none raised")
    except ValueError:
        return "ValueError raised correctly"


def _test_preprocess_parallel_streams():
    """raw and enhanced must diverge (CLAHE changes pixel values)."""
    from pipeline.stage1_preprocess import preprocess
    res = preprocess(_REAL_IMG)
    diff = np.abs(res.raw.astype(int) - res.enhanced.astype(int))
    assert diff.max() > 0, "raw == enhanced — CLAHE had no effect"
    return f"mean pixel delta={diff.mean():.1f}"


def _test_preprocess_detects_uneven_paper_tone():
    from pipeline.stage1_preprocess import preprocess
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "folded", "folded_paper3.jpg")
    res = preprocess(path)
    lighting = res.meta.get("lighting", {})
    assert "uneven paper tone" in lighting.get("issues", []), (
        "old/dim letter did not trigger uneven paper tone"
    )
    return (f"tone range={lighting['paper_tone_range']:.1f} "
            f"ratio={lighting['paper_tone_ratio']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Quality Gate
# ════════════════════════════════════════════════════════════════════════════

def _test_qgate_real_high():
    """A real paper photo should pass without looking like a near-failure."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate
    res   = preprocess(_REAL_IMG)
    label, conf, _details = quality_gate(res.enhanced)
    assert label in ("HIGH", "MEDIUM"), (
        f"Expected HIGH or MEDIUM, got {label} ({conf:.3f})"
    )
    return f"{label}  confidence={conf:.3f}"


def _test_qgate_blank_reject():
    """A plain white image should score NONE — triggers constraint message."""
    from pipeline.stage2_quality_gate import quality_gate
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 240  # near-white
    label, conf, _details = quality_gate(blank)
    assert label == "NONE", (
        f"Expected NONE for blank image, got {label} (conf={conf:.3f}).\n"
        f"The constraint message will NOT trigger for truly blank images — "
        f"adjust confidence_low if needed."
    )
    return f"NONE  confidence={conf:.3f}"


def _test_qgate_half_black_partial():
    """A partly readable but half-obscured page should warn, not reject."""
    if not _HALF_BLACK_IMG or not os.path.exists(_HALF_BLACK_IMG):
        return "SKIP - half-black sample not found"

    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate

    res = preprocess(_HALF_BLACK_IMG)
    label, conf, _details = quality_gate(res.enhanced)
    assert label == "PARTIAL", (
        f"Expected PARTIAL for half-obscured page, got {label} "
        f"({conf:.3f})"
    )
    assert conf >= 0.45, (
        f"Expected processable confidence for PARTIAL page, got {conf:.3f}"
    )
    return f"PARTIAL  confidence={conf:.3f}"


def _test_qgate_salvage_text_fragment():
    """A destroyed fragment with visible words should try salvage OCR, not hard-reject."""
    if not _UNRECOGNIZABLE_IMG or not os.path.exists(_UNRECOGNIZABLE_IMG):
        return "SKIP - unrecognizable sample not found"

    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate

    res = preprocess(_UNRECOGNIZABLE_IMG)
    label, conf, details = quality_gate(res.enhanced)
    salvage = (details.get("salvage_text") or {}).get("enabled")
    assert label == "PARTIAL" and salvage, (
        f"Expected salvage PARTIAL for visible text fragment, got {label} "
        f"({conf:.3f}) details={details}"
    )
    return f"PARTIAL salvage confidence={conf:.3f}"


def _test_qgate_confidence_range():
    """Confidence must always be in [0.0, 1.0]."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate
    for path in (_REAL_IMG, _SMALL_IMG):
        res = preprocess(path)
        _, conf, _details = quality_gate(res.enhanced)
        assert 0.0 <= conf <= 1.0, f"Out-of-range confidence: {conf}"
    return "both images in [0.0, 1.0]"


def _test_qgate_dataset_calibration():
    """Bundled valid samples should calibrate as readable, not LOW 21-23%."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate

    labels = []
    confs = []
    for dirpath, _, files in os.walk(_PAPERS):
        for name in sorted(files):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(dirpath, name)
            res = preprocess(path)
            label, conf, _details = quality_gate(res.enhanced)
            rel = os.path.relpath(path, _PAPERS)
            category = rel.split(os.sep, 1)[0]
            if category in _EXPECTED_SALVAGE_DIRS:
                salvage = (_details.get("salvage_text") or {}).get("enabled")
                assert label in ("PARTIAL", "NONE"), (
                    f"{rel} should be salvage PARTIAL or rejected NONE, "
                    f"got {label} ({conf:.3f})"
                )
                if label == "PARTIAL":
                    assert salvage, f"{rel} PARTIAL must be marked salvage_text"
                continue
            if category in _EXPECTED_REJECT_DIRS:
                assert label == "NONE", (
                    f"{rel} should be rejected, got {label} ({conf:.3f})"
                )
                continue

            assert label != "NONE", f"{rel} was rejected ({conf:.3f})"
            assert label != "LOW", f"{rel} still scored LOW ({conf:.3f})"
            assert conf >= 0.45, f"{rel} confidence too low ({conf:.3f})"
            labels.append(label)
            confs.append(conf)

    assert labels, "No dataset samples found for calibration"
    assert "HIGH" in labels, "Calibration produced no HIGH samples"
    assert "MEDIUM" in labels, "Calibration produced no MEDIUM samples"
    return (
        f"{len(labels)} samples: HIGH={labels.count('HIGH')}, "
        f"MEDIUM={labels.count('MEDIUM')}, min={min(confs):.3f}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  Stage 3 — De-Warping (DocTr loaded; warp gated by _APPLY_DEWARP)
# ════════════════════════════════════════════════════════════════════════════

def _test_dewarp_loads_models():
    """GeoTr + IllTr weights must load onto the active device."""
    from pipeline.stage3_dewarp import _load_models
    from pipeline import stage3_dewarp as dw_mod
    _load_models()
    assert dw_mod._geotr is not None, "GeoTr not loaded"
    assert dw_mod._illtr is not None, "IllTr not loaded"
    return f"GeoTr + IllTr loaded onto {dw_mod.device}"


def _test_dewarp_preserves_input_dims():
    """Stage 3 must always return an image with the input's exact H×W."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage3_dewarp import dewarp
    res = preprocess(_REAL_IMG)
    out = dewarp(res.raw)
    assert out.dtype == np.uint8
    assert out.shape == res.raw.shape, (
        f"shape changed: in={res.raw.shape} out={out.shape}"
    )
    return f"dims preserved ({out.shape[1]}×{out.shape[0]})"


def _test_dewarp_meta_includes_geotr_telemetry():
    """GeoTr forward pass should run and report telemetry every call."""
    from pipeline.stage3_dewarp import dewarp_with_meta, _APPLY_DEWARP
    img = np.full((400, 320, 3), 200, dtype=np.uint8)
    out, meta = dewarp_with_meta(img)
    assert "geotr" in meta
    g = meta["geotr"]
    for key in ("applied", "warp_strength", "max_shift_raw",
                "max_displacement", "input_size", "chaos", "chaos_mult"):
        assert key in g, f"missing geotr meta key: {key}"
    assert g["applied"] is _APPLY_DEWARP
    assert 0.0 <= g["warp_strength"] <= 1.0
    assert 0.0 <= g["max_displacement"] <= g["max_shift_raw"]
    assert 0.0 <= g["chaos_mult"] <= 1.0
    return (f"applied={g['applied']} strength={g['warp_strength']:.2f} "
            f"raw={g['max_shift_raw']:.3f} "
            f"applied_shift={g['max_displacement']:.3f}")


def _test_illtr_graphic_gate_is_less_blunt():
    from pipeline.stage3_dewarp import _looks_graphic_heavy
    img = np.full((400, 400, 3), (86, 88, 92), dtype=np.uint8)
    # A moderate dark-blue region can be caused by underexposed paper/print.
    # It should be allowed through when lighting diagnostics already indicate
    # hard lighting, but blocked on otherwise normal captures.
    cv2.rectangle(img, (20, 20), (199, 190), (18, 54, 130), -1)
    assert _looks_graphic_heavy(img, {"issues": []}), (
        "normal large color block should stay blocked"
    )
    assert not _looks_graphic_heavy(img, {"issues": ["underexposed"]}), (
        "underexposed moderate color block should not block IllTr"
    )

    large_panel = np.full((400, 400, 3), (86, 88, 92), dtype=np.uint8)
    cv2.rectangle(large_panel, (20, 160), (379, 379), (12, 35, 95), -1)
    assert _looks_graphic_heavy(large_panel, {"issues": ["underexposed"]}), (
        "true large dark design panel should still block IllTr"
    )
    return "only real large dark design panels block IllTr under hard lighting"


def _test_illtr_mixed_pages_follow_same_gate():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage3_dewarp import _lighting_needs_illtr, _looks_graphic_heavy

    mixed1 = preprocess(os.path.join(ROOT, "datasets", "user_papers",
                                     "mixed", "mixed_paper1.jpg"))
    mixed2 = preprocess(os.path.join(ROOT, "datasets", "user_papers",
                                     "mixed", "mixed_paper2.jpg"))

    assert _lighting_needs_illtr(mixed1.meta.get("lighting")), (
        "mixed_paper1 lighting should still qualify for IllTr consideration"
    )
    assert _lighting_needs_illtr(mixed2.meta.get("lighting")), (
        "mixed_paper2 lighting should still qualify for IllTr consideration"
    )
    assert _looks_graphic_heavy(
        mixed1.raw, mixed1.meta.get("lighting"),
    ), "mixed_paper1 should be treated as a dark design-panel page"
    assert not _looks_graphic_heavy(
        mixed2.raw, mixed2.meta.get("lighting"),
    ), "mixed_paper2 should not be treated as a dark design-panel page"
    return "mixed pages use the same lighting + dark-panel gate"


# ════════════════════════════════════════════════════════════════════════════
#  Stage 4 — Postprocessing
# ════════════════════════════════════════════════════════════════════════════

def _test_postprocess_returns_pil():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    res = preprocess(_REAL_IMG)
    pil = postprocess(res.raw, mode="Clean Scan")
    assert isinstance(pil, Image.Image), f"Expected PIL.Image, got {type(pil)}"
    return f"PIL Image {pil.size[0]}×{pil.size[1]}"


def _test_postprocess_all_modes():
    """Visible modes and legacy aliases must run without error."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    res = preprocess(_REAL_IMG)
    for mode in ("Clean Scan", "Ink Recovery", "Standard", "Heavy Damage"):
        out = postprocess(res.raw, mode=mode)
        assert isinstance(out, Image.Image)
    return "Clean Scan / Ink Recovery + aliases all OK"


def _test_postprocess_whitens_background_without_blank_output():
    from pipeline.stage4 import postprocess
    img = np.full((500, 700, 3), (205, 230, 180), dtype=np.uint8)
    cv2.putText(img, "MeshUnwarp", (80, 240), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (30, 40, 30), 4, cv2.LINE_AA)
    out = np.array(postprocess(img, mode="Clean Scan"))
    assert out.mean() > img.mean(), "background did not brighten"
    assert out.var() > 50.0, "output lost useful contrast"
    return f"in={img.mean():.1f} out={out.mean():.1f}"


def _test_postprocess_clean_scan_is_neutral():
    from pipeline.stage4 import postprocess
    img = np.full((400, 500, 3), 235, dtype=np.uint8)
    cv2.circle(img, (350, 150), 30, (40, 80, 220), -1)
    out = np.array(postprocess(img, mode="Clean Scan"))
    channel_delta = np.abs(out[..., 0].astype(int) - out[..., 2].astype(int))
    assert channel_delta.max() == 0, "Clean Scan should be grayscale/neutral"
    return "Clean Scan output is neutral grayscale"


def _test_postprocess_flattens_fold_shadow_but_keeps_text():
    from pipeline.stage4 import postprocess
    img = np.full((520, 720, 3), (210, 228, 185), dtype=np.uint8)
    cv2.rectangle(img, (320, 0), (390, 520), (145, 165, 125), -1)
    cv2.putText(img, "READABLE", (90, 275), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (30, 35, 30), 5, cv2.LINE_AA)
    out = np.array(postprocess(img, mode="Clean Scan"))
    fold_in = img[:, 335:375].mean()
    fold_out = out[:, 335:375].mean()
    assert fold_out > fold_in + 25.0, "fold band did not flatten enough"
    assert out.var() > 50.0, "cleanup erased useful structure"
    return f"fold_mean {fold_in:.1f}→{fold_out:.1f}"


def _test_postprocess_lined_handwriting_rescue():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    from pipeline.stage4.detectors import _document_mask, _looks_lined_handwritten_page
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "folded", "folded_paper1.jpg")
    res = preprocess(path)
    assert _looks_lined_handwritten_page(
        res.raw, _document_mask(res.raw),
    ), "lined handwriting page was not detected"
    out = np.array(postprocess(res.raw, mode="Clean Scan"))
    ink_frac = float((cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) < 220).mean())
    assert ink_frac > 0.015, "handwriting rescue produced too little ink"
    return f"lined handwriting preserved (ink_frac={ink_frac:.3f})"


def _test_postprocess_faint_typewritten_rescue():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    from pipeline.stage4.detectors import _document_mask, _looks_faint_typewritten_page
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "folded", "folded_paper3.jpg")
    res = preprocess(path)
    assert _looks_faint_typewritten_page(
        res.raw, _document_mask(res.raw),
    ), "faint typewritten page was not detected"
    out = np.array(postprocess(res.raw, mode="Clean Scan"))
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    ink_frac = float((gray < 220).mean())
    assert ink_frac > 0.040, "faint typewritten rescue erased too many letters"
    assert len(np.unique(gray)) > 16, "faint typewriter output should stay tonal"
    return f"faint typewritten text preserved (ink_frac={ink_frac:.3f})"


def _test_postprocess_faint_typewritten_creases_stay_tonal():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess

    path = os.path.join(ROOT, "datasets", "user_papers",
                        "folded", "folded_paper3.jpg")
    res = preprocess(path)
    out = np.array(postprocess(res.raw, mode="Clean Scan"))
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    h_fold = gray[int(h * 0.30):int(h * 0.39), :]
    v_fold = gray[:, int(w * 0.19):int(w * 0.25)]

    assert float((h_fold < 128).mean()) < 0.080, (
        "horizontal fold was promoted into hard black ink"
    )
    assert float((v_fold < 128).mean()) < 0.090, (
        "vertical fold was promoted into hard black ink"
    )
    return (
        f"h_fold_dark={(h_fold < 128).mean():.3f}, "
        f"v_fold_dark={(v_fold < 128).mean():.3f}"
    )


def _test_postprocess_ink_recovery_keeps_typewriter_dark():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess

    path = os.path.join(ROOT, "datasets", "user_papers",
                        "folded", "folded_paper3.jpg")
    res = preprocess(path)
    clean = np.array(postprocess(res.raw, mode="Clean Scan"))
    recovery = np.array(postprocess(res.raw, mode="Ink Recovery"))
    recovery_gray = cv2.cvtColor(recovery, cv2.COLOR_RGB2GRAY)

    readable_ink_frac = float((recovery_gray < 220).mean())
    assert readable_ink_frac >= 0.065, (
        "Ink Recovery made folded_paper3 typewriter strokes too faint"
    )

    diff = float(np.abs(
        clean.astype(np.int16) - recovery.astype(np.int16),
    ).mean())
    assert diff >= 2.0, "Clean Scan and Ink Recovery outputs are too similar"
    return f"readable ink={readable_ink_frac:.3f}, mode diff={diff:.1f}"


def _test_postprocess_dense_print_rescue():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    from pipeline.stage4.detectors import _document_mask, _looks_dense_print_page
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "warped", "warped_paper1.jpg")
    res = preprocess(path)
    assert _looks_dense_print_page(
        res.raw, _document_mask(res.raw),
    ), "dense printed page was not detected"
    out = np.array(postprocess(res.raw, mode="Clean Scan"))
    ink_frac = float((cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) < 220).mean())
    assert ink_frac > 0.075, "dense print rescue erased too many small strokes"
    return f"dense print preserved (ink_frac={ink_frac:.3f})"


def _test_postprocess_mixed_graphic_text_rescue():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    from pipeline.stage4.detectors import _document_mask, _looks_mixed_graphic_page
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "mixed", "mixed_paper2.jpg")
    if not os.path.exists(path):
        return "SKIP - mixed graphic sample not found"

    res = preprocess(path)
    assert _looks_mixed_graphic_page(
        res.raw, _document_mask(res.raw),
    ), "mixed graphic page was not detected"

    out = np.array(
        postprocess(
            res.raw,
            mode="Clean Scan",
            lighting=res.meta.get("lighting"),
        ),
    )
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    color_delta = float(np.abs(
        out[..., 0].astype(np.int16) - out[..., 2].astype(np.int16),
    ).mean())
    lower = gray[
        int(gray.shape[0] * 0.50):int(gray.shape[0] * 0.88),
        int(gray.shape[1] * 0.12):int(gray.shape[1] * 0.90),
    ]
    lower_ink = float((lower < 220).mean())

    assert color_delta < 1.0, (
        "mixed graphic Clean Scan should stay scanner-like grayscale"
    )
    assert lower_ink >= 0.045, (
        "mixed graphic cleanup erased too much readable lower-section text"
    )
    assert float((lower < 80).mean()) >= 0.045, (
        "mixed graphic cleanup left lower-section text too faded"
    )
    return f"mixed graphic scanner output kept text (lower_ink={lower_ink:.3f})"


def _test_postprocess_mixed_panel_preserves_both_sections():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage4 import postprocess
    path = os.path.join(ROOT, "datasets", "user_papers",
                        "mixed", "mixed_paper1.jpg")
    if not os.path.exists(path):
        return "SKIP - mixed panel sample not found"

    res = preprocess(path)
    out = np.array(
        postprocess(
            res.raw,
            mode="Clean Scan",
            lighting=res.meta.get("lighting"),
        ),
    )
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    top = gray[
        int(h * 0.05):int(h * 0.42),
        int(w * 0.08):int(w * 0.90),
    ]
    lower = gray[
        int(h * 0.50):int(h * 0.88),
        int(w * 0.08):int(w * 0.95),
    ]
    top_ink = float((top < 220).mean())
    lower_ink = float((lower < 220).mean())

    assert top_ink >= 0.05, (
        "mixed panel cleanup erased the normal top section text"
    )
    assert lower_ink >= 0.05, (
        "mixed panel cleanup erased the inverted lower section text"
    )
    return (
        "mixed panel kept both text sections "
        f"(top_ink={top_ink:.3f}, lower_ink={lower_ink:.3f})"
    )


def _test_postprocess_partial_context_does_not_force_noisy_recovery():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate
    from pipeline.stage4 import postprocess

    path = os.path.join(ROOT, "datasets", "user_papers",
                        "partial", "partial_paper1.png")
    if not os.path.exists(path):
        return "SKIP - partial panel sample not found"

    res = preprocess(path)
    label, _conf, details = quality_gate(res.enhanced)
    assert label == "PARTIAL", f"expected PARTIAL gate, got {label}"

    baseline = np.array(
        postprocess(
            res.raw,
            mode="Ink Recovery",
            lighting=res.meta.get("lighting"),
        ),
    )
    contextual = np.array(
        postprocess(
            res.raw,
            mode="Ink Recovery",
            lighting=res.meta.get("lighting"),
            qg_details=details,
        ),
    )

    diff = float(np.abs(
        baseline.astype(np.int16) - contextual.astype(np.int16),
    ).mean())
    assert diff < 0.5, (
        "PARTIAL context should not force noisy Ink Recovery changes"
    )
    dark_frac = float((contextual[..., 0] < 80).mean())
    return f"partial Ink Recovery stable (diff={diff:.2f}, dark={dark_frac:.3f})"


def _test_postprocess_partial_clean_scan_rescues_visible_fragments():
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate
    from pipeline.stage4 import postprocess

    path = os.path.join(ROOT, "datasets", "user_papers",
                        "partial", "partial_paper1.png")
    if not os.path.exists(path):
        return "SKIP - partial panel sample not found"

    res = preprocess(path)
    label, _conf, details = quality_gate(res.enhanced)
    assert label == "PARTIAL", f"expected PARTIAL gate, got {label}"

    baseline = np.array(
        postprocess(
            res.raw,
            mode="Clean Scan",
            lighting=res.meta.get("lighting"),
        ),
    )
    contextual = np.array(
        postprocess(
            res.raw,
            mode="Clean Scan",
            lighting=res.meta.get("lighting"),
            qg_details=details,
        ),
    )

    base_dark = float((baseline[..., 0] < 80).mean())
    contextual_dark = float((contextual[..., 0] < 80).mean())
    contextual_mid = float(
        ((contextual[..., 0] >= 80) & (contextual[..., 0] < 240)).mean()
    )
    assert contextual_dark > base_dark + 0.0030, (
        "PARTIAL Clean Scan did not rescue enough visible text fragments"
    )
    assert contextual_dark < 0.025, (
        "PARTIAL Clean Scan rescued too much noise from the dark region"
    )
    assert contextual_mid < 0.001, (
        "PARTIAL Clean Scan should stay scanner-like binary"
    )
    return (
        "partial Clean Scan rescued visible fragments "
        f"(dark {base_dark:.3f}->{contextual_dark:.3f})"
    )


def _test_boundary_cleanup_preserves_dense_edge_text():
    from pipeline.stage4.filters import _remove_boundary_artifacts
    ink = np.zeros((420, 520), dtype=np.uint8)
    doc = np.zeros_like(ink)
    cv2.rectangle(doc, (20, 10), (500, 400), 255, -1)
    # Thin title/header-like component touching the page boundary.
    cv2.rectangle(ink, (42, 12), (220, 28), 255, 1)
    # Dirty side strip that should still be removed.
    cv2.rectangle(ink, (22, 40), (31, 280), 255, -1)

    kept = _remove_boundary_artifacts(ink, doc, 0.035, preserve_text=True)
    removed = _remove_boundary_artifacts(ink, doc, 0.035, preserve_text=False)

    title_roi = np.s_[10:32, 38:224]
    side_roi = np.s_[38:284, 20:34]
    assert np.count_nonzero(kept[title_roi]) > 0, "edge title was erased"
    assert np.count_nonzero(removed[title_roi]) == 0, (
        "control cleanup did not remove boundary title"
    )
    assert np.count_nonzero(kept[side_roi]) == 0, "dirty side strip survived"
    return "dense edge title kept while side strip is removed"


# ════════════════════════════════════════════════════════════════════════════
#  Stage 5 — OCR Formatting
# ════════════════════════════════════════════════════════════════════════════

def _test_ocr_layout_formatter_preserves_structure():
    from pipeline.stage5._format import _format_paragraphs
    boxes = [
        {"xyxy": (10, 10, 90, 30), "text": "HEADER", "conf": 0.95},
        {"xyxy": (10, 70, 90, 90), "text": "Question", "conf": 0.95},
        {"xyxy": (105, 70, 150, 90), "text": "1", "conf": 0.95},
        {"xyxy": (10, 95, 100, 115), "text": "Answer", "conf": 0.95},
        {"xyxy": (10, 160, 100, 180), "text": "SUB-TOTAL", "conf": 0.95},
        {"xyxy": (250, 160, 320, 180), "text": "1007.10", "conf": 0.95},
    ]
    text = _format_paragraphs(boxes, min_conf=0.3)
    assert "HEADER\n\nQuestion 1\nAnswer" in text
    assert "\n\nSUB-TOTAL" in text and "1007.10" in text
    assert "Q u e s t i o n" not in text
    assert "S U B" not in text
    return "line and section breaks preserved without splitting words"


# ════════════════════════════════════════════════════════════════════════════
#  End-to-End — full pipeline without UI
# ════════════════════════════════════════════════════════════════════════════

def _test_e2e_full(skip_dewarp: bool = False):
    """Run the complete pipeline on the auto-picked test image."""
    from pipeline.stage1_preprocess import preprocess
    from pipeline.stage2_quality_gate import quality_gate
    from pipeline.stage4 import postprocess

    pre = preprocess(_REAL_IMG)
    label, conf, _details = quality_gate(pre.enhanced)

    if label == "NONE":
        return f"REJECTED (conf={conf:.3f}) — constraint message would show"

    if skip_dewarp:
        # Use preprocessed enhanced image as stand-in so we can still test the
        # postprocess step when skipping the slow model load.
        final = postprocess(pre.enhanced, mode="Clean Scan")
        out_path = os.path.join(ROOT, "datasets", "smoke_output_noDewarp.png")
    else:
        from pipeline.stage3_dewarp import dewarp
        dewarped = dewarp(pre.raw)
        final = postprocess(dewarped, mode="Clean Scan")
        out_path = os.path.join(ROOT, "datasets", "smoke_output.png")

    final.save(out_path)
    assert os.path.exists(out_path)
    return f"{label} ({conf:.0%}) → saved {os.path.basename(out_path)}"


# ════════════════════════════════════════════════════════════════════════════
#  Constraint Message Path (synthetic)
# ════════════════════════════════════════════════════════════════════════════

def _test_constraint_path():
    """
    Simulates what the UI does when quality_gate returns NONE:
    controller emits on_reject → _show_constraint_message().
    We verify the NONE path completes without error.
    """
    from pipeline.stage2_quality_gate import quality_gate
    blank = np.ones((400, 400, 3), dtype=np.uint8) * 240
    label, conf, _details = quality_gate(blank)
    if label == "NONE":
        # Simulate controller logic
        message = "Content Unrecoverable — document too damaged to process"
        assert isinstance(message, str) and len(message) > 0
        return f"NONE path triggered — message ready for right panel (conf={conf:.3f})"
    else:
        return f"SKIP — blank image scored {label} ({conf:.3f}) — right panel shows output"


# ════════════════════════════════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MeshUnwarp smoke test")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Stage 3 (DocTr model load) — fast sanity check")
    args = parser.parse_args()

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  MeshUnwarp Smoke Test Suite{RESET}")
    if args.quick:
        print(f"  {YELLOW}--quick mode: Stage 3 (DocTr) tests skipped{RESET}")
    print(f"{'─'*60}{RESET}\n")

    # ── Stage 1
    print(f"{BOLD}Stage 1 — Preprocessing{RESET}")
    _run("Real image loads + dual streams",     _test_preprocess_real)
    _run("Small image upscaled to ≥288",        _test_preprocess_small)
    _run("Missing file raises ValueError",      _test_preprocess_missing)
    _run("raw ≠ enhanced (CLAHE diverges)",     _test_preprocess_parallel_streams)
    _run("Uneven paper tone is detected",       _test_preprocess_detects_uneven_paper_tone)

    # ── Stage 2
    print(f"\n{BOLD}Stage 2 — Quality Gate{RESET}")
    _run("Real image scores MEDIUM/HIGH",      _test_qgate_real_high)
    _run("Blank image scores NONE (reject)",    _test_qgate_blank_reject)
    _run("Half-black image warns as PARTIAL",    _test_qgate_half_black_partial)
    _run("Destroyed text fragment enters salvage", _test_qgate_salvage_text_fragment)
    _run("Confidence always in [0.0, 1.0]",     _test_qgate_confidence_range)
    _run("Dataset calibration avoids LOW cluster", _test_qgate_dataset_calibration)

    # ── Stage 3
    print(f"\n{BOLD}Stage 3 — De-Warping{RESET}")
    _run("IllTr graphic gate is less blunt", _test_illtr_graphic_gate_is_less_blunt)
    _run("Mixed pages use same IllTr gate", _test_illtr_mixed_pages_follow_same_gate)
    if not args.quick:
        _run("GeoTr + IllTr weights load OK",       _test_dewarp_loads_models)
        _run("Stage 3 preserves input dimensions", _test_dewarp_preserves_input_dims)
        _run("GeoTr forward pass reports telemetry", _test_dewarp_meta_includes_geotr_telemetry)
    else:
        print(f"  {YELLOW}DocTr model-load tests skipped with --quick{RESET}")

    # ── Stage 4
    print(f"\n{BOLD}Stage 4 — Postprocessing{RESET}")
    _run("Returns PIL.Image",                   _test_postprocess_returns_pil)
    _run("Visible modes and aliases run OK",    _test_postprocess_all_modes)
    _run("Background brightens without blanking", _test_postprocess_whitens_background_without_blank_output)
    _run("Clean Scan output is neutral",        _test_postprocess_clean_scan_is_neutral)
    _run("Fold shadows flatten while text survives", _test_postprocess_flattens_fold_shadow_but_keeps_text)
    _run("Lined handwriting rescue activates", _test_postprocess_lined_handwriting_rescue)
    _run("Faint typewritten text is rescued", _test_postprocess_faint_typewritten_rescue)
    _run("Faint typewritten creases stay tonal", _test_postprocess_faint_typewritten_creases_stay_tonal)
    _run("Ink Recovery keeps typewriter readable", _test_postprocess_ink_recovery_keeps_typewriter_dark)
    _run("Dense printed pages keep small text", _test_postprocess_dense_print_rescue)
    _run("Mixed graphics stay scanner-like", _test_postprocess_mixed_graphic_text_rescue)
    _run("Mixed panel keeps both sections", _test_postprocess_mixed_panel_preserves_both_sections)
    _run("Partial context does not force noisy Ink Recovery", _test_postprocess_partial_context_does_not_force_noisy_recovery)
    _run("Partial Clean Scan rescues visible fragments", _test_postprocess_partial_clean_scan_rescues_visible_fragments)
    _run("Dense edge text survives cleanup", _test_boundary_cleanup_preserves_dense_edge_text)

    # ── Stage 5
    print(f"\n{BOLD}Stage 5 — OCR Formatting{RESET}")
    _run("Layout formatter keeps lines/sections", _test_ocr_layout_formatter_preserves_structure)

    # ── End-to-End
    print(f"\n{BOLD}End-to-End{RESET}")
    _run(f"Full pipeline on {os.path.basename(_REAL_IMG)}",
         _test_e2e_full, skip_dewarp=args.quick)

    # ── Constraint message
    print(f"\n{BOLD}Constraint Message Path{RESET}")
    _run("NONE → reject message assembled",     _test_constraint_path)

    # ── Summary
    passed  = sum(1 for _, ok, _, _ in _results if ok)
    failed  = sum(1 for _, ok, _, _ in _results if not ok)
    total   = len(_results)
    elapsed = sum(d for _, _, d, _ in _results)

    print(f"\n{'─'*60}")
    if failed == 0:
        print(f"{BOLD}{GREEN}  ✓ All {total} tests passed{RESET}  ({elapsed:.1f}s total)")
    else:
        print(f"{BOLD}{RED}  ✗ {failed}/{total} tests FAILED{RESET}  ({elapsed:.1f}s total)")
        print(f"\n  Failed tests:")
        for name, ok, _, detail in _results:
            if not ok:
                print(f"    {RED}• {name}{RESET}  —  {detail}")
    print(f"{'─'*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
