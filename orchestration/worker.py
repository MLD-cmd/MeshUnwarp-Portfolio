import os
import re
import sys
import time
import traceback

# Cap preview longest edge to keep queue payload small; fit-to-panel so no visible cost.
_PREVIEW_MAX_EDGE = 1200

# Minimum chip dwell so fast stages stay visible long enough to read.
_STAGE_DWELL_MS = int(os.environ.get("MESHUNWARP_STAGE_DWELL_MS", "120"))


def worker_main(req_q, resp_q) -> None:
    """Subprocess entrypoint. Do not call from the parent process."""
    # Make the MeshUnwarp repo root importable under spawn start method —
    # the subprocess inherits cwd but not the parent's sys.path additions.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)

    resp_q.put({"type": "worker_ready", "state": "loading"})

    from PIL import Image

    from core.logging_setup   import init_logging
    from pipeline.stage1_preprocess    import preprocess
    from pipeline.stage2_quality_gate  import quality_gate
    from pipeline.stage3_dewarp        import dewarp_with_meta, _load_models
    from pipeline.stage4               import postprocess, salvage_scan
    from pipeline.stage5               import ocr_with_meta, prewarm as ocr_prewarm

    log = init_logging(role="worker")
    log.info("worker subprocess started")

    _load_models()
    ocr_prewarm()
    resp_q.put({"type": "worker_ready", "state": "ready"})

    current_paper = "-"

    def stage_start(n, msg):
        resp_q.put({"type": "stage", "stage": n, "state": "start", "message": msg})
        return time.monotonic()

    def stage_done(n, t0, msg=""):
        # Pad fast stages so the chip stays visible long enough to read.
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        remaining = (_STAGE_DWELL_MS - elapsed_ms) / 1000.0
        if remaining > 0:
            time.sleep(remaining)
        # Report un-padded elapsed so the UI shows real compute cost.
        resp_q.put({
            "type": "stage", "stage": n, "state": "done",
            "message": msg, "elapsed_ms": elapsed_ms,
        })
        log.info("[%s] stage %d done in %.1f ms %s", current_paper, n, elapsed_ms, msg)
        return elapsed_ms

    def stage3_message(meta: dict) -> str:
        geotr = (meta or {}).get("geotr", {})
        illum = (meta or {}).get("illumination", {})
        warp_state = "applied" if geotr.get("applied", True) else "bypassed"
        shift = geotr.get("max_displacement")
        chaos = geotr.get("chaos")
        parts = [f"GeoTr {warp_state}"]
        if isinstance(shift, (int, float)):
            parts.append(f"shift {shift:.4f}")
        if isinstance(chaos, (int, float)):
            parts.append(f"chaos {chaos:.4f}")

        if illum.get("applied"):
            illtr = "IllTr applied"
        else:
            reason = illum.get("reason", "unknown")
            reason_label = {
                "lighting_clean": "lighting clean",
                "tone_only": "tone only",
                "graphic_heavy": "graphic-heavy",
                "dark_design_panel": "dark design panel",
                "disabled": "disabled",
                "salvage_mode": "not used",
            }.get(reason, str(reason).replace("_", " "))
            illtr = f"IllTr skipped · {reason_label}"
        return " · ".join(parts) + f"; {illtr}"

    def ocr_message(meta: dict) -> str:
        label = (meta or {}).get("engine_label", "OCR")
        chars = (meta or {}).get("chars", 0)
        msg = f"OCR: {label} · {chars} chars"
        if (meta or {}).get("fallback_used"):
            msg += " · fallback after Vision sparse"
        return msg

    def has_meaningful_text(text: str) -> bool:
        alnum = sum(ch.isalnum() for ch in text)
        words = re.findall(r"[A-Za-z]{3,}", text or "")
        return alnum >= 12 and len(words) >= 2

    def diagnostics(
        *,
        path: str | None,
        mode: str,
        qg: dict,
        dewarp_meta: dict,
        ocr_meta: dict,
        total_s: float,
        cached: bool,
    ) -> dict:
        return {
            "source_file": os.path.basename(path) if path else None,
            "source_path": path,
            "processing_mode": mode,
            "quality_gate": qg,
            "geotr": (dewarp_meta or {}).get("geotr", {}),
            "illtr": (dewarp_meta or {}).get("illumination", {}),
            "ocr": ocr_meta or {},
            "runtime": {
                "total_seconds": round(total_s, 3),
                "cached": bool(cached),
            },
        }

    last_dewarp = {
        "path": None,
        "array": None,
        "ocr_source": None,
        "qg_message": "",
        "qg": {},
        "stage3_meta": {},
        "stage3_message": "",
        "lighting": None,
    }

    while True:
        req = req_q.get()
        if req is None:
            return

        try:
            if req.get("action") == "repolish":
                if last_dewarp["array"] is None:
                    resp_q.put({
                        "type": "error",
                        "message": "No cached dewarp result to repolish",
                    })
                    continue
                mode = req.get("mode", "Clean Scan")
                current_paper = os.path.basename(last_dewarp["path"] or "cached")
                log.info("[%s] repolish started mode=%s", current_paper, mode)
                run_t0 = time.monotonic()
                t = stage_start(4, "Polishing…")
                final = postprocess(
                    last_dewarp["array"], mode=mode,
                    lighting=last_dewarp.get("lighting"),
                    stage3_meta=last_dewarp.get("stage3_meta"),
                    qg_details=(
                        (last_dewarp.get("qg") or {}).get("details") or {}
                    ),
                )
                stage_done(4, t)
                t = stage_start(5, "OCR…")
                # Apple Vision reads natural photos far better than Stage 4's binarized output.
                extra = (
                    [last_dewarp["ocr_source"]]
                    if last_dewarp["ocr_source"] is not None else None
                )
                text, ocr_meta = ocr_with_meta(last_dewarp["array"], extra_images=extra)
                stage_done(5, t, ocr_message(ocr_meta))
                total_s = time.monotonic() - run_t0
                log.info(
                    "[%s] repolish complete in %.1f s mode=%s",
                    current_paper, total_s, mode,
                )
                resp_q.put({
                    "type": "complete", "image": final, "text": text,
                    "total_seconds": total_s, "cached": True,
                    "diagnostics": diagnostics(
                        path=last_dewarp["path"],
                        mode=mode,
                        qg=last_dewarp.get("qg", {}),
                        dewarp_meta=last_dewarp.get("stage3_meta", {}),
                        ocr_meta=ocr_meta,
                        total_s=total_s,
                        cached=True,
                    ),
                })
                continue

            path = req["path"]
            mode = req.get("mode", "Clean Scan")
            current_paper = os.path.basename(path)
            log.info("[%s] run started mode=%s path=%s", current_paper, mode, path)
            run_t0 = time.monotonic()

            # Synthetic stage events keep UI chips animating on re-click cache hit.
            if last_dewarp["path"] == path and last_dewarp["array"] is not None:
                for n, msg in (
                    (1, "Preprocessing… (cached)"),
                    (2, "Quality gate… (cached)"),
                    (3, "De-warping… (cached)"),
                ):
                    t = stage_start(n, msg)
                    done_msg = ""
                    if n == 2:
                        done_msg = last_dewarp["qg_message"]
                    elif n == 3:
                        done_msg = last_dewarp.get("stage3_message", "")
                    stage_done(n, t, done_msg)
                _cg = last_dewarp.get("stage3_meta", {}).get("geotr", {})
                log.info(
                    "[cached] GeoTr max_displacement=%.4f → apply warp",
                    _cg.get("max_displacement", float("nan")),
                )
                resp_q.put({
                    "type":  "preview",
                    "stage": 3,
                    "image": _thumb(Image.fromarray(last_dewarp["array"])),
                    "meta":  {"cached": True, **last_dewarp.get("stage3_meta", {})},
                })
                t = stage_start(4, "Polishing…")
                final = postprocess(
                    last_dewarp["array"], mode=mode,
                    lighting=last_dewarp.get("lighting"),
                    stage3_meta=last_dewarp.get("stage3_meta"),
                    qg_details=(
                        (last_dewarp.get("qg") or {}).get("details") or {}
                    ),
                )
                stage_done(4, t)
                t = stage_start(5, "OCR…")
                extra = (
                    [last_dewarp["ocr_source"]]
                    if last_dewarp["ocr_source"] is not None else None
                )
                text, ocr_meta = ocr_with_meta(last_dewarp["array"], extra_images=extra)
                stage_done(5, t, ocr_message(ocr_meta))
                total_s = time.monotonic() - run_t0
                log.info(
                    "[%s] cached run complete in %.1f s mode=%s",
                    current_paper, total_s, mode,
                )
                resp_q.put({
                    "type": "complete", "image": final, "text": text,
                    "total_seconds": total_s, "cached": True,
                    "diagnostics": diagnostics(
                        path=path,
                        mode=mode,
                        qg=last_dewarp.get("qg", {}),
                        dewarp_meta=last_dewarp.get("stage3_meta", {}),
                        ocr_meta=ocr_meta,
                        total_s=total_s,
                        cached=True,
                    ),
                })
                continue

            t = stage_start(1, "Preprocessing…")
            pre = preprocess(path)
            resp_q.put({
                "type":  "preview",
                "stage": 1,
                "image": _thumb(Image.fromarray(pre.enhanced)),
                "meta":  pre.meta,
            })
            stage_done(1, t)

            t = stage_start(2, "Quality gate…")
            result, confidence, qg_details = quality_gate(pre.enhanced)

            features = qg_details.get("features", {})
            damage = qg_details.get("damage", 0.0)
            reject_reason = qg_details.get("reject_reason", "")
            salvage_text = bool(
                (qg_details.get("salvage_text") or {}).get("enabled")
            )
            qg_warning = (
                "salvage text only"
                if salvage_text else
                "large dark/obscured region"
                if result == "PARTIAL" else
                ""
            )
            qg_message = (
                f"{result} · {confidence:.0%} · "
                f"damage={damage:.2f} · "
                f"edge={features.get('edge_score', 0.0):.2f} · "
                f"text={features.get('text_area_score', 0.0):.2f}"
            )
            if reject_reason:
                qg_message += f" · reason={reject_reason}"
            if qg_warning:
                qg_message += f" · {qg_warning}"

            if result == "NONE":
                log.info("[%s] rejected at quality gate: %s", current_paper, qg_message)
                resp_q.put({
                    "type": "reject",
                    "message": "Content Unrecoverable — document too damaged",
                    "confidence": confidence,
                })
                continue

            qg_meta = {
                "result": result,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100.0, 1),
                "warning": qg_warning,
                "details": qg_details,
            }
            stage_done(2, t, qg_message)

            if result == "PARTIAL" and salvage_text:
                t = stage_start(3, "Skipping de-warp…")
                dewarp_meta = {
                    "geotr": {
                        "applied": False,
                        "reason": "salvage_mode",
                    },
                    "illumination": {
                        "applied": False,
                        "reason": "salvage_mode",
                    },
                }
                resp_q.put({
                    "type": "preview",
                    "stage": 3,
                    "image": _thumb(Image.fromarray(pre.enhanced)),
                    "meta": dewarp_meta,
                })
                stage_done(3, t, "GeoTr skipped")

                t = stage_start(4, "Preparing salvage scan…")
                final = salvage_scan(pre.enhanced)
                stage_done(4, t, "document salvage scan")

                t = stage_start(5, "OCR salvage text…")
                text, ocr_meta = ocr_with_meta(
                    final,
                    extra_images=[pre.enhanced, pre.raw],
                )
                stage_done(5, t, ocr_message(ocr_meta) + " · salvage")
                if not has_meaningful_text(text):
                    log.info(
                        "[%s] rejected after salvage OCR: no meaningful text",
                        current_paper,
                    )
                    resp_q.put({
                        "type": "reject",
                        "message": "Content Unrecoverable — no readable text recovered",
                        "confidence": confidence,
                    })
                    continue

                last_dewarp.update({
                    "path": None,
                    "array": None,
                    "ocr_source": None,
                    "qg_message": "",
                    "qg": {},
                    "stage3_meta": {},
                    "stage3_message": "",
                    "lighting": None,
                })
                total_s = time.monotonic() - run_t0
                log.info(
                    "[%s] salvage OCR complete in %.1f s mode=%s",
                    current_paper, total_s, mode,
                )
                resp_q.put({
                    "type": "complete", "image": final, "text": text,
                    "total_seconds": total_s, "cached": False,
                    "diagnostics": diagnostics(
                        path=path,
                        mode=mode,
                        qg=qg_meta,
                        dewarp_meta=dewarp_meta,
                        ocr_meta=ocr_meta,
                        total_s=total_s,
                        cached=False,
                    ),
                })
                continue

            t = stage_start(3, "De-warping…")
            # Lighting diagnostics let stage 3 skip IllTr — the dominant cost — when already-even.
            dewarped, dewarp_meta = dewarp_with_meta(
                pre.raw, lighting=pre.meta.get("lighting"),
                source_name=current_paper,
            )
            # Cache for fast mode-change repolish + re-click skip.
            last_dewarp["path"]       = path
            last_dewarp["array"]      = dewarped
            last_dewarp["ocr_source"] = pre.raw
            last_dewarp["qg_message"] = qg_message
            last_dewarp["qg"] = qg_meta
            last_dewarp["stage3_meta"] = dewarp_meta
            last_dewarp["stage3_message"] = stage3_message(dewarp_meta)
            last_dewarp["lighting"] = pre.meta.get("lighting")
            resp_q.put({
                "type":  "preview",
                "stage": 3,
                "image": _thumb(Image.fromarray(dewarped)),
                "meta":  dewarp_meta,
            })
            stage_done(3, t, last_dewarp["stage3_message"])

            t = stage_start(4, "Polishing…")
            final = postprocess(
                dewarped, mode=mode, lighting=pre.meta.get("lighting"),
                stage3_meta=dewarp_meta,
                qg_details=qg_details,
            )
            stage_done(4, t)

            t = stage_start(5, "OCR…")
            text, ocr_meta = ocr_with_meta(dewarped, extra_images=[pre.raw])
            stage_done(5, t, ocr_message(ocr_meta))

            total_s = time.monotonic() - run_t0
            log.info(
                "[%s] run complete in %.1f s mode=%s",
                current_paper, total_s, mode,
            )
            resp_q.put({
                "type": "complete", "image": final, "text": text,
                "total_seconds": total_s, "cached": False,
                "diagnostics": diagnostics(
                    path=path,
                    mode=mode,
                    qg=qg_meta,
                    dewarp_meta=dewarp_meta,
                    ocr_meta=ocr_meta,
                    total_s=total_s,
                    cached=False,
                ),
            })

        except Exception as exc:
            log.exception("[%s] worker error", current_paper)
            resp_q.put({
                "type": "error",
                "message": str(exc) or exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            })


def _thumb(img):
    """Downscale an intermediate preview to keep queue payload small."""
    w, h = img.size
    m = max(w, h)
    if m <= _PREVIEW_MAX_EDGE:
        return img
    scale = _PREVIEW_MAX_EDGE / m
    return img.resize((int(w * scale), int(h * scale)))
