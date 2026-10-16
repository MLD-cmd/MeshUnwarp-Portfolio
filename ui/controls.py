import tkinter as tk
from tkinter import ttk

from .theme import (
    BG_MAIN, BG_ELEMENT, BG_PANEL, BORDER_COLOR,
    ACCENT_INPUT, ACCENT_OUTPUT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, TEXT_MUTED,
    STATUS_IDLE, STATUS_OK, STATUS_RUN, STATUS_ERROR,
    PILL_IDLE_BG, PILL_IDLE_BORDER, PILL_IDLE_TEXT,
    PILL_OK_BG, PILL_OK_BORDER, PILL_OK_TEXT,
    PILL_RUN_BG, PILL_RUN_BORDER, PILL_RUN_TEXT,
    PILL_ERR_BG, PILL_ERR_BORDER, PILL_ERR_TEXT,
)
from ._utils import MODE_HINTS


class ControlsMixin:

    _STAGE_LABELS = ("① Pre", "② Gate", "③ Warp", "④ Polish", "⑤ OCR")

    def _build_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Eng.TCombobox",
            fieldbackground=BG_ELEMENT,
            background=BG_ELEMENT,
            foreground=TEXT_PRIMARY,
            arrowcolor=ACCENT_INPUT,
            bordercolor=BORDER_COLOR,
            lightcolor=BG_ELEMENT,
            darkcolor=BG_ELEMENT,
            insertcolor=TEXT_PRIMARY,
            selectforeground=TEXT_PRIMARY,
            selectbackground=BG_ELEMENT,
            font=("Menlo", 10),
        )
        style.map(
            "Eng.TCombobox",
            fieldbackground=[("readonly", BG_ELEMENT)],
            selectbackground=[("readonly", BG_ELEMENT)],
            selectforeground=[("readonly", TEXT_PRIMARY)],
        )

        style.configure(
            "Dark.Vertical.TScrollbar",
            troughcolor=BG_MAIN,
            background=BORDER_COLOR,
            darkcolor=BG_MAIN,
            lightcolor=BG_MAIN,
            bordercolor=BG_MAIN,
            arrowcolor=TEXT_DIM,
            relief="flat",
            gripcount=0,
            width=10,
            arrowsize=0,
        )
        style.map(
            "Dark.Vertical.TScrollbar",
            background=[("active", TEXT_MUTED), ("pressed", TEXT_SECONDARY)],
        )

    def _build_controls(self, parent, row):
        wrapper = tk.Frame(parent, bg=BG_MAIN)
        wrapper.grid(row=row, column=0, columnspan=2, sticky="ew",
                     padx=12, pady=(0, 12))
        wrapper.columnconfigure(0, weight=1)

        self._progress_bar = tk.Canvas(
            wrapper, height=2, bg=BORDER_COLOR,
            highlightthickness=0, bd=0)

        self._build_diagnostics_panel(wrapper).grid(
            row=1, column=0, sticky="ew", pady=(6, 6))

        strip = tk.Frame(wrapper, bg=BG_MAIN, height=56)
        strip.grid(row=2, column=0, sticky="ew")
        strip.columnconfigure(0, weight=1)
        strip.grid_propagate(False)

        left_group = tk.Frame(strip, bg=BG_MAIN)
        left_group.grid(row=0, column=0, sticky="w", padx=4, pady=15)

        self._pill_frame = tk.Frame(
            left_group, bg=PILL_IDLE_BG,
            highlightbackground=PILL_IDLE_BORDER,
            highlightthickness=1)
        self._pill_frame.pack(side="left")

        self._pill_dot = tk.Canvas(
            self._pill_frame, width=8, height=8,
            bg=PILL_IDLE_BG, highlightthickness=0)
        self._pill_dot.pack(side="left", padx=(10, 0))
        self._pill_dot.create_oval(1, 1, 7, 7,
                                   fill=STATUS_IDLE, outline="", tags="dot")

        self._pill_text = tk.Label(
            self._pill_frame,
            text="Idle — select an image to begin",
            font=self.fonts["status"],
            fg=PILL_IDLE_TEXT, bg=PILL_IDLE_BG, padx=6)
        self._pill_text.pack(side="left", padx=(4, 10), pady=5)

        self._build_stage_chips(left_group).pack(
            side="left", padx=(12, 0))

        btn_group = tk.Frame(strip, bg=BG_MAIN)
        btn_group.grid(row=0, column=1, sticky="e", padx=4)

        # Label-based buttons — tk.Button on macOS Aqua flashes white on press
        self._save_btn = tk.Label(
            btn_group, text="↓  Save Output",
            font=self.fonts["btn"],
            bg=BG_MAIN, fg=STATUS_OK,
            padx=16, pady=10, cursor="hand2",
            highlightbackground=STATUS_OK, highlightthickness=1,
        )
        self._save_btn.bind("<Enter>",
            lambda e: self._save_btn.config(bg="#0d1a10"))
        self._save_btn.bind("<Leave>",
            lambda e: self._save_btn.config(bg=BG_MAIN))
        self._save_btn.bind("<ButtonPress-1>",
            lambda e: self._save_btn.config(bg="#061008"))
        self._save_btn.bind("<ButtonRelease-1>",
            lambda e: self._save_btn.config(bg="#0d1a10"))
        self._save_btn.bind("<Button-1>",
            lambda e: self._save_output())

        self.start_btn = tk.Label(
            btn_group, text="▶  Start De-Warp",
            font=self.fonts["btn"],
            bg=BG_ELEMENT, fg=TEXT_MUTED,
            padx=28, pady=10,
            cursor="arrow",
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        self.start_btn.grid(row=0, column=2)

    def _build_stage_chips(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent, bg=BG_MAIN)
        self._chip_labels: list[tk.Label] = []
        for i, label in enumerate(self._STAGE_LABELS):
            if i > 0:
                tk.Label(frame, text="›", font=self.fonts["status"],
                         fg=TEXT_DIM, bg=BG_MAIN).pack(side="left", padx=3)
            chip = tk.Label(
                frame, text=label,
                font=self.fonts["status"],
                fg=TEXT_DIM, bg=BG_MAIN,
                padx=4, pady=2,
            )
            chip.pack(side="left")
            self._chip_labels.append(chip)
        return frame

    def _set_chip_state(self, idx: int, state: str) -> None:
        if not (0 <= idx < len(self._chip_labels)):
            return
        color = {
            "pending":  TEXT_DIM,
            "running":  STATUS_RUN,
            "complete": STATUS_OK,
        }.get(state, TEXT_DIM)
        self._chip_labels[idx].config(fg=color)

    def _reset_chips(self) -> None:
        for i in range(len(self._chip_labels)):
            self._set_chip_state(i, "pending")

    def _make_btn(self, parent, text, command,
                  bg, fg, active_bg,
                  padx=12, pady=6, font=None):
        """Label-based button — avoids macOS Aqua active-state white flash from tk.Button."""
        f = font or self.fonts["label"]
        lbl = tk.Label(
            parent, text=text,
            font=f, bg=bg, fg=fg,
            padx=padx, pady=pady,
            cursor="hand2",
            highlightthickness=0,
        )
        lbl.bind("<Enter>",        lambda e: lbl.config(bg=active_bg))
        lbl.bind("<Leave>",        lambda e: lbl.config(bg=bg))
        lbl.bind("<ButtonPress-1>", lambda e: lbl.config(bg=active_bg))
        lbl.bind("<Button-1>",     lambda e: command())
        return lbl

    def _on_mode_change(self, *args) -> None:
        mode = self._model_var.get()
        hint = MODE_HINTS.get(mode, "")
        self._mode_hint.config(text=hint)
        self._set_diag("mode", mode)

        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and pipeline.is_running():
            return
        if self._output_pil is not None:
            self._output_pil = None
            self._output_photo = None
            self._last_qg_message = ""
            self._last_lighting_warning = ""
            self._last_dewarp_message = ""
            self._last_ocr_message = ""
            self._last_diagnostics = {}
            self._last_total_seconds = 0.0
            self._last_run_cached = False
            self._stage_timings.clear()
            self._clear_output()
            self._reset_chips()
            self._reset_diagnostics(mode)
            if self._current_path is not None:
                if self._current_pil is not None and self._current_cat is not None:
                    self._update_input_meta(
                        self._current_path,
                        self._current_pil,
                        self._current_cat,
                    )
                self._set_state("ready", self._current_path.name)

    def _set_mode_picker_enabled(self, enabled: bool) -> None:
        combo = getattr(self, "model_combo", None)
        if combo is not None:
            combo.config(state="readonly" if enabled else "disabled")

    def _pulse_panel(self, frame: tk.Frame) -> None:
        frame.config(highlightbackground=ACCENT_INPUT, highlightthickness=1)
        self.after(400, lambda: frame.config(highlightbackground=BORDER_COLOR))

    def _set_state(self, state: str, detail: str = ""):
        # Don't leak past "loading" while DocTr weights are still loading.
        if (state in ("idle", "ready")
                and getattr(self, "_pipeline", None)
                and not self._pipeline.is_ready()):
            state = "loading"
            detail = ""

        if state == "complete":
            self._enable_pan_zoom()
        else:
            self._disable_pan_zoom()

        if state == "idle":
            self._set_mode_picker_enabled(True)
            self._reset_chips()
            self._update_pill(STATUS_IDLE, PILL_IDLE_BG, PILL_IDLE_BORDER,
                              PILL_IDLE_TEXT, "Idle — select an image to begin")
            self._hide_progress()
            self._set_start_disabled()
            self._hide_save_btn()

        elif state == "loading":
            self._set_mode_picker_enabled(False)
            self._update_pill(STATUS_RUN, PILL_RUN_BG, PILL_RUN_BORDER,
                              PILL_RUN_TEXT, "Loading DocTr weights…")
            self._show_progress()
            self._set_start_disabled()
            self._hide_save_btn()

        elif state == "ready":
            self._set_mode_picker_enabled(True)
            self._update_pill(STATUS_OK, PILL_OK_BG, PILL_OK_BORDER,
                              PILL_OK_TEXT, f"Ready · {detail}")
            self._hide_progress()
            self._set_start_enabled()
            self._hide_save_btn()

        elif state == "running":
            self._set_mode_picker_enabled(False)
            self._update_pill(STATUS_RUN, PILL_RUN_BG, PILL_RUN_BORDER,
                              PILL_RUN_TEXT, f"Processing · {detail} mode")
            self._show_progress()
            self._set_start_cancel()
            self._hide_save_btn()

        elif state == "complete":
            self._set_mode_picker_enabled(True)
            self._update_pill(STATUS_OK, PILL_OK_BG, PILL_OK_BORDER,
                              PILL_OK_TEXT, "Complete · output ready")
            self._hide_progress()
            self._set_start_enabled()
            self._show_save_btn()

        elif state == "error":
            self._set_mode_picker_enabled(True)
            self._update_pill(STATUS_ERROR, PILL_ERR_BG, PILL_ERR_BORDER,
                              PILL_ERR_TEXT, detail[:60])
            self._hide_progress()
            self._set_start_enabled()
            self._hide_save_btn()

    def _update_pill(self, dot_color, bg, border, text_color, text):
        self._pill_dot.itemconfig("dot", fill=dot_color)
        self._pill_dot.config(bg=bg)
        self._pill_frame.config(bg=bg, highlightbackground=border)
        self._pill_text.config(text=text, fg=text_color, bg=bg)

    def _set_start_disabled(self):
        self.start_btn.config(
            text="▶  Start De-Warp",
            bg=BG_ELEMENT, fg=TEXT_MUTED,
            cursor="arrow",
            highlightbackground=BORDER_COLOR, highlightthickness=1,
        )
        for seq in ("<Button-1>", "<ButtonPress-1>", "<ButtonRelease-1>",
                    "<Enter>", "<Leave>"):
            self.start_btn.unbind(seq)

    def _set_start_enabled(self):
        self.start_btn.config(
            text="▶  Start De-Warp",
            bg=ACCENT_INPUT, fg="#ffffff",
            cursor="hand2",
            highlightthickness=0,
        )
        # Unbind first — prevents stacking if called from multiple states
        for seq in ("<Button-1>", "<ButtonPress-1>", "<ButtonRelease-1>",
                    "<Enter>", "<Leave>"):
            self.start_btn.unbind(seq)
        self.start_btn.bind("<Enter>",
            lambda e: self.start_btn.config(bg="#4a7ee0"))
        self.start_btn.bind("<Leave>",
            lambda e: self.start_btn.config(bg=ACCENT_INPUT))
        self.start_btn.bind("<ButtonPress-1>",
            lambda e: self.start_btn.config(bg="#3a6bc4"))
        self.start_btn.bind("<ButtonRelease-1>",
            lambda e: self.start_btn.config(bg="#4a7ee0"))
        self.start_btn.bind("<Button-1>",
            lambda e: self._run_dewarp())

    def _set_start_cancel(self):
        self.start_btn.config(
            text="Cancel",
            bg="#2a1414", fg=STATUS_ERROR,
            cursor="hand2",
            highlightbackground=STATUS_ERROR, highlightthickness=1,
        )
        for seq in ("<Button-1>", "<ButtonPress-1>", "<ButtonRelease-1>",
                    "<Enter>", "<Leave>"):
            self.start_btn.unbind(seq)
        self.start_btn.bind("<Enter>",
            lambda e: self.start_btn.config(bg="#3a1c1c"))
        self.start_btn.bind("<Leave>",
            lambda e: self.start_btn.config(bg="#2a1414"))
        self.start_btn.bind("<ButtonPress-1>",
            lambda e: self.start_btn.config(bg="#4a2323"))
        self.start_btn.bind("<ButtonRelease-1>",
            lambda e: self.start_btn.config(bg="#3a1c1c"))
        self.start_btn.bind("<Button-1>",
            lambda e: self._cancel_dewarp())

    def _build_diagnostics_panel(self, parent: tk.Frame) -> tk.Frame:
        panel = tk.Frame(parent, bg=BG_PANEL, height=38)
        panel.grid_propagate(False)

        tk.Frame(panel, bg=ACCENT_OUTPUT, width=3).pack(side="left", fill="y")
        tk.Label(
            panel, text="  DIAGNOSTICS",
            font=self.fonts["meta"], fg=ACCENT_OUTPUT, bg=BG_PANEL,
        ).pack(side="left", padx=(0, 12))

        for key, label in (
            ("quality", "Quality"),
            ("geotr", "GeoTr"),
            ("illtr", "IllTr"),
            ("ocr", "OCR"),
            ("mode", "Mode"),
        ):
            group = tk.Frame(panel, bg=BG_PANEL)
            group.pack(side="left", padx=(0, 18))
            tk.Label(
                group, text=f"{label}:",
                font=self.fonts["meta"], fg=TEXT_MUTED, bg=BG_PANEL,
            ).pack(side="left")
            value = tk.Label(
                group, text="—",
                font=self.fonts["meta"], fg=TEXT_SECONDARY, bg=BG_PANEL,
            )
            value.pack(side="left", padx=(4, 0))
            self._diag_labels[key] = value

        self._reset_diagnostics()
        return panel

    def _set_diag(self, key: str, value: str, color: str = TEXT_SECONDARY) -> None:
        label = self._diag_labels.get(key)
        if label is not None:
            label.config(text=value or "—", fg=color)

    def _reset_diagnostics(self, mode: str | None = None) -> None:
        self._set_diag("quality", "—")
        self._set_diag("geotr", "—")
        self._set_diag("illtr", "—")
        self._set_diag("ocr", "—")
        self._set_diag("mode", mode or self._model_var.get())

    def _format_geotr_diag(self, geotr: dict) -> str:
        if not geotr:
            return "—"
        applied = bool(geotr.get("applied", True))
        if not applied:
            if geotr.get("reason") == "salvage_mode":
                return "not used"
            return "bypassed"
        shift = geotr.get("max_displacement")
        if not isinstance(shift, (int, float)):
            return "applied"
        if shift < 0.005:
            kind = "very subtle"
        elif shift < 0.030:
            kind = "subtle"
        else:
            kind = "visible"
        return f"applied, {kind} ({shift:.4f})"

    def _format_illtr_diag(self, illtr: dict) -> str:
        if not illtr:
            return "—"
        if illtr.get("applied"):
            return "applied"
        if illtr.get("reason") == "salvage_mode":
            return "not used"
        reason = str(illtr.get("reason", "unknown")).replace("_", " ")
        return f"skipped, {reason}"

    def _update_diagnostics_from_report(self, report: dict) -> None:
        qg = report.get("quality_gate", {}) or {}
        qg_result = qg.get("result")
        qg_conf = qg.get("confidence_percent")
        if qg_result is not None and qg_conf is not None:
            warning = qg.get("warning")
            suffix = f" · {warning}" if warning else ""
            self._set_diag("quality", f"{qg_result} · {qg_conf:.0f}%{suffix}")
        self._set_diag("geotr", self._format_geotr_diag(report.get("geotr", {})))
        self._set_diag("illtr", self._format_illtr_diag(report.get("illtr", {})))
        ocr = report.get("ocr", {}) or {}
        engine = ocr.get("engine_label") or ocr.get("engine")
        if engine:
            chars = ocr.get("chars")
            suffix = f" · {chars} chars" if isinstance(chars, int) else ""
            self._set_diag("ocr", f"{engine}{suffix}")
        self._set_diag("mode", report.get("processing_mode") or self._model_var.get())

    def _show_save_btn(self):
        self._save_btn.grid(row=0, column=0, padx=(0, 8))

    def _hide_save_btn(self):
        self._save_btn.grid_remove()

    def _show_progress(self):
        self._progress_bar.grid(row=0, column=0, sticky="ew")
        self._progress_running = True
        self._progress_pos = 0
        self._tick_progress()

    def _hide_progress(self):
        self._progress_running = False
        self._progress_bar.grid_remove()

    def _tick_progress(self):
        if not self._progress_running:
            return
        bar = self._progress_bar
        bar.delete("fill")
        w = bar.winfo_width()
        if w < 2:
            self.after(50, self._tick_progress)
            return
        fill_w = int(w * 0.35)
        x = int(self._progress_pos % (w + fill_w)) - fill_w // 2
        bar.create_rectangle(x, 0, x + fill_w, 2,
                             fill=ACCENT_INPUT, outline="", tags="fill")
        self._progress_pos += 4
        self.after(16, self._tick_progress)
