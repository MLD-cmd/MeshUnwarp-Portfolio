import json
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk

from .theme import (
    BG_PANEL, BG_PANEL_INNER, BORDER_COLOR,
    ACCENT_OUTPUT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, TEXT_MUTED,
    STATUS_ERROR,
    category_display_name,
)


class PanelsMixin:

    def _build_image_panel(self, parent, col, label, sublabel,
                            accent_color, empty_text, empty_hint,
                            show_close: bool = True,
                            show_toggle: bool = False,
                            show_zoom: bool = True):
        """Returns (canvas, meta_label, outer_frame)."""
        outer = tk.Frame(parent, bg=BORDER_COLOR,
                         highlightbackground=BORDER_COLOR, highlightthickness=1)
        outer.grid(row=0, column=col, sticky="nsew",
                   padx=(12 if col == 0 else 6, 6 if col == 0 else 12),
                   pady=12)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = tk.Frame(outer, bg=BG_PANEL, height=36)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        tk.Frame(header, bg=accent_color, width=3).pack(side="left", fill="y")
        tk.Label(header, text=f"  {label}",
                 font=self.fonts["heading"],
                 fg=accent_color, bg=BG_PANEL).pack(side="left")
        tk.Label(header, text=f"  · {sublabel}",
                 font=self.fonts["heading"],
                 fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")

        zoom_buttons = []
        if show_zoom:
            zoom_minus = tk.Label(
                header, text="−",
                font=self.fonts["heading"],
                fg=TEXT_MUTED, bg=BG_PANEL,
                cursor="hand2", padx=6,
            )
            zoom_minus.pack(side="right")
            zoom_minus.bind("<Enter>",
                lambda e, b=zoom_minus: b.config(fg=TEXT_PRIMARY))
            zoom_minus.bind("<Leave>",
                lambda e, b=zoom_minus: b.config(fg=TEXT_MUTED))
            zoom_buttons.append(("minus", zoom_minus))

            zoom_plus = tk.Label(
                header, text="+",
                font=self.fonts["heading"],
                fg=TEXT_MUTED, bg=BG_PANEL,
                cursor="hand2", padx=6,
            )
            zoom_plus.pack(side="right")
            zoom_plus.bind("<Enter>",
                lambda e, b=zoom_plus: b.config(fg=TEXT_PRIMARY))
            zoom_plus.bind("<Leave>",
                lambda e, b=zoom_plus: b.config(fg=TEXT_MUTED))
            zoom_buttons.append(("plus", zoom_plus))

            sep = tk.Label(header, text="    ", fg=TEXT_MUTED, bg=BG_PANEL)
            sep.pack(side="right")
            zoom_buttons.append(("separator", sep))

            if col == 0:
                self._input_zoom_buttons = zoom_buttons
            else:
                self._output_zoom_buttons = zoom_buttons

            for name, btn in zoom_buttons:
                btn.pack_forget()

        if show_close:
            close_lbl = tk.Label(
                header, text="×",
                font=self.fonts["heading"],
                fg=TEXT_MUTED, bg=BG_PANEL,
                cursor="hand2", padx=10,
            )
            close_lbl.pack(side="right")
            close_lbl.bind("<Button-1>", self._clear_workspace)
            close_lbl.bind("<Enter>",
                lambda e, b=close_lbl: b.config(fg=STATUS_ERROR))
            close_lbl.bind("<Leave>",
                lambda e, b=close_lbl: b.config(fg=TEXT_MUTED))

        if show_toggle:
            toggle_lbl = tk.Label(
                header, text="TEXT",
                font=self.fonts["meta"],
                fg=TEXT_MUTED, bg=BG_PANEL,
                cursor="hand2", padx=10,
            )
            toggle_lbl.pack(side="right")
            toggle_lbl.bind("<Button-1>", self._toggle_output_view)
            toggle_lbl.bind("<Enter>",
                lambda e, b=toggle_lbl: b.config(fg=TEXT_PRIMARY))
            toggle_lbl.bind("<Leave>",
                lambda e, b=toggle_lbl: b.config(fg=TEXT_MUTED))
            self._output_toggle_lbl = toggle_lbl

        detail_kind = label.lower()
        if detail_kind in ("input", "output"):
            details_lbl = tk.Label(
                header, text="DETAILS",
                font=self.fonts["meta"],
                fg=accent_color, bg=BG_PANEL,
                cursor="hand2", padx=10,
            )
            details_lbl.pack(side="right")
            details_lbl.bind(
                "<Button-1>",
                lambda _e, kind=detail_kind: self._show_panel_details(kind),
            )
            details_lbl.bind("<Enter>",
                lambda e, b=details_lbl: b.config(fg=TEXT_PRIMARY))
            details_lbl.bind("<Leave>",
                lambda e, b=details_lbl, c=accent_color: b.config(fg=c))

        canvas = tk.Canvas(outer, bg=BG_PANEL_INNER,
                           highlightthickness=0, bd=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        canvas._empty_text = empty_text  # type: ignore[attr-defined]
        # Guard: _on_resize handles loaded images and constraint messages;
        # don't paint placeholder art over an existing image on resize.
        def _on_canvas_configure(_e, c=canvas):
            if c is getattr(self, "output_panel", None):
                if self._output_mode != "placeholder":
                    return
            elif self._current_pil is not None:
                return
            self._draw_placeholder(c)
        canvas.bind("<Configure>", _on_canvas_configure)

        tk.Frame(outer, bg=BORDER_COLOR, height=1).grid(
            row=2, column=0, sticky="ew")

        meta_frame = tk.Frame(outer, bg=BG_PANEL, height=24)
        meta_frame.grid(row=3, column=0, sticky="ew")
        meta_frame.grid_propagate(False)

        meta_label = tk.Label(
            meta_frame,
            text=f"  {empty_hint}",
            font=self.fonts["meta"],
            fg=TEXT_DIM, bg=BG_PANEL, anchor="w",
        )
        meta_label.pack(fill="x", expand=True, padx=4, pady=4)

        return canvas, meta_label, outer

    def _draw_placeholder(self, canvas):
        canvas.delete("placeholder")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 2 or h < 2:
            return

        cx, cy = w // 2, h // 2
        s = 28

        canvas.create_rectangle(
            cx - s, cy - s, cx + s, cy + s,
            outline=TEXT_DIM, width=1, dash=(4, 4), tags="placeholder")
        canvas.create_line(
            cx - s + 8, cy, cx + s - 8, cy,
            fill=TEXT_DIM, width=1, tags="placeholder")
        canvas.create_line(
            cx, cy - s + 8, cx, cy + s - 8,
            fill=TEXT_DIM, width=1, tags="placeholder")
        canvas.create_text(
            cx, cy + s + 20,
            text=getattr(canvas, "_empty_text", ""),
            font=self.fonts["placeholder"],
            fill=TEXT_DIM, justify="center", tags="placeholder")

    def _build_output_text_view(self, outer: tk.Frame) -> None:
        """OCR text widget sharing row 1 with the image canvas; visibility via grid_remove/grid."""
        frame = tk.Frame(outer, bg=BG_PANEL_INNER)
        frame.grid(row=1, column=0, sticky="nsew")

        text = tk.Text(
            frame, wrap="word",
            bg=BG_PANEL_INNER, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            selectbackground=ACCENT_OUTPUT,
            selectforeground=BG_PANEL_INNER,
            font=self.fonts["ocr_text"],
            spacing1=2, spacing3=4,
            relief="flat", borderwidth=0, highlightthickness=0,
            padx=22, pady=18,
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._output_text_frame = frame
        self._output_text_widget = text

        # grid_remove/grid instead of tkraise() — Canvas overrides tkraise() with its own item-stacking method.
        frame.grid_remove()

    def _set_output_text(self, text: str) -> None:
        widget = getattr(self, "_output_text_widget", None)
        if widget is None:
            return
        widget.config(state="normal")
        widget.delete("1.0", "end")
        if text.strip():
            widget.insert("1.0", text)
        else:
            widget.insert("1.0", "— no text extracted —")
        widget.config(state="disabled")

    def _set_output_view(self, view: str) -> None:
        if view not in ("image", "text"):
            return
        self._output_view = view
        toggle = getattr(self, "_output_toggle_lbl", None)
        text_frame = getattr(self, "_output_text_frame", None)
        if view == "image":
            if text_frame is not None:
                text_frame.grid_remove()
            self.output_panel.grid()
            if toggle is not None:
                toggle.config(text="TEXT")
        else:
            self.output_panel.grid_remove()
            if text_frame is not None:
                text_frame.grid()
            if toggle is not None:
                toggle.config(text="IMAGE")

    def _toggle_output_view(self, _event=None) -> None:
        self._set_output_view("text" if self._output_view == "image" else "image")

    def _display_input(self, pil_img: Image.Image):
        self._render_to_canvas(
            pil_img, self.input_panel, "_input_photo",
            max_edge=self._DISPLAY_MAX_EDGE,
        )

    def _display_output(self, pil_img: Image.Image):
        self._output_mode = "image"
        self._render_to_canvas(
            pil_img, self.output_panel, "_output_photo",
            max_edge=self._DISPLAY_MAX_EDGE,
        )

    def _render_to_canvas(self, pil_img: Image.Image,
                          canvas: tk.Canvas, store_attr: str,
                          max_edge: int | None = None):
        # Cache a display-size base PIL so pan/zoom redraws do not repeatedly
        # resize 7k phone photos on Tk's main thread.
        base = pil_img.copy()
        if max_edge is not None and max(base.size) > max_edge:
            base.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

        self._view_state[canvas] = {
            "scale": 1.0, "ox": 0, "oy": 0,
            "base": base,
        }
        self._redraw_canvas(canvas, store_attr)

    def _redraw_canvas(self, canvas: tk.Canvas, store_attr: str):
        # update_idletasks() avoided — on macOS Tk it causes header buttons to
        # ghost at both pre- and post-reflow positions. Defer instead.
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 2 or h < 2:
            self.after(16, lambda: self._redraw_canvas(canvas, store_attr))
            return
        pad = self.PANEL_PADDING
        max_w = max(w - pad * 2, 1)
        max_h = max(h - pad * 2, 1)

        state = self._view_state.get(canvas)
        if state is None:
            return
        base = state["base"]
        orig_w, orig_h = base.size

        fit_scale = min(max_w / orig_w, max_h / orig_h)
        eff_scale = fit_scale * state["scale"]
        disp_w = max(1, int(orig_w * eff_scale))
        disp_h = max(1, int(orig_h * eff_scale))

        resized = base.resize((disp_w, disp_h), Image.Resampling.BICUBIC)
        photo = ImageTk.PhotoImage(resized)
        setattr(self, store_attr, photo)
        canvas.delete("all")
        cx = w // 2 + state["ox"]
        cy = h // 2 + state["oy"]
        canvas.create_image(cx, cy, anchor="center", image=photo)

    def _clear_output(self):
        self.output_panel.delete("all")
        self._output_mode = "placeholder"
        # Skip the dashed-square placeholder while a run is in flight —
        # otherwise it flashes for a few frames between clear and the
        # first stage preview, reading as a phantom "×-in-box" icon.
        if not (self._pipeline and self._pipeline.is_running()):
            self._draw_placeholder(self.output_panel)
        self._clear_output_meta()
        self._output_text = ""
        self._set_output_text("")
        self._set_output_view("image")

    def _update_input_meta(self, path: Path, pil_img: Image.Image, cat_name: str):
        w, h    = pil_img.size
        size_mb = path.stat().st_size / (1024 * 1024)
        cat     = category_display_name(cat_name)
        parts = [
            f"  {path.name}", f"{w}x{h} px", f"{size_mb:.1f} MB", cat,
        ]
        self._input_meta.config(
            text="   ·   ".join(parts), fg=TEXT_SECONDARY,
        )

    def _format_input_details(self) -> str:
        path = getattr(self, "_current_path", None)
        pil_img = getattr(self, "_current_pil", None)
        cat_name = getattr(self, "_current_cat", None)
        if path is None or pil_img is None:
            return "No input file is selected yet."

        w, h = pil_img.size
        size_mb = path.stat().st_size / (1024 * 1024)
        lines = [
            "INPUT DETAILS",
            "",
            f"File: {path.name}",
            f"Path: {path}",
            f"Category: {category_display_name(cat_name) if cat_name else '-'}",
            f"Original display size: {w}x{h} px",
            f"File size: {size_mb:.1f} MB",
            f"Selected mode: {self._model_var.get()}",
        ]

        if self._last_qg_message:
            lines.extend(["", f"Quality Gate: {self._last_qg_message}"])
        if self._last_lighting_warning:
            lines.append(f"Lighting warning: {self._last_lighting_warning}")

        qg = (self._last_diagnostics or {}).get("quality_gate", {})
        if qg:
            lines.extend([
                "",
                "Quality Gate Details:",
                json.dumps(qg, indent=2, sort_keys=True, default=str),
            ])
        return "\n".join(lines)

    def _update_output_meta(self, pil_img: Image.Image):
        w, h = pil_img.size
        name = (f"output_{self._current_path.name}"
                if self._current_path else "output.jpg")
        qg = (self._last_diagnostics or {}).get("quality_gate", {})
        is_salvage = bool(
            ((qg.get("details") or {}).get("salvage_text") or {}).get("enabled")
        )
        output_kind = "SALVAGE SCAN" if is_salvage else "DE-WARPED"
        parts = [f"  {name}", f"{w}x{h} px", output_kind]
        if self._last_total_seconds > 0:
            tag = "cached" if self._last_run_cached else "fresh"
            parts.append(f"{self._last_total_seconds:.1f}s ({tag})")
        self._output_meta.config(
            text="   ·   ".join(parts), fg=TEXT_SECONDARY,
        )

    def _format_output_details(self) -> str:
        pil_img = getattr(self, "_output_pil", None)
        if pil_img is None:
            return "No output has been generated yet."

        w, h = pil_img.size
        name = (f"output_{self._current_path.name}"
                if self._current_path else "output.jpg")
        qg = (self._last_diagnostics or {}).get("quality_gate", {})
        is_salvage = bool(
            ((qg.get("details") or {}).get("salvage_text") or {}).get("enabled")
        )
        output_kind = "SALVAGE SCAN" if is_salvage else "DE-WARPED"
        lines = [
            "OUTPUT DETAILS",
            "",
            f"File: {name}",
            f"Output size: {w}x{h} px",
            f"Output type: {output_kind}",
            f"Selected mode: {self._model_var.get()}",
        ]

        if self._last_total_seconds > 0:
            tag = "cached" if self._last_run_cached else "fresh"
            lines.append(f"Runtime: {self._last_total_seconds:.1f}s ({tag})")

        if self._stage_timings:
            lines.extend(["", "Stage timings:"])
            for stage in sorted(self._stage_timings):
                lines.append(f"  S{stage}: {int(self._stage_timings[stage])} ms")

        if self._last_qg_message:
            lines.extend(["", f"Quality Gate: {self._last_qg_message}"])
        if self._last_dewarp_message:
            lines.append(f"GeoTr / IllTr: {self._last_dewarp_message}")
        if self._last_ocr_message:
            lines.append(f"OCR: {self._last_ocr_message}")

        if self._last_diagnostics:
            lines.extend([
                "",
                "Full Diagnostics:",
                json.dumps(
                    self._last_diagnostics,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
            ])
        return "\n".join(lines)

    def _show_panel_details(self, panel: str) -> None:
        if panel == "input":
            title = "Input Details"
            body = self._format_input_details()
        elif panel == "output":
            title = "Output Details"
            body = self._format_output_details()
        else:
            return

        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg=BG_PANEL)
        win.geometry("720x520")
        win.minsize(480, 320)
        win.transient(self)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        header = tk.Frame(win, bg=BG_PANEL, height=40)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        tk.Label(
            header, text=f"  {title.upper()}",
            font=self.fonts["heading"],
            fg=TEXT_PRIMARY, bg=BG_PANEL,
        ).pack(side="left", fill="y")
        close_lbl = tk.Label(
            header, text="CLOSE",
            font=self.fonts["meta"],
            fg=TEXT_MUTED, bg=BG_PANEL,
            cursor="hand2", padx=12,
        )
        close_lbl.pack(side="right")
        close_lbl.bind("<Button-1>", lambda _e: win.destroy())
        close_lbl.bind("<Enter>",
            lambda e, b=close_lbl: b.config(fg=TEXT_PRIMARY))
        close_lbl.bind("<Leave>",
            lambda e, b=close_lbl: b.config(fg=TEXT_MUTED))

        frame = tk.Frame(win, bg=BG_PANEL_INNER)
        frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text = tk.Text(
            frame,
            wrap="word",
            bg=BG_PANEL_INNER,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            selectbackground=ACCENT_OUTPUT,
            selectforeground=BG_PANEL_INNER,
            font=self.fonts["details_text"],
            spacing1=3,
            spacing3=6,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=16,
            pady=14,
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        text.insert("1.0", body)
        text.config(state="disabled")
        win.bind("<Escape>", lambda _e: win.destroy())
        win.focus_set()

    def _clear_input_meta(self):
        self._input_meta.config(text="  — no file selected —", fg=TEXT_DIM)

    def _clear_output_meta(self):
        self._output_meta.config(
            text="  — run de-warp to generate output —", fg=TEXT_DIM)

    def _show_constraint_message(self):
        canvas = self.output_panel
        canvas.delete("all")
        self._output_mode = "constraint"
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        cx, cy = w // 2, h // 2
        # Tag all items so resize can delete and redraw without ghost copies.
        canvas.create_text(cx, cy - 20, text="⚠",
                           font=("Menlo", 32), fill=STATUS_ERROR,
                           anchor="center", tags="constraint")
        canvas.create_text(cx, cy + 24, text="Content Unrecoverable",
                           font=self.fonts["heading"], fill=STATUS_ERROR,
                           anchor="center", tags="constraint")
        canvas.create_text(cx, cy + 48,
                           text="The physical state of this document\nprevents reliable content extraction.",
                           font=self.fonts["label"], fill=TEXT_MUTED,
                           anchor="center", justify="center", tags="constraint")
