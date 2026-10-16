import json
import platform
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from pathlib import Path
from PIL import Image

from .theme import (
    BG_MAIN, BG_SIDEBAR, BG_ELEMENT,
    ACCENT_INPUT, ACCENT_OUTPUT,
    TEXT_PRIMARY, TEXT_MUTED,
    STATUS_RUN,
    PILL_RUN_BG, PILL_RUN_BORDER, PILL_RUN_TEXT,
    category_color, build_fonts,
)

from ._utils import (
    USER_PAPERS_DIR, CATEGORY_DIRS,
)
from .sidebar import SidebarMixin
from .panels import PanelsMixin
from .controls import ControlsMixin
from .pan_zoom import PanZoomMixin

USER_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
for _cat in CATEGORY_DIRS:
    (USER_PAPERS_DIR / _cat).mkdir(exist_ok=True)


class MeshUnwarpApp(SidebarMixin, PanelsMixin, ControlsMixin, PanZoomMixin, tk.Tk):

    PANEL_PADDING     = 12
    _DISPLAY_MAX_EDGE = 2400
    _BASE_TITLE       = "MeshUnwarp  ·  Document De-warping Tool"

    def __init__(self):
        super().__init__()
        self.title(self._BASE_TITLE)
        self.configure(bg=BG_MAIN)
        self.minsize(1100, 680)

        # macOS Retina HiDPI workaround
        if platform.system() == "Darwin":
            try:
                self.tk.call("tk", "scaling", 2.0)
            except Exception:
                pass

        self._current_pil:   Image.Image | None = None
        self._output_pil:    Image.Image | None = None
        self._input_photo:   object | None = None
        self._output_photo:  object | None = None
        self._current_path:  Path | None = None
        self._current_cat:   str | None = None

        self._stage_timings:         dict[int, float] = {}
        self._last_qg_message:       str = ""
        self._last_lighting_warning: str = ""
        self._last_dewarp_message:   str = ""
        self._last_ocr_message:      str = ""
        self._last_total_seconds:    float = 0.0
        self._last_run_cached:       bool = False
        self._last_diagnostics:      dict = {}
        self._diag_labels:           dict[str, tk.Label] = {}

        # Output panel mode: "placeholder" | "image" | "constraint"
        self._output_mode: str = "placeholder"
        self._output_text: str = ""
        self._output_view: str = "image"

        self._selected_item:     tk.Label | None = None
        self._selected_cat:      str | None = None
        self._cat_label_widgets: dict[str, tk.Label] = {}
        self._file_items:        list[tk.Label] = []

        self._progress_running = False
        self._progress_pos     = 0
        self._resize_after_id: str | None = None
        self._model_var = tk.StringVar(value="Clean Scan")

        # Pan/zoom enabled only after a successful dewarp run.
        self._pan_zoom_enabled = False
        self._view_state: dict[tk.Canvas, dict] = {}
        self._drag_anchor: tuple[int, int] | None = None
        self._drag_canvas: tk.Canvas | None = None
        self._input_zoom_buttons: list = []
        self._output_zoom_buttons: list = []

        self.fonts = build_fonts()
        self._build_ttk_styles()
        self._build_layout()
        self._refresh_list()
        self._bind_keyboard()
        self.bind("<Configure>", self._on_resize)
        self.after(100, self._center_window)

        # Callbacks fire on the listener thread — after() bounces them to Tk.
        from orchestration.runner import PipelineController
        self._pipeline = PipelineController()
        self._pipeline.on_ready    = lambda: self.after(0, self._on_pipeline_ready)
        self._pipeline.on_stage    = lambda s, st, m, e=None: self.after(0, self._on_pipeline_stage, s, st, m, e)
        self._pipeline.on_preview  = lambda s, img, meta: self.after(0, self._on_pipeline_preview, s, img, meta)
        self._pipeline.on_complete = lambda img, text, total, cached, diag=None: self.after(0, self._on_pipeline_complete, img, text, total, cached, diag)
        self._pipeline.on_reject   = lambda m: self.after(0, self._on_pipeline_reject, m)
        self._pipeline.on_error    = lambda m: self.after(0, self._on_pipeline_error, m)
        self._pipeline.on_cancel   = lambda: self.after(0, self._on_pipeline_cancel)
        self._set_state("loading")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self.columnconfigure(0, weight=0, minsize=220)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _build_main(self):
        main = tk.Frame(self, bg=BG_MAIN)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        self.input_panel, self._input_meta, self._input_outer = \
            self._build_image_panel(
                main, col=0,
                label="INPUT", sublabel="User's Paper",
                accent_color=ACCENT_INPUT,
                empty_text="Select an image\nfrom the sidebar",
                empty_hint="— no file selected —",
                show_close=False,
            )
        self.output_panel, self._output_meta, self._output_outer = \
            self._build_image_panel(
                main, col=1,
                label="OUTPUT", sublabel="Proper Content",
                accent_color=ACCENT_OUTPUT,
                empty_text="Awaiting\nProcessing",
                empty_hint="— run de-warp to generate output —",
                show_close=True,
                show_toggle=True,
            )
        self._build_output_text_view(self._output_outer)
        self._build_controls(main, row=1)

    def _on_pipeline_ready(self):
        if self._current_pil and self._current_path:
            self._set_state("ready", self._current_path.name)
        else:
            self._set_state("idle")

    def _on_pipeline_stage(self, stage: int, state: str, message: str,
                           elapsed_ms: float | None = None):
        idx = stage - 1
        if state == "start":
            self._set_chip_state(idx, "running")
            label = (self._STAGE_LABELS[idx]
                     if 0 <= idx < len(self._STAGE_LABELS)
                     else f"Stage {stage}")
            tail = f" · {message}" if message else ""
            self._update_pill(STATUS_RUN, PILL_RUN_BG, PILL_RUN_BORDER,
                              PILL_RUN_TEXT, f"Processing · {label}{tail}")
        elif state == "done":
            self._set_chip_state(idx, "complete")
            if elapsed_ms is not None:
                self._stage_timings[stage] = elapsed_ms
            if stage == 2 and message:
                self._last_qg_message = message
                self._set_diag("quality", message)
            elif stage == 3 and message:
                self._last_dewarp_message = message
            elif stage == 5 and message:
                self._last_ocr_message = message
                self._set_diag("ocr", message.replace("OCR: ", "", 1))

    def _on_pipeline_preview(self, stage: int, pil_image: Image.Image, meta: dict):
        self._display_output(pil_image)
        if stage == 1:
            warning = (meta or {}).get("lighting", {}).get("warning", "")
            self._last_lighting_warning = warning
            if warning:
                self._set_state("running", f"⚠ Lighting: {warning}")
        elif stage == 3:
            self._set_diag("geotr", self._format_geotr_diag((meta or {}).get("geotr", {})))
            self._set_diag("illtr", self._format_illtr_diag((meta or {}).get("illumination", {})))

    def _on_pipeline_complete(self, pil_image: Image.Image,
                              text: str = "",
                              total_seconds: float = 0.0,
                              cached: bool = False,
                              diagnostics: dict | None = None):
        self._output_pil = pil_image
        self._output_text = text
        self._last_total_seconds = total_seconds
        self._last_run_cached    = cached
        self._last_diagnostics   = diagnostics or {}
        self._update_diagnostics_from_report(self._last_diagnostics)
        self._set_output_text(text)
        self._display_output(pil_image)
        self._update_output_meta(pil_image)
        if self._current_path and self._current_pil and self._current_cat:
            self._update_input_meta(
                self._current_path, self._current_pil, self._current_cat,
            )
        self._set_state("complete")

    def _on_pipeline_reject(self, message: str):
        self._show_constraint_message()
        self._set_state("error", "Content Unrecoverable")

    def _on_pipeline_error(self, message: str):
        self._set_state("error", message)

    def _on_pipeline_cancel(self):
        self._reset_chips()
        self._clear_output()
        if self._current_pil and self._current_path:
            self._set_state("ready", self._current_path.name)
        else:
            self._set_state("idle")

    def _on_close(self):
        """Reap the worker subprocess before tearing down the Tk window."""
        try:
            self._pipeline.shutdown()
        except Exception:
            pass
        self.destroy()

    def _on_file_click(self, item: tk.Label, path: Path, cat_name: str):
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and pipeline.is_running():
            return "break"

        if self._selected_item is not None:
            self._selected_item.config(bg=BG_SIDEBAR, fg=TEXT_MUTED)
        if self._selected_cat and self._selected_cat in self._cat_label_widgets:
            self._cat_label_widgets[self._selected_cat].config(fg=TEXT_MUTED)

        item.config(bg=BG_ELEMENT, fg=TEXT_PRIMARY)
        self._selected_item = item
        self._selected_cat  = cat_name
        if cat_name in self._cat_label_widgets:
            self._cat_label_widgets[cat_name].config(fg=category_color(cat_name))

        if not path.exists():
            self._set_state("error", f"File not found: {path.name}")
            return

        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            self._set_state("error", f"Could not open: {exc}")
            return

        self._current_pil  = pil_img
        self._current_path = path
        self._current_cat  = cat_name
        self._output_pil   = None
        self._output_photo = None
        self._output_text = ""
        self._last_qg_message = ""
        self._last_lighting_warning = ""
        self._last_dewarp_message = ""
        self._last_ocr_message = ""
        self._last_total_seconds = 0.0
        self._last_run_cached = False
        self._last_diagnostics = {}
        self._stage_timings.clear()
        self._reset_diagnostics()
        self._reset_chips()

        self._clear_output()
        self._display_input(pil_img)
        self._update_input_meta(path, pil_img, cat_name)
        self._set_state("ready", path.name)

        self.title(f"MeshUnwarp  ·  {path.name}")

        self._pulse_panel(self._input_outer)

    def _clear_workspace(self, event=None) -> None:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and pipeline.is_running():
            return "break"
        # Disable pan/zoom before wiping state to avoid a phantom redraw.
        self._disable_pan_zoom()
        self._view_state.clear()

        self._current_pil  = None
        self._output_pil   = None
        self._current_path = None
        self._current_cat  = None
        self._input_photo  = None
        self._output_photo = None
        self._stage_timings.clear()
        self._last_qg_message = ""
        self._last_lighting_warning = ""
        self._last_dewarp_message = ""
        self._last_ocr_message = ""
        self._last_total_seconds = 0.0
        self._last_run_cached = False
        self._last_diagnostics = {}
        self._reset_diagnostics()
        self._reset_chips()
        self._output_mode = "placeholder"

        self.input_panel.delete("all")
        self._draw_placeholder(self.input_panel)
        self.output_panel.delete("all")
        self._draw_placeholder(self.output_panel)
        # Snap output back to image view (prevents TEXT frame obscuring cleared canvas).
        self._output_text = ""
        self._set_output_text("")
        self._set_output_view("image")

        self._clear_input_meta()
        self._clear_output_meta()

        if self._selected_item is not None:
            self._selected_item.config(bg=BG_SIDEBAR, fg=TEXT_MUTED)
            self._selected_item = None
        if self._selected_cat and self._selected_cat in self._cat_label_widgets:
            self._cat_label_widgets[self._selected_cat].config(fg=TEXT_MUTED)
            self._selected_cat = None

        self.title(self._BASE_TITLE)
        self._set_state("idle")

    def _run_dewarp(self):
        if self._current_pil is None or self._current_path is None:
            return
        if not self._pipeline.is_ready() or self._pipeline.is_running():
            return
        mode = self._model_var.get()
        self._disable_pan_zoom()
        self._stage_timings.clear()
        self._last_qg_message = ""
        self._last_lighting_warning = ""
        self._last_dewarp_message = ""
        self._last_ocr_message = ""
        self._last_total_seconds = 0.0
        self._last_run_cached = False
        self._last_diagnostics = {}
        self._reset_diagnostics(mode)
        self._reset_chips()
        self._display_input(self._current_pil)
        self._clear_output()
        self._set_state("running", mode)
        self._pipeline.run(str(self._current_path), mode)

    def _cancel_dewarp(self):
        if self._pipeline.is_running():
            self._pipeline.cancel()

    def _save_output(self):
        if self._output_pil is None:
            return
        default_name = (
            f"output_{self._current_path.stem}.jpg"
            if self._current_path else "output.jpg"
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")],
            initialfile=default_name,
        )
        if path:
            self._output_pil.save(path)
            self._save_diagnostics_report(Path(path))

    def _save_diagnostics_report(self, output_path: Path) -> None:
        report_path = output_path.with_name(f"{output_path.stem}_diagnostics.json")
        diagnostics = dict(self._last_diagnostics or {})
        diagnostics.update({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "output_file": output_path.name,
            "output_path": str(output_path),
        })
        diagnostics.setdefault("source_file", (
            self._current_path.name if self._current_path else None
        ))
        diagnostics.setdefault("processing_mode", self._model_var.get())
        diagnostics.setdefault("quality_gate", {})
        diagnostics.setdefault("geotr", {})
        diagnostics.setdefault("illtr", {})
        diagnostics.setdefault("ocr", {})

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, sort_keys=True)

        self._set_state("complete", f"saved {output_path.name} + diagnostics")

    def _bind_keyboard(self):
        self.bind_all("<Escape>", self._clear_workspace)
        self.bind("<Return>",
            lambda e: self._run_dewarp() if self._current_pil else None)
        self.bind("r", lambda e: self._refresh_list())
        self.bind("R", lambda e: self._refresh_list())
        self.bind("s",
            lambda e: self._save_output() if self._output_pil else None)
        self.bind("S",
            lambda e: self._save_output() if self._output_pil else None)
        self.bind("<Up>",   self._navigate_up)
        self.bind("<Down>", self._navigate_down)

    def _navigate_up(self, event=None):
        if self._pipeline.is_running():
            return "break"
        if not self._file_items:
            return
        if self._selected_item is None:
            target = self._file_items[-1]
        else:
            idx = self._file_items.index(self._selected_item)
            if idx == 0:
                return
            target = self._file_items[idx - 1]
        self._on_file_click(
            target,
            target._file_path,   # type: ignore[attr-defined]
            target._cat_name,    # type: ignore[attr-defined]
        )

    def _navigate_down(self, event=None):
        if self._pipeline.is_running():
            return "break"
        if not self._file_items:
            return
        if self._selected_item is None:
            target = self._file_items[0]
        else:
            idx = self._file_items.index(self._selected_item)
            if idx == len(self._file_items) - 1:
                return
            target = self._file_items[idx + 1]
        self._on_file_click(
            target,
            target._file_path,   # type: ignore[attr-defined]
            target._cat_name,    # type: ignore[attr-defined]
        )

    def _on_resize(self, event=None):
        if event is not None and event.widget is not self:
            return
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(90, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_after_id = None
        if self._current_pil is not None:
            if self.input_panel in self._view_state:
                self._redraw_canvas(self.input_panel, "_input_photo")
            else:
                self._display_input(self._current_pil)
        else:
            self._draw_placeholder(self.input_panel)

        if self._output_pil is not None and self._output_view == "image":
            if self.output_panel in self._view_state:
                self._redraw_canvas(self.output_panel, "_output_photo")
            else:
                self._display_output(self._output_pil)
        elif self._output_mode == "constraint":
            self._show_constraint_message()
        else:
            self._draw_placeholder(self.output_panel)

    def _center_window(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w  = min(1400, sw - 80)
        h  = min(860,  sh - 80)
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


if __name__ == "__main__":
    app = MeshUnwarpApp()
    app.mainloop()
