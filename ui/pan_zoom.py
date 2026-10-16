import tkinter as tk


class PanZoomMixin:

    _MIN_ZOOM  = 1.0
    _MAX_ZOOM  = 8.0
    _ZOOM_STEP = 1.15

    def _enable_pan_zoom(self):
        if self._pan_zoom_enabled:
            return
        self._pan_zoom_enabled = True
        for canvas in (self.input_panel, self.output_panel):
            canvas.config(cursor="fleur")
            canvas.bind("<ButtonPress-1>", self._on_pan_start)
            canvas.bind("<B1-Motion>", self._on_pan_drag)
            canvas.bind("<ButtonRelease-1>", self._on_pan_end)
            canvas.bind("<Double-Button-1>", self._on_view_reset)

        for name, btn in self._input_zoom_buttons:
            if name == "plus":
                btn.bind("<Button-1>",
                    lambda _: self._zoom(self.input_panel, self._ZOOM_STEP))
            elif name == "minus":
                btn.bind("<Button-1>",
                    lambda _: self._zoom(self.input_panel, 1 / self._ZOOM_STEP))
            btn.pack(side="right")

        for name, btn in self._output_zoom_buttons:
            if name == "plus":
                btn.bind("<Button-1>",
                    lambda _: self._zoom(self.output_panel, self._ZOOM_STEP))
            elif name == "minus":
                btn.bind("<Button-1>",
                    lambda _: self._zoom(self.output_panel, 1 / self._ZOOM_STEP))
            btn.pack(side="right")

    def _disable_pan_zoom(self):
        if not self._pan_zoom_enabled:
            return
        self._pan_zoom_enabled = False
        self._drag_anchor = None
        self._drag_canvas = None
        for canvas in (self.input_panel, self.output_panel):
            canvas.config(cursor="")
            for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>",
                        "<Double-Button-1>"):
                canvas.unbind(seq)

        for name, btn in self._input_zoom_buttons:
            btn.unbind("<Button-1>")
            btn.pack_forget()
        for name, btn in self._output_zoom_buttons:
            btn.unbind("<Button-1>")
            btn.pack_forget()

        for canvas, attr in ((self.input_panel, "_input_photo"),
                             (self.output_panel, "_output_photo")):
            state = self._view_state.get(canvas)
            if state is not None:
                state["scale"] = 1.0
                state["ox"] = 0
                state["oy"] = 0
                self._redraw_canvas(canvas, attr)

    def _store_attr_for(self, canvas: tk.Canvas) -> str:
        return "_input_photo" if canvas is self.input_panel else "_output_photo"

    def _on_pan_start(self, event):
        if not self._pan_zoom_enabled:
            return
        self._drag_canvas = event.widget
        self._drag_anchor = (event.x, event.y)

    def _on_pan_drag(self, event):
        if (not self._pan_zoom_enabled or self._drag_anchor is None
                or event.widget is not self._drag_canvas):
            return
        canvas = self._drag_canvas
        state = self._view_state.get(canvas)
        if state is None:
            return
        dx = event.x - self._drag_anchor[0]
        dy = event.y - self._drag_anchor[1]
        state["ox"] += dx
        state["oy"] += dy
        self._drag_anchor = (event.x, event.y)
        self._clamp_pan(canvas)
        self._redraw_canvas(canvas, self._store_attr_for(canvas))

    def _on_pan_end(self, event):
        self._drag_anchor = None
        self._drag_canvas = None

    def _zoom(self, canvas: tk.Canvas, factor: float):
        if not self._pan_zoom_enabled:
            return
        state = self._view_state.get(canvas)
        if state is None:
            return
        new_scale = max(self._MIN_ZOOM,
                        min(self._MAX_ZOOM, state["scale"] * factor))
        if abs(new_scale - state["scale"]) < 1e-3:
            return
        w = canvas.winfo_width(); h = canvas.winfo_height()
        cx, cy = w // 2 + state["ox"], h // 2 + state["oy"]
        ratio = new_scale / state["scale"]
        state["ox"] = int((cx - w // 2) * ratio + (w // 2 - cx) * (1 - ratio))
        state["oy"] = int((cy - h // 2) * ratio + (h // 2 - cy) * (1 - ratio))
        state["scale"] = new_scale
        self._clamp_pan(canvas)
        self._redraw_canvas(canvas, self._store_attr_for(canvas))

    def _on_view_reset(self, event):
        if not self._pan_zoom_enabled:
            return
        canvas = event.widget
        state = self._view_state.get(canvas)
        if state is None:
            return
        state["scale"] = 1.0
        state["ox"] = 0
        state["oy"] = 0
        self._redraw_canvas(canvas, self._store_attr_for(canvas))

    def _clamp_pan(self, canvas: tk.Canvas):
        state = self._view_state.get(canvas)
        if state is None:
            return
        w = canvas.winfo_width(); h = canvas.winfo_height()
        pad = self.PANEL_PADDING
        max_w = max(w - pad * 2, 1); max_h = max(h - pad * 2, 1)
        base_w, base_h = state["base"].size
        fit_scale = min(max_w / base_w, max_h / base_h)
        disp_w = base_w * fit_scale * state["scale"]
        disp_h = base_h * fit_scale * state["scale"]
        # Allow image edge to reach canvas center but no further.
        max_ox = int(disp_w / 2)
        max_oy = int(disp_h / 2)
        state["ox"] = max(-max_ox, min(max_ox, state["ox"]))
        state["oy"] = max(-max_oy, min(max_oy, state["oy"]))
