import tkinter as tk
from tkinter import ttk

from ._utils import USER_PAPERS_DIR, MODE_HINTS, truncate_filename, scan_categorized_papers, _Tooltip
from .theme import (
    BG_SIDEBAR, BG_ELEMENT, BORDER_COLOR,
    ACCENT_INPUT, TEXT_PRIMARY, TEXT_DIM, TEXT_MUTED,
    category_color, category_display_name,
)


class SidebarMixin:

    def _build_sidebar(self):
        sb = tk.Frame(self, bg=BG_SIDEBAR, width=220)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.columnconfigure(0, weight=1)
        sb.rowconfigure(2, weight=1)

        self._build_sidebar_logo(sb)

        tk.Frame(sb, bg=BORDER_COLOR, height=1).grid(
            row=1, column=0, sticky="ew", padx=10)

        self._build_sidebar_file_area(sb, row=2)
        self._build_sidebar_bottom(sb, row=3)

    def _build_sidebar_logo(self, parent):
        container = tk.Frame(parent, bg=BG_SIDEBAR)
        container.grid(row=0, column=0, sticky="ew", padx=14, pady=(18, 12))

        logo_row = tk.Frame(container, bg=BG_SIDEBAR)
        logo_row.pack(fill="x")
        tk.Label(logo_row, text="MESH", font=self.fonts["title"],
                 fg=ACCENT_INPUT, bg=BG_SIDEBAR).pack(side="left")
        tk.Label(logo_row, text="UNWARP", font=self.fonts["title"],
                 fg=TEXT_PRIMARY, bg=BG_SIDEBAR).pack(side="left")

        tk.Label(container, text="COMPUTER VISION TOOL",
                 font=self.fonts["tiny"],
                 fg=TEXT_DIM, bg=BG_SIDEBAR, anchor="w",
                 ).pack(fill="x", pady=(3, 0))

    def _build_sidebar_file_area(self, parent, row):
        container = tk.Frame(parent, bg=BG_SIDEBAR)
        container.grid(row=row, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._cat_canvas = tk.Canvas(
            container, bg=BG_SIDEBAR, highlightthickness=0, bd=0)
        self._cat_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self._cat_canvas.yview,
            style="Dark.Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._cat_scrollbar = scrollbar
        self._cat_canvas.configure(yscrollcommand=scrollbar.set)

        self._cat_frame = tk.Frame(self._cat_canvas, bg=BG_SIDEBAR)
        self._cat_win = self._cat_canvas.create_window(
            (0, 0), window=self._cat_frame, anchor="nw")

        self._cat_frame.bind(
            "<Configure>",
            lambda _e: self._update_sidebar_scrollregion())
        self._cat_canvas.bind(
            "<Configure>",
            lambda e: self._on_sidebar_canvas_configure(e))

        for widget in (self._cat_canvas, self._cat_frame):
            self._bind_scroll(widget)

    def _on_sidebar_canvas_configure(self, event) -> None:
        self._cat_canvas.itemconfigure(self._cat_win, width=event.width)
        self._update_sidebar_scrollregion()

    def _update_sidebar_scrollregion(self) -> None:
        bbox = self._cat_canvas.bbox("all")
        if bbox is None:
            self._cat_canvas.configure(scrollregion=(0, 0, 0, 0))
            return

        canvas_h = max(1, self._cat_canvas.winfo_height())
        content_h = max(1, bbox[3] - bbox[1])
        scroll_h = max(canvas_h, content_h)
        self._cat_canvas.configure(
            scrollregion=(0, 0, max(1, bbox[2]), scroll_h),
        )

        scrollbar = getattr(self, "_cat_scrollbar", None)
        if content_h <= canvas_h:
            self._cat_canvas.yview_moveto(0.0)
            if scrollbar is not None and scrollbar.winfo_ismapped():
                scrollbar.grid_remove()
        elif scrollbar is not None and not scrollbar.winfo_ismapped():
            scrollbar.grid(row=0, column=1, sticky="ns")

    def _scroll_sidebar(self, units: int) -> str:
        bbox = self._cat_canvas.bbox("all")
        if bbox is None:
            return "break"
        content_h = bbox[3] - bbox[1]
        canvas_h = self._cat_canvas.winfo_height()
        if content_h <= canvas_h:
            self._cat_canvas.yview_moveto(0.0)
            return "break"
        self._cat_canvas.yview_scroll(units, "units")
        return "break"

    def _bind_scroll(self, widget: tk.Widget) -> None:
        """Forward mousewheel events on widget to the sidebar canvas."""
        def _mousewheel(event):
            if event.delta == 0:
                return "break"
            units = int(-event.delta / 120)
            if units == 0:
                units = -1 if event.delta > 0 else 1
            return self._scroll_sidebar(units)

        widget.bind("<MouseWheel>",
            _mousewheel)
        widget.bind("<Button-4>",   # Linux scroll up
            lambda _e: self._scroll_sidebar(-1))
        widget.bind("<Button-5>",   # Linux scroll down
            lambda _e: self._scroll_sidebar(1))

    def _build_sidebar_bottom(self, parent, row):
        frame = tk.Frame(parent, bg=BG_SIDEBAR)
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        tk.Frame(frame, bg=BORDER_COLOR, height=1).grid(
            row=0, column=0, sticky="ew", padx=10, pady=(4, 0))

        self._make_btn(
            frame, text="↻  Refresh List",
            command=self._refresh_list,
            bg=BG_ELEMENT, fg=TEXT_MUTED,
            active_bg=BORDER_COLOR,
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=6)

        tk.Frame(frame, bg=BORDER_COLOR, height=1).grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 4))

        tk.Label(frame, text="PROCESSING MODE",
                 font=self.fonts["tiny"],
                 fg=TEXT_DIM, bg=BG_SIDEBAR, anchor="w",
                 ).grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 2))

        self.model_combo = ttk.Combobox(
            frame,
            textvariable=self._model_var,
            values=["Clean Scan", "Ink Recovery"],
            state="readonly",
            style="Eng.TCombobox",
            font=("Menlo", 10),
        )
        self.model_combo.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 4))

        self._mode_hint = tk.Label(
            frame,
            text=MODE_HINTS["Clean Scan"],
            font=self.fonts["tiny"],
            fg=TEXT_DIM, bg=BG_SIDEBAR,
            anchor="w", wraplength=180,
        )
        self._mode_hint.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))
        self._model_var.trace_add("write", self._on_mode_change)

    def _build_category_section(self, cat_name: str, filenames: list[str]):
        color   = category_color(cat_name)
        display = category_display_name(cat_name)

        hdr = tk.Frame(self._cat_frame, bg=BG_SIDEBAR)
        hdr.pack(fill="x", padx=8, pady=(6, 2))
        self._bind_scroll(hdr)

        lbl = tk.Label(hdr, text=display,
                       font=self.fonts["cat_label"],
                       fg=TEXT_MUTED, bg=BG_SIDEBAR, anchor="w")
        lbl.pack(side="left")
        self._bind_scroll(lbl)
        self._cat_label_widgets[cat_name] = lbl

        tk.Label(hdr, text=f" {len(filenames)} ",
                 font=self.fonts["tiny"],
                 fg=color, bg=BG_ELEMENT,
                 padx=4, pady=1, relief="flat",
                 ).pack(side="right", padx=2)

        for fname in filenames:
            full_path = (
                USER_PAPERS_DIR / fname
                if cat_name == "_other"
                else USER_PAPERS_DIR / cat_name / fname
            )
            item = tk.Label(
                self._cat_frame,
                text=f"  {truncate_filename(fname)}",
                font=self.fonts["listbox"],
                fg=TEXT_MUTED, bg=BG_SIDEBAR,
                anchor="w", cursor="hand2",
            )
            item.pack(fill="x", padx=4, pady=1)
            item._file_path = full_path   # type: ignore[attr-defined]
            item._cat_name  = cat_name    # type: ignore[attr-defined]

            item.bind("<Button-1>",
                lambda e, i=item, p=full_path, c=cat_name:
                    self._on_file_click(i, p, c))
            item.bind("<Enter>",
                lambda e, i=item: self._on_item_enter(i))
            item.bind("<Leave>",
                lambda e, i=item: self._on_item_leave(i))

            self._bind_scroll(item)

            if len(fname) > 22:
                _Tooltip(item, fname)

            self._file_items.append(item)

    def _on_item_enter(self, item: tk.Label):
        if item is not self._selected_item:
            item.config(bg=BG_ELEMENT, fg="#999999")

    def _on_item_leave(self, item: tk.Label):
        if item is self._selected_item:
            item.config(bg=BG_ELEMENT, fg=TEXT_PRIMARY)
        else:
            item.config(bg=BG_SIDEBAR, fg=TEXT_MUTED)

    def _refresh_list(self):
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and pipeline.is_running():
            return
        for w in self._cat_frame.winfo_children():
            w.destroy()
        self._cat_label_widgets.clear()
        self._file_items.clear()
        self._selected_item = None
        self._selected_cat  = None

        self.title(self._BASE_TITLE)

        categories = scan_categorized_papers()

        if not categories:
            tk.Label(
                self._cat_frame,
                text="  (no images found)",
                font=self.fonts["listbox"],
                fg=TEXT_DIM, bg=BG_SIDEBAR, anchor="w",
            ).pack(fill="x", padx=8, pady=8)
        else:
            first = True
            for cat_name, filenames in categories.items():
                if not first:
                    tk.Frame(self._cat_frame, bg=BORDER_COLOR, height=1).pack(
                        fill="x", padx=8, pady=4)
                self._build_category_section(cat_name, filenames)
                first = False

        self._cat_canvas.yview_moveto(0.0)
        self.after_idle(self._update_sidebar_scrollregion)
        self._set_state("idle")
