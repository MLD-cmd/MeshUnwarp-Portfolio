from pathlib import Path
import tkinter as tk

BASE_DIR        = Path(__file__).parent.parent
USER_PAPERS_DIR = BASE_DIR / "datasets" / "user_papers"
CATEGORY_DIRS   = ["crumpled", "wet", "warped", "low_ink", "mixed", "partial", "unrecognizable"]
SUPPORTED_EXT   = {".jpg", ".jpeg", ".png"}

MODE_HINTS: dict[str, str] = {
    "Clean Scan":   "Pristine white page with black text",
    "Ink Recovery": "Faint or low-ink text rescue",
}


def truncate_filename(name: str, max_len: int = 22) -> str:
    if len(name) <= max_len:
        return name
    p = Path(name)
    return p.stem[:18] + "…" + p.suffix


def scan_categorized_papers(base_dir: Path | None = None) -> dict[str, list[str]]:
    """
    Scan USER_PAPERS_DIR subfolders and root-level files.
    Returns {category_name: [sorted filenames]}.
    Root-level files are grouped under '_other' (Uncategorized).
    Accepts optional base_dir for testability.
    """
    root = base_dir if base_dir is not None else USER_PAPERS_DIR
    result: dict[str, list[str]] = {}

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            files = sorted(
                f.name for f in entry.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
            )
            if files:
                result[entry.name] = files

    root_files = sorted(
        f.name for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    )
    if root_files:
        result["_other"] = root_files

    return result


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._tip = None
        widget.bind("<Enter>", lambda e: self._show(widget, text), add="+")
        widget.bind("<Leave>", lambda e: self._hide(), add="+")

    def _show(self, widget, text):
        from .theme import TEXT_PRIMARY
        if self._tip:
            return
        x = widget.winfo_rootx() + 6
        y = widget.winfo_rooty() + widget.winfo_height() + 2
        self._tip = tk.Toplevel(widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=text, font=("Menlo", 8),
            fg=TEXT_PRIMARY, bg="#2a2a2f", padx=8, pady=4).pack()

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None
