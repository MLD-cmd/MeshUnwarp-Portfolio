"""
MeshUnwarp — Theme Constants
Modern Dark Studio visual language.
All palette, font, and category color definitions live here.
Import from main_dashboard.py — no logic, no behavior.
"""

import tkinter.font as tkfont

# Palette 
BG_MAIN        = "#111113"
BG_SIDEBAR     = "#161618"
BG_PANEL       = "#161618"
BG_PANEL_INNER = "#111113"
BG_ELEMENT     = "#1e1e20"
BORDER_COLOR   = "#252528"

ACCENT_INPUT   = "#5b8ff0"   # blue  — input panel
ACCENT_OUTPUT  = "#2ecc71"   # green — output panel

TEXT_PRIMARY   = "#f2f2f4"   # main headings
TEXT_SECONDARY = "#c8c8cc"   # metadata strip, file info
TEXT_MUTED     = "#b0b0b6"   # sidebar items, muted labels
TEXT_DIM       = "#8f8f98"   # placeholders, mode hints

STATUS_IDLE    = "#3a3a3f"
STATUS_OK      = "#2ecc71"
STATUS_RUN     = "#5b8ff0"
STATUS_ERROR   = "#e74c3c"

# Pill state backgrounds (simulated low-opacity tints on dark base)
PILL_IDLE_BG     = BG_ELEMENT
PILL_IDLE_BORDER = BORDER_COLOR
PILL_IDLE_TEXT   = TEXT_MUTED   # was TEXT_DIM (#3a3a3f → ~1.5:1 contrast — invisible)

PILL_OK_BG     = "#132218"
PILL_OK_BORDER = "#1e3828"
PILL_OK_TEXT   = "#2ecc71"

PILL_RUN_BG     = "#131828"
PILL_RUN_BORDER = "#1e2838"
PILL_RUN_TEXT   = "#5b8ff0"

PILL_ERR_BG     = "#221315"
PILL_ERR_BORDER = "#321520"
PILL_ERR_TEXT   = "#e74c3c"

# Category Colors 
CATEGORY_COLORS: dict[str, str] = {
    "crumpled": "#5b8ff0",   # blue
    "wet":      "#2ecc71",   # green
    "warped":   "#e67e22",   # orange
    "low_ink":  "#e74c3c",   # red
    "mixed":    "#9b59b6",   # purple
    "_other":   "#6b6b72",   # grey — Uncategorized
}


def category_color(name: str) -> str:
    """Return accent color for a category folder name; grey for unknown folders."""
    return CATEGORY_COLORS.get(name, "#6b6b72")


def category_display_name(name: str) -> str:
    """Return human-readable uppercase category label."""
    if name == "_other":
        return "UNCATEGORIZED"
    return name.upper().replace("_", " ")


# Fonts (must be called after Tk() is initialized)
def build_fonts() -> dict[str, tkfont.Font]:
    """Return named Font objects. Call once after Tk() is created."""
    return {
        "label":       tkfont.Font(family="Menlo", size=10, weight="normal"),
        "heading":     tkfont.Font(family="Menlo", size=12, weight="bold"),
        "title":       tkfont.Font(family="Menlo", size=14, weight="bold"),
        "btn":         tkfont.Font(family="Menlo", size=12, weight="bold"),
        "status":      tkfont.Font(family="Menlo", size=10, weight="bold"),
        "listbox":     tkfont.Font(family="Menlo", size=10),
        "cat_label":   tkfont.Font(family="Menlo", size=9,  weight="bold"),
        "meta":        tkfont.Font(family="Menlo", size=9,  weight="normal"),
        "tiny":        tkfont.Font(family="Menlo", size=8,  weight="bold"),
        "placeholder": tkfont.Font(family="Menlo", size=11),
        "details_text": tkfont.Font(family="Menlo", size=13, weight="normal"),
        "ocr_text":    tkfont.Font(family="Menlo", size=16, weight="normal"),
    }
