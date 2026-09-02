from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# A light, near-white palette, in place of ttk's own default gray.
BACKGROUND = "#fafafa"  # page background (root window, every Frame/Label)
SURFACE = "#ffffff"  # input-like widgets (Entry/Combobox fields, buttons, the output box)
TEXT = "#1a1a1a"
ACCENT = "#2563eb"  # the active tab, focused borders
ACCENT_SOFT = "#e8f0fe"  # hover/pressed/selected fill
BORDER = "#d9d9dc"


def apply_theme(root: tk.Tk) -> None:
    """A light theme built on ttk's 'clam' base theme (the least dated-looking of its built-in
    cross-platform themes) with colors overridden on top - falls back to whatever theme is already
    active if 'clam' isn't available on this Tk build.

    Call once, before building any other widget - `ttk.Style` configuration only affects widgets
    created (or restyled) after it runs.
    """
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(background=BACKGROUND)
    style.configure(".", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("TFrame", background=BACKGROUND)
    style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT)
    style.configure("TRadiobutton", background=BACKGROUND, foreground=TEXT)

    style.configure(
        "TButton", background=SURFACE, foreground=TEXT, borderwidth=1,
        relief="solid", bordercolor=BORDER, padding=(10, 5),
    )
    style.map(
        "TButton",
        background=[("active", ACCENT_SOFT), ("pressed", ACCENT_SOFT)],
        bordercolor=[("focus", ACCENT)],
    )

    style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT, padding=4, bordercolor=BORDER)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", fieldbackground=SURFACE, foreground=TEXT, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)])

    style.configure("TScrollbar", background=SURFACE, troughcolor=BACKGROUND, bordercolor=BORDER)

    # gui/tabbar.py's TabBar: flat buttons, the active one picked out with
    # the accent color rather than relying on a native OS tab look.
    style.configure(
        "Toolbutton", background=SURFACE, foreground=TEXT, padding=(10, 8),
        relief="flat", anchor="center", borderwidth=1, bordercolor=BORDER,
    )
    style.map(
        "Toolbutton",
        background=[("selected", ACCENT_SOFT), ("pressed", ACCENT_SOFT), ("active", ACCENT_SOFT)],
        foreground=[("selected", ACCENT)],
        bordercolor=[("selected", ACCENT)],
    )

    # gui/collapsible.py's CollapsibleSection header - left-aligned (reads
    # as a section title, not a centered action button) and bold, with no
    # border of its own since it's a fold/unfold toggle, not a real button.
    style.configure(
        "Section.TButton", background=BACKGROUND, foreground=TEXT, borderwidth=0,
        relief="flat", anchor="w", padding=(4, 6), font=("Segoe UI", 10, "bold"),
    )
    style.map("Section.TButton", background=[("active", ACCENT_SOFT)])
