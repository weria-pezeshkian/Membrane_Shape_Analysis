from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class TabBar(ttk.Frame):
    """A row of equal-width, centered-text buttons spanning the full available width - stretching
    or shrinking together as the window is resized - that switches between content frames stacked
    in the same cell.

    `ttk.Notebook`'s own tab strip has no option to stretch every tab to
    fill its container (each tab sizes to its own text/padding, full
    stop) - this replaces it with `ttk.Radiobutton`s using ttk's built-in
    "Toolbutton" style (flat, shows a pressed look for whichever one is
    selected), one per grid column with equal weight, so the row divides
    the available width evenly and grows/shrinks with the window.
    """

    def __init__(self, parent: tk.Misc, auto_select_first: bool = True) -> None:
        """`auto_select_first` (default True) selects and raises the first tab as soon as it's
        added - the right behavior for a bar the user is meant to land on a real tab of (e.g. a
        module's own subcommand bar). Pass False for a bar that should start on neither: nothing
        is selected or raised until `select` is called explicitly, e.g. to leave a separate
        placeholder frame (gridded into `.content` directly, bypassing `add`) visible until the
        user actually picks a tab - see gui/app.py's outer module bar.
        """
        super().__init__(parent)
        self._auto_select_first = auto_select_first

        self._button_row = ttk.Frame(self)
        self._button_row.pack(fill="x")
        self._content = ttk.Frame(self)
        self._content.pack(fill="both", expand=True)
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        self._active_var = tk.StringVar()
        self._frames: dict[str, ttk.Frame] = {}
        self._on_change: Callable[[], None] | None = None
        self._n_columns = 0

    @property
    def content(self) -> ttk.Frame:
        """The shared cell every tab's own frame is stacked into - exposed so a caller can grid a
        placeholder frame into the same cell directly, without going through `add` (which always
        creates a matching button too)."""
        return self._content

    def add(self, label: str) -> ttk.Frame:
        """Add one tab, returning its own (empty) content frame to build the tab's contents into."""
        column = self._n_columns
        self._n_columns += 1
        self._button_row.columnconfigure(column, weight=1)

        ttk.Radiobutton(
            self._button_row, text=label, value=label, variable=self._active_var,
            style="Toolbutton", command=self._on_select,
        ).grid(row=0, column=column, sticky="we")

        frame = ttk.Frame(self._content)
        frame.grid(row=0, column=0, sticky="nsew")
        self._frames[label] = frame

        if self._auto_select_first and not self._active_var.get():
            self._active_var.set(label)

        # Every newly-gridded frame lands on top of its siblings by
        # default, silently displacing whichever tab was previously
        # selected and raised - re-assert the actually-active one every
        # time a tab is added (not just the first), so a bar with several
        # tabs built in sequence (gui/app.py's outer bar builds all of
        # Analyze/Link/Map after Calibrate is already selected) stays
        # visually correct regardless of what gets added afterward.
        active = self._active_var.get()
        if active in self._frames:
            self._frames[active].tkraise()
        return frame

    def select(self, label: str) -> None:
        self._active_var.set(label)
        self._frames[label].tkraise()

    def _on_select(self) -> None:
        self._frames[self._active_var.get()].tkraise()
        if self._on_change is not None:
            self._on_change()

    @property
    def active(self) -> str:
        return self._active_var.get()

    def bind_change(self, callback: Callable[[], None]) -> None:
        """`callback` runs every time the user picks a different tab (not when `select` is called
        programmatically)."""
        self._on_change = callback
