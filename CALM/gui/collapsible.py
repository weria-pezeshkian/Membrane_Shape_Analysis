from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class CollapsibleSection(ttk.Frame):
    """A titled section whose body can be folded (hidden) or unfolded (shown) by clicking its own
    header - starts folded. Anything gridded into `.body` (its own row/column layout) shows or
    hides as a whole when the section is toggled; `.grid_remove()` (not `.grid_forget()`) is used
    so the body's own grid configuration survives being hidden and doesn't need rebuilding on
    every unfold.
    """

    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(parent)
        self._folded = True
        self._title = title
        self._label_var = tk.StringVar()

        self.columnconfigure(0, weight=1)
        header = ttk.Button(
            self, textvariable=self._label_var, command=self.toggle, style="Section.TButton",
        )
        header.grid(row=0, column=0, sticky="we", pady=(8, 2))

        self.body = ttk.Frame(self)
        self.body.grid(row=1, column=0, sticky="we")

        self._sync()

    def toggle(self) -> None:
        self._folded = not self._folded
        self._sync()

    def _sync(self) -> None:
        if self._folded:
            self.body.grid_remove()
            self._label_var.set(f"▸ {self._title}")
        else:
            self.body.grid()
            self._label_var.set(f"▾ {self._title}")
