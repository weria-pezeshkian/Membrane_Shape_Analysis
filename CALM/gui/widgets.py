from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk

from . import theme
from .introspect import FieldSpec
from .path_specs import PathSpec


class _Tooltip:
    """A small popup showing `text` while the mouse hovers over `widget`.

    Tkinter has no built-in tooltip widget - this is the standard
    bind-Enter/Leave-to-a-borderless-Toplevel pattern. A no-op if `text`
    is empty (an argparse action with no help string).
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip: tk.Toplevel | None = None
        if not text:
            return
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: object = None) -> None:
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 5
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip, text=self._text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, wraplength=420, padx=6, pady=3,
        )
        label.pack()

    def _hide(self, _event: object = None) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class FieldWidget:
    """One field's row of widgets, plus a way to read (for Start) and set (for loading a replay
    file) its current value, in the shape `gui/runner.py::build_argv` expects: str for text/choice,
    bool for bool/bool_optional, list[str] for multi/multichoice."""

    def __init__(self, get_value: Callable[[], object], set_value: Callable[[object], None]) -> None:
        self.get_value = get_value
        self.set_value = set_value


def _browse(entry_var: tk.StringVar, path_spec: PathSpec, cwd_getter: Callable[[], str]) -> None:
    initialdir = cwd_getter() or None
    if path_spec.mode == "dir":
        chosen = filedialog.askdirectory(initialdir=initialdir)
    else:
        filetypes = []
        if path_spec.extensions:
            pattern = " ".join(f"*{ext}" for ext in path_spec.extensions)
            filetypes.append((", ".join(path_spec.extensions), pattern))
        filetypes.append(("All files", "*.*"))
        if path_spec.mode == "open":
            chosen = filedialog.askopenfilename(initialdir=initialdir, filetypes=filetypes)
        else:
            chosen = filedialog.asksaveasfilename(initialdir=initialdir, filetypes=filetypes)
    if chosen:
        entry_var.set(chosen)


def build_field_row(
    parent: tk.Widget, row: int, spec: FieldSpec, path_spec: PathSpec | None, cwd_getter: Callable[[], str],
) -> FieldWidget:
    """Build one field's label + input (+ Browse button, for a path-like field) as a grid row in
    `parent` at `row`, and return a `FieldWidget` to read its current value back.

    `path_spec` is None for every field except the ones `gui/path_specs.py`
    explicitly lists - -n/--index, in particular, is deliberately absent
    there (it is a plain MDAnalysis selection string after the
    --index-file split, not a path), so it falls through to the plain text
    entry below with no Browse button, same as any other string field.
    """
    label_text = spec.flag + (" *" if spec.required else "")
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
    _Tooltip(label, spec.help)

    if spec.kind in ("bool", "bool_optional"):
        var = tk.BooleanVar(value=bool(spec.default))
        widget: tk.Widget = ttk.Checkbutton(parent, variable=var)
        widget.grid(row=row, column=1, sticky="w", pady=2)
        _Tooltip(widget, spec.help)
        return FieldWidget(var.get, lambda v: var.set(bool(v)))

    if spec.kind == "choice":
        default = str(spec.default) if spec.default is not None else (spec.choices or [""])[0]
        text_var = tk.StringVar(value=default)
        widget = ttk.Combobox(parent, textvariable=text_var, values=spec.choices or [], state="readonly")
        widget.grid(row=row, column=1, sticky="we", pady=2)
        _Tooltip(widget, spec.help)
        return FieldWidget(text_var.get, lambda v: text_var.set(str(v) if v is not None else default))

    if spec.kind == "multichoice":
        default_selected = set(spec.default) if spec.default else set()
        choice_vars = {choice: tk.BooleanVar(value=choice in default_selected) for choice in (spec.choices or [])}
        summary_var = tk.StringVar()

        def update_summary() -> None:
            selected = [c for c, v in choice_vars.items() if v.get()]
            summary_var.set(", ".join(selected) if selected else "(none selected)")

        def open_popup() -> None:
            popup = tk.Toplevel(parent, background=theme.BACKGROUND)
            popup.title(spec.flag)
            popup.transient(parent.winfo_toplevel())
            for choice, var in choice_vars.items():
                ttk.Checkbutton(popup, text=choice, variable=var, command=update_summary).pack(
                    anchor="w", padx=10, pady=2
                )
            ttk.Button(popup, text="Done", command=popup.destroy).pack(pady=8)

        def set_multichoice(values: object) -> None:
            selected: set[str] = set(values) if isinstance(values, (list, tuple, set)) else set()
            for choice, var in choice_vars.items():
                var.set(choice in selected)
            update_summary()

        update_summary()
        widget = ttk.Button(parent, textvariable=summary_var, command=open_popup)
        widget.grid(row=row, column=1, sticky="we", pady=2)
        _Tooltip(widget, spec.help)
        return FieldWidget(lambda: [c for c, v in choice_vars.items() if v.get()], set_multichoice)

    # "multi" and "text" both start as a plain Entry; "multi" only differs
    # in how its value is read back below (space-separated -> a list).
    if isinstance(spec.default, (list, tuple)):
        default_text = " ".join(str(v) for v in spec.default)
    elif spec.default is None:
        default_text = ""
    else:
        default_text = str(spec.default)
    text_var = tk.StringVar(value=default_text)
    entry = ttk.Entry(parent, textvariable=text_var)
    entry.grid(row=row, column=1, sticky="we", pady=2)
    _Tooltip(entry, spec.help)

    if path_spec is not None:
        browse_button = ttk.Button(
            parent, text="Browse...", command=lambda: _browse(text_var, path_spec, cwd_getter)
        )
        browse_button.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=2)

    def set_text(value: object) -> None:
        if isinstance(value, (list, tuple)):
            text_var.set(" ".join(str(v) for v in value))
        elif value is None:
            text_var.set("")
        else:
            text_var.set(str(value))

    if spec.kind == "multi":
        return FieldWidget(lambda: text_var.get().split(), set_text)
    return FieldWidget(text_var.get, set_text)
