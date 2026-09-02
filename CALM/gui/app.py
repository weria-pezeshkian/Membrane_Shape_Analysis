from __future__ import annotations

import queue
import tempfile
import tkinter as tk
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from ..core.argument_parser import load_replay_args
from ..core.manual import load_manual_markdown, render_markdown_as_html
from . import theme
from .collapsible import CollapsibleSection
from .introspect import MODULES, CommandSpec, FieldSpec, field_specs
from .path_specs import path_spec_for
from .runner import CommandRunner, build_argv
from .tabbar import TabBar
from .widgets import FieldWidget, build_field_row

_TITLE = "Keep CALM and analyze lipid membranes"


def _bind_mousewheel(canvas: tk.Canvas) -> None:
    """Scroll `canvas` on the mouse wheel while the pointer is over it.

    A plain `canvas.bind("<MouseWheel>", ...)` only fires while the
    pointer is directly over the canvas's own (mostly empty) background,
    not any of the field widgets gridded on top of it - the standard fix
    is to bind app-wide only while the pointer is inside the canvas
    (tracked via Enter/Leave on the canvas itself) and unbind again on
    Leave, so a different tab's own canvas isn't hijacked once the
    pointer moves there. `<Button-4>`/`<Button-5>` are the wheel events on
    X11 (Linux); `<MouseWheel>` (with `event.delta`) covers Windows/macOS -
    both are bound unconditionally since only the platform's own real one
    ever actually fires.

    `canvas.yview()` is checked before scrolling: when the whole form
    already fits in the visible area (e.g. a short command like
    'link vmd_xtc'), or the wheel would move past whichever edge (top/
    bottom) is already fully in view, `yview_scroll` is skipped entirely -
    otherwise the canvas happily scrolls its content into empty space
    above or below itself with nothing left to show there.
    """

    def on_wheel(event: tk.Event) -> None:
        top, bottom = canvas.yview()
        if top <= 0.0 and bottom >= 1.0:
            return  # the whole form already fits - nothing to scroll

        scroll_up = event.num == 4 or (event.num != 5 and event.delta > 0)
        if scroll_up:
            if top > 0.0:
                canvas.yview_scroll(-1, "units")
        else:
            if bottom < 1.0:
                canvas.yview_scroll(1, "units")

    def bind_wheel(_event: object = None) -> None:
        canvas.bind_all("<MouseWheel>", on_wheel)
        canvas.bind_all("<Button-4>", on_wheel)
        canvas.bind_all("<Button-5>", on_wheel)

    def unbind_wheel(_event: object = None) -> None:
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", bind_wheel)
    canvas.bind("<Leave>", unbind_wheel)


class CommandTab:
    """One subcommand's auto-built form: a scrollable frame holding a "Required" and an
    "Optional" `CollapsibleSection` (each starting folded), sorting `FieldSpec`s by their own
    `required` flag - the same split every CALM parser's own `--help` output already uses. A
    Manual button and a Load-replay button sit fixed above the scroll area so both stay reachable
    regardless of how far a long form (e.g. 'analyze diffusion', 31 fields) has been scrolled.
    """

    def __init__(self, parent: tk.Widget, command: CommandSpec, cwd_getter: Callable[[], str]) -> None:
        self.command = command
        self.specs: list[FieldSpec] = field_specs(command.build_parser())
        self._widgets: dict[str, FieldWidget] = {}

        self.frame = ttk.Frame(parent)

        # Right-aligned as a group, "Manual" reading left of "Load
        # replay..." - pack(side="right") stacks each newly-packed widget
        # to the left of the previous one, so the one meant to sit at the
        # far right edge has to be packed first.
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(toolbar, text="Load replay...", command=self._load_replay_dialog).pack(side="right")
        ttk.Button(toolbar, text="Manual", command=self._open_manual).pack(side="right", padx=(0, 6))

        scroll_area = ttk.Frame(self.frame)
        scroll_area.pack(fill="both", expand=True)
        canvas = tk.Canvas(scroll_area, highlightthickness=0, background=theme.BACKGROUND)
        scrollbar = ttk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        _bind_mousewheel(canvas)

        form.columnconfigure(0, weight=1)
        required_specs = [s for s in self.specs if s.required]
        optional_specs = [s for s in self.specs if not s.required]
        for title, specs in (("Required", required_specs), ("Optional", optional_specs)):
            if not specs:
                continue
            section = CollapsibleSection(form, title)
            section.pack(fill="x")
            section.body.columnconfigure(1, weight=1)
            for row, spec in enumerate(specs):
                path_spec = path_spec_for(command.label, spec.dest)
                self._widgets[spec.dest] = build_field_row(section.body, row, spec, path_spec, cwd_getter)

    def _open_manual(self) -> None:
        markdown = load_manual_markdown(self.command.manual_name)
        title = f"CALM {' '.join(self.command.argv_prefix)}"
        html_page = render_markdown_as_html(markdown, title=title)
        tmp_path = Path(tempfile.gettempdir()) / f"calm_manual_{self.command.manual_name}.html"
        tmp_path.write_text(html_page, encoding="utf-8")
        webbrowser.open(tmp_path.as_uri())

    def _load_replay_dialog(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Replay log", "*.log"), ("All files", "*.*")])
        if path:
            self.load_replay(path)

    def load_replay(self, path: str) -> None:
        """Repopulate every field from a saved replay file.

        `load_replay_args` (already used by every CLI command's own
        `--replay`) returns a flat, shlex-split token list; re-parsing it
        with this command's own parser gives back the same Namespace a
        real run would have used, so no separate replay-format parsing is
        needed here.
        """
        tokens = load_replay_args(path)
        ns = self.command.build_parser().parse_args(tokens)
        for spec in self.specs:
            self._widgets[spec.dest].set_value(getattr(ns, spec.dest, None))

    def values(self) -> dict[str, object]:
        return {dest: widget.get_value() for dest, widget in self._widgets.items()}


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(_TITLE)
        root.geometry("960x880")
        theme.apply_theme(root)

        ttk.Label(root, text=_TITLE, font=("Segoe UI", 16, "bold")).pack(pady=(10, 4))

        cwd_frame = ttk.Frame(root)
        cwd_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(cwd_frame, text="Working directory:").pack(side="left")
        self.cwd_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(cwd_frame, textvariable=self.cwd_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(cwd_frame, text="Browse...", command=self._browse_cwd).pack(side="left")

        self.outer_bar = TabBar(root, auto_select_first=False)
        self.outer_bar.pack(fill="both", expand=True, padx=10, pady=6)

        # Shown until the user picks a real module tab - landing on an
        # already-selected Calibrate tab with Map's inner tabs still visible
        # underneath (the bug fixed by outer_bar's own auto_select_first
        # handling above) read as broken; a blank landing page makes "no
        # module is chosen yet" visually obvious instead of implying one is.
        landing = ttk.Frame(self.outer_bar.content)
        landing.grid(row=0, column=0, sticky="nsew")
        ttk.Label(landing, text="CALM", font=("Segoe UI", 48, "bold")).pack(expand=True)

        # Maps each outer tab's own module name to how to find its
        # currently-active CommandTab - see _active_tab. "Calibrate" has
        # no subcommands (a single CommandSpec, not a list), so it goes
        # straight into _single_tabs; every other module gets its own
        # inner TabBar (whose own `.active` gives the selected subcommand's
        # label) plus a label -> CommandTab lookup, in _multi_tabs.
        self._single_tabs: dict[str, CommandTab] = {}
        self._multi_tabs: dict[str, tuple[TabBar, dict[str, CommandTab]]] = {}

        for module_name, entry in MODULES.items():
            outer_tab = self.outer_bar.add(module_name)

            if isinstance(entry, CommandSpec):
                tab = CommandTab(outer_tab, entry, self.cwd_var.get)
                tab.frame.pack(fill="both", expand=True)
                self._single_tabs[module_name] = tab
            else:
                inner_bar = TabBar(outer_tab)
                inner_bar.pack(fill="both", expand=True)
                command_tabs: dict[str, CommandTab] = {}
                for command in entry:
                    inner_tab = inner_bar.add(command.label)
                    tab = CommandTab(inner_tab, command, self.cwd_var.get)
                    tab.frame.pack(fill="both", expand=True)
                    command_tabs[command.label] = tab
                self._multi_tabs[module_name] = (inner_bar, command_tabs)

        # Every module's own frame lands on top of its siblings as it's
        # gridded in (same reason TabBar.add re-raises the active tab on
        # every call) - re-raise the landing page last so it's what's
        # actually on top at startup, regardless of build order, until the
        # user clicks a real tab and TabBar's own _on_select raises that
        # tab's frame above it.
        landing.tkraise()
        self.outer_bar.bind_change(self._on_outer_tab_selected)

        ttk.Label(root, text="Output:").pack(anchor="w", padx=10)
        self.output_box = scrolledtext.ScrolledText(
            root, height=12, font=("Courier", 10), state="disabled",
            background=theme.SURFACE, foreground=theme.TEXT,
            relief="solid", borderwidth=1, highlightthickness=0,
        )
        self.output_box.pack(fill="both", expand=False, padx=10, pady=(0, 6))
        self.output_box.vbar.configure(background=theme.SURFACE, troughcolor=theme.BACKGROUND)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=(0, 10))
        self.start_button = ttk.Button(button_frame, text="Start", command=self._start)
        self.start_button.pack(side="left", padx=4)
        self.start_button.state(["disabled"])  # re-enabled once a real module tab is picked
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)

        self.runner = CommandRunner()

    def _browse_cwd(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.cwd_var.get() or None)
        if chosen:
            self.cwd_var.set(chosen)

    def _active_tab(self) -> CommandTab:
        """Whichever CommandTab the outer (and, if applicable, inner) TabBar currently has
        selected - Start always runs this one."""
        module_name = self.outer_bar.active
        if module_name in self._single_tabs:
            return self._single_tabs[module_name]
        inner_bar, command_tabs = self._multi_tabs[module_name]
        return command_tabs[inner_bar.active]

    def _on_outer_tab_selected(self) -> None:
        """Runs the first time the user picks a real module tab (dismissing the landing page) and
        every time after - Start stays disabled until there's an actual command tab to run, and
        (guarded by is_running) doesn't get re-enabled by switching tabs mid-run."""
        if not self.runner.is_running():
            self.start_button.state(["!disabled"])

    def _start(self) -> None:
        if self.runner.is_running():
            return
        tab = self._active_tab()
        argv = build_argv(tab.command.argv_prefix, tab.specs, tab.values())
        self._append_output(f"$ {' '.join(argv)}\n")
        try:
            self.runner.start(argv, cwd=self.cwd_var.get())
        except Exception as exc:
            self._append_output(f"Failed to start: {exc}\n")
            return
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.root.after(100, self._poll_output)

    def _stop(self) -> None:
        self.runner.stop()

    def _poll_output(self) -> None:
        finished = False
        while True:
            try:
                item = self.runner.output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                finished = True
                break
            self._append_output(item + "\n")

        if finished:
            self.start_button.state(["!disabled"])
            self.stop_button.state(["disabled"])
            self._append_output("(finished)\n")
        else:
            self.root.after(100, self._poll_output)

    def _append_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.insert("end", text)
        self.output_box.see("end")
        self.output_box.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
