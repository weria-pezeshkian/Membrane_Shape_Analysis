from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..analyze.diffusion import _build_diffusion_parser
from ..analyze.full import _build_full_parser
from ..analyze.lipids import _build_lipids_parser
from ..analyze.sft import _build_sft_parser
from ..calibrate.enter import _build_calibrate_parser
from ..map.diffusion_plot import _build_diffusion_plot_parser
from ..map.dynamic_plot import _build_dynamic_plot_parser
from ..map.lipids_plot import _build_lipids_plot_parser
from ..map.plot import _build_plot_parser
from ..map.radial_plot import _build_radial_plot_parser
from ..utilize.vmd_vectors import _build_vmd_vectors_parser
from ..utilize.vmd_xtc import _build_vmd_xtc_parser
from ..utilize.write_ndx import _build_write_ndx_parser


@dataclass
class FieldSpec:
    """One argparse Action's shape, exactly as the GUI's form generator needs it - nothing more.

    `flag` is the long option string used to build argv (or the only
    option string, for a long-only flag like `--index-file`). `kind` is
    one of:
    - "bool" - store_true/store_false: a plain Checkbutton.
    - "bool_optional" - BooleanOptionalAction (`--clear`): a Checkbutton
      that emits an explicit `--flag`/`--no-flag` either way.
    - "choice" - `choices` given, single value: a single-select dropdown.
    - "multichoice" - `choices` given, `nargs` in ("+", "*"): a
      multi-select (several of the fixed choices at once, e.g. --method).
    - "multi" - `nargs` in ("+", "*"), no `choices` (e.g. --lipids): a
      text entry, space-separated, same syntax as the CLI itself.
    - "text" - everything else (str/int/float/a custom `type=` callable):
      a plain text entry. The GUI never needs to know what a custom type=
      callable does - the value is passed through as a string and
      validated for real by the actual CLI subprocess at Start.

    `required` drives which `CollapsibleSection` (gui/app.py) a field's
    row lands in - it is *not* simply the action's own argparse `required=`
    flag (see `_field_spec`): several fields are `required=False` at the
    argparse level but still genuinely required in a broader either/or
    sense argparse itself can't express (--sft vs. -f/-s/-n in 'analyze
    full'; -n/--index vs. --index-file everywhere), and the command's own
    parser already says so by which argument group it put the flag in.
    """

    dest: str
    option_strings: list[str]
    flag: str
    kind: str
    help: str
    required: bool
    default: Any
    choices: list[str] | None = None


_REQUIRED_GROUP_TITLE = "Required arguments"
_OPTIONAL_GROUP_TITLE = "Optional arguments"


def field_specs(parser: argparse.ArgumentParser) -> list[FieldSpec]:
    """Every user-facing argument `parser` accepts, as `FieldSpec`s - the single source the GUI's
    form generator builds every widget from. Skips fire-and-exit actions (-h/--help, --man,
    -v/--version, SUPPRESS-defaulted) and positionals (no CALM command has one)."""
    group_titles = {
        id(action): group.title
        for group in parser._action_groups
        for action in group._group_actions
    }
    return [
        _field_spec(action, group_titles.get(id(action))) for action in parser._actions
        if action.option_strings and action.default is not argparse.SUPPRESS
    ]


def _field_spec(action: argparse.Action, group_title: str | None) -> FieldSpec:
    flag = next((s for s in action.option_strings if s.startswith("--")), action.option_strings[-1])

    if isinstance(action, argparse.BooleanOptionalAction):
        kind = "bool_optional"
    elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        kind = "bool"
    elif action.choices is not None and action.nargs in ("+", "*"):
        kind = "multichoice"
    elif action.choices is not None:
        kind = "choice"
    elif action.nargs in ("+", "*"):
        kind = "multi"
    else:
        kind = "text"

    # Prefer the argparse group a command explicitly placed this action in
    # over the action's own `required=` flag: several commands (every one
    # built on `add_build_arguments`) put a flag under "Required arguments"
    # even though it's `required=False` at the argparse level, because the
    # real requirement is an either/or argparse itself can't express -
    # --sft vs. -f/-s/-n in 'analyze full', or -n/--index vs. --index-file
    # everywhere. Falling back to `action.required` only matters for the
    # handful of commands with no custom "Required"/"Optional" groups at
    # all (e.g. 'link vmd_vectors'), where it's the only signal available.
    if group_title == _REQUIRED_GROUP_TITLE:
        required = True
    elif group_title == _OPTIONAL_GROUP_TITLE:
        required = False
    else:
        required = bool(getattr(action, "required", False))

    return FieldSpec(
        dest=action.dest,
        option_strings=list(action.option_strings),
        flag=flag,
        kind=kind,
        help=action.help or "",
        required=required,
        default=action.default,
        choices=[str(c) for c in action.choices] if action.choices is not None else None,
    )


@dataclass
class CommandSpec:
    """One leaf command's identity for the GUI: its tab label, the argv prefix `CALM` itself needs
    (e.g. `["analyze", "sft"]`, or just `["calibrate"]` for the one command with no submodule), the
    manual name `core/manual.py` knows it by, and its own parser builder."""

    label: str
    argv_prefix: list[str]
    manual_name: str
    build_parser: Callable[[], argparse.ArgumentParser]


# One entry per top-level GUI tab. Calibrate has no subcommands of its own
# (a single CommandSpec, not a list) - every other module is a list of its
# own subcommands, in the same order `CALM <module> -h` lists them.
MODULES: dict[str, CommandSpec | list[CommandSpec]] = {
    "Calibrate": CommandSpec("Calibrate", ["calibrate"], "calibrate", _build_calibrate_parser),
    "Analyze": [
        CommandSpec("sft", ["analyze", "sft"], "analyze_sft", _build_sft_parser),
        CommandSpec("full", ["analyze", "full"], "analyze_full", _build_full_parser),
        CommandSpec("lipids", ["analyze", "lipids"], "analyze_lipids", _build_lipids_parser),
        CommandSpec("diffusion", ["analyze", "diffusion"], "analyze_diffusion", _build_diffusion_parser),
    ],
    "Link": [
        CommandSpec("write_ndx", ["link", "write_ndx"], "link_write_ndx", _build_write_ndx_parser),
        CommandSpec("vmd_xtc", ["link", "vmd_xtc"], "link_vmd_xtc", _build_vmd_xtc_parser),
        CommandSpec("vmd_vectors", ["link", "vmd_vectors"], "link_vmd_vectors", _build_vmd_vectors_parser),
    ],
    "Map": [
        CommandSpec("plot", ["map", "plot"], "map_plot", _build_plot_parser),
        CommandSpec("dynamic_plot", ["map", "dynamic_plot"], "map_dynamic_plot", _build_dynamic_plot_parser),
        CommandSpec("radial_plot", ["map", "radial_plot"], "map_radial_plot", _build_radial_plot_parser),
        CommandSpec("lipids_plot", ["map", "lipids_plot"], "map_lipids_plot", _build_lipids_plot_parser),
        CommandSpec(
            "diffusion_plot", ["map", "diffusion_plot"], "map_diffusion_plot", _build_diffusion_plot_parser
        ),
    ],
}
