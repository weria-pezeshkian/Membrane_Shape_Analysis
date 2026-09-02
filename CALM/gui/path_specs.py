from __future__ import annotations

from dataclasses import dataclass

_TRAJECTORY_EXTS = [".xtc", ".trr", ".dcd", ".nc", ".pdb"]
_STRUCTURE_EXTS = [".tpr", ".gro", ".pdb", ".psf"]


@dataclass
class PathSpec:
    """How a Browse button should behave for one field.

    `mode` is "open" (pick an existing file), "save" (choose a path to
    write - doesn't need to exist yet), or "dir" (choose a directory).
    `extensions` (unused for "dir") restricts which files the dialog
    highlights - argparse has no native concept of "this is a path", let
    alone which extensions are legal, so this table is the one place that
    has to be curated by hand (see `CALM/gui/introspect.py::FieldSpec`).
    """

    mode: str
    extensions: list[str] | None = None


# Keyed by (command label, argparse dest) - the same dest can mean a
# directory in one command and a file in another (-o/--out is an output
# *directory* for every 'analyze' command, but a single .ndx *file* for
# 'write_ndx'), so scoping by command is required, not optional.
#
# -n/--index is deliberately absent everywhere: after the --index-file
# split, it is purely a dynamic MDAnalysis selection string, not a path at
# all, so the generic fallback (plain text entry, no Browse button, from
# CALM/gui/widgets.py) applies to it automatically.
PATH_SPECS: dict[tuple[str, str], PathSpec] = {}

for _command in ("sft", "full", "lipids", "diffusion"):
    PATH_SPECS[(_command, "trajectory")] = PathSpec("open", _TRAJECTORY_EXTS)
    PATH_SPECS[(_command, "structure")] = PathSpec("open", _STRUCTURE_EXTS)
    PATH_SPECS[(_command, "index_file")] = PathSpec("open", [".ndx"])
    PATH_SPECS[(_command, "out")] = PathSpec("dir")
    PATH_SPECS[(_command, "replay")] = PathSpec("open", [".log"])
    PATH_SPECS[(_command, "out_replay")] = PathSpec("save", [".log"])

PATH_SPECS[("full", "sft")] = PathSpec("dir")  # reusing a precomputed analyze-sft/full output dir

PATH_SPECS[("Calibrate", "sft")] = PathSpec("dir")
PATH_SPECS[("Calibrate", "out")] = PathSpec("save")  # physics-implementation-defined format

PATH_SPECS[("write_ndx", "trajectory")] = PathSpec("open", _TRAJECTORY_EXTS)
PATH_SPECS[("write_ndx", "structure")] = PathSpec("open", _STRUCTURE_EXTS)
PATH_SPECS[("write_ndx", "out")] = PathSpec("save", [".ndx"])

for _command in ("vmd_xtc", "vmd_vectors"):
    PATH_SPECS[(_command, "input")] = PathSpec("dir")
    PATH_SPECS[(_command, "output")] = PathSpec("dir")

for _command, _ext in (
    ("plot", ".png"), ("dynamic_plot", ".gif"), ("radial_plot", ".png"),
    ("lipids_plot", ".png"), ("diffusion_plot", ".png"),
):
    PATH_SPECS[(_command, "numpys_directory")] = PathSpec("dir")
    PATH_SPECS[(_command, "outfile")] = PathSpec("save", [_ext])


def path_spec_for(command_label: str, dest: str) -> PathSpec | None:
    """The PathSpec for this command's own `dest`, or None if it isn't a path at all - the generic
    plain-text-entry fallback (see `CALM/gui/widgets.py`) applies then."""
    return PATH_SPECS.get((command_label, dest))
