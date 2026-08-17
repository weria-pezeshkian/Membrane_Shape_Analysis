from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.manual import add_manual
from ..core.rotation import fixed_circle_radius
from .plot import _clip_to_circle, _frame_filtered_glob, get_XY

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"


def _read_species(Dir: str) -> list[str]:
    """Species names, in the fixed order indexing every {frame}_lipid_fractions.npy's leading axis."""
    return Path(Dir, "lipid_species.txt").read_text().split()


def _mean_fractions(Dir: str, frame_numbers: Iterable[int] | None) -> np.ndarray:
    """Trajectory-mean per-species occupancy: shape (n_species, 2, gridsize, gridsize), NaN where holed.

    Each {frame}_lipid_fractions.npy is a hard per-point species
    assignment (1 for that frame's own nearest species, 0 for the rest -
    see 'CALM analyze lipids --man'). The plain mean across frames (not a
    NaN-aware one - same convention 'CALM map plot' uses) turns this into
    each point's own occupancy *frequency* over the averaging window - 1.0
    where one species always wins there, lower where the winner changes
    frame to frame (e.g. a boundary between two species' territory).

    If a matching {frame}_hole_mask.npy exists (--Remove-TMD was used),
    every species' value at a holed point in that frame is set to NaN
    before averaging, so a point holed in even one frame ends up NaN in
    the trajectory mean too, rather than being silently diluted by frames
    where it had no real lipid competing at all.
    """
    files = _frame_filtered_glob(os.path.join(Dir, "*_lipid_fractions.npy"), frame_numbers)
    if not files:
        raise FileNotFoundError(f"No *_lipid_fractions.npy files found in {Dir}")

    frames = []
    for f in files:
        arr = np.load(f).astype(float)
        hole_file = f.replace("_lipid_fractions.npy", "_hole_mask.npy")
        if os.path.exists(hole_file):
            hole = np.load(hole_file)  # (2, gridsize, gridsize), [upper, lower]
            for leaflet_idx in range(2):
                arr[:, leaflet_idx, hole[leaflet_idx]] = np.nan
        frames.append(arr)

    return np.mean(np.stack(frames), axis=0)


def _leaflet_counts(Dir: str, leaflet: str) -> dict[str, float] | None:
    """species -> mean_count for one leaflet, from area_per_lipid.csv - None if that file isn't there."""
    path = Path(Dir, "area_per_lipid.csv")
    if not path.exists():
        return None
    counts: dict[str, float] = {}
    lines = path.read_text().splitlines()[1:]  # header
    for line in lines:
        row_leaflet, species, _flat, _curved, mean_count = line.split(",")
        if row_leaflet == leaflet:
            counts[species] = float(mean_count)
    return counts


def _species_outfile(filename: str, name: str) -> str:
    """'{stem}_{name}{suffix}' next to `filename` - e.g. 'lipids.png' + 'POPC' -> 'lipids_POPC.png'."""
    path = Path(filename)
    return str(path.with_name(f"{path.stem}_{name}{path.suffix}"))


def _was_rotated(Dir: str) -> bool:
    """Whether --rotate was used to build this output - see rotated.npy in analyze/lipids.py's calc_lipids.

    Missing (an output directory from before this file existed, or a
    genuinely --rotate-less run) reads as False, same as an explicit one.
    """
    path = Path(Dir, "rotated.npy")
    return bool(np.load(path)) if path.exists() else False


def _lipids_dimensions(Dir: str, frame_numbers: Iterable[int] | None) -> np.ndarray:
    """Every {frame}_dimensions.npy, stacked: shape (n_frames, 3) - that frame's own Lx, Ly, Lz.

    Feeds core.rotation.fixed_circle_radius (built for an SFT's own
    `dimensions` there) - 'analyze lipids' has no SFT of its own, but
    writes the same per-frame box size directly, one file per frame (see
    analyze/lipids.py's _one_lipid_frame).
    """
    files = _frame_filtered_glob(os.path.join(Dir, "*_dimensions.npy"), frame_numbers)
    if not files:
        raise FileNotFoundError(f"No *_dimensions.npy files found in {Dir}")
    return np.stack([np.load(f) for f in files])


def _render_panel(
    ax, X: np.ndarray, Y: np.ndarray, frac: np.ndarray, levels: np.ndarray,
    circle_radius: float | None, box_size: np.ndarray, title: str, fontsize: int,
):
    """One leaflet's own occupancy-frequency panel: contourf + circle-clip + ticks/aspect. Returns the contour set."""
    contour = ax.contourf(X, Y, frac, cmap="viridis", levels=levels, vmin=0, vmax=1)
    _clip_to_circle(contour, ax, circle_radius, box_size)
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=12)
    ax.set_xticks([0, box_size[0]])
    ax.set_yticks([0, box_size[1]])
    ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
    ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)
    ax.set_aspect(box_size[0] / box_size[1])
    return contour


def draw(
    Dir: str,
    filename: str = "",
    frame_numbers: Iterable[int] | None = None,
) -> None:
    """Render every species' own continuous occupancy-frequency field: one combined overview, one file per species.

    Every panel is a species' own `mean_fractions` slice directly (no
    collapsing species against each other into a single "dominant
    species" per point), a continuous value in [0, 1] - how often that
    species owns each grid point across the averaged frames. A point held
    out by `--Remove-TMD` in any averaged frame is NaN (`_mean_fractions`'s
    own all-or-nothing convention) and left unfilled by `contourf`, the
    same way 'CALM map plot' leaves its own holes unfilled.

    A single combined "which species wins here" map (one color per point,
    the winning species) was tried and rejected: collapsing every species
    down to one categorical winner per point throws away the graded
    competition CALM already computes, and with only two species it
    degenerates into a plain binary map that shows less than the two
    continuous fields it was built from.

    Two things are written: `filename` itself, one combined figure with
    every species' own row (Upper/Lower columns, one shared colorbar) for
    a quick overview; and `_species_outfile(filename, name)` per species,
    each its own two-panel figure with its own colorbar, for closer
    inspection of one species at a time. `filename == ""` shows each
    figure interactively in turn (`plt.show()` blocks until closed)
    instead of saving either.
    """
    fontsize = 20
    species = _read_species(Dir)
    n_species = len(species)

    dims = _lipids_dimensions(Dir, frame_numbers)
    box_size = dims[0]

    mean_fractions = _mean_fractions(Dir, frame_numbers)
    gridsize = mean_fractions.shape[-1]
    X, Y = get_XY(box_size, gridsize)
    levels = np.linspace(0, 1, 21)

    # With --rotate, only the largest box-centered circle that stays inside
    # every averaged frame's own box is meaningful (rotated_grid says so
    # explicitly) - same restriction 'CALM map plot' applies to rotated
    # curvature/thickness output, via the same fixed_circle_radius formula.
    circle_radius = fixed_circle_radius(dims) if _was_rotated(Dir) else None

    leaflet_names = ("Upper", "Lower")
    counts = {leaflet: _leaflet_counts(Dir, leaflet) for leaflet in ("upper", "lower")}

    def panel_title(name: str, leaflet_idx: int) -> str:
        leaflet = leaflet_names[leaflet_idx].lower()
        leaflet_counts = counts[leaflet]
        count_str = f" ({leaflet_counts[name]:.1f})" if leaflet_counts and name in leaflet_counts else ""
        return f"{leaflet_names[leaflet_idx]} Surface: {name}{count_str}"

    # Combined overview: one row per species, one shared colorbar.
    fig, axes = plt.subplots(n_species, 2, figsize=(18, 8 * n_species), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.86, bottom=0.04, top=0.96, hspace=0.35, wspace=0.1)
    for row, name in enumerate(species):
        for leaflet_idx, ax in enumerate(axes[row]):
            frac = mean_fractions[row, leaflet_idx]  # (gridsize, gridsize), NaN where holed
            contour = _render_panel(
                ax, X, Y, frac, levels, circle_radius, box_size, panel_title(name, leaflet_idx), fontsize
            )
    cbar_ax = fig.add_axes((0.89, 0.08, 0.02, 0.84))
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("Occupancy frequency", fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)
    if filename == "":
        plt.show()
    else:
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    # Per-species detail: one two-panel figure, own colorbar, per species.
    for row, name in enumerate(species):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.subplots_adjust(left=0.05, right=0.86, bottom=0.1, top=0.88, wspace=0.1)

        for leaflet_idx, ax in enumerate(axes):
            frac = mean_fractions[row, leaflet_idx]  # (gridsize, gridsize), NaN where holed
            contour = _render_panel(
                ax, X, Y, frac, levels, circle_radius, box_size, panel_title(name, leaflet_idx), fontsize
            )

        cbar_ax = fig.add_axes((0.89, 0.1, 0.02, 0.78))
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Occupancy frequency", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)

        if filename == "":
            plt.show()
        else:
            fig.savefig(_species_outfile(filename, name), dpi=300)
            plt.close(fig)


def lipids_plot(argv: list[str]) -> None:
    """CLI entry: render every species' own per-leaflet occupancy map from 'CALM analyze lipids' output."""
    parser = argparse.ArgumentParser(description="Render CALM analyze lipids output (composition/density maps)")
    parser.add_argument(
        '-i', '--numpys_directory', type=str, required=True,
        help="'CALM analyze lipids' output directory",
    )
    parser.add_argument(
        '-o', '--outfile', type=str, default="lipids.png",
        help="output image path (default: lipids.png)",
    )
    add_manual(parser, "map_lipids_plot")

    ns = parser.parse_args(argv)
    draw(Dir=ns.numpys_directory, filename=ns.outfile)


if __name__ == "__main__":
    pass
