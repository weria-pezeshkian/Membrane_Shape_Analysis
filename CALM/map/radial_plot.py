from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from ..core.rotation import fixed_circle_radius, rotation_was_used
from .plot import _frame_filtered_glob, _load_and_mask

plt.rcParams["font.family"] = "serif"

_QUANTITY_FILES = {
    "mean": ("*_mean_curvature.npy", "Mean Curvature (nm$^{-1}$)"),
    "height": ("*_Z_fitted.npy", "Height (nm)"),
}


def _hole_radius(r: np.ndarray, valid: np.ndarray) -> float:
    """Radius of the largest circle centered at r=0 containing no valid point.

    Equal to the smallest radius among the valid points themselves - any
    circle smaller than that contains only NaN by definition. Reported as
    the full extent of `r` if every point is NaN.
    """
    if not valid.any():
        return float(r.max())
    return float(r[valid].min())


def _quantile_bin_edges(
    r: np.ndarray, valid: np.ndarray, r_hole: float, r_max: float, bin_width: float
) -> np.ndarray:
    """Radial bin edges from `r_hole` to `r_max`, each containing roughly
    equal counts of valid points.

    Quantile-splits the valid points' own radii, so each bin's width
    adapts to wherever those points actually sit along the radius -
    narrow where they're dense, wide where they're sparse. The target bin
    count is `(r_max - r_hole)` divided by `bin_width`, capped to the
    number of valid points available.
    """
    in_range = r[valid & (r >= r_hole) & (r <= r_max)]
    if len(in_range) == 0:
        return np.array([r_hole, r_max])

    n_bins = max(1, int((r_max - r_hole) / bin_width)) if bin_width > 0 else 1
    n_bins = max(1, min(n_bins, len(in_range)))

    edges = np.unique(np.quantile(in_range, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = r_hole, r_max
    return edges


def _radial_profile(values: np.ndarray, r: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of `values` (may contain NaN) within each radial annulus defined by `edges`.

    Each bin is `[edges[i], edges[i+1])`, except the last, which is
    `[edges[-2], edges[-1]]` (upper bound inclusive) so a point sitting
    exactly at `edges[-1]` (e.g. `r_max` itself) still counts.
    """
    n_bins = len(edges) - 1
    profile = np.full(n_bins, np.nan)
    valid = ~np.isnan(values)
    for i in range(n_bins):
        upper = (r <= edges[i + 1]) if i == n_bins - 1 else (r < edges[i + 1])
        in_bin = valid & (r >= edges[i]) & upper
        if in_bin.any():
            profile[i] = values[in_bin].mean()
    return profile


def _no_value_wedge(
    upper_start: tuple[float, float], lower_start: tuple[float, float], y_min: float, y_max: float
) -> tuple[list[float], list[float]]:
    """(x, y) polygon vertices for the "no value" region: the y-axis on one
    side, and the line through `upper_start` and `lower_start`, spanning
    the full `[y_min, y_max]` range, on the other.
    """
    if upper_start[1] == lower_start[1]:
        # A horizontal line through the two points has the same x at every
        # y; use the further of the two points' own x positions for both ends.
        x_at_ymin = x_at_ymax = max(upper_start[0], lower_start[0])
    else:
        dx_dy = (lower_start[0] - upper_start[0]) / (lower_start[1] - upper_start[1])
        x_at_ymin = upper_start[0] + dx_dy * (y_min - upper_start[1])
        x_at_ymax = upper_start[0] + dx_dy * (y_max - upper_start[1])

    return [0.0, x_at_ymax, x_at_ymin, 0.0], [y_max, y_max, y_min, y_min]


def _radial_series(
    values: np.ndarray, r: np.ndarray, r_max: float, bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """(x positions, per-bin mean) for one leaflet's own values.

    Bin edges are quantile-split from this leaflet's own hole radius and
    its own valid-point distribution, independently of the other leaflet,
    so each leaflet's bin count and width reflect only its own density.
    Each bin's *left* edge is used as its x position, placing the first
    plotted point exactly at this leaflet's own hole radius - `x[0]` -
    directly usable by a caller.
    """
    valid = ~np.isnan(values)
    r_hole = _hole_radius(r, valid)
    edges = _quantile_bin_edges(r, valid, r_hole, r_max, bin_width)
    profile = _radial_profile(values, r, edges)
    return edges[:-1], profile


def draw(
    Dir: str,
    filename: str = "radial.png",
    minmax: list[float] | None = None,
    quantity: str = "mean",
) -> None:
    """Render upper/lower mean curvature or fitted height, radially averaged outward from the box center.

    `quantity` is `"mean"` (mean curvature, default) or `"height"` (fitted
    surface height, expressed relative to the mid-surface's own height at
    that same point - the membrane's absolute position in the box is
    arbitrary, but its distance from its own mid-surface is physical).
    Middle is loaded at its raw value everywhere: for `"mean"` it's never
    read at all (only upper/lower are plotted); for `"height"` it's a
    subtraction reference whose value at a point holds regardless of
    either leaflet's own hole there, keeping upper's and lower's own
    relative-height values independent of each other's holes. Only upper
    and lower are plotted either way, each drawn exactly as computed.

    Each leaflet's own curve starts at its own `_hole_radius` (the largest
    all-NaN circle centered on the box center, e.g. where --Remove-TMD
    masked a protein) and runs to `r_max` (the fixed circle radius under
    --rotate, or the box's own inscribed-circle radius), quantile-binned
    (`_radial_series`) so each point on the curve reflects a comparable
    number of grid points. The x-axis starts at 0. `_no_value_wedge`
    shades the region between the y-axis and the line through the two
    curves' own starting points, spanning the full y-range, labeled
    "No value" in the legend.
    """
    if not Dir.endswith("/"):
        Dir += "/"

    sft = SFT.from_directory(Dir)
    assert sft.dimensions is not None
    box_size = sft.dimensions[0]

    circle_radius = None
    if rotation_was_used(sft):
        circle_radius = fixed_circle_radius(sft.dimensions)

    pattern, ylabel = _QUANTITY_FILES[quantity]
    # Middle is loaded with its raw value at every grid point. For
    # --quantity mean it's loaded but only upper/lower are plotted; for
    # --quantity height it's a subtraction reference, so its value at a
    # point stays the same regardless of either leaflet's own hole there.
    field_mean = _load_and_mask(
        _frame_filtered_glob(Dir + pattern, None), pattern, Dir, sft, layer_sources=["upper", "lower", None]
    )
    gridsize = field_mean.shape[-1]

    if quantity == "height":
        # Height relative to the mid-surface's own height at that same
        # point - the membrane's distance from its own mid-surface is
        # physical, regardless of where the box places the membrane.
        upper_values = field_mean[0] - field_mean[2]
        lower_values = field_mean[1] - field_mean[2]
        ylabel = "Height relative to mid-surface (nm)"
    else:
        upper_values = field_mean[0]
        lower_values = field_mean[1]

    x = np.linspace(0, box_size[0], gridsize, endpoint=False)
    y = np.linspace(0, box_size[1], gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cx, cy = box_size[0] / 2.0, box_size[1] / 2.0
    r = np.hypot(X - cx, Y - cy)

    r_max = circle_radius if circle_radius is not None else min(box_size[0], box_size[1]) / 2.0
    bin_width = box_size[0] / gridsize

    upper_x, upper_profile = _radial_series(upper_values, r, r_max, bin_width)
    lower_x, lower_profile = _radial_series(lower_values, r, r_max, bin_width)

    linewidth = 3
    fontsize = 22

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(0, color="black", linewidth=linewidth, linestyle="--", alpha=0.8)
    ax.plot(upper_x, upper_profile, label="Upper", color="tab:red", linewidth=linewidth)
    ax.plot(lower_x, lower_profile, label="Lower", color="tab:blue", linewidth=linewidth)
    ax.set_xlabel("Radial distance from center (Angstrom)", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize * 0.8, width=linewidth, length=2 * linewidth)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.set_xlim(left=0, right=r_max)
    if minmax is not None:
        ax.set_ylim(minmax[0], minmax[1])

    # A wedge from the y-axis to the line through the two curves' own
    # starting points, spanning the axes' own y-range, drawn behind the
    # data lines.
    y_min, y_max = ax.get_ylim()
    no_value_x, no_value_y = _no_value_wedge(
        (upper_x[0], upper_profile[0]), (lower_x[0], lower_profile[0]), y_min, y_max
    )
    ax.fill(no_value_x, no_value_y, color="gray", alpha=0.15, label="No value", zorder=0)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=fontsize * 0.8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _build_radial_plot_parser() -> argparse.ArgumentParser:
    """The 'CALM map radial_plot' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(description="Render a radial mean-curvature or height profile (upper/lower)")
    parser.add_argument(
        '-i', '--numpys_directory', type=str, required=True,
        help="'CALM analyze full' output directory",
    )
    parser.add_argument(
        '-o', '--outfile', type=str, default="radial.png",
        help="output image path (default: radial.png)",
    )
    parser.add_argument(
        '--quantity', choices=["mean", "height"], default="mean",
        help="quantity to plot: mean curvature or fitted height (default: mean)",
    )
    parser.add_argument('--minimum', type=float, default=None, help="fix the y-axis lower bound")
    parser.add_argument('--maximum', type=float, default=None, help="fix the y-axis upper bound")
    add_manual(parser, "map_radial_plot")
    return parser


def radial_plot(argv: list[str]) -> None:
    """CLI entry: render a radial mean-curvature or height profile (upper/lower) from a 'CALM analyze full' output directory."""
    parser = _build_radial_plot_parser()

    ns = parser.parse_args(argv)
    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None

    draw(Dir=ns.numpys_directory, filename=ns.outfile, minmax=minmax, quantity=ns.quantity)


if __name__ == "__main__":
    pass
