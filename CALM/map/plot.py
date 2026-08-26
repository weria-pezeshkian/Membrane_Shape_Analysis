from __future__ import annotations

import argparse
import glob
import logging
import os
import warnings
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from ..core.rotation import (
    fixed_circle_radius,
    lookup_mask_at_rotated_grid,
    recover_all_rotation_angles,
    rotation_was_used,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "serif"


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def get_XY(box_size: np.ndarray, gridsize: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, box_size[0], gridsize)
    y = np.linspace(0, box_size[1], gridsize)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _frame_number(path: str) -> int:
    return int(Path(path).stem.split("_")[0])


def _frame_filtered_glob(pattern_path: str, frame_numbers: Iterable[int] | None) -> list[str]:
    """Sorted glob of `pattern_path`, kept to files whose frame number is in `frame_numbers` (all, if None)."""
    files = sorted(glob.glob(pattern_path))
    if frame_numbers is None:
        return files
    keep = set(frame_numbers)
    return [f for f in files if _frame_number(f) in keep]


def _hole_masks_for_frame(sft: SFT, frame_idx: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower) hole masks for one frame, remapped onto the output grid if theta != 0.

    Callers only reach this once they've already confirmed `sft.hole_mask
    is not None`; the assert documents and enforces that precondition.
    """
    assert sft.hole_mask is not None
    upper, lower = sft.hole_mask[frame_idx]
    if theta == 0.0:
        return upper, lower

    assert sft.dimensions is not None
    Lx, Ly = sft.dimensions[frame_idx, 0], sft.dimensions[frame_idx, 1]
    gridsize = upper.shape[0]
    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cx, cy = Lx / 2.0, Ly / 2.0
    upper = lookup_mask_at_rotated_grid(upper, X, Y, Lx, Ly, cx, cy, theta)
    lower = lookup_mask_at_rotated_grid(lower, X, Y, Lx, Ly, cx, cy, theta)
    return upper, lower


def _masked_frame_stack(
    files: list[str],
    pattern: str,
    Dir: str,
    sft: SFT | None,
    layer_sources: Sequence[str | None] | None,
) -> np.ndarray:
    """Stack of every file in `files`, loaded and hole-masked (NaN'd) per frame, one frame per leading index.

    `layer_sources` is None if the array has no leading layer axis (e.g.
    thickness), in which case the union (upper OR lower) hole mask is
    applied directly; otherwise it is a list, one entry per layer along
    axis 0, each "upper", "lower", "union", or None (skip that layer).
    """
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {Dir}")

    sft_with_holes = sft if (sft is not None and sft.hole_mask is not None) else None
    thetas = None
    if sft_with_holes is not None and rotation_was_used(sft_with_holes):
        thetas = recover_all_rotation_angles(sft_with_holes)

    frames = []
    for f in files:
        arr = np.load(f)
        if sft_with_holes is not None:
            assert sft_with_holes.frame_indices is not None
            matches = np.nonzero(sft_with_holes.frame_indices == _frame_number(f))[0]
            if matches.size:
                idx = matches[0]
                theta = thetas[idx] if thetas is not None else 0.0
                upper_hole, lower_hole = _hole_masks_for_frame(sft_with_holes, idx, theta)
                union_hole = upper_hole | lower_hole
                arr = arr.copy()
                if layer_sources is None:
                    arr[union_hole] = np.nan
                else:
                    sources = {"upper": upper_hole, "lower": lower_hole, "union": union_hole}
                    for layer_idx, source in enumerate(layer_sources):
                        if source is not None:
                            arr[layer_idx][sources[source]] = np.nan
        frames.append(arr)

    return np.asarray(frames)


def _load_and_mask(
    files: list[str],
    pattern: str,
    Dir: str,
    sft: SFT | None,
    layer_sources: Sequence[str | None] | None,
) -> np.ndarray:
    """Load and mean-average a set of per-frame .npy files, applying the (optional) hole mask per frame first.

    Uses a plain np.mean, not np.nanmean: a NaN at a given point in any one
    frame makes the averaged point NaN too.
    """
    return np.mean(_masked_frame_stack(files, pattern, Dir, sft, layer_sources), axis=0)


def _average_principal_directions(
    files: list[str],
    pattern: str,
    Dir: str,
    sft: SFT | None,
    layer_sources: Sequence[str | None] | None,
) -> np.ndarray:
    """Nematic-tensor average of a set of per-frame unit tangent-vector fields (shape (..., 3) per frame).

    A principal direction's sign is arbitrary per point per frame (it's an
    eigenvector). This sums each point's outer product n (x) n across frames
    - sign-invariant, since (-n)(-n)^T equals n n^T - and takes the dominant
    eigenvector of that summed 3x3 tensor as the consensus direction. The
    result is signed to match the first frame's own direction at that point.
    A single file's own direction is recovered exactly (the dominant
    eigenvector of n (x) n, signed to match n itself, is n). Follows the same
    all-or-nothing NaN rule as `_load_and_mask`: a point NaN in any one frame
    stays NaN here.
    """
    stack = _masked_frame_stack(files, pattern, Dir, sft, layer_sources)
    nan_mask = np.isnan(stack).any(axis=(0, -1))
    filled = np.where(np.isnan(stack), 0.0, stack)

    tensor = np.einsum("f...a,f...b->...ab", filled, filled)
    _, eigvecs = np.linalg.eigh(tensor)
    dominant = eigvecs[..., -1]

    sign = np.sign(np.sum(dominant * filled[0], axis=-1))
    sign = np.where(sign == 0, 1.0, sign)
    dominant = dominant * sign[..., None]

    dominant[nan_mask] = np.nan
    return dominant


def _align_signs_to_lower_z(dirs_slice: np.ndarray, flat_tol: float = 1e-9) -> np.ndarray:
    """Sign-align a (Ny, Nx, 3) unit-direction field so neighboring arrows point consistently.

    A principal direction's sign is arbitrary (it's an eigenvector), so
    naively drawing each grid point's own stored direction can flip sign
    between neighbors that represent the same physical axis, looking
    disordered. This picks one consistent sign per point in two stages:

    1. Wherever a point's z-component has a definite sign (`abs(dz) >
       flat_tol`), it's flipped, if needed, to point toward lower z - a
       physically meaningful, unambiguous choice.
    2. Points with `abs(dz) <= flat_tol` (locally flat, where z can't
       disambiguate) are flipped to match the first already-oriented
       4-connected, periodic neighbor reached by breadth-first search
       outward from every point resolved in stage 1. A point with no
       oriented neighbor reachable this way (e.g. an entirely flat, isolated
       region) keeps its original sign.

    `NaN` points (holes, outside the --rotate circle) are skipped and left
    as `NaN`.
    """
    ny, nx = dirs_slice.shape[:2]
    aligned = dirs_slice.copy()
    valid = ~np.isnan(aligned).any(axis=-1)

    oriented = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            dz = aligned[i, j, 2]
            if dz > flat_tol:
                aligned[i, j] *= -1
                oriented[i, j] = True
            elif dz < -flat_tol:
                oriented[i, j] = True

    queue = deque((i, j) for i in range(ny) for j in range(nx) if oriented[i, j])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = (i + di) % ny, (j + dj) % nx
            if not valid[ni, nj] or oriented[ni, nj]:
                continue
            if np.dot(aligned[ni, nj], aligned[i, j]) < 0:
                aligned[ni, nj] *= -1
            oriented[ni, nj] = True
            queue.append((ni, nj))

    return aligned




def _auto_minmax(arr: np.ndarray) -> tuple[float, float]:
    """Min/max of `arr`'s non-NaN values, widened by a hair if they're equal."""
    valid = arr[~np.isnan(arr)]
    Minimum, Maximum = float(np.min(valid)), float(np.max(valid))
    if Minimum == Maximum:
        Maximum = Minimum + 1e-6
    return Minimum, Maximum


def _clip_to_circle(contour_set, ax, circle_radius: float | None, box_size: np.ndarray) -> None:
    """Confine a contourf's rendering to the shared circle; outside is left as plain white background."""
    if circle_radius is not None:
        circle = mpatches.Circle(
            (box_size[0] / 2.0, box_size[1] / 2.0), circle_radius,
            transform=ax.transData,
        )
        contour_set.set_clip_path(circle)


def _add_colorbar_histogram(fig, cbar_ax, values: np.ndarray, levels: np.ndarray, side: str) -> None:
    """Bar strip flush against `cbar_ax` showing how many of `values`' points fall in each of `levels`' bins.

    Reuses `levels` exactly as the colorbar's own bin edges, so the strip's
    vertical axis lines up with the colorbar's color mapping one-to-one.
    `side` ("left" or "right") places the strip on the side of `cbar_ax`
    facing away from the data panels, bars growing outward from it. Sits
    with no gap against the colorbar; `cbar_ax`'s own ticks and axis label
    are pushed out past the strip's own width so neither overlaps it.
    """
    valid = values[~np.isnan(values)]
    counts, _ = np.histogram(valid, bins=levels)
    centers = (levels[:-1] + levels[1:]) / 2
    heights = np.diff(levels)

    pos = cbar_ax.get_position()
    width = 0.025
    x0 = pos.x0 - width if side == "left" else pos.x1
    hist_ax = fig.add_axes((x0, pos.y0, width, pos.height))

    hist_ax.barh(centers, counts, height=heights, color="0.4", alpha=0.7, linewidth=0)
    hist_ax.set_ylim(levels[0], levels[-1])
    if side == "left":
        hist_ax.invert_xaxis()

    # The colorbar's own tick lines, redrawn across the histogram (its own
    # axes, so they sit on top of the bars regardless of axes draw order),
    # so each level's line reads directly across its own bar.
    tick_locs = [t for t in cbar_ax.get_yticks() if levels[0] <= t <= levels[-1]]
    hist_ax.hlines(tick_locs, *hist_ax.get_xlim(), color="0.6", linewidth=1, zorder=3)
    hist_ax.axis("off")

    hist_width_points = width * fig.get_figwidth() * 72
    cbar_ax.tick_params(pad=hist_width_points + 4)
    cbar_ax.yaxis.labelpad = hist_width_points + 55


def draw(
    Dir: str,
    mode: str = "mean",
    layer1: str = "Upper",
    layer2: str = "Lower",
    layer3: str = "Middle",
    minmax: list[float] | None = None,
    thickness_minmax: list[float] | None = None,
    filename: str = "",
    show_vectors: bool = True,
    title_pad: int = 12,
    frame_numbers: Iterable[int] | None = None,
    vector_frame: int | None = None,
    histogram: bool = False,
) -> None:
    """Render mean curvature or thickness.

    `frame_numbers`, if given, restricts averaging to that subset of frames
    instead of every frame found in `Dir` (used by `dynamic_plot` to average
    over a rolling window instead of the whole trajectory). `thickness_minmax`
    fixes the thickness subpanel's own color scale in `--mode mean` (auto-scaled
    from its own data when omitted); it has no effect in `--mode thickness`,
    which uses `minmax` directly since thickness is the only panel drawn there.

    `vector_frame`, in `--mode principal`, selects that single frame's own
    principal directions for the vector overlay (`dynamic_plot` sets this to
    each video frame's own trajectory frame number, so the overlay always
    shows that frame's own instantaneous directions, while the curvature
    background it's drawn over keeps averaging over `frame_numbers` as usual).
    In `--mode principal`, each direction field is also sign-aligned
    (`_align_signs_to_lower_z`) before display.

    `histogram` adds a bar-chart strip beside each colorbar (via
    `_add_colorbar_histogram`) showing how this call's own data distributes
    across the colorbar's fixed range - opt-in on both `map plot` and
    `dynamic_plot`'s own `--histogram` flag, off by default on both.
    """
    fontsize = 20
    # Extra figure width (inches) a colorbar histogram strip, plus its
    # colorbar's own pushed-out ticks and axis label, need beyond a plain
    # colorbar alone - added to figsize rather than reclaimed from the
    # existing margins, so the data panels keep the exact same absolute
    # size (and set_aspect keeps fitting them the exact same way) whether
    # or not histogram is on.
    hist_room_in = 2.6

    if not Dir.endswith("/"):
        Dir += "/"

    # Amn/qmn/dimensions.npy are always written together (see SFT.write) -
    # box_size comes straight from here now, not a separate dimensions.csv.
    sft = SFT.from_directory(Dir)
    assert sft.dimensions is not None
    box_size = sft.dimensions[0]

    circle_radius = None
    if rotation_was_used(sft):
        circle_radius = fixed_circle_radius(sft.dimensions)

    if mode == "thickness":
        thickness_mean = _load_and_mask(
            _frame_filtered_glob(Dir + "*_thickness.npy", frame_numbers),
            "*_thickness.npy", Dir, sft, layer_sources=None,
        )
        gridsize = thickness_mean.shape[-1]
        X, Y = get_XY(box_size, gridsize)

        Minimum, Maximum = minmax if minmax is not None else _auto_minmax(thickness_mean)
        thickness_levels = np.linspace(Minimum, Maximum, 20)

        # Extra room is added to the figure's own width, not reclaimed from
        # the plot's margins, so the plot's absolute size (and set_aspect's
        # fit) is identical whether or not histogram is on.
        base_w = 12.0
        fig_w = base_w + hist_room_in if histogram else base_w
        fig, ax = plt.subplots(figsize=(fig_w, 10))
        fig.subplots_adjust(left=1.2 / fig_w, right=10.2 / fig_w, bottom=0.1, top=0.92)

        contour = ax.contourf(X, Y, thickness_mean, cmap="viridis", levels=thickness_levels)
        _clip_to_circle(contour, ax, circle_radius, box_size)
        ax.set_title("Bilayer Thickness", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes((10.44 / fig_w, 0.1, 0.36 / fig_w, 0.8))
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Thickness (nm)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        if histogram:
            _add_colorbar_histogram(fig, cbar_ax, thickness_mean, thickness_levels, side="right")

        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

        if filename == "":
            plt.show()
        else:
            fig.savefig(filename, dpi=300)
            plt.close(fig)
        return

    if mode == "mean":
        pattern = "*_mean_curvature.npy"
    elif mode == "gaussian":
        pattern = "*_gaussian_curvature.npy"
    elif mode == "principal":
        pattern = "*_principal_curvatures.npy"
    else:
        raise ValueError("mode must be 'mean', 'gaussian', 'principal', or 'thickness'")

    # Layer order along axis 0, per mode - see analyze/analyze.py's np.stack
    # calls when saving each of these files.
    if mode == "principal":
        curvature_layer_sources = ["upper", "upper", "lower", "lower", "union", "union"]
    else:
        curvature_layer_sources = ["upper", "lower", "union"]

    curvature_mean = _load_and_mask(
        _frame_filtered_glob(Dir + pattern, frame_numbers), pattern, Dir, sft, curvature_layer_sources
    )

    gridsize = curvature_mean.shape[-1]
    X, Y = get_XY(box_size, gridsize)

    have_thickness = False
    thickness_min = thickness_max = 0.0

    if mode == "mean":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Mean Curvature"

        # Thickness is optional here: "mean" was computed without it if
        # --method didn't include "thickness". Layout adapts below.
        thickness_files = _frame_filtered_glob(Dir + "*_thickness.npy", frame_numbers)
        have_thickness = bool(thickness_files)
        if have_thickness:
            thickness_mean = _load_and_mask(thickness_files, "*_thickness.npy", Dir, sft, layer_sources=None)
            thickness_min, thickness_max = (
                thickness_minmax if thickness_minmax is not None else _auto_minmax(thickness_mean)
            )

    elif mode == "gaussian":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Gaussian Curvature"

    elif mode == "principal":
        curvature_k1 = [curvature_mean[0], curvature_mean[2], curvature_mean[4]]
        curvature_k2 = [curvature_mean[1], curvature_mean[3], curvature_mean[5]]

        dir_frame_numbers = [vector_frame] if vector_frame is not None else frame_numbers
        dir_files = _frame_filtered_glob(Dir + "*_principal_dirs.npy", dir_frame_numbers)
        dir_mean = _average_principal_directions(dir_files, "*_principal_dirs.npy", Dir, sft, curvature_layer_sources)

        dirs_k1 = [normalize(_align_signs_to_lower_z(dir_mean[i])) for i in (0, 2, 4)]
        dirs_k2 = [normalize(_align_signs_to_lower_z(dir_mean[i])) for i in (1, 3, 5)]

    if minmax is None:
        if mode == "principal":
            all_vals = np.concatenate([c.flatten() for c in curvature_k1 + curvature_k2])
        else:
            all_vals = np.concatenate([c.flatten() for c in [curvature_data1, curvature_data2, curvature_data3]])
        all_vals = all_vals[~np.isnan(all_vals)]
        Minimum = np.min(all_vals)
        Maximum = np.max(all_vals)
        if Minimum == Maximum:
            Maximum = Minimum + 1e-6
    else:
        Minimum, Maximum = minmax

    levels = np.linspace(Minimum, Maximum, 20)
    norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)

    if mode == "mean" and have_thickness:
        # Extra room (both sides: thickness's histogram on the left,
        # curvature's on the right) is added to the figure's own width,
        # not reclaimed from the plots' margins, so the plots' absolute
        # size (and set_aspect's fit) is identical whether or not
        # histogram is on.
        base_w = 32.0
        shift = hist_room_in if histogram else 0.0
        fig_w = base_w + 2 * shift
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, 26), gridspec_kw={"hspace": 0.075, "wspace": 0.001})
        axes = axes.flatten()
        fig.subplots_adjust(left=(2.24 + shift) / fig_w, right=(28.48 + shift) / fig_w, bottom=0.03, top=0.97)

        thickness_levels = np.linspace(thickness_min, thickness_max, 20)
        contour0 = axes[0].contourf(X, Y, thickness_mean, cmap="viridis", levels=thickness_levels)
        _clip_to_circle(contour0, axes[0], circle_radius, box_size)
        axes[0].set_title("Bilayer Thickness", fontsize=fontsize, fontweight="bold", pad=title_pad)

        contour1 = axes[1].contourf(X, Y, curvature_data1, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour1, axes[1], circle_radius, box_size)
        contour2 = axes[2].contourf(X, Y, curvature_data2, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour2, axes[2], circle_radius, box_size)
        contour3 = axes[3].contourf(X, Y, curvature_data3, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour3, axes[3], circle_radius, box_size)

        axes[1].set_title(f"{layer1} Surface: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)
        axes[2].set_title(f"{layer2} Surface: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)
        axes[3].set_title(f"{layer3} Surface: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes(((1.728 + shift) / fig_w, 0.15, 0.64 / fig_w, 0.7))
        fig.colorbar(contour0, cax=cbar_ax).set_label("Thickness (nm)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')
        if histogram:
            _add_colorbar_histogram(fig, cbar_ax, thickness_mean, thickness_levels, side="left")

        cbar_ax2 = fig.add_axes(((28.16 + shift) / fig_w, 0.15, 0.64 / fig_w, 0.7))
        cbar2 = fig.colorbar(contour1, cax=cbar_ax2)
        cbar2.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar2.ax.tick_params(labelsize=fontsize)
        if histogram:
            combined = np.concatenate([curvature_data1.ravel(), curvature_data2.ravel(), curvature_data3.ravel()])
            _add_colorbar_histogram(fig, cbar_ax2, combined, levels, side="right")

    elif mode == "mean" and not have_thickness:
        # Extra room is added to the figure's own width, not reclaimed from
        # the plots' margins, so the plots' absolute size (and
        # set_aspect's fit) is identical whether or not histogram is on.
        base_w = 30.0
        fig_w = base_w + hist_room_in if histogram else base_w
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 10))
        fig.subplots_adjust(left=0.6 / fig_w, right=26.4 / fig_w, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data3, curvature_data2]  # upper, middle, lower
        layers = [layer1, layer3, layer2]

        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, axes[i], circle_radius, box_size)
            axes[i].set_title(f"{layers[i]} Surface: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes((27.0 / fig_w, 0.08, 0.6 / fig_w, 0.8))
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize, labelpad=title_pad)
        cbar_ax.tick_params(labelsize=fontsize)
        if histogram:
            combined = np.concatenate([layer_data.ravel() for layer_data in curvatures])
            _add_colorbar_histogram(fig, cbar_ax, combined, levels, side="right")

    elif mode == "gaussian":
        # Extra room is added to the figure's own width, not reclaimed from
        # the plots' margins, so the plots' absolute size (and
        # set_aspect's fit) is identical whether or not histogram is on.
        base_w = 30.0
        fig_w = base_w + hist_room_in if histogram else base_w
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, 10))
        fig.subplots_adjust(left=0.6 / fig_w, right=26.4 / fig_w, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data2, curvature_data3]
        layers = [layer1, layer2, layer3]

        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, axes[i], circle_radius, box_size)
            axes[i].set_title(f"{layers[i]} Surface: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes((27.0 / fig_w, 0.08, 0.6 / fig_w, 0.8))
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize, labelpad=title_pad)
        cbar_ax.tick_params(labelsize=fontsize)
        if histogram:
            combined = np.concatenate([layer_data.ravel() for layer_data in curvatures])
            _add_colorbar_histogram(fig, cbar_ax, combined, levels, side="right")

    elif mode == "principal":
        # Extra room is added to the figure's own width, not reclaimed from
        # the plots' margins, so the plots' absolute size (and
        # set_aspect's fit) is identical whether or not histogram is on.
        base_w = 30.0
        fig_w = base_w + hist_room_in if histogram else base_w
        fig, axes = plt.subplots(2, 3, figsize=(fig_w, 26))
        fig.subplots_adjust(
            left=1.5 / fig_w, right=27.3 / fig_w, bottom=0.05, top=0.90, wspace=0.08, hspace=0.0,
        )

        for i in range(3):
            ax = axes[0, i]
            c = ax.contourf(X, Y, curvature_k1[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, ax, circle_radius, box_size)
            ax.set_title(f"{layer1 if i==0 else layer2 if i==1 else layer3} Surface: k1",
                         fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k1[i][::step, ::step, 0], dirs_k1[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)

        for i in range(3):
            ax = axes[1, i]
            c = ax.contourf(X, Y, curvature_k2[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, ax, circle_radius, box_size)
            ax.set_title(f"{layer1 if i==0 else layer2 if i==1 else layer3} Surface: k2",
                         fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k2[i][::step, ::step, 0], dirs_k2[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)

        cbar_ax = fig.add_axes((27.9 / fig_w, 0.08, 0.6 / fig_w, 0.8))
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if histogram:
            combined = np.concatenate([layer_data.ravel() for layer_data in curvature_k1 + curvature_k2])
            _add_colorbar_histogram(fig, cbar_ax, combined, levels, side="right")

    for ax in axes.flatten():
        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

    if filename == "":
        plt.show()
    else:
        fig.savefig(filename, dpi=300)
        plt.close(fig)


def plot(args: list[str]) -> None:
    """CLI entry: plot mean curvature or thickness from a 'CALM analyze full' output directory."""
    parser = argparse.ArgumentParser(description="Plot mean curvature or thickness")
    parser.add_argument('-i', '--numpys_directory', type=str, help="'CALM analyze full' output directory")
    parser.add_argument(
        '--mode', choices=["mean", "gaussian", "principal", "thickness"], default="mean",
        help="quantity to plot (default: mean)",
    )
    parser.add_argument('-o', '--outfile', type=str, default="mean.png", help="output image path (default: mean.png)")
    parser.add_argument('--minimum', type=float, default=None, help="fix the color scale's lower bound")
    parser.add_argument('--maximum', type=float, default=None, help="fix the color scale's upper bound")
    parser.add_argument('--vectors', action="store_true", default=False, help="overlay principal-direction vectors")
    parser.add_argument(
        '--histogram', action="store_true", default=False,
        help="add a distribution strip beside each colorbar",
    )
    add_manual(parser, "map_plot")

    ns = parser.parse_args(args)
    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None

    draw(
        Dir=ns.numpys_directory, mode=ns.mode, minmax=minmax, filename=ns.outfile, show_vectors=ns.vectors,
        histogram=ns.histogram,
    )


if __name__ == "__main__":
    pass
