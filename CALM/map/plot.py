from __future__ import annotations

import argparse
import glob
import logging
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    return np.divide(v, norm, where=norm > 0)


def get_XY(box_size: np.ndarray, gridsize: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, box_size[0], gridsize)
    y = np.linspace(0, box_size[1], gridsize)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _frame_number(path: str) -> int:
    return int(Path(path).stem.split("_")[0])


def _frame_filtered_glob(pattern_path: str, frame_numbers: Optional[Iterable[int]]) -> List[str]:
    """Sorted glob of `pattern_path`, kept to files whose frame number is in `frame_numbers` (all, if None)."""
    files = sorted(glob.glob(pattern_path))
    if frame_numbers is None:
        return files
    keep = set(frame_numbers)
    return [f for f in files if _frame_number(f) in keep]


def _hole_masks_for_frame(sft: SFT, frame_idx: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """(upper, lower) hole masks for one frame, remapped onto the output grid if theta != 0."""
    upper, lower = sft.hole_mask[frame_idx]
    if theta == 0.0:
        return upper, lower

    Lx, Ly = sft.dimensions[frame_idx, 0], sft.dimensions[frame_idx, 1]
    gridsize = upper.shape[0]
    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cx, cy = Lx / 2.0, Ly / 2.0
    upper = lookup_mask_at_rotated_grid(upper, X, Y, Lx, Ly, cx, cy, theta)
    lower = lookup_mask_at_rotated_grid(lower, X, Y, Lx, Ly, cx, cy, theta)
    return upper, lower


def _load_and_mask(
    files: List[str],
    pattern: str,
    Dir: str,
    sft: Optional[SFT],
    layer_sources: Optional[List[Optional[str]]],
) -> np.ndarray:
    """Load and mean-average a set of per-frame .npy files, applying the (optional) hole mask per frame first.

    `layer_sources` is None if the array has no leading layer axis (e.g.
    thickness), in which case the union (upper OR lower) hole mask is
    applied directly; otherwise it is a list, one entry per layer along
    axis 0, each "upper", "lower", "union", or None (skip that layer).

    Uses a plain np.mean, not np.nanmean: a NaN at a given point in any one
    frame makes the averaged point NaN too, rather than silently averaging
    over whichever frames had data there.
    """
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {Dir}")

    have_holes = sft is not None and sft.hole_mask is not None
    thetas = recover_all_rotation_angles(sft) if have_holes and rotation_was_used(sft) else None

    frames = []
    for f in files:
        arr = np.load(f)
        if have_holes:
            matches = np.nonzero(sft.frame_indices == _frame_number(f))[0]
            if matches.size:
                idx = matches[0]
                theta = thetas[idx] if thetas is not None else 0.0
                upper_hole, lower_hole = _hole_masks_for_frame(sft, idx, theta)
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

    return np.mean(np.asarray(frames), axis=0)


def draw(
    Dir: str,
    mode: str = "mean",
    layer1: str = "Upper",
    layer2: str = "Lower",
    layer3: str = "Middle",
    minmax: Optional[List[float]] = None,
    filename: str = "",
    show_vectors: bool = True,
    title_pad: int = 12,
    frame_numbers: Optional[Iterable[int]] = None,
) -> None:
    """Render mean curvature or thickness.

    `frame_numbers`, if given, restricts averaging to that subset of frames
    instead of every frame found in `Dir` (used by `dynamic_plot` to average
    over a rolling window instead of the whole trajectory).
    """
    fontsize = 20

    if not Dir.endswith("/"):
        Dir += "/"

    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1, max_rows=1, usecols=(1, 2, 3))

    # Detect whether --rotate / --Remove-TMD were used for this run.
    sft = None
    circle_radius = None
    try:
        sft = SFT.from_directory(Dir)
        if rotation_was_used(sft):
            circle_radius = fixed_circle_radius(sft)
    except FileNotFoundError:
        pass

    def _clip_to_circle(contour_set, ax) -> None:
        """Confine a contourf's rendering to the shared circle; outside is left as plain white background."""
        if circle_radius is not None:
            circle = mpatches.Circle(
                (box_size[0] / 2.0, box_size[1] / 2.0), circle_radius,
                transform=ax.transData,
            )
            contour_set.set_clip_path(circle)

    if mode == "thickness":
        thickness_mean = _load_and_mask(_frame_filtered_glob(Dir + "*_thickness.npy", frame_numbers), "*_thickness.npy", Dir, sft, layer_sources=None)
        gridsize = thickness_mean.shape[-1]
        X, Y = get_XY(box_size, gridsize)

        if minmax is None:
            valid_vals = thickness_mean[~np.isnan(thickness_mean)]
            Minimum, Maximum = np.min(valid_vals), np.max(valid_vals)
            if Minimum == Maximum:
                Maximum = Minimum + 1e-6
        else:
            Minimum, Maximum = minmax

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.92)

        contour = ax.contourf(X, Y, thickness_mean, cmap="viridis", levels=np.linspace(Minimum, Maximum, 20))
        _clip_to_circle(contour, ax)
        ax.set_title("Bilayer Thickness", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("Thickness (nm)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)

        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

        if filename == "":
            plt.show()
        else:
            plt.savefig(filename, dpi=300)
            plt.close()
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

    curvature_mean = _load_and_mask(_frame_filtered_glob(Dir + pattern, frame_numbers), pattern, Dir, sft, curvature_layer_sources)

    gridsize = curvature_mean.shape[-1]
    X, Y = get_XY(box_size, gridsize)

    have_thickness = False

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

    elif mode == "gaussian":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Gaussian Curvature"

    elif mode == "principal":
        curvature_k1 = [curvature_mean[0], curvature_mean[2], curvature_mean[4]]
        curvature_k2 = [curvature_mean[1], curvature_mean[3], curvature_mean[5]]

        dir_files = _frame_filtered_glob(Dir + "*_principal_dirs.npy", frame_numbers)
        dir_mean = _load_and_mask(dir_files, "*_principal_dirs.npy", Dir, sft, curvature_layer_sources)

        dirs_k1 = [normalize(dir_mean[0]), normalize(dir_mean[2]), normalize(dir_mean[4])]
        dirs_k2 = [normalize(dir_mean[1]), normalize(dir_mean[3]), normalize(dir_mean[5])]

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
        fig, axes = plt.subplots(2, 2, figsize=(32, 26), gridspec_kw={"hspace": 0.075, "wspace": 0.001})
        axes = axes.flatten()
        fig.subplots_adjust(left=0.07, right=0.89, bottom=0.03, top=0.97)

        contour0 = axes[0].contourf(X, Y, thickness_mean, cmap="viridis")
        _clip_to_circle(contour0, axes[0])
        axes[0].set_title("Bilayer Thickness", fontsize=fontsize, fontweight="bold", pad=title_pad)

        contour1 = axes[1].contourf(X, Y, curvature_data1, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour1, axes[1])
        contour2 = axes[2].contourf(X, Y, curvature_data2, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour2, axes[2])
        contour3 = axes[3].contourf(X, Y, curvature_data3, cmap="plasma", norm=norm, levels=levels)
        _clip_to_circle(contour3, axes[3])

        axes[1].set_title(f"{layer1} Bilayer: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)
        axes[2].set_title(f"{layer2} Bilayer: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)
        axes[3].set_title(f"{layer3} Bilayer: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes([0.054, 0.15, 0.02, 0.7])
        fig.colorbar(contour0, cax=cbar_ax).set_label("Thickness (nm)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

        cbar_ax2 = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar2 = fig.colorbar(contour1, cax=cbar_ax2)
        cbar2.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar2.ax.tick_params(labelsize=fontsize)

    elif mode == "mean" and not have_thickness:
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data3, curvature_data2]  # upper, middle, lower
        layers = [layer1, layer3, layer2]

        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, axes[i])
            axes[i].set_title(f"{layers[i]} Bilayer: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize, labelpad=title_pad)
        cbar_ax.tick_params(labelsize=fontsize)

    elif mode == "gaussian":
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data2, curvature_data3]
        layers = [layer1, layer2, layer3]

        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, axes[i])
            axes[i].set_title(f"{layers[i]} Bilayer: {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize, labelpad=title_pad)
        cbar_ax.tick_params(labelsize=fontsize)

    elif mode == "principal":
        fig, axes = plt.subplots(2, 3, figsize=(30, 26))
        fig.subplots_adjust(left=0.05, right=0.91, bottom=0.05, top=0.90, wspace=0.08, hspace=0.0)

        for i in range(3):
            ax = axes[0, i]
            c = ax.contourf(X, Y, curvature_k1[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, ax)
            ax.set_title(f"{layer1 if i==0 else layer2 if i==1 else layer3} Bilayer: k1",
                         fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k1[i][::step, ::step, 0], dirs_k1[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)

        for i in range(3):
            ax = axes[1, i]
            c = ax.contourf(X, Y, curvature_k2[i], cmap="plasma", norm=norm, levels=levels)
            _clip_to_circle(c, ax)
            ax.set_title(f"{layer1 if i==0 else layer2 if i==1 else layer3} Bilayer: k2",
                         fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k2[i][::step, ::step, 0], dirs_k2[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)

        cbar_ax = fig.add_axes([0.93, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for ax in axes.flatten():
        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
        plt.close()


def plot(args: List[str]) -> None:
    """CLI entry: plot mean curvature or thickness from a 'CALM analyze full' output directory."""
    parser = argparse.ArgumentParser(description="Plot mean curvature or thickness")
    parser.add_argument('-i', '--numpys_directory', type=str, help="'CALM analyze full' output directory")
    parser.add_argument('--mode', choices=["mean", "gaussian", "principal", "thickness"], default="mean", help="quantity to plot (default: mean)")
    parser.add_argument('-o', '--outfile', type=str, default="mean.png", help="output image path (default: mean.png)")
    parser.add_argument('--minimum', type=float, default=None, help="fix the color scale's lower bound")
    parser.add_argument('--maximum', type=float, default=None, help="fix the color scale's upper bound")
    parser.add_argument('--vectors', action="store_true", default=False, help="overlay principal-direction vectors")
    add_manual(parser, "map_plot")

    ns = parser.parse_args(args)
    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None

    draw(Dir=ns.numpys_directory, mode=ns.mode, minmax=minmax, filename=ns.outfile, show_vectors=ns.vectors)


if __name__ == "__main__":
    pass
