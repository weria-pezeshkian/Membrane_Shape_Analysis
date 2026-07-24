import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import matplotlib.colors as mcolors
import warnings
import os
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

from ..core.fourier_sft import SFT
from ..core.rotation import (
    rotation_was_used,
    fixed_circle_radius,
    recover_all_rotation_angles,
    lookup_mask_at_rotated_grid,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "serif"

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, where=norm > 0)

def get_XY(box_size, gridsize):
    x = np.linspace(0, box_size[0], gridsize)
    y = np.linspace(0, box_size[1], gridsize)
    X, Y = np.meshgrid(x, y)
    return X, Y

def _frame_number(path):
    return int(Path(path).stem.split("_")[0])

def _hole_masks_for_frame(sft, frame_idx, theta):
    """(upper, lower) hole masks for one frame of sft, remapped onto the
    output grid via the rotation-aware lookup if theta != 0. Safe only
    within fixed_circle_radius (see core/rotation.py) - fine here, since
    everything outside that circle is already excluded for the unrelated
    circle_cutter reason."""
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

def _load_and_mask(files, pattern, Dir, sft, layer_sources):
    """Load and average a set of per-frame .npy files, applying the (optional)
    hole mask per frame before averaging.

    layer_sources:
      - None: the array has no leading layer axis (e.g. thickness, already
        the combined bilayer quantity) - the union (upper OR lower) hole
        mask is applied directly.
      - a list, one entry per layer along axis 0: "upper", "lower", "union"
        (upper OR lower - e.g. a middle-surface layer), or None (skip).

    Averaging is conservative: a single NaN across the stacked frames at a
    given point (from circle_cutter already baked into the file, or from the
    hole mask applied here) makes the *averaged* point NaN too - plain
    np.mean, not np.nanmean, so a point isn't silently averaged over
    whichever frames happened to have data there. This relies on the hole
    mask's own per-frame false-positive rate being low (see
    core/fourier_build.py's median_multiple_threshold comment) - an under-
    tuned per-frame threshold would make this poison almost every point over
    a long trajectory; the fix belongs in the per-frame threshold, not here.
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

def draw(Dir, mode="mean", layer1="Upper", layer2="Lower", layer3="Middle", minmax=None, filename="", show_vectors=True, title_pad=12,):
    fontsize = 20

    if not Dir.endswith("/"):
        Dir += "/"

    # --- Load box dimensions ---
    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1, max_rows=1, usecols=(1, 2, 3))

    # --- Detect whether --rotate / --Remove-TMD were used for this run ---
    sft = None
    circle_radius = None
    try:
        sft = SFT.from_directory(Dir)
        if rotation_was_used(sft):
            circle_radius = fixed_circle_radius(sft)
    except FileNotFoundError:
        pass

    def _clip_to_circle(contour_set, ax):
        """Confine a contourf's rendering to the shared circle: everything
        outside is left as plain white background, not drawn at all - no
        outline, no reliance on the underlying NaN mask's grid granularity
        for a clean edge."""
        if circle_radius is not None:
            circle = mpatches.Circle(
                (box_size[0] / 2.0, box_size[1] / 2.0), circle_radius,
                transform=ax.transData,
            )
            contour_set.set_clip_path(circle)

    if mode == "thickness":
        # Single panel, single colorbar - thickness has no upper/lower/middle
        # stacking (it's already the combined bilayer quantity).
        thickness_mean = _load_and_mask(sorted(glob.glob(Dir + "*_thickness.npy")), "*_thickness.npy", Dir, sft, layer_sources=None)
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

    # --- Load curvatures ---
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

    curvature_mean = _load_and_mask(sorted(glob.glob(Dir + pattern)), pattern, Dir, sft, curvature_layer_sources)

    # --- Grid, sized to match the actual saved data rather than assuming a
    # fixed resolution - the data's own --gridsize may differ from any
    # particular default. ---
    gridsize = curvature_mean.shape[-1]
    X, Y = get_XY(box_size, gridsize)

    have_thickness = False

    if mode == "mean":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Mean Curvature"

        # thickness is optional here - "mean" was computed without it if
        # --method didn't include "thickness". Layout adapts below rather
        # than requiring it.
        thickness_files = sorted(glob.glob(Dir + "*_thickness.npy"))
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

        dir_files = sorted(glob.glob(Dir + "*_principal_dirs.npy"))
        dir_mean = _load_and_mask(dir_files, "*_principal_dirs.npy", Dir, sft, curvature_layer_sources)

        dirs_k1 = [normalize(dir_mean[0]), normalize(dir_mean[2]), normalize(dir_mean[4])]
        dirs_k2 = [normalize(dir_mean[1]), normalize(dir_mean[3]), normalize(dir_mean[5])]

    # --- Determine min/max ---
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

    #========================
    #--- PLOTTING ----------
    #========================

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

        # Colorbars
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
        # No thickness available - single row, upper/middle/lower curvature
        # only, one shared colorscale (same structure as "gaussian" below).
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
        #cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    elif mode == "principal":
        fig, axes = plt.subplots(2, 3, figsize=(30, 26))
        fig.subplots_adjust(left=0.05, right=0.91, bottom=0.05, top=0.90, wspace=0.08, hspace=0.0)

        # --- Plot k1 ---
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

        # --- Plot k2 ---
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

    # --- Axis formatting ---
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
    parser = argparse.ArgumentParser(description="Plot membrane curvature")
    parser.add_argument('-i', '--numpys_directory', type=str)
    parser.add_argument('--mode', choices=["mean", "gaussian", "principal", "thickness"], default="mean", help="Choose which curvature to plot, default=mean")
    parser.add_argument('-o', '--outfile', type=str, default="mean.png")
    parser.add_argument('--minimum', type=float, default=None, help="Choose maximum, default=None")
    parser.add_argument('--maximum', type=float, default=None, help="Choose minimum, default=None")
    parser.add_argument('--vectors', action="store_true", help="Show principal direction vectors, default=False", default=False)

    args = parser.parse_args(args)
    minmax = [args.minimum, args.maximum] if args.minimum is not None and args.maximum is not None else None

    draw(Dir=args.numpys_directory, mode=args.mode, minmax=minmax, filename=args.outfile, show_vectors=args.vectors)

if __name__ == "__main__":
    pass
