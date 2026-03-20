import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as mcolors
import warnings
import os
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "serif"

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, where=norm > 0)

def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    X, Y = np.meshgrid(x, y)
    return X, Y

def draw(Dir, mode="mean", layer1="Upper", layer2="Lower", layer3="Middle", minmax=None, filename="", show_vectors=True, title_pad=12,):
    fontsize = 20

    if not Dir.endswith("/"):
        Dir += "/"

    # --- Load box dimensions ---
    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1, max_rows=1, usecols=(1, 2, 3))

    # --- Grid ---
    X, Y = get_XY(box_size)

    # --- Load curvatures ---
    if mode == "mean":
        pattern = "*_mean_curvature.npy"
    elif mode == "gaussian":
        pattern = "*_gaussian_curvature.npy"
    elif mode == "principal":
        pattern = "*_principal_curvatures.npy"
    else:
        raise ValueError("mode must be 'mean', 'gaussian', or 'principal'")

    curvature_frames = [np.load(f) for f in sorted(glob.glob(Dir + pattern))]
    curvature_frames = np.asarray(curvature_frames)
    curvature_mean = np.nanmean(curvature_frames, axis=0)

    if mode == "mean":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Mean Curvature"

        thickness_frames = [np.load(f) for f in sorted(glob.glob(Dir + "*_thickness.npy"))]
        thickness_frames = np.asarray(thickness_frames)
        thickness_mean = np.nanmean(thickness_frames, axis=0)

    elif mode == "gaussian":
        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]
        quantity = "Gaussian Curvature"

    elif mode == "principal":
        curvature_k1 = [curvature_mean[0], curvature_mean[2], curvature_mean[4]]
        curvature_k2 = [curvature_mean[1], curvature_mean[3], curvature_mean[5]]

        dir_files = sorted(glob.glob(Dir + "*_principal_dirs.npy"))
        dir_frames = [np.load(f) for f in dir_files]
        dir_frames = np.asarray(dir_frames)
        dir_mean = np.nanmean(dir_frames, axis=0)

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

    if mode == "mean":
        fig, axes = plt.subplots(2, 2, figsize=(32, 26), gridspec_kw={"hspace": 0.075, "wspace": 0.001})
        axes = axes.flatten()
        fig.subplots_adjust(left=0.07, right=0.89, bottom=0.03, top=0.97)

        contour0 = axes[0].contourf(X, Y, thickness_mean, cmap="viridis")
        axes[0].set_title("Bilayer Thickness", fontsize=fontsize, fontweight="bold", pad=title_pad)

        contour1 = axes[1].contourf(X, Y, curvature_data1, cmap="plasma", norm=norm, levels=levels)
        contour2 = axes[2].contourf(X, Y, curvature_data2, cmap="plasma", norm=norm, levels=levels)
        contour3 = axes[3].contourf(X, Y, curvature_data3, cmap="plasma", norm=norm, levels=levels)

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

    elif mode == "gaussian":
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data2, curvature_data3]
        layers = [layer1, layer2, layer3]

        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
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
    parser.add_argument('-d', '--numpys_directory', type=str)
    parser.add_argument('--mode', choices=["mean", "gaussian", "principal"], default="mean")
    parser.add_argument('-o', '--outfile', type=str, default="")
    parser.add_argument('--minimum', type=float, default=None)
    parser.add_argument('--maximum', type=float, default=None)
    parser.add_argument('--vectors', action="store_true", help="Show principal direction vectors")

    args = parser.parse_args(args)
    minmax = [args.minimum, args.maximum] if args.minimum is not None and args.maximum is not None else None

    draw(Dir=args.numpys_directory, mode=args.mode, minmax=minmax, filename=args.outfile, show_vectors=args.vectors)

if __name__ == "__main__":
    pass