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

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    X, Y = np.meshgrid(x, y)
    return X, Y


def draw(Dir, layer1="Upper", layer2="Lower", layer3="Middle",
         minmax=None, t_minmax=None, filename=""):

    fontsize = 24

    if not Dir.endswith("/"):
        Dir += "/"

    # --- Load box dimensions ---
    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",",
                          skiprows=1, max_rows=1,
                          usecols=(1, 2, 3))

    # --- Load curvature files ---
    curvature_frames = []
    for file_path in sorted(glob.glob(Dir + "*_mean_curvature.npy")):
        curvature_frames.append(np.load(file_path))


    if len(curvature_frames) == 0:
        raise FileNotFoundError("No *_mean_curvature.npy files found.")

    curvature_frames = np.asarray(curvature_frames)
    curvature_mean = np.mean(curvature_frames, axis=0)
    # shape: (3, Nx, Ny)

    curvature_data1 = curvature_mean[0]
    curvature_data2 = curvature_mean[1]
    curvature_data3 = curvature_mean[2]

    # --- Load thickness files ---
    thickness_frames = []
    for file_path in sorted(glob.glob(Dir + "*_thickness.npy")):
        thickness_frames.append(np.load(file_path))

    if len(thickness_frames) == 0:
        raise FileNotFoundError("No *_thickness.npy files found.")

    thickness_frames = np.asarray(thickness_frames)
    thickness_mean = np.mean(thickness_frames, axis=0)
    # shape: (Nx, Ny)

    # --- Create X,Y grid from data shape ---
    Nx, Ny = curvature_data1.shape
    x = np.linspace(0, box_size[0], Nx)
    y = np.linspace(0, box_size[1], Ny)
    X, Y = np.meshgrid(x, y)

    # --- Global curvature min/max ---
    catted = [curvature_data1, curvature_data2, curvature_data3]

    if minmax is None:
        Minimum = np.min([np.min(x) for x in catted])
        Maximum = np.max([np.max(x) for x in catted])
    else:
        Minimum, Maximum = minmax

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(24, 28))
    axes = axes.flatten()

    for ax in axes:
        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'L$_x$'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'L$_y$'], fontsize=fontsize)

    # ---- Thickness plot ----
    if t_minmax is not None:
        t_norm = mcolors.Normalize(vmin=t_minmax[0], vmax=t_minmax[1])
        t_levels = np.linspace(t_minmax[0], t_minmax[1], 20)
        contour1 = axes[0].contourf(
            X, Y, thickness_mean,
            cmap="viridis",
            norm=t_norm,
            levels=t_levels
        )
    else:
        contour1 = axes[0].contourf(
            X, Y, thickness_mean,
            cmap="viridis"
        )

    axes[0].set_title("Thickness", fontsize=fontsize)

    # ---- Curvature plots ----
    levels = np.linspace(Minimum, Maximum, 20)
    norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)

    contour2 = axes[1].contourf(
        X, Y, curvature_data1,
        cmap="plasma", norm=norm,
        levels=levels
    )
    axes[1].set_title(f"Curvature {layer1}", fontsize=fontsize)

    contour3 = axes[2].contourf(
        X, Y, curvature_data2,
        cmap="plasma", norm=norm,
        levels=levels
    )
    axes[2].set_title(f"Curvature {layer2}", fontsize=fontsize)

    contour4 = axes[3].contourf(
        X, Y, curvature_data3,
        cmap="plasma", norm=norm,
        levels=levels
    )
    axes[3].set_title(f"Curvature {layer3}", fontsize=fontsize)

    # ---- Colorbars ----
    cbar_ax = fig.add_axes([0.08, 0.15, 0.02, 0.7])
    fig.colorbar(contour1, cax=cbar_ax)
    cbar_ax.set_ylabel("Thickness (nm)", fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize)

    cbar_ax2 = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(contour2, cax=cbar_ax2)
    cbar_ax2.set_ylabel("Curvature (nm$^{-1}$)", fontsize=fontsize)
    cbar_ax2.tick_params(labelsize=fontsize)

    plt.tight_layout(rect=[0.1, 0, .9, 1])


    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=300)


def plot(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Plot the curvature of a membrane",formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--numpys_directory', type=str,help="Path to numpy directory (output folder from calculate).")
    parser.add_argument('-l1', '--layer1', type=str, default="Upper")
    parser.add_argument('-l2', '--layer2', type=str, default="Lower")
    parser.add_argument('-l3', '--layer3', type=str, default="Middle")
    parser.add_argument('--minimum', type=float, default=None)
    parser.add_argument('--maximum', type=float, default=None)
    parser.add_argument('--thickness_minimum', type=float, default=None)
    parser.add_argument('--thickness_maximum', type=float, default=None)
    parser.add_argument('-o', '--outfile', type=str, default="")

    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    if args.minimum is not None and args.maximum is not None:
        minmax = [args.minimum, args.maximum]
    else:
        minmax = None

    if args.thickness_minimum is not None and args.thickness_maximum is not None:
        thickness_minmax = [args.thickness_minimum, args.thickness_maximum]
    else:
        thickness_minmax = None

    try:
        draw(Dir=args.numpys_directory,layer1=args.layer1,layer2=args.layer2,layer3=args.layer3,minmax=minmax,t_minmax=thickness_minmax,filename=args.outfile)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    import sys
    plot(sys.argv[1:])
