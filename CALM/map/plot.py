import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
import os
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, where=norm > 0)


def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
    return np.meshgrid(x, y)


# -------------------------
# core plotting
# -------------------------

def draw(Dir,mode="mean",layer1="Upper",layer2="Lower",layer3="Middle",minmax=None,filename="",show_vectors=True,avg_surface=False,title_pad=12,):

    fontsize = 20

    if not Dir.endswith("/"):
        Dir += "/"

    # -------------------------
    # box
    # -------------------------
    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1, max_rows=1, usecols=(1, 2, 3))

    X, Y = get_XY(box_size)

    # -------------------------
    # mode selection
    # -------------------------
    pattern = None

    if mode == "mean":
        pattern = "*_mean_curvature.npy"
    elif mode == "gaussian":
        pattern = "*_gaussian_curvature.npy"
    elif mode == "principal":
        pattern = "*_principal_curvatures.npy"
    elif mode == "thickness":
        pattern = "*_thickness.npy"
    else:
        raise ValueError("mode must be mean, gaussian, principal, or thickness")

    # -------------------------
    # LOAD DATA
    # -------------------------

    if avg_surface:

        if mode == "mean":
            curvature_mean = np.load(os.path.join(Dir, "avg_surface_mean_curvature.npy"))
        elif mode == "gaussian":
            curvature_mean = np.load(os.path.join(Dir, "avg_surface_gaussian_curvature.npy"))
        elif mode == "principal":
            curvature_mean = np.load(os.path.join(Dir, "avg_surface_principal_curvatures.npy"))
        elif mode == "thickness":
            curvature_mean = np.load(os.path.join(Dir, "avg_surface_thickness.npy"))

    else:

        frames = [np.load(f) for f in sorted(glob.glob(Dir + pattern))]
        frames = np.asarray(frames)
        curvature_mean = np.nanmean(frames, axis=0)


    # -------------------------
    # PLOTTING
    # -------------------------

    if mode == "thickness":

        fig, ax = plt.subplots(figsize=(12, 10))

        Minimum = np.nanmin(curvature_mean)
        Maximum = np.nanmax(curvature_mean)

        norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)

        c = ax.contourf(X, Y,curvature_mean,cmap="viridis",norm=norm)

        ax.set_title("Bilayer Thickness",fontsize=fontsize,fontweight="bold",pad=title_pad)

        sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Thickness (nm)", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        axes = np.array([ax])


    elif mode == "mean":

        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data2, curvature_data3]
        layers = [layer1, layer2, layer3]

        all_vals = np.concatenate([c.flatten() for c in curvatures])
        all_vals = all_vals[~np.isnan(all_vals)]

        Minimum = np.min(all_vals) if minmax is None else minmax[0]
        Maximum = np.max(all_vals) if minmax is None else minmax[1]

        norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)
        levels = np.linspace(Minimum, Maximum, 20)

        sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
        sm.set_array([])

        for i in range(3):
            axes[i].contourf(X, Y, curvatures[i],cmap="plasma", norm=norm, levels=levels)
            axes[i].set_title(f"{layers[i]} Bilayer: Mean Curvature",fontsize=fontsize,fontweight="bold",pad=title_pad)

        cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)

    # -------------------------
    # GAUSSIAN
    # -------------------------
    elif mode == "gaussian":

        curvature_data1 = curvature_mean[0]
        curvature_data2 = curvature_mean[1]
        curvature_data3 = curvature_mean[2]

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        curvatures = [curvature_data1, curvature_data2, curvature_data3]
        layers = [layer1, layer2, layer3]

        all_vals = np.concatenate([c.flatten() for c in curvatures])
        all_vals = all_vals[~np.isnan(all_vals)]

        Minimum = np.min(all_vals) if minmax is None else minmax[0]
        Maximum = np.max(all_vals) if minmax is None else minmax[1]

        norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)
        levels = np.linspace(Minimum, Maximum, 20)

        sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
        sm.set_array([])

        for i in range(3):
            axes[i].contourf(X, Y, curvatures[i], cmap="plasma", norm=norm, levels=levels)
            axes[i].set_title(f"{layers[i]} Bilayer: Gaussian Curvature",fontsize=fontsize,fontweight="bold",pad=title_pad)

        cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)

    # -------------------------
    # PRINCIPAL
    # -------------------------
    elif mode == "principal":

        curvature_k1 = [curvature_mean[0], curvature_mean[2], curvature_mean[4]]
        curvature_k2 = [curvature_mean[1], curvature_mean[3], curvature_mean[5]]

        fig, axes = plt.subplots(2, 3, figsize=(30, 26))
        fig.subplots_adjust(left=0.05, right=0.91, bottom=0.05, top=0.90, wspace=0.08, hspace=0.0)

        all_vals = np.concatenate([c.flatten() for c in curvature_k1 + curvature_k2])
        all_vals = all_vals[~np.isnan(all_vals)]

        Minimum = np.min(all_vals) if minmax is None else minmax[0]
        Maximum = np.max(all_vals) if minmax is None else minmax[1]

        norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)
        levels = np.linspace(Minimum, Maximum, 20)

        sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
        sm.set_array([])

        for i in range(3):
            axes[0, i].contourf(X, Y, curvature_k1[i], cmap="plasma", norm=norm, levels=levels)
            axes[1, i].contourf(X, Y, curvature_k2[i], cmap="plasma", norm=norm, levels=levels)

            axes[0, i].set_title(f"{['Upper','Lower','Middle'][i]} k1", fontsize=fontsize)
            axes[1, i].set_title(f"{['Upper','Lower','Middle'][i]} k2", fontsize=fontsize)

        cbar_ax = fig.add_axes([0.93, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
    # -------------------------
    # FINAL AXES FORMATTING
    # -------------------------
    for ax in np.ravel(axes):
        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(['0', 'Lx'], fontsize=fontsize)
        ax.set_yticklabels(['0', 'Ly'], fontsize=fontsize)

    # -------------------------
    # OUTPUT
    # -------------------------
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
        plt.close()


# -------------------------
# CLI
# -------------------------

def plot(args):
    parser = argparse.ArgumentParser(description="Plot bilayer thickness and curvature")
    parser.add_argument("-i", "--numpys_directory", type=str)
    parser.add_argument("--mode", choices=["mean", "gaussian", "principal", "thickness"], default="mean", help="Choose which curvature to plot, default=mean")
    parser.add_argument("-o", "--outfile", default="mean.png", help="Choose what to call the output .png, deafult=mean.png")
    parser.add_argument("--vectors", action="store_true",help="Show principal direction vectors, default=False", default=False)
    parser.add_argument("--avg-surface", action="store_true",help="Plot curvature of the averaged surface.")
    parser.add_argument("--minimum", type=float,default=None , help="Choose maximum curvature, default=None")
    parser.add_argument("--maximum", type=float,default=None, help="Choose minimum curvature, default=None")

    args = parser.parse_args(args)

    minmax = None
    if args.minimum is not None and args.maximum is not None:
        minmax = [args.minimum, args.maximum]

    draw(Dir=args.numpys_directory,mode=args.mode,minmax=minmax,filename=args.outfile,show_vectors=args.vectors,avg_surface=args.avg_surface,)


if __name__ == "__main__":
    pass






