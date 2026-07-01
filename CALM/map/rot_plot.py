import argparse
import glob
import logging
import os
import warnings
from typing import List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import rotate
from tqdm import tqdm

from ..core.calc_vectors import get_rotation_and_protein

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
plt.rcParams["font.family"] = "serif"


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v, dtype=float), where=norm > 0)


def get_XY(box_size: Sequence[float], shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape
    x = np.linspace(0, float(box_size[0]), W)
    y = np.linspace(0, float(box_size[1]), H)
    return np.meshgrid(x, y)


def _inscribed_disk_mask(shape: Tuple[int, int], inset_px: float = 0.5) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    radius = min(cx, cy) - inset_px
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= radius


def _theta_from_OP_box(O_box: np.ndarray, P_box: np.ndarray, Lx: float, Ly: float) -> float:
    """Signed angle between O->P and the direction from O to the top-center of the box."""
    O = np.asarray(O_box, float)
    P = np.asarray(P_box, float)
    top_center = np.array([Lx / 2.0, Ly], float)

    OP = P - O
    Oref = top_center - O
    denom = np.linalg.norm(OP) * np.linalg.norm(Oref)
    if denom == 0:
        return 0.0

    theta = np.degrees(np.arccos(np.clip(np.dot(OP, Oref) / denom, -1.0, 1.0)))
    cross = OP[0] * Oref[1] - OP[1] * Oref[0]
    return theta if cross > 0 else -theta


def _rotate_points_about_center(points_xy: np.ndarray, theta_deg: float, center_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, float)
    center = np.asarray(center_xy, float)
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (pts - center) @ R.T + center


def rotate_vector_components(vec_xy: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate 2D vector components by theta_deg without moving their grid positions."""
    theta = np.radians(theta_deg)
    vx = vec_xy[..., 0]
    vy = vec_xy[..., 1]
    out = np.empty_like(vec_xy, dtype=float)
    out[..., 0] = vx * np.cos(theta) - vy * np.sin(theta)
    out[..., 1] = vx * np.sin(theta) + vy * np.cos(theta)
    return out


def _calc_frame_recenter_and_theta(arr2d: np.ndarray, O_box: np.ndarray, P_box: np.ndarray, Lx: float, Ly: float):
    """Recenter a 2D field around O and calculate the rotation angle from the shifted O->P vector."""
    H, W = arr2d.shape
    center_px = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    O_px = np.array([O_box[0] / Lx * (W - 1), O_box[1] / Ly * (H - 1)])

    sx = int(round(center_px[0] - O_px[0]))
    sy = int(round(center_px[1] - O_px[1]))
    recentered = np.roll(np.roll(arr2d, sx, axis=1), sy, axis=0)

    dx_box = sx * (Lx / (W - 1))
    dy_box = sy * (Ly / (H - 1))
    O_shifted = np.array([Lx / 2.0, Ly / 2.0], float)
    P_shifted = np.array([P_box[0] + dx_box, P_box[1] + dy_box], float)
    theta = _theta_from_OP_box(O_shifted, P_shifted, Lx, Ly)

    return recentered, theta, dx_box, dy_box


def _scatter_protein(ax, pts: np.ndarray, O_idx: Optional[np.ndarray], P_idx: Optional[np.ndarray]) -> None:
    if pts.size == 0:
        return

    ax.scatter(pts[:, 0], pts[:, 1], s=20, c="white", edgecolors="black", linewidths=0.5, alpha=0.9, zorder=10)

    def com(indices):
        if indices is None:
            return None
        valid = [idx for idx in indices if 0 <= idx < pts.shape[0]]
        return pts[valid].mean(axis=0) if valid else None

    comO = com(O_idx)
    comP = com(P_idx)
    if comO is not None:
        ax.scatter([comO[0]], [comO[1]], s=45, c="red", edgecolors="black", linewidths=0.8, zorder=15)
    if comP is not None:
        ax.scatter([comP[0]], [comP[1]], s=45, c="lime", edgecolors="black", linewidths=0.8, zorder=15)


def _set_axes_style_all(axes: np.ndarray, box_size: np.ndarray, fontsize: int) -> None:
    for ax in axes.flatten():
        ax.set_aspect(box_size[0] / box_size[1])
        ax.set_xticks([0, box_size[0]])
        ax.set_yticks([0, box_size[1]])
        ax.set_xticklabels(["0", "L$_x$"], fontsize=fontsize)
        ax.set_yticklabels(["0", "L$_y$"], fontsize=fontsize)


def draw(
    Dir: str,
    mode: str = "mean",
    layer1: str = "Upper",
    layer2: str = "Lower",
    layer3: str = "Middle",
    minmax: Optional[Sequence[float]] = None,
    filename: str = "",
    show_vectors: bool = False,
    title_pad: int = 12,
) -> None:
    fontsize = 20
    Dir = os.path.abspath(Dir)
    layers = [layer1, layer2, layer3]

    dim_file = os.path.join(Dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1, max_rows=1, usecols=(1, 2, 3)).astype(float)
    Lx, Ly = float(box_size[0]), float(box_size[1])

    o_all = np.asarray(np.load(os.path.join(Dir, "rotation_vectors_o.npy")))[:, :2].astype(float)
    p_all = np.asarray(np.load(os.path.join(Dir, "rotation_vectors_p.npy")))[:, :2].astype(float)

    prot_raw = np.asarray(np.load(os.path.join(Dir, "protein_atom_positions_rotation.npy")))
    if prot_raw.ndim == 3:
        prot_all_box = prot_raw[:, :, :2].astype(float)
    elif prot_raw.ndim == 2:
        prot_all_box = prot_raw[None, :, :2].astype(float)
    elif prot_raw.ndim == 1:
        prot_all_box = prot_raw[None, None, :2].astype(float)
    else:
        raise ValueError(f"Unexpected protein array shape: {prot_raw.shape}")

    origin_file = os.path.join(Dir, "rotation_origin_indices.npy")
    p2_file = os.path.join(Dir, "rotation_p2_indices.npy")
    O_idx = np.asarray(np.load(origin_file)).astype(int).ravel() if os.path.exists(origin_file) else None
    P_idx = np.asarray(np.load(p2_file)).astype(int).ravel() if os.path.exists(p2_file) else None

    if mode == "mean":
        pattern = "*_mean_curvature.npy"
        quantity = "Mean Curvature"
    elif mode == "gaussian":
        pattern = "*_gaussian_curvature.npy"
        quantity = "Gaussian Curvature"
    elif mode == "principal":
        pattern = "*_principal_curvatures.npy"
        quantity = "Principal Curvature"
    else:
        raise ValueError("mode must be 'mean', 'gaussian', or 'principal'")

    files = sorted(glob.glob(os.path.join(Dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(Dir, pattern)}")

    curvature_frames = np.asarray([np.load(f) for f in files])
    n_frames = min(curvature_frames.shape[0], o_all.shape[0], p_all.shape[0])
    curvature_frames = curvature_frames[:n_frames]
    o_all = o_all[:n_frames]
    p_all = p_all[:n_frames]

    # Align every scalar curvature matrix: recenter by O, rotate by -theta, then average.
    aligned_curvatures = np.empty_like(curvature_frames, dtype=float)
    if curvature_frames.ndim != 4:
        raise ValueError(f"Expected curvature shape (F,N,H,W), got {curvature_frames.shape}")

    for i in tqdm(range(n_frames), desc="Align curvature frames", unit="frame"):
        rec_i, theta_i, _, _ = _calc_frame_recenter_and_theta(curvature_frames[i, 0], o_all[i], p_all[i], Lx, Ly)
        aligned_curvatures[i, 0] = rotate(rec_i, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)

        # Use the same frame-specific protein transform for the other layers/curvatures.
        for j in range(1, curvature_frames.shape[1]):
            H, W = curvature_frames[i, j].shape
            center_px = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
            O_px = np.array([o_all[i, 0] / Lx * (W - 1), o_all[i, 1] / Ly * (H - 1)])
            sx = int(round(center_px[0] - O_px[0]))
            sy = int(round(center_px[1] - O_px[1]))
            recentered = np.roll(np.roll(curvature_frames[i, j], sx, axis=1), sy, axis=0)
            aligned_curvatures[i, j] = rotate(recentered, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)

    curvature_mean = np.nanmean(aligned_curvatures, axis=0)
    disk_mask = _inscribed_disk_mask(curvature_mean.shape[-2:])
    curvature_mean = np.where(disk_mask, curvature_mean, np.nan)

    if mode in ["mean", "gaussian"]:
        curvatures = [curvature_mean[0], curvature_mean[1], curvature_mean[2]]
    else:
        curvature_k1 = [curvature_mean[0], curvature_mean[2], curvature_mean[4]]
        curvature_k2 = [curvature_mean[1], curvature_mean[3], curvature_mean[5]]

    # For mean curvature mode, keep the old first-panel logic: align and average the
    # Z-fit/Fourier-approximation maps. If those files are not present, fall back to
    # the newer thickness maps used by plot.py.
    if mode == "mean":
        zfit_files = sorted(glob.glob(os.path.join(Dir, "Z_fitted_*_Both.npy")))

        if zfit_files:
            zfit_frames = np.asarray([np.load(f) for f in zfit_files])
            n_zfit = min(zfit_frames.shape[0], o_all.shape[0], p_all.shape[0])
            zfit_frames = zfit_frames[:n_zfit]
            aligned_zfit = np.empty_like(zfit_frames, dtype=float)

            for i in tqdm(range(n_zfit), desc="Align Z-fit frames", unit="frame"):
                rec_i, theta_i, _, _ = _calc_frame_recenter_and_theta(zfit_frames[i], o_all[i], p_all[i], Lx, Ly)
                aligned_zfit[i] = rotate(rec_i, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)

            first_panel_data = np.nanmean(aligned_zfit, axis=0)
            first_panel_data = np.where(disk_mask, first_panel_data, np.nan)
            first_panel_title = "Fourier Approximation"
            first_panel_cbar = "Z-fit height (nm)"

        else:
            thickness_files = sorted(glob.glob(os.path.join(Dir, "*_thickness.npy")))
            if not thickness_files:
                raise FileNotFoundError(
                    f"No Z-fit files or thickness files found in {Dir}"
                )

            thickness_frames = np.asarray([np.load(f) for f in thickness_files])
            n_thick = min(thickness_frames.shape[0], o_all.shape[0], p_all.shape[0])
            thickness_frames = thickness_frames[:n_thick]
            aligned_thickness = np.empty_like(thickness_frames, dtype=float)

            for i in tqdm(range(n_thick), desc="Align thickness frames", unit="frame"):
                rec_i, theta_i, _, _ = _calc_frame_recenter_and_theta(thickness_frames[i], o_all[i], p_all[i], Lx, Ly)
                aligned_thickness[i] = rotate(rec_i, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)

            first_panel_data = np.nanmean(aligned_thickness, axis=0)
            first_panel_data = np.where(disk_mask, first_panel_data, np.nan)
            first_panel_title = "Protein-aligned Bilayer Thickness"
            first_panel_cbar = "Thickness (nm)"

    # For principal mode, optionally align direction vectors too.
    if mode == "principal":
        dirs_k1 = dirs_k2 = None
        if show_vectors:
            dir_files = sorted(glob.glob(os.path.join(Dir, "*_principal_dirs.npy")))
            if not dir_files:
                raise FileNotFoundError(f"No files matched {os.path.join(Dir, '*_principal_dirs.npy')}")

            dir_frames = np.asarray([np.load(f) for f in dir_files])
            n_dirs = min(dir_frames.shape[0], o_all.shape[0], p_all.shape[0])
            dir_frames = dir_frames[:n_dirs]
            aligned_dirs = np.empty_like(dir_frames, dtype=float)

            for i in tqdm(range(n_dirs), desc="Align direction frames", unit="frame"):
                _, theta_i, _, _ = _calc_frame_recenter_and_theta(dir_frames[i, 0, :, :, 0], o_all[i], p_all[i], Lx, Ly)

                for j in range(dir_frames.shape[1]):
                    H, W = dir_frames[i, j, :, :, 0].shape
                    center_px = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
                    O_px = np.array([o_all[i, 0] / Lx * (W - 1), o_all[i, 1] / Ly * (H - 1)])
                    sx = int(round(center_px[0] - O_px[0]))
                    sy = int(round(center_px[1] - O_px[1]))

                    recentered_x = np.roll(np.roll(dir_frames[i, j, :, :, 0], sx, axis=1), sy, axis=0)
                    recentered_y = np.roll(np.roll(dir_frames[i, j, :, :, 1], sx, axis=1), sy, axis=0)
                    rotated_x = rotate(recentered_x, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)
                    rotated_y = rotate(recentered_y, -theta_i, reshape=False, order=1, mode="nearest", prefilter=False)

                    # Rotate vector components by +theta because the field itself was spatially rotated by -theta.
                    aligned_dirs[i, j] = rotate_vector_components(np.stack([rotated_x, rotated_y], axis=-1), theta_i)

            dir_mean = normalize(np.nanmean(normalize(aligned_dirs), axis=0))
            dirs_k1 = [np.where(disk_mask[..., None], dir_mean[0], np.nan),
                       np.where(disk_mask[..., None], dir_mean[2], np.nan),
                       np.where(disk_mask[..., None], dir_mean[4], np.nan)]
            dirs_k2 = [np.where(disk_mask[..., None], dir_mean[1], np.nan),
                       np.where(disk_mask[..., None], dir_mean[3], np.nan),
                       np.where(disk_mask[..., None], dir_mean[5], np.nan)]

    # Color scale after rotation and masking.
    if minmax is None:
        if mode == "principal":
            all_vals = np.concatenate([c.ravel() for c in curvature_k1 + curvature_k2])
        else:
            all_vals = np.concatenate([c.ravel() for c in curvatures])
        all_vals = all_vals[~np.isnan(all_vals)]
        Minimum = float(np.min(all_vals))
        Maximum = float(np.max(all_vals))
        if Minimum == Maximum:
            Maximum = Minimum + 1e-6
    else:
        Minimum, Maximum = map(float, minmax)

    levels = np.linspace(Minimum, Maximum, 20)
    norm = mcolors.Normalize(vmin=Minimum, vmax=Maximum)
    cmap = "plasma"

    # Representative protein overlay in the aligned coordinate system.
    i_rep = n_frames // 2
    example_field = curvature_frames[i_rep, 0]
    rec_i, theta_i, dx_box, dy_box = _calc_frame_recenter_and_theta(example_field, o_all[i_rep], p_all[i_rep], Lx, Ly)
    prot_rep = prot_all_box[i_rep if prot_all_box.shape[0] > 1 else 0]
    prot_recent = prot_rep + np.array([dx_box, dy_box], float)
    prot_rot = _rotate_points_about_center(prot_recent, theta_i, np.array([Lx / 2.0, Ly / 2.0]))

    # ====================== plotting ======================
    if mode == "mean":
        X, Y = get_XY(box_size, curvatures[0].shape)
        fig, axes = plt.subplots(2, 2, figsize=(32, 26), gridspec_kw={"hspace": 0.075, "wspace": 0.001})
        axes = axes.flatten()
        fig.subplots_adjust(left=0.07, right=0.89, bottom=0.03, top=0.97)

        contour0 = axes[0].contourf(X, Y, first_panel_data, cmap="viridis")
        axes[0].set_title(first_panel_title, fontsize=fontsize, fontweight="bold", pad=title_pad)

        contour1 = axes[1].contourf(X, Y, curvatures[0], cmap=cmap, norm=norm, levels=levels)
        axes[2].contourf(X, Y, curvatures[1], cmap=cmap, norm=norm, levels=levels)
        axes[3].contourf(X, Y, curvatures[2], cmap=cmap, norm=norm, levels=levels)

        for ax, layer in zip(axes[1:], layers):
            ax.set_title(f"{layer} Bilayer: Protein-aligned {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)

        cbar_ax = fig.add_axes([0.054, 0.15, 0.02, 0.7])
        fig.colorbar(contour0, cax=cbar_ax).set_label(first_panel_cbar, fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar_ax.yaxis.set_ticks_position("left")
        cbar_ax.yaxis.set_label_position("left")

        cbar_ax2 = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar2 = fig.colorbar(contour1, cax=cbar_ax2)
        cbar2.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar2.ax.tick_params(labelsize=fontsize)

        for ax in axes:
            _scatter_protein(ax, prot_rot, O_idx, P_idx)

    elif mode == "gaussian":
        X, Y = get_XY(box_size, curvatures[0].shape)
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.88, wspace=0.05)

        c = None
        for i in range(3):
            c = axes[i].contourf(X, Y, curvatures[i], cmap=cmap, norm=norm, levels=levels)
            axes[i].set_title(f"{layers[i]} Bilayer: Protein-aligned {quantity}", fontsize=fontsize, fontweight="bold", pad=title_pad)
            _scatter_protein(axes[i], prot_rot, O_idx, P_idx)

        cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize, labelpad=title_pad)
        cbar_ax.tick_params(labelsize=fontsize)

    elif mode == "principal":
        X, Y = get_XY(box_size, curvature_k1[0].shape)
        fig, axes = plt.subplots(2, 3, figsize=(30, 26))
        fig.subplots_adjust(left=0.05, right=0.91, bottom=0.05, top=0.90, wspace=0.08, hspace=0.0)

        c = None
        for i in range(3):
            ax = axes[0, i]
            c = ax.contourf(X, Y, curvature_k1[i], cmap=cmap, norm=norm, levels=levels)
            ax.set_title(f"{layers[i]} Bilayer: Protein-aligned k1", fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors and dirs_k1 is not None:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k1[i][::step, ::step, 0], dirs_k1[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)
            _scatter_protein(ax, prot_rot, O_idx, P_idx)

        for i in range(3):
            ax = axes[1, i]
            c = ax.contourf(X, Y, curvature_k2[i], cmap=cmap, norm=norm, levels=levels)
            ax.set_title(f"{layers[i]} Bilayer: Protein-aligned k2", fontsize=fontsize, fontweight="bold", pad=title_pad)
            if show_vectors and dirs_k2 is not None:
                step = 5
                ax.quiver(X[::step, ::step], Y[::step, ::step],
                          dirs_k2[i][::step, ::step, 0], dirs_k2[i][::step, ::step, 1],
                          color="black", scale=30, width=0.002, alpha=0.6)
            _scatter_protein(ax, prot_rot, O_idx, P_idx)

        cbar_ax = fig.add_axes([0.93, 0.08, 0.02, 0.8])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Curvature (nm$^{-1}$)", fontsize=fontsize)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    _set_axes_style_all(np.asarray(axes), box_size, fontsize)

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Saved {filename}")


def rot_plot(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Protein-aligned membrane curvature plots for the new numpy layout")
    parser.add_argument("-i", "--numpys_directory", required=True, type=str)
    parser.add_argument("--mode", choices=["mean", "gaussian", "principal"], default="mean")
    parser.add_argument("-o", "--outfile", type=str, default="rotated_mean.png")
    parser.add_argument("--minimum", type=float, default=None)
    parser.add_argument("--maximum", type=float, default=None)
    parser.add_argument("--vectors", action="store_true", default=False, help="Show principal direction vectors for mode='principal'")
    parser.add_argument("-f", "--trajectory", type=str, required=True, help="Trajectory file, e.g. .xtc/.dcd")
    parser.add_argument("-s", "--structure", type=str, required=True, help="Structure/topology file, e.g. .gro/.pdb/.psf")
    parser.add_argument("-F", "--From", default=0, type=int, help="First trajectory frame index")
    parser.add_argument("-U", "--Until", default=None, type=int, help="Stop before this frame index")
    parser.add_argument("-S", "--Step", default=1, type=int, help="Trajectory stride")
    parser.add_argument("-p1", "--selection1", type=str, required=True, help="Atom selection for reference point O")
    parser.add_argument("-p2", "--selection2", type=str, required=True, help="Atom selection for reference point P")
    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    np_dir = os.path.abspath(ns.numpys_directory)
    logger.info("Calculating rotation vectors/protein overlay arrays")
    u = mda.Universe(ns.structure, ns.trajectory)
    get_rotation_and_protein(
        out_dir=np_dir,
        u=u,
        From=ns.From,
        Until=ns.Until,
        Step=ns.Step,
        sele1=ns.selection1,
        sele2=ns.selection2,
    )

    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None
    draw(
        Dir=np_dir,
        mode=ns.mode,
        minmax=minmax,
        filename=ns.outfile,
        show_vectors=ns.vectors,
    )


# Alias matching the newer script style.
def plot(args: List[str]) -> None:
    rot_plot(args)


if __name__ == "__main__":
    rot_plot(None)
