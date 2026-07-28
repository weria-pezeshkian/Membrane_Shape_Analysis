from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import MDAnalysis as mda
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from tqdm import tqdm

from ..core.curvature import shape_operator_curvatures
from ..core.fourier_core import Fourier_Series_Function, get_fourier_modes
from ..core.fourier_sft import SFT
from ..core.rotation import recover_rotation_angle, rotated_grid

logger = logging.getLogger(__name__)


def circle_cutter(arr: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """NaN-mask `arr` outside the largest box-centered circle that fits within it.

    Spatial axes are always axes 1 and 2 (`arr.shape[1:3]`); axis 0 is a
    layer/component axis (e.g. upper/lower/middle), and any further
    trailing axes (e.g. principal_dirs' 2 vector components) are untouched.
    """
    Lx, Ly = dimensions[:2]

    if arr.ndim >= 3:
        n, m = arr.shape[1:3]
    else:
        n, m = arr.shape

    x = (np.arange(n) + 0.5) * Lx / n - Lx / 2.0
    y = (np.arange(m) + 0.5) * Ly / m - Ly / 2.0

    X, Y = np.meshgrid(x, y, indexing="ij")

    radius = min(Lx, Ly) / 2.0
    mask = X**2 + Y**2 <= radius**2

    if arr.ndim >= 3:
        arr[:, ~mask] = np.nan
    else:
        arr[~mask] = np.nan

    return arr


def periodic_gradient(
    Z: np.ndarray, dx: float, dy: float, periodic_x: bool = True, periodic_y: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dZ/dy, dZ/dx) of grid Z (axis 0 = y, axis 1 = x), central-differenced with wraparound where periodic."""
    if periodic_x:
        dz_dx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * dx)
    else:
        dz_dx = np.gradient(Z, dx, axis=1)

    if periodic_y:
        dz_dy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * dy)
    else:
        dz_dy = np.gradient(Z, dy, axis=0)

    return dz_dy, dz_dx


def f(
    t: float,
    interp: RectBivariateSpline,
    mx: float, my: float, mz: float,
    nx: float, ny: float, nz: float,
    Lx: float, Ly: float,
) -> float:
    """Root function for brentq: signed distance along a normal ray between
    a query point and the interpolated surface."""
    xq = mx + t * nx
    yq = my + t * ny
    zq = mz + t * nz

    xq = np.mod(xq, Lx)
    yq = np.mod(yq, Ly)

    return zq - interp(yq, xq, grid=False)[()]


def _rotate_direction_vectors(vecs: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an array of 2D tangent-plane direction vectors (shape (..., 2)) by `angle`."""
    vx = vecs[..., 0]
    vy = vecs[..., 1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.stack([cos_a * vx - sin_a * vy, sin_a * vx + cos_a * vy], axis=-1)


def analysis(
    universe: mda.Universe | None,
    sft: SFT,
    methods: Sequence[str],
    args: argparse.Namespace,
) -> None:
    """Compute the requested `methods` for every frame in `sft` and save each to `args.out`."""
    # sft is always fully populated here: built fresh by build_sft or loaded
    # by SFT.from_directory, both of which set every one of these fields.
    assert sft.frame_indices is not None
    assert sft.dimensions is not None
    assert sft.A_mn is not None
    assert sft.q_mn is not None
    rotated = args.rotate

    for i, frame in tqdm(enumerate(sft.frame_indices), total=len(sft.frame_indices)):
        if universe is not None:
            try:
                universe.trajectory[frame]
            except IndexError:
                universe.trajectory[i]
            dimensions = universe.dimensions[:3]
        else:
            # No trajectory (SFT was loaded via --sft): use the box size
            # captured per-frame at build time instead.
            dimensions = sft.dimensions[i]
        x = np.linspace(0, dimensions[:3][0], args.gridsize, endpoint=False)
        y = np.linspace(0, dimensions[:3][1], args.gridsize, endpoint=False)
        X, Y = np.meshgrid(x, y)

        num_digits = len(str(max(sft.frame_indices)))
        Nx, Ny = get_fourier_modes(dimensions[:3], args.lambda_x, args.lambda_y)
        fourier0 = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
        fourier0.setAnm(sft.A_mn[i, 0])
        fourier1 = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
        fourier1.setAnm(sft.A_mn[i, 1])
        fourier2 = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
        fourier2.setAnm(sft.A_mn[i, 2])

        # Curvature/height are rotation-invariant scalars: instead of
        # rotating Anm (would need refitting), evaluate the as-fit surface
        # at the corresponding unrotated coordinate for each output grid
        # point. Only direction vectors (dirs1/dirs2 below) need rotating
        # afterward.
        if rotated:
            theta = recover_rotation_angle(sft.q_mn[i], dimensions[:3][0], dimensions[:3][1], Nx, Ny)
            X_eval, Y_eval = rotated_grid(X, Y, dimensions[:3][0] / 2.0, dimensions[:3][1] / 2.0, theta)
        else:
            theta = 0.0
            X_eval, Y_eval = X, Y

        if "Z_fitted" in methods or "thickness" in methods:
            Z_fitted_1 = fourier0.Z(X_eval, Y_eval)
            Z_fitted_2 = fourier1.Z(X_eval, Y_eval)
            Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2
            Z_fitted = np.stack([Z_fitted_1, Z_fitted_2, Z_fitted_vmd], axis=0)

            if rotated:
                Z_fitted = circle_cutter(Z_fitted, dimensions)

            if "Z_fitted" in methods:
                np.save(f"{args.out}/{frame:0{num_digits}d}_Z_fitted.npy", Z_fitted / 10)

            if "thickness" in methods:
                try:
                    interp_upper = RectBivariateSpline(Y[:, 0], X[0, :], Z_fitted_1)
                    interp_lower = RectBivariateSpline(Y[:, 0], X[0, :], Z_fitted_2)

                    dx = dimensions[:3][0] / (X.shape[1] - 1)
                    dy = dimensions[:3][1] / (Y.shape[0] - 1)
                    dz_dy, dz_dx = periodic_gradient(Z_fitted_vmd, dx, dy, periodic_x=True, periodic_y=True)

                    Nx_arr, Ny_arr = -dz_dx, -dz_dy
                    Nz_arr = np.ones_like(Z_fitted_vmd)
                    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)
                    N /= np.linalg.norm(N, axis=-1, keepdims=True)

                    thickness_map = np.full_like(Z_fitted_vmd, np.nan, dtype=float)

                    t_max = np.nanmax(np.abs(Z_fitted_1 - Z_fitted_2)) * 2
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            x0, y0, z0 = X[i, j], Y[i, j], Z_fitted_vmd[i, j]
                            nvecx, nvecy, nvecz = N[i, j]

                            brentq_args = (x0, y0, z0, nvecx, nvecy, nvecz, dimensions[:3][0], dimensions[:3][1])
                            l1 = brentq(f, 0.0, t_max, args=(interp_upper, *brentq_args))
                            l2 = brentq(f, -t_max, 0.0, args=(interp_lower, *brentq_args))

                            thickness_map[i, j] = l1 - l2

                    if rotated:
                        thickness_map = circle_cutter(thickness_map, dimensions)

                    np.save(f"{args.out}/{frame:0{num_digits}d}_thickness.npy", thickness_map / 10)
                except ValueError:
                    logger.warning(
                        f"frame {frame}: thickness could not be calculated - this can "
                        "indicate the curvature is too high or lambda_x/lambda_y are too small."
                    )

        if any(m in methods for m in ("mean", "gaussian", "principal", "principal_directions")):
            # H, K, k1, k2 are rotation-invariant scalars: evaluating at
            # (X_eval, Y_eval) directly gives the rotated surface's value at
            # output position (X, Y). dirs1/dirs2 are direction vectors and
            # get an extra +theta rotation to express them in the
            # rotated/aligned frame's basis.
            H1, K1, k1_1, k2_1, dirs1_1, dirs2_1 = shape_operator_curvatures(fourier0, X_eval, Y_eval)
            H2, K2, k1_2, k2_2, dirs1_2, dirs2_2 = shape_operator_curvatures(fourier1, X_eval, Y_eval)
            Hmid, Kmid, k1_mid, k2_mid, dirs1_mid, dirs2_mid = shape_operator_curvatures(fourier2, X_eval, Y_eval)

            if theta != 0.0:
                dirs1_1 = _rotate_direction_vectors(dirs1_1, theta)
                dirs2_1 = _rotate_direction_vectors(dirs2_1, theta)
                dirs1_2 = _rotate_direction_vectors(dirs1_2, theta)
                dirs2_2 = _rotate_direction_vectors(dirs2_2, theta)
                dirs1_mid = _rotate_direction_vectors(dirs1_mid, theta)
                dirs2_mid = _rotate_direction_vectors(dirs2_mid, theta)

            if "mean" in methods:
                mean_curvature = np.stack([H1, H2, Hmid], axis=0) * 10
                if rotated:
                    mean_curvature = circle_cutter(mean_curvature, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_mean_curvature.npy", mean_curvature)

            if "gaussian" in methods:
                gaussian_curvature = np.stack([K1, K2, Kmid], axis=0) * 10
                if rotated:
                    gaussian_curvature = circle_cutter(gaussian_curvature, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_gaussian_curvature.npy", gaussian_curvature)

            if "principal" in methods:
                principal_curvatures = np.stack([k1_1, k2_1, k1_2, k2_2, k1_mid, k2_mid], axis=0) * 10
                if rotated:
                    principal_curvatures = circle_cutter(principal_curvatures, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_principal_curvatures.npy", principal_curvatures)

            if "principal_directions" in methods:
                principal_dirs = np.stack([dirs1_1, dirs2_1, dirs1_2, dirs2_2, dirs1_mid, dirs2_mid], axis=0)
                if rotated:
                    principal_dirs = circle_cutter(principal_dirs, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_principal_dirs.npy", principal_dirs)


if __name__ == "__main__":
    pass
