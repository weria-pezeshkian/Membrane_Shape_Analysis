from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq


def shape_operator_curvatures(
    surface: Any, X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curvature of `surface` at grid points (X, Y) via the shape operator.

    `surface` must expose Zx, Zy, Zxx, Zyy, Zxy (e.g. Fourier_Series_Function).

    Returns:
        H: mean curvature
        K: gaussian curvature
        k1, k2: principal curvatures, ordered k1 >= k2
        dirs1, dirs2: principal directions as unit 3D tangent vectors on the surface
    """
    fx = surface.Zx(X, Y)
    fy = surface.Zy(X, Y)
    fxx = surface.Zxx(X, Y)
    fyy = surface.Zyy(X, Y)
    fxy = surface.Zxy(X, Y)

    # First fundamental form
    E = 1 + fx**2
    F = fx * fy
    G = 1 + fy**2

    # Second fundamental form
    L = fxx / np.sqrt(1 + fx**2 + fy**2)
    M = fxy / np.sqrt(1 + fx**2 + fy**2)
    N = fyy / np.sqrt(1 + fx**2 + fy**2)

    # Shape operator, S = (first fundamental form)^-1 (second fundamental form)
    det = E * G - F**2
    S11 = (G * L - F * M) / det
    S12 = (G * M - F * N) / det
    S21 = (-F * L + E * M) / det
    S22 = (-F * M + E * N) / det

    k1 = np.zeros_like(X)
    k2 = np.zeros_like(X)
    dirs1 = np.zeros(X.shape + (3,))
    dirs2 = np.zeros(X.shape + (3,))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            S = np.array([[S11[i, j], S12[i, j]],
                          [S21[i, j], S22[i, j]]])
            vals, vecs = np.linalg.eigh(S)
            k1[i, j], k2[i, j] = vals[1], vals[0]
            dx1, dy1 = vecs[:, 1]
            dx2, dy2 = vecs[:, 0]
            dz1 = fx[i, j] * dx1 + fy[i, j] * dy1
            dz2 = fx[i, j] * dx2 + fy[i, j] * dy2
            dirs1[i, j, :] = np.array([dx1, dy1, dz1]) / np.linalg.norm([dx1, dy1, dz1])
            dirs2[i, j, :] = np.array([dx2, dy2, dz2]) / np.linalg.norm([dx2, dy2, dz2])

    H = 0.5 * (k1 + k2)
    K = k1 * k2

    return H, K, k1, k2, dirs1, dirs2


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


def surface_normal(surface: Any, x: float, y: float) -> np.ndarray:
    """Unit normal of `surface` (e.g. a Fourier_Series_Function) at (x, y), from its own analytic
    gradient: (-Zx, -Zy, 1), normalized - the same sign convention analyze.py's thickness
    calculation already uses for the middle surface, evaluated here directly from `surface`'s own
    Zx/Zy rather than a discrete grid gradient, and for any surface (upper or lower), not just the
    middle."""
    zx = float(surface.Zx(np.asarray(x), np.asarray(y)))
    zy = float(surface.Zy(np.asarray(x), np.asarray(y)))
    n = np.array([-zx, -zy, 1.0])
    return n / np.linalg.norm(n)


def _surface_root(
    t: float, surface: Any,
    mx: float, my: float, mz: float,
    nx: float, ny: float, nz: float,
    Lx: float, Ly: float,
) -> float:
    """Root function for brentq: signed distance along a normal ray between a query point and
    `surface`'s own analytic height - the same idea as `f` above, but evaluating `surface.Z`
    directly (e.g. a Fourier_Series_Function) rather than through a RectBivariateSpline, so no
    separate discretize-then-interpolate step is needed."""
    xq = np.mod(mx + t * nx, Lx)
    yq = np.mod(my + t * ny, Ly)
    zq = mz + t * nz
    return float(zq - surface.Z(np.asarray(xq), np.asarray(yq)))


def nearest_surface_intersection(
    surface: Any,
    mx: float, my: float, mz: float,
    nx: float, ny: float, nz: float,
    Lx: float, Ly: float,
    t_max_base: float,
) -> float | None:
    """Signed distance t along the normal ray (mx,my,mz)+t*(nx,ny,nz) to `surface`'s own height -
    whichever of the two directions (t>0, t<0) crosses closer to the query point - widening the
    search bracket up to 3 times (matching `_thickness_root`'s own widening) if neither direction
    brackets a root at the current width.

    Unlike `_thickness_root` (which always starts on the middle surface, so it already knows
    upper lies in the t>0 direction and lower in t<0), the query point here can be an arbitrary
    point - e.g. a protein atom that might sit on either side of `surface`, at any distance from
    it - so both directions are tried and the smaller-magnitude root wins.

    Unlike `_thickness_root`, a root found only via widening is NOT rejected for being far from
    the query point: how far the query point is from `surface` is meaningless on its own here
    (the caller decides what, if anything, that distance should be used for) - this only ever
    reports whether a crossing exists at all, and where. `t_max_base` only sets the starting
    search width; widening exists purely to help `brentq` bracket a genuine root, not to bound
    how far away an accepted one may be.

    Returns None only if every widened bracket, in both directions, still fails to bracket any
    root at all.
    """
    t_max = t_max_base
    for _ in range(4):
        candidates = []
        for lo, hi in ((0.0, t_max), (-t_max, 0.0)):
            try:
                candidates.append(brentq(_surface_root, lo, hi, args=(surface, mx, my, mz, nx, ny, nz, Lx, Ly)))
            except ValueError:
                continue
        if candidates:
            return min(candidates, key=abs)
        t_max *= 2
    return None


def _thickness_root(
    interp: RectBivariateSpline,
    mx: float, my: float, mz: float,
    nx: float, ny: float, nz: float,
    Lx: float, Ly: float,
    t_max_base: float,
    upper: bool,
) -> float | None:
    """Root of `f` along the local normal ray, widening the search bracket up to 3 times if the previous one fails to bracket a root.

    A root is only accepted if it falls within the original, un-widened
    `t_max_base` of the query point, so a genuine nearby surface
    intersection is comfortably inside it. The wider brackets (2x, 4x, 8x)
    exist only to help `brentq` find a sign change to bracket; a root that
    only exists that much farther out most likely crosses an unrelated,
    distant part of the periodic surface rather than the true nearby one,
    so it's treated the same as a bracket that never found a root at all.

    Returns None if every widened bracket (t_max_base, 2x, 4x, 8x) still
    fails to bracket a root, or if the only root found lies beyond
    `t_max_base` itself.
    """
    t_max = t_max_base
    for _ in range(4):
        try:
            if upper:
                root = brentq(f, 0.0, t_max, args=(interp, mx, my, mz, nx, ny, nz, Lx, Ly))
            else:
                root = brentq(f, -t_max, 0.0, args=(interp, mx, my, mz, nx, ny, nz, Lx, Ly))
            return root if abs(root) <= t_max_base else None
        except ValueError:
            t_max *= 2
    return None


if __name__ == "__main__":
    pass
