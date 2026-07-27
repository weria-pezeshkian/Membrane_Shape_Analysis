from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def shape_operator_curvatures(
    surface: Any, X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curvature of `surface` at grid points (X, Y) via the shape operator.

    `surface` must expose Zx, Zy, Zxx, Zyy, Zxy (e.g. Fourier_Series_Function).

    Returns:
        H: mean curvature
        K: gaussian curvature
        k1, k2: principal curvatures, ordered k1 >= k2
        dirs1, dirs2: principal directions as 2D vectors in the tangent plane
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
    dirs1 = np.zeros(X.shape + (2,))
    dirs2 = np.zeros(X.shape + (2,))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            S = np.array([[S11[i, j], S12[i, j]],
                          [S21[i, j], S22[i, j]]])
            vals, vecs = np.linalg.eigh(S)
            k1[i, j], k2[i, j] = vals[1], vals[0]
            dirs1[i, j, :] = vecs[:, 1]
            dirs2[i, j, :] = vecs[:, 0]

    H = 0.5 * (k1 + k2)
    K = k1 * k2

    return H, K, k1, k2, dirs1, dirs2


if __name__ == "__main__":
    pass
