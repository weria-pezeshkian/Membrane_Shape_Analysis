"""Tests for shape_operator_curvatures."""

from __future__ import annotations

import numpy as np

from CALM.core.curvature import shape_operator_curvatures
from CALM.core.fourier_core import Fourier_Series_Function


def make_surface(
    Nx: int = 2, Ny: int = 2, Lx: float = 100.0, Ly: float = 80.0, seed: int = 3
) -> Fourier_Series_Function:
    rng = np.random.default_rng(seed)
    f = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    f.setAnm(rng.uniform(-1.0, 1.0, size=f.Anm.shape))
    return f


def test_shape_operator_curvature_self_consistency() -> None:
    f = make_surface()
    x = np.linspace(10, f.Lx - 10, 4)
    y = np.linspace(10, f.Ly - 10, 4)
    X, Y = np.meshgrid(x, y)

    H, K, k1, k2, dirs1, dirs2 = shape_operator_curvatures(f, X, Y)

    assert np.all(np.isfinite(H))
    assert np.allclose(H, 0.5 * (k1 + k2))
    assert np.allclose(K, k1 * k2)
    assert np.all(k1 >= k2 - 1e-12)  # convention: k1 is the larger eigenvalue
    assert dirs1.shape == X.shape + (3,)
    assert dirs2.shape == X.shape + (3,)


def test_shape_operator_curvature_directions_are_unit_tangent_vectors() -> None:
    f = make_surface()
    x = np.linspace(10, f.Lx - 10, 4)
    y = np.linspace(10, f.Ly - 10, 4)
    X, Y = np.meshgrid(x, y)

    _, _, _, _, dirs1, dirs2 = shape_operator_curvatures(f, X, Y)
    fx = f.Zx(X, Y)
    fy = f.Zy(X, Y)

    for dirs in (dirs1, dirs2):
        assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)
        # Tangency: (dx, dy, dz) must lie in the plane spanned by the surface's
        # own local tangent basis (1, 0, fx) and (0, 1, fy), i.e. dz == fx*dx + fy*dy.
        # Linear in (dx, dy, dz), so it holds after the unit-length renormalization too.
        assert np.allclose(dirs[..., 2], fx * dirs[..., 0] + fy * dirs[..., 1])
