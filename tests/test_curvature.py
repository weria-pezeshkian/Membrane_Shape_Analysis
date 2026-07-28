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
    assert dirs1.shape == X.shape + (2,)
    assert dirs2.shape == X.shape + (2,)
