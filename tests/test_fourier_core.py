"""Characterization tests for Fourier_Series_Function (representation + evaluation only)."""

from __future__ import annotations

import numpy as np
import pytest

from CALM.core.fourier_core import Fourier_Series_Function, average_coefficients, get_fourier_modes


def make_surface(Nx: int = 2, Ny: int = 2, Lx: float = 100.0, Ly: float = 80.0, seed: int = 0) -> Fourier_Series_Function:
    rng = np.random.default_rng(seed)
    f = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    f.setAnm(rng.uniform(-1.0, 1.0, size=f.Anm.shape))
    return f


def test_zero_coefficients_give_zero_surface() -> None:
    f = Fourier_Series_Function(Lx=50.0, Ly=50.0, Nx=2, Ny=2)
    x = np.linspace(0, 50, 7)
    y = np.linspace(0, 50, 7)
    X, Y = np.meshgrid(x, y)
    assert np.allclose(f.Z(X, Y), 0.0)


def test_constant_mode_is_constant() -> None:
    f = Fourier_Series_Function(Lx=50.0, Ly=50.0, Nx=1, Ny=1)
    f.Anm[f.Nx, f.Ny] = 3.5  # i=j=0 term: cos(0)+sin(0) == 1
    x = np.linspace(0, 50, 5)
    y = np.linspace(0, 50, 5)
    X, Y = np.meshgrid(x, y)
    assert np.allclose(f.Z(X, Y), 3.5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_derivatives_match_finite_difference(seed: int) -> None:
    f = make_surface(seed=seed)
    rng = np.random.default_rng(seed + 100)
    x = rng.uniform(5, f.Lx - 5, size=6)
    y = rng.uniform(5, f.Ly - 5, size=6)
    h = 1e-4

    zx_fd = (f.Z(x + h, y) - f.Z(x - h, y)) / (2 * h)
    zy_fd = (f.Z(x, y + h) - f.Z(x, y - h)) / (2 * h)

    assert np.allclose(f.Zx(x, y), zx_fd, atol=1e-4, rtol=1e-4)
    assert np.allclose(f.Zy(x, y), zy_fd, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_second_derivatives_match_finite_difference(seed: int) -> None:
    f = make_surface(seed=seed)
    rng = np.random.default_rng(seed + 200)
    x = rng.uniform(5, f.Lx - 5, size=6)
    y = rng.uniform(5, f.Ly - 5, size=6)
    h = 1e-4

    zxx_fd = (f.Zx(x + h, y) - f.Zx(x - h, y)) / (2 * h)
    zyy_fd = (f.Zy(x, y + h) - f.Zy(x, y - h)) / (2 * h)
    zxy_fd = (f.Zx(x, y + h) - f.Zx(x, y - h)) / (2 * h)

    assert np.allclose(f.Zxx(x, y), zxx_fd, atol=1e-3, rtol=1e-3)
    assert np.allclose(f.Zyy(x, y), zyy_fd, atol=1e-3, rtol=1e-3)
    assert np.allclose(f.Zxy(x, y), zxy_fd, atol=1e-3, rtol=1e-3)


def test_average_coefficients() -> None:
    a = np.ones((5, 5))
    b = np.zeros((5, 5))
    assert np.allclose(average_coefficients(a, b), 0.5)


def test_get_fourier_modes_records_diagnostic_when_lambda_too_large() -> None:
    diagnostics: list = []
    Nx, Ny = get_fourier_modes([50.0, 50.0], lambda_x=100.0, lambda_y=100.0, diagnostics=diagnostics)
    assert (Nx, Ny) == (1, 1)
    levels = {level for level, _ in diagnostics}
    assert levels == {"warning"}
    assert any("Nx" in message for _, message in diagnostics)
    assert any("Ny" in message for _, message in diagnostics)


def test_get_fourier_modes_silent_when_lambda_fits() -> None:
    diagnostics: list = []
    Nx, Ny = get_fourier_modes([500.0, 500.0], lambda_x=10.0, lambda_y=10.0, diagnostics=diagnostics)
    assert Nx > 1 and Ny > 1
    assert diagnostics == []
