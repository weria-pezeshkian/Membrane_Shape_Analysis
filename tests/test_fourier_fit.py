"""Tests for fit_coefficients (extracted from Fourier_Series_Function.Fit)."""

from __future__ import annotations

import numpy as np
import pytest

from CALM.core.fourier_core import Fourier_Series_Function
from CALM.core.fourier_fit import fit_coefficients


def test_fit_recovers_known_coefficients() -> None:
    # regularize=False: sanity check of the basis/solve math against
    # noiseless data, unrelated to the Tikhonov regularization feature.
    Lx, Ly, Nx, Ny = 100.0, 80.0, 2, 2
    rng = np.random.default_rng(42)
    truth = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    truth.setAnm(rng.uniform(-1.0, 1.0, size=truth.Anm.shape))

    n_points = 400
    x = rng.uniform(0, Lx, n_points)
    y = rng.uniform(0, Ly, n_points)
    z = truth.Z(x, y)

    Anm = fit_coefficients(np.vstack([x, y, z]), Lx, Ly, Nx, Ny, regularize=False)

    assert np.allclose(Anm, truth.Anm, atol=1e-6)


def test_fit_rejects_wrong_shaped_input() -> None:
    with pytest.raises(ValueError):
        fit_coefficients(np.zeros((2, 5)), Lx=10.0, Ly=10.0, Nx=1, Ny=1)


# ---- Tikhonov (|q|^2-weighted) regularization ----

def test_regularization_negligible_when_well_determined() -> None:
    # M >> Length (5000 atoms for 25 coefficients): the auto-scaled penalty
    # should stay small enough to still recover the true coefficients closely.
    Lx, Ly, Nx, Ny = 100.0, 80.0, 2, 2
    rng = np.random.default_rng(1)
    truth = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    truth.setAnm(rng.uniform(-1.0, 1.0, size=truth.Anm.shape))

    n_points = 5000
    x = rng.uniform(0, Lx, n_points)
    y = rng.uniform(0, Ly, n_points)
    z = truth.Z(x, y)

    Anm_reg = fit_coefficients(np.vstack([x, y, z]), Lx, Ly, Nx, Ny, regularize=True)
    assert np.allclose(Anm_reg, truth.Anm, atol=1e-3)


def test_regularization_bounds_curvature_in_overfit_regime() -> None:
    # Few atoms (M=200) relative to a fine-resolution fit (Length close to
    # M) makes the plain (regularize=False) fit produce unphysically large
    # curvature; regularized should stay bounded for the same noisy data.
    from CALM.core.fourier_core import get_fourier_modes
    from CALM.core.curvature import shape_operator_curvatures

    Lx = Ly = 100.0
    M = 200
    rng = np.random.default_rng(0)
    x = rng.uniform(0, Lx, M)
    y = rng.uniform(0, Ly, M)
    z = 5.0 * np.sin(2 * np.pi * x / Lx) + rng.normal(0, 1.5, M)

    # grid=40: keep in sync with whatever grid the max|H| thresholds below
    # were derived from - the overfit "ringing" is spiky/localized, so the
    # sampled max is sensitive to grid resolution.
    grid = np.linspace(0, Lx, 40, endpoint=False)
    X, Y = np.meshgrid(grid, grid)

    Nx, Ny = get_fourier_modes([Lx, Ly], 1.5, 1.5)  # Length=169, close to M=200

    Anm_plain = fit_coefficients(np.stack([x, y, z]), Lx, Ly, Nx, Ny, regularize=False)
    f_plain = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    f_plain.setAnm(Anm_plain)
    H_plain, *_ = shape_operator_curvatures(f_plain, X, Y)

    Anm_reg = fit_coefficients(np.stack([x, y, z]), Lx, Ly, Nx, Ny, regularize=True)
    f_reg = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    f_reg.setAnm(Anm_reg)
    H_reg, *_ = shape_operator_curvatures(f_reg, X, Y)

    max_plain = np.nanmax(np.abs(H_plain))
    max_reg = np.nanmax(np.abs(H_reg))

    assert max_plain > 1.0  # confirms this is the overfit regime (A^-1)
    assert max_reg < max_plain / 3  # regularization meaningfully tames it
    assert max_reg < 1.0  # and lands back in a physically sane range (A^-1)


def test_regularization_recovers_flat_surface_with_realistic_noise() -> None:
    # A near-flat membrane patch (tiny thermal jitter) should recover close
    # to the constant height with no meaningful spurious curvature, even
    # heavily underdetermined (M=5 atoms, Length=121). At exactly zero data
    # variance (all-identical z), alpha collapses to 0 and no regularization
    # is applied - not a case real trajectory data hits, so realistic noise
    # is used here rather than exact-zero.
    Lx, Ly, Nx, Ny = 100.0, 100.0, 5, 5
    C = 7.3
    x = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    y = np.array([15.0, 35.0, 55.0, 75.0, 95.0])
    rng = np.random.default_rng(2)
    z = C + rng.normal(0, 0.01, 5)  # sub-Angstrom thermal-scale jitter

    Anm = fit_coefficients(np.vstack([x, y, z]), Lx, Ly, Nx, Ny, regularize=True)

    center_i, center_j = Nx, Ny  # index of mode (0,0)
    assert np.isclose(Anm[center_i, center_j], C, atol=0.01)
    off_center = Anm.copy()
    off_center[center_i, center_j] = 0.0
    assert np.max(np.abs(off_center)) < 0.01


# ---- curvature-ballpark warning tier (3x < M/Length < 10x) ----

def test_curvature_ballpark_warning_fires_between_3x_and_10x_oversampling() -> None:
    # M=300, Nx=Ny=3 -> Length=49, M/Length~6.1: passes the 3x numerical
    # check but is still below the stricter 10x ballpark tier.
    Lx = Ly = 100.0
    rng = np.random.default_rng(0)
    x = rng.uniform(0, Lx, 300)
    y = rng.uniform(0, Ly, 300)
    z = 5.0 * np.sin(2 * np.pi * x / Lx) + rng.normal(0, 1.5, 300)

    diagnostics: list = []
    fit_coefficients(np.stack([x, y, z]), Lx, Ly, 3, 3, regularize=False, diagnostics=diagnostics)
    messages = [message for _, message in diagnostics]
    assert any("physically reasonable membrane ballpark" in m for m in messages)
    assert any("6.1x oversampling" in m for m in messages)


def test_curvature_ballpark_warning_silent_when_well_oversampled() -> None:
    # Same data, coarser Nx=Ny=1 -> Length=9, M/Length~33: past both the 3x
    # and 10x tiers, no warning expected.
    Lx = Ly = 100.0
    rng = np.random.default_rng(0)
    x = rng.uniform(0, Lx, 300)
    y = rng.uniform(0, Ly, 300)
    z = 5.0 * np.sin(2 * np.pi * x / Lx) + rng.normal(0, 1.5, 300)

    diagnostics: list = []
    fit_coefficients(np.stack([x, y, z]), Lx, Ly, 1, 1, regularize=False, diagnostics=diagnostics)
    assert diagnostics == []
