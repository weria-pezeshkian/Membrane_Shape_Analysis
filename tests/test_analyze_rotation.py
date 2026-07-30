"""Integration tests for the rotation path in analyze/analyze.py::analysis().

These build a synthetic SFT (no trajectory needed) with a known rotation
angle baked into q_mn the same way core/fourier_build.py::_rotate_q does,
then check that analysis() recovers that angle and produces a genuinely
rotated (not just relabeled) output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from CALM.analyze.analyze import _rotate_direction_vectors, analysis, circle_cutter
from CALM.core.curvature import shape_operator_curvatures
from CALM.core.fourier_core import Fourier_Series_Function
from CALM.core.fourier_sft import SFT
from CALM.core.rotation import recover_rotation_angle, rotated_grid


def _build_q_mn(Lx: float, Ly: float, Nx: int, Ny: int, theta: float) -> np.ndarray:
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.arange(M)
    n = np.arange(N)
    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)
    qx = 2 * np.pi * m / Lx
    qy = 2 * np.pi * n / Ly
    qx_grid, qy_grid = np.meshgrid(qx, qy, indexing="ij")

    cos_a, sin_a = np.cos(theta), np.sin(theta)
    qx_rot = cos_a * qx_grid - sin_a * qy_grid
    qy_rot = sin_a * qx_grid + cos_a * qy_grid
    return np.stack([qx_rot, qy_rot], axis=0)


def make_args(out: Path, gridsize: int = 20, lambda_x: float = 3.0, lambda_y: float = 2.5, rotate: bool = True) -> Any:
    # lambda_x=3.0/lambda_y=2.5 with Lx=100/Ly=80 (used throughout this file)
    # reproduce Nx=Ny=3 via get_fourier_modes, matching the test surfaces'
    # Nx=Ny=3 below - analysis() recomputes Nx/Ny from lambda_x/lambda_y
    # rather than reading them off sft.A_mn's shape, so these must agree.
    class Args:
        pass
    a = Args()
    a.out = str(out)
    a.gridsize = gridsize
    a.lambda_x = lambda_x
    a.lambda_y = lambda_y
    a.rotate = rotate
    return a


def test_recover_rotation_angle_matches_known_value() -> None:
    Lx, Ly, Nx, Ny, theta = 100.0, 80.0, 3, 3, 0.4
    q_mn_frame = _build_q_mn(Lx, Ly, Nx, Ny, theta)
    recovered = recover_rotation_angle(q_mn_frame, Lx, Ly, Nx, Ny)
    assert np.isclose(recovered, theta)


def test_analysis_rotated_zfitted_matches_manual_grid_rotation(tmp_path: Path) -> None:
    Lx, Ly, Nx, Ny = 100.0, 80.0, 3, 3
    theta = 0.4
    rng = np.random.default_rng(0)

    surface = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    surface.setAnm(rng.uniform(-1, 1, size=surface.Anm.shape))

    sft = SFT()
    sft.A_mn = np.stack([surface.Anm, surface.Anm, surface.Anm])[None, ...]  # 1 frame, 3 layers, identical
    sft.q_mn = _build_q_mn(Lx, Ly, Nx, Ny, theta)[None, ...]
    sft.frame_indices = np.array([0])
    sft.dimensions = np.array([[Lx, Ly, 60.0]])

    args = make_args(tmp_path, gridsize=20, rotate=True)
    analysis(None, sft, ("Z_fitted",), args)

    Z_fitted = np.load(tmp_path / "0_Z_fitted.npy") * 10  # analysis() saves in nm, undo /10

    x = np.linspace(0, Lx, args.gridsize, endpoint=False)
    y = np.linspace(0, Ly, args.gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    X_old, Y_old = rotated_grid(X, Y, Lx / 2.0, Ly / 2.0, theta)
    expected = surface.Z(X_old, Y_old)

    valid = ~np.isnan(Z_fitted[0])
    assert valid.any()
    assert np.allclose(Z_fitted[0][valid], expected[valid], atol=1e-4)


def test_analysis_rotation_curvature_and_directions_match_manual_reference(tmp_path: Path) -> None:
    Lx, Ly, Nx, Ny = 100.0, 80.0, 3, 3
    theta = 0.6
    gridsize = 12
    rng = np.random.default_rng(2)

    surface = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    surface.setAnm(rng.uniform(-1, 1, size=surface.Anm.shape))

    sft = SFT()
    sft.A_mn = np.stack([surface.Anm, surface.Anm, surface.Anm])[None, ...]
    sft.q_mn = _build_q_mn(Lx, Ly, Nx, Ny, theta)[None, ...]
    sft.frame_indices = np.array([0])
    sft.dimensions = np.array([[Lx, Ly, 60.0]])

    methods = ("mean", "principal_directions")
    args = make_args(tmp_path, gridsize=gridsize, rotate=True)
    analysis(None, sft, methods, args)

    mean_out = np.load(tmp_path / "0_mean_curvature.npy") / 10  # undo analysis()'s *10
    dirs_out = np.load(tmp_path / "0_principal_dirs.npy")

    # Manual reference: H/K are rotation-invariant scalars, so evaluating
    # shape_operator_curvatures directly at the transformed (old-frame) grid
    # must match analysis()'s output exactly at every point. rotate=True also
    # means analysis() applies circle_cutter to mean/principal_dirs, so the
    # reference gets the same masking for a like-for-like comparison.
    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    X_old, Y_old = rotated_grid(X, Y, Lx / 2.0, Ly / 2.0, theta)
    H_expected, K_expected, k1_expected, k2_expected, dirs1_expected, dirs2_expected = (
        shape_operator_curvatures(surface, X_old, Y_old)
    )
    dimensions = np.array([Lx, Ly, 60.0])
    H_expected_masked = circle_cutter(np.stack([H_expected]), dimensions)[0]

    valid = ~np.isnan(H_expected_masked)
    assert valid.any()
    assert np.array_equal(np.isnan(mean_out[0]), np.isnan(H_expected_masked))  # same mask
    assert np.allclose(mean_out[0][valid], H_expected_masked[valid], atol=1e-6)  # upper leaflet
    assert np.allclose(mean_out[2][valid], H_expected_masked[valid], atol=1e-6)  # middle surface (identical Anm here)

    # dirs1 (upper leaflet, index 0 in the stack) must match the manually
    # rotated direction vectors (also circle_cutter-masked), not the
    # unrotated ones.
    dirs1_rotated_expected = _rotate_direction_vectors(dirs1_expected, theta)
    dirs1_rotated_expected_masked = circle_cutter(np.stack([dirs1_rotated_expected]), dimensions)[0]
    assert np.allclose(dirs_out[0][valid], dirs1_rotated_expected_masked[valid], atol=1e-6)
    assert not np.allclose(dirs_out[0][valid], dirs1_expected[valid], atol=1e-3)
