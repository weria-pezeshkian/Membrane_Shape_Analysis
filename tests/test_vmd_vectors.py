"""Tests for vmd_vectors's static/dynamic principal-direction TCL scripts:
arrow endpoints, --which/--layer filtering, --Remove-TMD hole exclusion,
per-frame (not averaged) directions in the dynamic script, --scale, and the
k1/k2 families that continue across neighbors and frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CALM.core.fourier_sft import SFT
from CALM.utilize.vmd_vectors import (
    build_dynamic_vectors_tcl,
    build_static_vectors_tcl,
    family_swap_mask,
    vmd_vectors,
)


def _make_q_mn(Lx: float, Ly: float, Nx: int, Ny: int, theta: float) -> np.ndarray:
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.arange(M)
    n = np.arange(N)
    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)
    qx_grid, qy_grid = np.meshgrid(2 * np.pi * m / Lx, 2 * np.pi * n / Ly, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * qx_grid - s * qy_grid, s * qx_grid + c * qy_grid], axis=0)


def _make_sft(thetas: list[float], Lx: float = 100.0, Ly: float = 80.0, Nx: int = 3, Ny: int = 3) -> SFT:
    n_frames = len(thetas)
    rng = np.random.default_rng(0)
    s = SFT()
    s.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    s.q_mn = np.stack([_make_q_mn(Lx, Ly, Nx, Ny, t) for t in thetas]).astype(np.float32)
    s.frame_indices = np.arange(n_frames)
    s.dimensions = np.tile([Lx, Ly, 60.0], (n_frames, 1))
    return s


def _n_arrow_calls(content: str) -> int:
    """Count of calm_draw_arrow call sites (not its proc definition, which also contains the name)."""
    return sum(1 for line in content.splitlines() if line.startswith("calm_draw_arrow "))


def test_vmd_vectors_raises_without_principal_dirs(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _make_sft([0.0], Lx=100.0, Ly=100.0).write(in_dir)

    with pytest.raises(FileNotFoundError, match="principal_dirs"):
        vmd_vectors(["-i", str(in_dir), "-o", str(out_dir)])


def test_build_static_vectors_tcl_known_point_endpoint(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 1
    dirs = np.zeros((6, gridsize, gridsize, 3))
    dirs[0, 0, 0] = [1.0, 0.0, 0.0]  # upper k1
    np.save(in_dir / "0_principal_dirs.npy", dirs)

    z = np.zeros((3, gridsize, gridsize))
    z[0, 0, 0] = 5.0  # upper layer height in nm -> *10 = 50 Angstrom
    np.save(in_dir / "0_Z_fitted.npy", z)
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    out_path = tmp_path / "static.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, scale=1.0)

    content = out_path.read_text()
    # (x, y) = (0, 0); z = 50.0; direction (1, 0, 0); fixed base length 15 Angstrom at scale=1.
    assert "calm_draw_arrow {0.0000 0.0000 50.0000} {15.0000 0.0000 50.0000} red" in content


def test_static_vectors_which_and_layer_filter_arrow_count(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 2
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(6, gridsize, gridsize, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    np.save(in_dir / "0_principal_dirs.npy", dirs)
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    out_both = tmp_path / "both.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_both), which="both", layers=["upper", "lower", "middle"], step=1)
    out_k1_upper = tmp_path / "k1_upper.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_k1_upper), which="k1", layers=["upper"], step=1)

    assert _n_arrow_calls(out_both.read_text()) == gridsize * gridsize * 2 * 3  # k1+k2, 3 layers
    assert _n_arrow_calls(out_k1_upper.read_text()) == gridsize * gridsize  # k1 only, upper only


def test_static_vectors_skips_hole_flagged_points(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 2
    dirs = np.zeros((6, gridsize, gridsize, 3))
    dirs[..., 0] = 1.0  # every direction points along +x
    np.save(in_dir / "0_principal_dirs.npy", dirs)
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))

    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)
    hole = np.zeros((1, 2, gridsize, gridsize), dtype=bool)
    hole[0, 0, 0, 0] = True  # upper hole at (0, 0)
    sft.hole_mask = hole

    out_path = tmp_path / "static.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1)

    assert _n_arrow_calls(out_path.read_text()) == gridsize * gridsize - 1


def test_dynamic_vectors_tcl_uses_each_frames_own_directions(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 1
    dirs0 = np.zeros((6, gridsize, gridsize, 3))
    dirs0[0, 0, 0] = [1.0, 0.0, 0.0]
    dirs1 = np.zeros((6, gridsize, gridsize, 3))
    dirs1[0, 0, 0] = [0.0, 1.0, 0.0]
    np.save(in_dir / "0_principal_dirs.npy", dirs0)
    np.save(in_dir / "1_principal_dirs.npy", dirs1)
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    np.save(in_dir / "1_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    sft = _make_sft([0.0, 0.0], Lx=100.0, Ly=100.0)

    out_path = tmp_path / "dynamic.tcl"
    build_dynamic_vectors_tcl(sft, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, scale=1.0)

    content = out_path.read_text()
    assert "trace add variable vmd_frame($calm_molid) write calm_redraw_vectors" in content
    assert "{0.0000 0.0000 0.0000 15.0000 0.0000 0.0000 red}" in content  # frame 0: +x
    assert "{0.0000 0.0000 0.0000 0.0000 15.0000 0.0000 red}" in content  # frame 1: +y


def test_dynamic_length_scales_endpoint_with_curvature(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 1
    dirs = np.zeros((6, gridsize, gridsize, 3))
    dirs[0, 0, 0] = [1.0, 0.0, 0.0]
    np.save(in_dir / "0_principal_dirs.npy", dirs)
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    curv = np.zeros((6, gridsize, gridsize))
    curv[0, 0, 0] = 2.0  # k1 = 2.0 nm^-1 at this point
    np.save(in_dir / "0_principal_curvatures.npy", curv)
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    out_fixed = tmp_path / "fixed.tcl"
    out_dyn = tmp_path / "dyn.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_fixed), which="k1", layers=["upper"], step=1, scale=1.0)
    build_static_vectors_tcl(
        sft, str(in_dir), str(out_dyn), which="k1", layers=["upper"], step=1, dynamic_length=True, scale=1.0
    )

    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {15.0000 0.0000 0.0000} red" in out_fixed.read_text()
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {20.0000 0.0000 0.0000} red" in out_dyn.read_text()  # 10 * |2.0|


def test_scale_multiplies_arrow_length_and_defaults_to_ten(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    gridsize = 1
    dirs = np.zeros((6, gridsize, gridsize, 3))
    dirs[0, 0, 0] = [1.0, 0.0, 0.0]
    np.save(in_dir / "0_principal_dirs.npy", dirs)
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    out_default = tmp_path / "default.tcl"
    out_explicit = tmp_path / "explicit.tcl"
    build_static_vectors_tcl(sft, str(in_dir), str(out_default), which="k1", layers=["upper"], step=1)
    build_static_vectors_tcl(
        sft, str(in_dir), str(out_explicit), which="k1", layers=["upper"], step=1, scale=2.0
    )

    # Default scale is 10: base fixed length 15 Angstrom -> 150.
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {150.0000 0.0000 0.0000} red" in out_default.read_text()
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {30.0000 0.0000 0.0000} red" in out_explicit.read_text()


def _stripe_dirs(gridsize: int, swapped_columns: slice) -> np.ndarray:
    """A 6-slice principal_dirs stack whose stored k1 points along +x and k2 along +y, except
    in `swapped_columns`, where the stored labels are the other way round."""
    dirs = np.zeros((6, gridsize, gridsize, 3))
    dirs[0::2, ..., 0] = 1.0
    dirs[1::2, ..., 1] = 1.0
    dirs[0::2, :, swapped_columns] = [0.0, 1.0, 0.0]
    dirs[1::2, :, swapped_columns] = [1.0, 0.0, 0.0]
    return dirs


def test_family_swap_mask_swaps_the_smaller_stripe() -> None:
    dirs = _stripe_dirs(4, slice(3, 4))
    swap = family_swap_mask(dirs[0], dirs[1])
    assert swap[:, 3].all()
    assert not swap[:, :3].any()


def test_family_swap_mask_never_swaps_nan_points() -> None:
    dirs = _stripe_dirs(4, slice(3, 4))
    dirs[:, 0, 3] = np.nan
    swap = family_swap_mask(dirs[0], dirs[1])
    assert not swap[0, 3]
    assert swap[1:, 3].all()


def test_static_red_arrows_keep_one_axis_and_carry_their_own_curvature(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    gridsize = 4
    np.save(in_dir / "0_principal_dirs.npy", _stripe_dirs(gridsize, slice(3, 4)))
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    curv = np.zeros((6, gridsize, gridsize))
    curv[0::2] = 2.0  # stored k1
    curv[1::2] = 1.0  # stored k2
    curv[0::2, :, 3] = 3.0  # stored k1 of the swapped column, along y
    curv[1::2, :, 3] = 0.5  # stored k2 of the swapped column, along x
    np.save(in_dir / "0_principal_curvatures.npy", curv)
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    out_path = tmp_path / "static.tcl"
    build_static_vectors_tcl(
        sft, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, dynamic_length=True, scale=1.0
    )

    content = out_path.read_text()
    # Column 3 sits at x = 75: its red arrow runs along x with the stored k2 of 0.5 (length 5).
    assert "calm_draw_arrow {75.0000 0.0000 0.0000} {80.0000 0.0000 0.0000} red" in content
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {20.0000 0.0000 0.0000} red" in content
    assert " {0.0000 15.0000 0.0000} red" not in content


def test_dynamic_red_arrows_keep_the_previous_frames_axis(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    gridsize = 2
    np.save(in_dir / "0_principal_dirs.npy", _stripe_dirs(gridsize, slice(0, 0)))
    np.save(in_dir / "1_principal_dirs.npy", _stripe_dirs(gridsize, slice(0, gridsize)))
    np.save(in_dir / "0_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    np.save(in_dir / "1_Z_fitted.npy", np.zeros((3, gridsize, gridsize)))
    sft = _make_sft([0.0, 0.0], Lx=100.0, Ly=100.0)

    out_path = tmp_path / "dynamic.tcl"
    build_dynamic_vectors_tcl(sft, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, scale=1.0)

    frames = [line for line in out_path.read_text().splitlines() if line.startswith("    { {")]
    assert len(frames) == 2
    for frame in frames:
        assert "{0.0000 0.0000 0.0000 15.0000 0.0000 0.0000 red}" in frame
