"""Tests for vmd_vectors's static/dynamic principal-direction TCL scripts:
arrow endpoints, --which/--layer filtering, --Remove-TMD hole exclusion,
per-frame (not averaged) directions in the dynamic script, and --scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CALM.core.fourier_sft import SFT
from CALM.utilize.vmd_vectors import build_dynamic_vectors_tcl, build_static_vectors_tcl, vmd_vectors


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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

    out_path = tmp_path / "static.tcl"
    build_static_vectors_tcl(None, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, scale=1.0)

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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

    out_both = tmp_path / "both.tcl"
    build_static_vectors_tcl(None, str(in_dir), str(out_both), which="both", layers=["upper", "lower", "middle"], step=1)
    out_k1_upper = tmp_path / "k1_upper.tcl"
    build_static_vectors_tcl(None, str(in_dir), str(out_k1_upper), which="k1", layers=["upper"], step=1)

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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n1,100.0,100.0,60.0\n")

    out_path = tmp_path / "dynamic.tcl"
    build_dynamic_vectors_tcl(None, str(in_dir), str(out_path), which="k1", layers=["upper"], step=1, scale=1.0)

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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

    out_fixed = tmp_path / "fixed.tcl"
    out_dyn = tmp_path / "dyn.tcl"
    build_static_vectors_tcl(None, str(in_dir), str(out_fixed), which="k1", layers=["upper"], step=1, scale=1.0)
    build_static_vectors_tcl(
        None, str(in_dir), str(out_dyn), which="k1", layers=["upper"], step=1, dynamic_length=True, scale=1.0
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
    (in_dir / "dimensions.csv").write_text("#header\n0,100.0,100.0,60.0\n")

    out_default = tmp_path / "default.tcl"
    out_explicit = tmp_path / "explicit.tcl"
    build_static_vectors_tcl(None, str(in_dir), str(out_default), which="k1", layers=["upper"], step=1)
    build_static_vectors_tcl(
        None, str(in_dir), str(out_explicit), which="k1", layers=["upper"], step=1, scale=2.0
    )

    # Default scale is 10: base fixed length 15 Angstrom -> 150.
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {150.0000 0.0000 0.0000} red" in out_default.read_text()
    assert "calm_draw_arrow {0.0000 0.0000 0.0000} {30.0000 0.0000 0.0000} red" in out_explicit.read_text()
