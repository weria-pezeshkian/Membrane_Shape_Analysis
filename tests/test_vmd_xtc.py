"""Tests for get_vmd_visualisation's handling of NaN grid points (from
circle_cutter masking in analyze/analyze.py when --rotate is used), for
build_rotation_tcl / vmd_xtc's auto-detection of whether q_mn shows a real
rotation was used, and for _trajectory_hole_union's per-trajectory hole
combining (including gaps created by the union itself).
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from CALM.core.fourier_sft import SFT
from CALM.utilize.vmd_xtc import (
    _trajectory_hole_union,
    build_rotation_tcl,
    get_vmd_visualisation,
    vmd_xtc,
)


def test_nan_grid_points_are_dropped_not_written(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    rng = np.random.default_rng(0)

    # Two frames with DIFFERENT NaN regions, as could happen across an NPT
    # run where circle_cutter's radius (min(Lx,Ly)/2) varies per frame.
    z0 = rng.uniform(-1, 1, size=(3, 10, 10))
    z0[:, 8:, 8:] = np.nan
    np.save(in_dir / "0_Z_fitted.npy", z0)

    z1 = rng.uniform(-1, 1, size=(3, 10, 10))
    z1[:, 7:, 7:] = np.nan
    np.save(in_dir / "1_Z_fitted.npy", z1)

    sft = _make_sft([0.0, 0.0])
    sft.dimensions = np.array([[100.0, 100.0, 60.0], [98.0, 98.0, 60.0]])

    get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)

    expected_valid = np.ones((10, 10), dtype=bool)
    expected_valid[8:, 8:] = False
    expected_valid[7:, 7:] = False
    expected_n_atoms = int(expected_valid.sum()) * 3  # 3 layers

    u = mda.Universe(str(out_dir / "first_frame.gro"))
    assert u.atoms.n_atoms == expected_n_atoms
    assert not np.any(np.isnan(u.atoms.positions))

    u_avg = mda.Universe(str(out_dir / "average_structure.gro"))
    assert u_avg.atoms.n_atoms == expected_n_atoms
    assert not np.any(np.isnan(u_avg.atoms.positions))

    assert (out_dir / "trajectory.xtc").exists()


def test_all_nan_raises_descriptive_error(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    z = np.full((3, 5, 5), np.nan)
    np.save(in_dir / "0_Z_fitted.npy", z)
    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)

    with pytest.raises(ValueError, match="No grid points are valid"):
        get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)


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


def test_build_rotation_tcl_writes_script_when_rotation_used(tmp_path: Path) -> None:
    sft = _make_sft([0.0, 0.1, 0.3, 0.6])  # frame 0's theta is always ~0 (base frame)
    out_path = tmp_path / "rotate_and_select.tcl"

    assert build_rotation_tcl(sft, str(out_path)) is True
    assert out_path.exists()

    content = out_path.read_text()
    assert "transaxis z" in content
    assert "same residue as" in content
    assert "mol modselect" in content
    # radius = min(Lx,Ly)/2 = min(100,80)/2 = 40
    assert "set radius 40.000000" in content


def test_build_rotation_tcl_skips_when_no_rotation_used(tmp_path: Path) -> None:
    sft = _make_sft([0.0, 0.0, 0.0, 0.0])
    out_path = tmp_path / "rotate_and_select.tcl"

    assert build_rotation_tcl(sft, str(out_path)) is False
    assert not out_path.exists()


def test_vmd_xtc_generates_tcl_only_when_rotation_present(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    Lx, Ly = 100.0, 80.0
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, size=(3, 10, 10))
    np.save(in_dir / "0_Z_fitted.npy", z)

    sft = _make_sft([0.4], Lx=Lx, Ly=Ly)
    sft.write(in_dir)

    vmd_xtc(["-i", str(in_dir), "-o", str(out_dir)])

    assert (out_dir / "rotate_and_select.tcl").exists()
    assert "Rotation detected" in capsys.readouterr().out


# ---- --Remove-TMD hole mask atom naming ----

def test_hole_grid_points_are_renamed_not_dropped(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    gridsize = 10
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, size=(3, gridsize, gridsize))
    np.save(in_dir / "0_Z_fitted.npy", z)

    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)
    hole = np.zeros((1, 2, gridsize, gridsize), dtype=bool)
    hole[0, 0, 2, 3] = True  # upper hole at grid cell (2,3)
    hole[0, 1, 5, 5] = True  # lower hole at grid cell (5,5), disjoint from upper
    sft.hole_mask = hole

    get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)

    u = mda.Universe(str(out_dir / "first_frame.gro"))
    # No atoms dropped: hole cells are renamed, not excluded (all grid points
    # valid here - no NaN/circle exclusion in play).
    assert u.atoms.n_atoms == gridsize * gridsize * 3

    n_valid = gridsize * gridsize
    upper_names = u.atoms.names[0:n_valid]
    lower_names = u.atoms.names[n_valid:2 * n_valid]
    middle_names = u.atoms.names[2 * n_valid:3 * n_valid]

    assert np.count_nonzero(upper_names == "S") == 1
    assert np.count_nonzero(lower_names == "S") == 1
    # middle is the union of upper OR lower holes -> 2 distinct cells flagged
    assert np.count_nonzero(middle_names == "S") == 2
    assert np.count_nonzero(u.atoms.names == "C") == gridsize * gridsize * 3 - 4


def test_no_hole_mask_names_everything_carbon(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, size=(3, 6, 6))
    np.save(in_dir / "0_Z_fitted.npy", z)

    sft = _make_sft([0.0], Lx=100.0, Ly=100.0)  # hole_mask left as None

    get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)

    u = mda.Universe(str(out_dir / "first_frame.gro"))
    assert np.all(u.atoms.names == "C")


def test_hole_mask_union_across_trajectory_frames(tmp_path: Path) -> None:
    # A hole that only appears in frame 1 must still be renamed for the whole
    # (single, fixed-atom-count) trajectory, not just the frames it was
    # actually present in.
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    gridsize = 8
    rng = np.random.default_rng(0)
    z0 = rng.uniform(-1, 1, size=(3, gridsize, gridsize))
    z1 = rng.uniform(-1, 1, size=(3, gridsize, gridsize))
    np.save(in_dir / "0_Z_fitted.npy", z0)
    np.save(in_dir / "1_Z_fitted.npy", z1)

    sft = _make_sft([0.0, 0.0], Lx=100.0, Ly=100.0)
    hole = np.zeros((2, 2, gridsize, gridsize), dtype=bool)
    hole[1, 0, 4, 4] = True  # only frame 1 has this hole
    sft.hole_mask = hole

    get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)

    u = mda.Universe(str(out_dir / "first_frame.gro"))
    n_valid = gridsize * gridsize
    upper_names = u.atoms.names[0:n_valid]
    assert np.count_nonzero(upper_names == "S") == 1


def test_hole_mask_lookup_is_rotation_aware(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    gridsize = 21
    Lx = Ly = 100.0
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, size=(3, gridsize, gridsize))
    np.save(in_dir / "0_Z_fitted.npy", z)

    sft = _make_sft([0.7], Lx=Lx, Ly=Ly)
    hole = np.zeros((1, 2, gridsize, gridsize), dtype=bool)
    center_idx = gridsize // 2
    hole[0, 0, center_idx, center_idx] = True  # hole exactly at the pivot
    sft.hole_mask = hole

    get_vmd_visualisation(str(in_dir), str(out_dir), sft=sft)

    u = mda.Universe(str(out_dir / "first_frame.gro"))
    n_valid = gridsize * gridsize
    upper_names = u.atoms.names[0:n_valid].reshape(gridsize, gridsize)
    # A hole exactly at the rotation pivot is invariant under rotation about
    # that same pivot - it must still be flagged post-rotation-remap.
    assert upper_names[center_idx, center_idx] == "S"


def test_trajectory_hole_union_closes_gaps_created_by_the_union_itself(tmp_path: Path) -> None:
    # Each frame's own hole mask has a single gap in an otherwise complete
    # ring around the center cell - at a DIFFERENT ring position per frame -
    # so neither frame's own mask encloses the center (a gap in the ring
    # connects it to the exterior). The union of the two masks fills both
    # gaps at once, completing the ring and enclosing the center: a small
    # island that no single frame's own build-time closing pass ever saw.
    gridsize = 5
    ring = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]

    hole0 = np.zeros((gridsize, gridsize), dtype=bool)
    for i, j in ring:
        if (i, j) != (1, 2):
            hole0[i, j] = True

    hole1 = np.zeros((gridsize, gridsize), dtype=bool)
    for i, j in ring:
        if (i, j) != (3, 2):
            hole1[i, j] = True

    sft = _make_sft([0.0, 0.0], Lx=100.0, Ly=100.0)
    sft.hole_mask = np.stack([np.stack([hole0, hole0]), np.stack([hole1, hole1])])

    z_files = []
    for i in range(2):
        path = tmp_path / f"{i}_Z_fitted.npy"
        np.save(path, np.zeros((3, gridsize, gridsize)))
        z_files.append(str(path))

    upper_union, lower_union = _trajectory_hole_union(sft, sorted(z_files), (gridsize, gridsize))
    assert upper_union[2, 2]
    assert lower_union[2, 2]
