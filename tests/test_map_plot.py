"""Tests for map/plot.py:
- rotation-aware circular-vs-box view (circle clipping, not an outline)
- grid resolution taken from the actual saved data, not a hardcoded assumption
- clear errors instead of a cryptic crash when an expected pattern matches nothing
- mean mode's dynamic 2x2 (with thickness) / 1x3 (without) layout
- the dedicated single-panel "thickness" mode
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import pytest

from CALM.core.fourier_sft import SFT
from CALM.map.plot import _load_and_mask, draw


def _make_q_mn(Lx: float, Ly: float, Nx: int, Ny: int, theta: float) -> np.ndarray:
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.arange(M)
    n = np.arange(N)
    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)
    qx_grid, qy_grid = np.meshgrid(2 * np.pi * m / Lx, 2 * np.pi * n / Ly, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * qx_grid - s * qy_grid, s * qx_grid + c * qy_grid], axis=0)


def _write_dimensions_csv(d: Path, n_frames: int, Lx: float, Ly: float) -> None:
    with open(d / "dimensions.csv", "w") as f:
        f.write("#header\n")
        for i in range(n_frames):
            f.write(f"{i},{Lx},{Ly},60.0\n")


def _write_sft(d: Path, theta: float, n_frames: int, Lx: float, Ly: float, Nx: int = 3, Ny: int = 3) -> None:
    rng = np.random.default_rng(0)
    sft = SFT()
    sft.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    sft.q_mn = np.stack([_make_q_mn(Lx, Ly, Nx, Ny, theta) for _ in range(n_frames)]).astype(np.float32)
    sft.frame_indices = np.arange(n_frames)
    sft.dimensions = np.tile([Lx, Ly, 60.0], (n_frames, 1))
    sft.write(d)


def _setup_plot_dir(
    tmp_path: Path, theta: float, n_frames: int = 2, Lx: float = 100.0, Ly: float = 80.0, grid: int = 100,
    write_sft: bool = True, write_mean: bool = True, write_thickness: bool = True,
) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path

    if write_sft:
        _write_sft(d, theta, n_frames, Lx, Ly)

    _write_dimensions_csv(d, n_frames, Lx, Ly)

    for i in range(n_frames):
        if write_mean:
            np.save(d / f"{i}_mean_curvature.npy", rng.uniform(-0.1, 0.1, size=(3, grid, grid)))
        if write_thickness:
            np.save(d / f"{i}_thickness.npy", rng.uniform(3, 5, size=(grid, grid)))
    return d


def test_circle_clipped_when_rotation_used(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.4)
    with patch.object(mpatches, "Circle", wraps=mpatches.Circle) as mock_circle:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    assert mock_circle.call_count > 0


def test_no_circle_when_rotation_not_used(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0)
    with patch.object(mpatches, "Circle", wraps=mpatches.Circle) as mock_circle:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    assert mock_circle.call_count == 0


def test_no_circle_when_no_sft_present(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False)
    with patch.object(mpatches, "Circle", wraps=mpatches.Circle) as mock_circle:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    assert mock_circle.call_count == 0


def test_grid_resolution_matches_actual_data_not_hardcoded(tmp_path: Path) -> None:
    # Deliberately NOT 100: get_XY must size itself from the actual data,
    # not assume a fixed grid regardless of --gridsize.
    _setup_plot_dir(tmp_path, theta=0.0, grid=37, write_sft=False)
    draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    assert (tmp_path / "out.png").exists()


def test_missing_mean_curvature_files_raises_clear_error(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False, write_mean=False, write_thickness=True)
    with pytest.raises(FileNotFoundError, match="mean_curvature"):
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))


def test_missing_thickness_files_raises_clear_error_in_thickness_mode(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False, write_mean=True, write_thickness=False)
    with pytest.raises(FileNotFoundError, match="thickness"):
        draw(str(tmp_path), mode="thickness", filename=str(tmp_path / "out.png"))


def test_mean_mode_uses_2x2_layout_when_thickness_present(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False, write_mean=True, write_thickness=True)
    import matplotlib.pyplot as plt
    with patch.object(plt, "subplots", wraps=plt.subplots) as mock_subplots:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    (nrows, ncols), _ = mock_subplots.call_args
    assert (nrows, ncols) == (2, 2)


def test_mean_mode_uses_1x3_layout_when_thickness_absent(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False, write_mean=True, write_thickness=False)
    import matplotlib.pyplot as plt
    with patch.object(plt, "subplots", wraps=plt.subplots) as mock_subplots:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))
    (nrows, ncols), _ = mock_subplots.call_args
    assert (nrows, ncols) == (1, 3)


def test_thickness_mode_uses_single_panel(tmp_path: Path) -> None:
    _setup_plot_dir(tmp_path, theta=0.0, write_sft=False, write_mean=False, write_thickness=True)
    import matplotlib.pyplot as plt
    with patch.object(plt, "subplots", wraps=plt.subplots) as mock_subplots:
        draw(str(tmp_path), mode="thickness", filename=str(tmp_path / "out.png"))
    args, kwargs = mock_subplots.call_args
    # single-panel call: no (nrows, ncols) positional args, just figsize
    assert args == ()
    assert "figsize" in kwargs


# ---- --Remove-TMD hole mask consumption ----

def _sft_with_hole(
    tmp_path: Path, hole: np.ndarray, theta: float = 0.0, n_frames: int = 2,
    Lx: float = 100.0, Ly: float = 100.0, gridsize: int = 5, Nx: int = 3, Ny: int = 3,
) -> SFT:
    """hole: boolean array (n_frames, 2, gridsize, gridsize) [upper, lower]."""
    sft = SFT()
    rng = np.random.default_rng(0)
    sft.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    sft.q_mn = np.stack([_make_q_mn(Lx, Ly, Nx, Ny, theta) for _ in range(n_frames)]).astype(np.float32)
    sft.frame_indices = np.arange(n_frames)
    sft.dimensions = np.tile([Lx, Ly, 60.0], (n_frames, 1))
    sft.hole_mask = hole
    return sft


def test_load_and_mask_applies_upper_lower_union_per_layer(tmp_path: Path) -> None:
    # Hole-mask NaNs poison the average exactly like circle-cutter NaNs
    # (plain np.mean, not nanmean): relies on the per-frame hole threshold
    # (core/packing.py's median_multiple_threshold) having a low false-
    # positive rate, rather than averaging compensating for it.
    gridsize = 5
    frame0 = np.ones((3, gridsize, gridsize))
    frame1 = np.ones((3, gridsize, gridsize)) * 3.0
    np.save(tmp_path / "0_mean_curvature.npy", frame0)
    np.save(tmp_path / "1_mean_curvature.npy", frame1)

    hole = np.zeros((2, 2, gridsize, gridsize), dtype=bool)
    hole[0, 0, 0, 0] = True  # frame 0, upper leaflet, grid cell (0,0)
    sft = _sft_with_hole(tmp_path, hole, gridsize=gridsize)

    files = sorted(str(p) for p in tmp_path.glob("*_mean_curvature.npy"))
    result = _load_and_mask(files, "pattern", str(tmp_path), sft, layer_sources=["upper", "lower", "union"])

    assert np.isnan(result[0, 0, 0])       # upper: masked in frame 0 -> poisons the average
    assert result[0, 1, 1] == 2.0          # upper: untouched cell averages normally
    assert result[1, 0, 0] == 2.0          # lower: no mask applied to lower at all
    assert np.isnan(result[2, 0, 0])       # middle (union): masked via upper's hole


def test_load_and_mask_no_hole_mask_falls_back_to_plain_average(tmp_path: Path) -> None:
    gridsize = 5
    frame0 = np.ones((3, gridsize, gridsize))
    frame1 = np.ones((3, gridsize, gridsize)) * 3.0
    np.save(tmp_path / "0_mean_curvature.npy", frame0)
    np.save(tmp_path / "1_mean_curvature.npy", frame1)

    files = sorted(str(p) for p in tmp_path.glob("*_mean_curvature.npy"))
    result = _load_and_mask(files, "pattern", str(tmp_path), sft=None, layer_sources=["upper", "lower", "union"])
    assert np.allclose(result, 2.0)


def test_load_and_mask_conservative_averaging_poisons_on_any_nan(tmp_path: Path) -> None:
    # Circle-cutter-style NaN already baked into one frame (not from a hole
    # mask at all): the averaged point must still come out NaN, not silently
    # averaged over the remaining valid frame.
    gridsize = 3
    frame0 = np.ones((3, gridsize, gridsize))
    frame0[0, 0, 0] = np.nan
    frame1 = np.ones((3, gridsize, gridsize)) * 5.0
    np.save(tmp_path / "0_mean_curvature.npy", frame0)
    np.save(tmp_path / "1_mean_curvature.npy", frame1)

    files = sorted(str(p) for p in tmp_path.glob("*_mean_curvature.npy"))
    result = _load_and_mask(files, "pattern", str(tmp_path), sft=None, layer_sources=["upper", "lower", "union"])
    assert np.isnan(result[0, 0, 0])
    assert result[0, 1, 1] == 3.0  # unaffected cell still averages normally


def test_hole_mask_lookup_is_rotation_aware(tmp_path: Path) -> None:
    # A hole placed at the box center should still register as invalid after
    # rotation is baked in (recovers theta from q_mn, remaps via
    # lookup_mask_at_rotated_grid): the identity check that this isn't just
    # ignoring rotation.
    gridsize = 21
    Lx = Ly = 100.0
    frame0 = np.ones((3, gridsize, gridsize))
    np.save(tmp_path / "0_mean_curvature.npy", frame0)

    hole = np.zeros((1, 2, gridsize, gridsize), dtype=bool)
    center_idx = gridsize // 2
    hole[0, 0, center_idx, center_idx] = True  # upper hole at box center
    sft = _sft_with_hole(tmp_path, hole, theta=0.7, n_frames=1, Lx=Lx, Ly=Ly, gridsize=gridsize)

    files = sorted(str(p) for p in tmp_path.glob("*_mean_curvature.npy"))
    result = _load_and_mask(files, "pattern", str(tmp_path), sft, layer_sources=["upper", "lower", "union"])
    # A hole exactly at the pivot is invariant under rotation around that
    # same pivot: it must still show up as NaN post-rotation.
    assert np.isnan(result[0, center_idx, center_idx])
