"""Tests for map/radial_plot.py: the all-NaN center hole (_hole_radius),
equal-count radial binning from there outward (_quantile_bin_edges,
_radial_series - independent per leaflet), the per-annulus averaging
(_radial_profile), and draw()'s upper/lower-only rendering, quantity
selection, and styling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from CALM.core.fourier_sft import SFT
from CALM.map.radial_plot import (
    _hole_radius,
    _no_value_wedge,
    _quantile_bin_edges,
    _radial_profile,
    _radial_series,
    draw,
)


def _make_sft_with_hole(Lx: float, Ly: float, Nx: int, Ny: int, hole: np.ndarray) -> SFT:
    """A single-frame, non-rotated SFT carrying `hole` as its (upper, lower) hole_mask."""
    rng = np.random.default_rng(0)
    s = SFT()
    s.A_mn = rng.uniform(-1, 1, size=(1, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.where(np.arange(M) > M // 2, np.arange(M) - M, np.arange(M))
    n = np.where(np.arange(N) > N // 2, np.arange(N) - N, np.arange(N))
    qx, qy = np.meshgrid(2 * np.pi * m / Lx, 2 * np.pi * n / Ly, indexing="ij")
    s.q_mn = np.stack([np.stack([qx, qy])]).astype(np.float32)  # theta=0 for the one frame -> no rotation
    s.frame_indices = np.array([0])
    s.dimensions = np.array([[Lx, Ly, 60.0]])
    s.hole_mask = hole[None, ...]  # (1 frame, 2 layers, gridsize, gridsize)
    return s


def test_hole_radius_is_smallest_valid_point_radius() -> None:
    r = np.array([1.0, 5.0, 10.0, 20.0])
    valid = np.array([False, False, True, True])  # points at r=1, 5 are NaN
    assert _hole_radius(r, valid) == 10.0


def test_hole_radius_is_full_extent_when_everything_is_nan() -> None:
    r = np.array([1.0, 5.0, 10.0])
    valid = np.array([False, False, False])
    assert _hole_radius(r, valid) == 10.0


def test_quantile_bin_edges_have_roughly_equal_point_counts() -> None:
    rng = np.random.default_rng(0)
    # Non-uniform radius distribution: many points near r=10, few near r=40.
    r = np.concatenate([rng.uniform(8, 12, 200), rng.uniform(35, 45, 20)])
    valid = np.ones_like(r, dtype=bool)
    edges = _quantile_bin_edges(r, valid, r_hole=0.0, r_max=50.0, bin_width=5.0)

    counts = [np.sum((r >= edges[i]) & (r <= edges[i + 1])) for i in range(len(edges) - 1)]
    assert max(counts) - min(counts) <= 1  # equal-count by construction (quantile split)


def test_quantile_bin_edges_span_from_r_hole_to_r_max() -> None:
    r = np.linspace(0, 50, 100)
    valid = np.ones_like(r, dtype=bool)
    edges = _quantile_bin_edges(r, valid, r_hole=10.0, r_max=50.0, bin_width=4.0)
    assert edges[0] == 10.0
    assert edges[-1] == 50.0


def test_radial_profile_recovers_a_known_linear_radial_trend() -> None:
    gridsize = 40
    Lx = Ly = 100.0
    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cx, cy = Lx / 2, Ly / 2
    r = np.hypot(X - cx, Y - cy)

    values = 0.01 * r  # exact linear trend in r
    edges = np.linspace(0, 40, 9)
    profile = _radial_profile(values, r, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    assert np.allclose(profile, 0.01 * centers, atol=0.01)


def test_radial_profile_ignores_nan_points() -> None:
    r = np.array([1.0, 1.0, 1.0])
    values = np.array([1.0, np.nan, 3.0])
    edges = np.array([0.0, 2.0])
    profile = _radial_profile(values, r, edges)
    assert profile[0] == 2.0  # mean of 1.0 and 3.0, NaN excluded


def test_radial_series_uses_each_layers_own_density_independently() -> None:
    # A sparse layer sharing radius-space with a much denser one: if bin
    # edges were shared/combined, the dense layer would force narrow bins
    # onto the sparse one, starving several of them of any point at all.
    # Computed independently, the sparse layer still gets one point per bin.
    rng = np.random.default_rng(0)
    dense_r = rng.uniform(10.0, 40.0, 500)
    sparse_r = np.linspace(10.0, 40.0, 5)

    dense_x, dense_profile = _radial_series(np.ones_like(dense_r), dense_r, r_max=40.0, bin_width=1.0)
    sparse_x, sparse_profile = _radial_series(np.ones_like(sparse_r), sparse_r, r_max=40.0, bin_width=1.0)

    assert np.isfinite(sparse_profile).all()  # no bin left empty by the dense layer's own binning
    assert len(dense_x) > len(sparse_x)  # each layer's own bin count reflects its own data


def test_radial_series_first_x_is_exactly_its_own_hole_radius() -> None:
    r = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    values = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])  # valid starting at r=15
    x, _ = _radial_series(values, r, r_max=30.0, bin_width=5.0)
    assert x[0] == 15.0


def test_no_value_wedge_extends_the_line_through_both_points_to_the_full_range() -> None:
    x, y = _no_value_wedge(upper_start=(5.0, 2.0), lower_start=(15.0, -2.0), y_min=-10.0, y_max=10.0)
    # dx/dy = (15-5)/(-2-2) = -2.5; x(y=10) = 5 + -2.5*(10-2) = -15; x(y=-10) = 5 + -2.5*(-10-2) = 35
    assert x == pytest.approx([0.0, -15.0, 35.0, 0.0])
    assert y == pytest.approx([10.0, 10.0, -10.0, -10.0])
    # The extended line still passes through both original points exactly.
    x_at_ymax, x_at_ymin = x[1], x[2]
    t_upper = (2.0 - 10.0) / (-10.0 - 10.0)
    assert x_at_ymax + t_upper * (x_at_ymin - x_at_ymax) == pytest.approx(5.0)
    t_lower = (-2.0 - 10.0) / (-10.0 - 10.0)
    assert x_at_ymax + t_lower * (x_at_ymin - x_at_ymax) == pytest.approx(15.0)


def test_no_value_wedge_horizontal_tie_falls_back_to_the_further_x() -> None:
    x, y = _no_value_wedge(upper_start=(5.0, 1.0), lower_start=(15.0, 1.0), y_min=-10.0, y_max=10.0)
    assert x == [0.0, 15.0, 15.0, 0.0]
    assert y == [10.0, 10.0, -10.0, -10.0]


def test_quantity_files_mapping_covers_mean_and_height() -> None:
    from CALM.map.radial_plot import _QUANTITY_FILES

    assert _QUANTITY_FILES["mean"][0] == "*_mean_curvature.npy"
    assert _QUANTITY_FILES["height"][0] == "*_Z_fitted.npy"


def _write_dimensions_sft(d: Path, Lx: float, Ly: float, Nx: int = 1, Ny: int = 1) -> None:
    """A minimal single-frame, non-rotated SFT - just enough for draw()'s own box_size read."""
    rng = np.random.default_rng(0)
    s = SFT()
    s.A_mn = rng.uniform(-1, 1, size=(1, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.where(np.arange(M) > M // 2, np.arange(M) - M, np.arange(M))
    n = np.where(np.arange(N) > N // 2, np.arange(N) - N, np.arange(N))
    qx, qy = np.meshgrid(2 * np.pi * m / Lx, 2 * np.pi * n / Ly, indexing="ij")
    s.q_mn = np.stack([np.stack([qx, qy])]).astype(np.float32)  # theta=0 -> no rotation
    s.frame_indices = np.array([0])
    s.dimensions = np.array([[Lx, Ly, 60.0]])
    s.write(d)


def _draw_with_captured_axes(tmp_path: Path, **draw_kwargs) -> plt.Axes:
    real_fig, real_ax = plt.subplots()
    with patch("CALM.map.radial_plot.plt.subplots", return_value=(real_fig, real_ax)):
        draw(str(tmp_path), filename=str(tmp_path / "out.png"), **draw_kwargs)
    return real_ax


def test_draw_plots_only_upper_and_lower_with_no_sign_change(tmp_path: Path) -> None:
    gridsize = 20
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - Lx / 2, Y - Ly / 2)
    upper = 0.01 * r
    lower = 0.02 * r  # both positive; a "negate lower" convention would make this negative
    middle = 0.03 * r
    np.save(tmp_path / "0_mean_curvature.npy", np.stack([upper, lower, middle]))

    ax = _draw_with_captured_axes(tmp_path)
    try:
        data_lines = [line for line in ax.lines if line.get_label() in ("Upper", "Lower")]
        assert len(data_lines) == 2

        upper_line = next(line for line in data_lines if line.get_label() == "Upper")
        lower_line = next(line for line in data_lines if line.get_label() == "Lower")
        assert np.all(upper_line.get_ydata() >= 0)
        assert np.all(lower_line.get_ydata() >= 0)
        assert np.nanmean(lower_line.get_ydata()) > np.nanmean(upper_line.get_ydata())
    finally:
        plt.close(ax.figure)

    assert (tmp_path / "out.png").exists()


def test_draw_skips_a_masked_center_but_xaxis_still_starts_at_zero(tmp_path: Path) -> None:
    gridsize = 40
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - Lx / 2, Y - Ly / 2)

    hole = r < 20.0  # a protein occupying the center, out to r=20
    upper = np.where(hole, np.nan, 0.01 * r)
    lower = np.where(hole, np.nan, 0.01 * r)
    middle = np.where(hole, np.nan, 0.01 * r)
    np.save(tmp_path / "0_mean_curvature.npy", np.stack([upper, lower, middle]))

    ax = _draw_with_captured_axes(tmp_path)
    try:
        upper_line = next(line for line in ax.lines if line.get_label() == "Upper")
        xdata = upper_line.get_xdata()
        assert np.all(np.isfinite(xdata))  # no bins were even created inside the hole
        assert xdata.min() >= 19.0  # first bin starts essentially at the hole's own radius

        assert ax.get_xlim()[0] == 0.0  # axis itself still starts at 0, hole visible as empty space

        no_value_patches = [p for p in ax.patches if p.get_label() == "No value"]
        assert len(no_value_patches) == 1
        vertices = no_value_patches[0].get_xy()
        assert vertices[:, 0].min() == 0.0  # wedge is bounded by the y-axis
        # Wedge spans the axes' own final y-range exactly.
        y_min, y_max = ax.get_ylim()
        assert vertices[:, 1].min() == pytest.approx(y_min)
        assert vertices[:, 1].max() == pytest.approx(y_max)
    finally:
        plt.close(ax.figure)


def test_draw_wedge_line_passes_through_each_leaflets_own_starting_point(tmp_path: Path) -> None:
    gridsize = 40
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - Lx / 2, Y - Ly / 2)

    # Upper's hole is smaller (r<10) than lower's (r<20) - an asymmetric case.
    upper = np.where(r < 10.0, np.nan, 0.01 * r)
    lower = np.where(r < 20.0, np.nan, 0.01 * r)
    middle = np.where(r < 20.0, np.nan, 0.01 * r)
    np.save(tmp_path / "0_mean_curvature.npy", np.stack([upper, lower, middle]))

    ax = _draw_with_captured_axes(tmp_path)
    try:
        upper_line = next(line for line in ax.lines if line.get_label() == "Upper")
        lower_line = next(line for line in ax.lines if line.get_label() == "Lower")
        no_value_patches = [p for p in ax.patches if p.get_label() == "No value"]
        vertices = no_value_patches[0].get_xy()
        y_min, y_max = ax.get_ylim()

        # The wedge's boundary line, evaluated at each leaflet's own
        # starting y-value, lands back on that leaflet's own starting x -
        # regardless of how different upper's and lower's own hole radii are.
        x_at_ymax, x_at_ymin = vertices[1, 0], vertices[2, 0]
        for line in (upper_line, lower_line):
            x0, y0 = line.get_xdata()[0], line.get_ydata()[0]
            t = (y0 - y_max) / (y_min - y_max)
            x_on_wedge = x_at_ymax + t * (x_at_ymin - x_at_ymax)
            assert x_on_wedge == pytest.approx(x0)
    finally:
        plt.close(ax.figure)


def test_draw_height_quantity_is_relative_to_mid_surface(tmp_path: Path) -> None:
    gridsize = 20
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    upper = np.full((gridsize, gridsize), 7.0)
    lower = np.full((gridsize, gridsize), 3.0)
    middle = np.full((gridsize, gridsize), 5.0)  # exact midpoint of upper/lower
    np.save(tmp_path / "0_Z_fitted.npy", np.stack([upper, lower, middle]))

    ax = _draw_with_captured_axes(tmp_path, quantity="height")
    try:
        upper_line = next(line for line in ax.lines if line.get_label() == "Upper")
        lower_line = next(line for line in ax.lines if line.get_label() == "Lower")
        # Relative to the mid-surface (5.0): upper -> +2.0, lower -> -2.0.
        assert np.allclose(upper_line.get_ydata(), 2.0)
        assert np.allclose(lower_line.get_ydata(), -2.0)
        assert ax.get_ylabel() == "Height relative to mid-surface (nm)"
    finally:
        plt.close(ax.figure)


def test_draw_height_quantity_not_contaminated_by_the_other_leaflets_hole(tmp_path: Path) -> None:
    # A real --Remove-TMD hole_mask (not baked directly into the .npy file,
    # so this actually exercises _load_and_mask's masking, unlike the
    # earlier tests which inject NaN straight into the saved arrays): only
    # upper has a hole. Lower's own relative-height curve must not be
    # pulled out to upper's hole radius by the mid-surface reference.
    gridsize = 20
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    x = np.linspace(0, Lx, gridsize, endpoint=False)
    y = np.linspace(0, Ly, gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - Lx / 2, Y - Ly / 2)

    upper_hole = r < 20.0
    lower_hole = np.zeros_like(upper_hole)
    sft = _make_sft_with_hole(Lx, Ly, Nx=3, Ny=3, hole=np.stack([upper_hole, lower_hole]))
    sft.write(str(tmp_path))

    # Raw Z_fitted has real values everywhere - --Remove-TMD masking is
    # applied later, at load time, from sft.hole_mask, not baked into the file.
    upper = np.full((gridsize, gridsize), 7.0)
    lower = np.full((gridsize, gridsize), 3.0)
    middle = (upper + lower) / 2.0
    np.save(tmp_path / "0_Z_fitted.npy", np.stack([upper, lower, middle]))

    ax = _draw_with_captured_axes(tmp_path, quantity="height")
    try:
        lower_line = next(line for line in ax.lines if line.get_label() == "Lower")
        # Lower has no hole of its own - its curve starts near r=0, not out
        # at upper's r=20.
        assert lower_line.get_xdata().min() < 5.0
    finally:
        plt.close(ax.figure)


def test_draw_styling_linewidth_and_hidden_spines(tmp_path: Path) -> None:
    gridsize = 20
    Lx = Ly = 100.0
    _write_dimensions_sft(tmp_path, Lx, Ly)

    rng = np.random.default_rng(0)
    np.save(tmp_path / "0_mean_curvature.npy", rng.uniform(-0.1, 0.1, size=(3, gridsize, gridsize)))

    ax = _draw_with_captured_axes(tmp_path)
    try:
        data_lines = [line for line in ax.lines if line.get_label() in ("Upper", "Lower")]
        assert all(line.get_linewidth() == 3 for line in data_lines)
        assert not ax.spines["right"].get_visible()
        assert not ax.spines["top"].get_visible()
        assert ax.spines["left"].get_linewidth() == 3
        assert ax.spines["bottom"].get_linewidth() == 3
    finally:
        plt.close(ax.figure)
