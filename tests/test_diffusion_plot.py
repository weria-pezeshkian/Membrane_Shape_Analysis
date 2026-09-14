"""Tests for map/diffusion_plot.py: draw()'s single-directory rendering (previously untested) and
its --replica averaging across independent directories (_load_diffusion_directory, the
align_and_average integration, and the legend's mean-D-across-replicas reporting).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from CALM.analyze.diffusion import _DIFFUSION_DTYPE, _MSD_DTYPE
from CALM.map.diffusion_plot import draw

_TAU = np.array([10.0, 20.0, 30.0, 40.0])


def _write_diffusion_dir(
    directory: Path, curves: dict[tuple[str, str], tuple[np.ndarray, float, float]]
) -> None:
    """Write diffusion.npy/msd_curves.npy for one directory from `{(species, leaflet): (msd, D, D_stderr)}`."""
    diffusion_rows = []
    msd_rows = []
    for (species, leaflet), (msd, d_value, d_stderr) in curves.items():
        diffusion_rows.append((leaflet, species, d_value, d_stderr, 1, len(msd), 10.0, 40.0, 0.99, 1.0, 0))
        for tau, value in zip(_TAU, msd):
            msd_rows.append((leaflet, species, tau, value, 1))
    np.save(directory / "diffusion.npy", np.array(diffusion_rows, dtype=_DIFFUSION_DTYPE))
    np.save(directory / "msd_curves.npy", np.array(msd_rows, dtype=_MSD_DTYPE))


def _draw_with_captured_axes(**draw_kwargs) -> plt.Axes:
    real_fig, real_ax = plt.subplots()
    with patch("CALM.map.diffusion_plot.plt.subplots", return_value=(real_fig, real_ax)):
        draw(**draw_kwargs)
    return real_ax


def test_draw_single_directory_plots_one_curve_per_species_leaflet(tmp_path: Path) -> None:
    _write_diffusion_dir(tmp_path, {
        ("POPC", "upper"): (np.array([1.0, 2.0, 3.0, 4.0]), 1.5e-7, 1.0e-8),
        ("CHOL", "lower"): (np.array([2.0, 4.0, 6.0, 8.0]), 3.0e-7, 2.0e-8),
    })

    ax = _draw_with_captured_axes(Dir=str(tmp_path), filename=str(tmp_path / "out.png"))
    try:
        assert len(ax.lines) >= 2
        labels = [line.get_label() for line in ax.lines]
        assert any("POPC" in label and "1.5e-07" in label for label in labels)
        assert any("CHOL" in label and "3e-07" in label for label in labels)
        assert not ax.collections  # no replica band for a single directory
    finally:
        plt.close(ax.figure)
    assert (tmp_path / "out.png").exists()


def test_draw_single_directory_has_no_shaded_band(tmp_path: Path) -> None:
    _write_diffusion_dir(tmp_path, {("POPC", "upper"): (np.array([1.0, 2.0, 3.0, 4.0]), 1.5e-7, 1.0e-8)})
    ax = _draw_with_captured_axes(Dir=str(tmp_path), filename=str(tmp_path / "out.png"))
    try:
        assert len(ax.collections) == 0
    finally:
        plt.close(ax.figure)


def test_draw_replica_average_reports_mean_d_and_replica_count(tmp_path: Path) -> None:
    rep1, rep2 = tmp_path / "rep1", tmp_path / "rep2"
    rep1.mkdir()
    rep2.mkdir()
    _write_diffusion_dir(rep1, {("POPC", "upper"): (np.array([1.0, 2.0, 3.0, 4.0]), 1.0e-7, 1.0e-8)})
    _write_diffusion_dir(rep2, {("POPC", "upper"): (np.array([1.2, 2.2, 3.2, 4.2]), 3.0e-7, 1.0e-8)})

    ax = _draw_with_captured_axes(
        Dir=[str(rep1), str(rep2)], filename=str(tmp_path / "out.png"),
    )
    try:
        line = next(line for line in ax.lines if "POPC" in line.get_label())
        label = line.get_label()
        assert "n=2 replicas" in label
        assert "D=2e-07" in label  # mean of 1e-7 and 3e-7
        assert len(ax.collections) == 1  # one shaded band, for the averaged curve
    finally:
        plt.close(ax.figure)


def test_draw_replica_average_skips_a_replica_missing_a_species(tmp_path: Path) -> None:
    rep1, rep2 = tmp_path / "rep1", tmp_path / "rep2"
    rep1.mkdir()
    rep2.mkdir()
    _write_diffusion_dir(rep1, {
        ("POPC", "upper"): (np.array([1.0, 2.0, 3.0, 4.0]), 1.0e-7, 1.0e-8),
        ("CHOL", "upper"): (np.array([0.5, 1.0, 1.5, 2.0]), 5.0e-8, 1.0e-8),
    })
    _write_diffusion_dir(rep2, {("POPC", "upper"): (np.array([1.2, 2.2, 3.2, 4.2]), 1.2e-7, 1.0e-8)})

    ax = _draw_with_captured_axes(Dir=[str(rep1), str(rep2)], filename=str(tmp_path / "out.png"))
    try:
        labels = [line.get_label() for line in ax.lines]
        assert any("POPC" in label and "n=2 replicas" in label for label in labels)
        assert any("CHOL" in label and "n=1 replicas" in label for label in labels)
    finally:
        plt.close(ax.figure)


def test_draw_replica_average_aligns_onto_the_shortest_replicas_own_tau(tmp_path: Path) -> None:
    rep1, rep2 = tmp_path / "rep1", tmp_path / "rep2"
    rep1.mkdir()
    rep2.mkdir()
    _write_diffusion_dir(rep1, {("POPC", "upper"): (np.array([1.0, 2.0, 3.0, 4.0]), 1.0e-7, 1.0e-8)})
    # rep2's own tau grid runs further (values still linear in tau=10..40, plus one extra beyond).
    diffusion_rows = [("upper", "POPC", 1.0e-7, 1.0e-8, 1, 5, 10.0, 50.0, 0.99, 1.0, 0)]
    msd_rows = [("upper", "POPC", tau, msd, 1) for tau, msd in zip([10, 20, 30, 40, 50], [1, 2, 3, 4, 5])]
    np.save(rep2 / "diffusion.npy", np.array(diffusion_rows, dtype=_DIFFUSION_DTYPE))
    np.save(rep2 / "msd_curves.npy", np.array(msd_rows, dtype=_MSD_DTYPE))

    ax = _draw_with_captured_axes(Dir=[str(rep1), str(rep2)], filename=str(tmp_path / "out.png"))
    try:
        line = next(line for line in ax.lines if "POPC" in line.get_label())
        assert line.get_xdata().max() == 40.0  # rep1's own (shorter) range, not rep2's tau=50
    finally:
        plt.close(ax.figure)
