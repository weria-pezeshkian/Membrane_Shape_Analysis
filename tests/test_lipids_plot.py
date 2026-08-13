"""Tests for map/lipids_plot.py: per-species, per-leaflet occupancy-frequency
maps from 'CALM analyze lipids' output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CALM.map.lipids_plot import (
    _leaflet_counts,
    _lipids_dimensions,
    _mean_fractions,
    _read_species,
    _species_outfile,
    _was_rotated,
    draw,
)


def test_species_outfile_inserts_the_name_before_the_suffix() -> None:
    assert _species_outfile("lipids.png", "POPC") == "lipids_POPC.png"
    assert _species_outfile("/out/dir/lipids.png", "TOCL1") == "/out/dir/lipids_TOCL1.png"


def test_was_rotated_defaults_false_when_the_marker_is_missing(tmp_path: Path) -> None:
    assert _was_rotated(str(tmp_path)) is False


def test_was_rotated_reads_the_saved_marker(tmp_path: Path) -> None:
    np.save(tmp_path / "rotated.npy", np.array(True))
    assert _was_rotated(str(tmp_path)) is True
    np.save(tmp_path / "rotated.npy", np.array(False))
    assert _was_rotated(str(tmp_path)) is False


def test_lipids_dimensions_filters_by_frame_number(tmp_path: Path) -> None:
    (tmp_path / "dimensions.csv").write_text("0,100.0,100.0,60.0\n1,90.0,110.0,60.0\n2,80.0,120.0,60.0\n")
    dims = _lipids_dimensions(str(tmp_path), frame_numbers=[0, 2])
    np.testing.assert_array_equal(dims, [[100.0, 100.0, 60.0], [80.0, 120.0, 60.0]])
    dims_all = _lipids_dimensions(str(tmp_path), frame_numbers=None)
    assert dims_all.shape == (3, 3)


def _write_lipids_output(
    out_dir: Path, species: list[str], fractions: np.ndarray, box: tuple[float, float, float] = (10.0, 10.0, 8.0),
) -> None:
    """A minimal 'CALM analyze lipids' output directory: one frame's own fractions, species list, dimensions.

    `fractions` is (n_species, 2, gridsize, gridsize) - the same shape
    '{frame}_lipid_fractions.npy' saves. dimensions.csv is written with no
    header row, matching calc_lipids's own convention (unlike calc_fourier's).
    """
    (out_dir / "lipid_species.txt").write_text("\n".join(species) + "\n")
    np.save(out_dir / "00_lipid_fractions.npy", fractions)
    (out_dir / "dimensions.csv").write_text(f"0,{box[0]},{box[1]},{box[2]}\n")


def test_read_species_returns_the_saved_order(tmp_path: Path) -> None:
    (tmp_path / "lipid_species.txt").write_text("POPC\nPOPE\nTOCL1\n")
    assert _read_species(str(tmp_path)) == ["POPC", "POPE", "TOCL1"]


def test_mean_fractions_averages_hard_assignments_into_occupancy_frequency(tmp_path: Path) -> None:
    # Frame 0: species 0 owns everything. Frame 1: species 1 owns everything.
    # A point's mean fraction is then 0.5/0.5 - its occupancy frequency, not a blend.
    frame0 = np.zeros((2, 2, 3, 3))
    frame0[0] = 1.0
    frame1 = np.zeros((2, 2, 3, 3))
    frame1[1] = 1.0
    (tmp_path / "lipid_species.txt").write_text("A\nB\n")
    np.save(tmp_path / "00_lipid_fractions.npy", frame0)
    np.save(tmp_path / "01_lipid_fractions.npy", frame1)

    mean = _mean_fractions(str(tmp_path), frame_numbers=None)
    assert mean.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(mean[:, 0, 0, 0], [0.5, 0.5])


def test_mean_fractions_raises_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _mean_fractions(str(tmp_path), frame_numbers=None)


def test_mean_fractions_poisons_a_point_holed_in_even_one_frame(tmp_path: Path) -> None:
    # Both frames: species 0 owns everything. Frame 0 additionally holes
    # point (0, 0) in the upper leaflet via --Remove-TMD.
    frame0 = np.zeros((2, 2, 3, 3))
    frame0[0] = 1.0
    frame1 = np.zeros((2, 2, 3, 3))
    frame1[0] = 1.0
    (tmp_path / "lipid_species.txt").write_text("A\nB\n")
    np.save(tmp_path / "00_lipid_fractions.npy", frame0)
    np.save(tmp_path / "01_lipid_fractions.npy", frame1)

    hole = np.zeros((2, 3, 3), dtype=bool)
    hole[0, 0, 0] = True  # upper leaflet, point (0, 0)
    np.save(tmp_path / "00_hole_mask.npy", hole)

    mean = _mean_fractions(str(tmp_path), frame_numbers=None)
    assert np.all(np.isnan(mean[:, 0, 0, 0]))  # holed in frame 0 - NaN for every species
    assert np.all(np.isfinite(mean[:, 1, 0, 0]))  # lower leaflet, same point - never holed
    np.testing.assert_allclose(mean[:, 0, 0, 1], [1.0, 0.0])  # untouched neighboring point


def test_leaflet_counts_reads_mean_count_for_the_requested_leaflet(tmp_path: Path) -> None:
    (tmp_path / "area_per_lipid.csv").write_text(
        "leaflet,species,area_per_lipid_flat,area_per_lipid_curved,mean_count\n"
        "upper,POPC,60.0,61.0,100.0\n"
        "lower,POPC,59.0,60.0,90.0\n"
    )
    assert _leaflet_counts(str(tmp_path), "upper") == {"POPC": 100.0}
    assert _leaflet_counts(str(tmp_path), "lower") == {"POPC": 90.0}


def test_leaflet_counts_returns_none_when_csv_is_absent(tmp_path: Path) -> None:
    assert _leaflet_counts(str(tmp_path), "upper") is None


def test_draw_saves_a_combined_overview_and_one_file_per_species(tmp_path: Path) -> None:
    fractions = np.zeros((3, 2, 5, 5))
    fractions[0] = 1.0  # species 0 owns every point, both leaflets, the only frame
    _write_lipids_output(tmp_path, ["POPC", "POPE", "TOCL1"], fractions)

    draw(Dir=str(tmp_path), filename=str(tmp_path / "out.png"))

    combined = tmp_path / "out.png"
    assert combined.exists()
    assert combined.stat().st_size > 0
    for name in ("POPC", "POPE", "TOCL1"):
        out = tmp_path / f"out_{name}.png"
        assert out.exists()
        assert out.stat().st_size > 0


def test_draw_clips_to_the_fixed_circle_only_when_rotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import CALM.map.lipids_plot as lipids_plot_module

    fractions = np.zeros((1, 2, 5, 5))
    fractions[0] = 1.0
    _write_lipids_output(tmp_path, ["POPC"], fractions, box=(100.0, 80.0, 60.0))

    radii: list[float | None] = []
    monkeypatch.setattr(
        lipids_plot_module, "_clip_to_circle",
        lambda contour, ax, circle_radius, box_size: radii.append(circle_radius),
    )

    # 4 calls total: 2 panels in the combined overview + 2 in the one
    # per-species figure (a single species here).
    draw(Dir=str(tmp_path), filename=str(tmp_path / "out.png"))
    assert radii == [None] * 4  # not rotated - no clipping

    radii.clear()
    np.save(tmp_path / "rotated.npy", np.array(True))
    draw(Dir=str(tmp_path), filename=str(tmp_path / "out.png"))
    assert radii == [40.0] * 4  # min(100, 80) / 2
