"""Tests for map/dynamic_plot.py:
- _windows: full-size rolling-window construction, including strided frame
  numbers and shared boundary windows at the sequence's edges
- draw()'s frame_numbers filter (in map/plot.py), as consumed by
  draw_dynamic: correct subset when given, unchanged (all-frames) behavior
  when omitted
- draw_dynamic end-to-end: one output GIF frame per available trajectory frame
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from CALM.map import plot as plot_module
from CALM.map.dynamic_plot import _windows, draw_dynamic
from CALM.map.plot import draw


def test_windows_are_full_size_with_shared_boundary_windows_at_the_edges() -> None:
    assert _windows([0, 1, 2, 3, 4, 5, 6], 5) == [
        [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6],
    ]


def test_windows_uses_position_not_frame_number_for_stride() -> None:
    # Frame numbers 0,5,10,15,20 (e.g. --Step 5): a window of 3 should still
    # average the 3 nearest AVAILABLE frames, not a numeric +/-N range.
    assert _windows([0, 5, 10, 15, 20], 3) == [
        [0, 5, 10], [0, 5, 10], [5, 10, 15], [10, 15, 20], [10, 15, 20],
    ]


def test_windows_size_one_disables_smoothing() -> None:
    assert _windows([0, 1, 2], 1) == [[0], [1], [2]]


def test_windows_larger_than_sequence_includes_everything() -> None:
    assert _windows([0, 1, 2], 100) == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]


def _write_dimensions_csv(d: Path, n_frames: int, Lx: float, Ly: float) -> None:
    with open(d / "dimensions.csv", "w") as f:
        f.write("#header\n")
        for i in range(n_frames):
            f.write(f"{i},{Lx},{Ly},60.0\n")


def _write_mean_curvature_frames(d: Path, n_frames: int, gridsize: int = 10) -> None:
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        np.save(d / f"{i}_mean_curvature.npy", rng.uniform(-0.1, 0.1, size=(3, gridsize, gridsize)))


def _write_thickness_frames(d: Path, n_frames: int, gridsize: int = 10) -> None:
    rng = np.random.default_rng(1)
    for i in range(n_frames):
        np.save(d / f"{i}_thickness.npy", rng.uniform(3.0, 5.0, size=(gridsize, gridsize)))


def test_draw_frame_numbers_filters_files_passed_to_load_and_mask(tmp_path: Path) -> None:
    _write_dimensions_csv(tmp_path, 3, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, 3)

    with patch.object(plot_module, "_load_and_mask", wraps=plot_module._load_and_mask) as mock_load:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"), frame_numbers=[0, 2])

    files_arg = mock_load.call_args_list[0][0][0]
    frame_numbers_used = sorted(int(Path(f).stem.split("_")[0]) for f in files_arg)
    assert frame_numbers_used == [0, 2]


def test_draw_frame_numbers_omitted_uses_every_file(tmp_path: Path) -> None:
    _write_dimensions_csv(tmp_path, 3, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, 3)

    with patch.object(plot_module, "_load_and_mask", wraps=plot_module._load_and_mask) as mock_load:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))

    files_arg = mock_load.call_args_list[0][0][0]
    frame_numbers_used = sorted(int(Path(f).stem.split("_")[0]) for f in files_arg)
    assert frame_numbers_used == [0, 1, 2]


def test_draw_dynamic_renders_one_frame_per_available_frame(tmp_path: Path) -> None:
    n_frames = 4
    _write_dimensions_csv(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)

    out_gif = tmp_path / "out.gif"
    with patch("CALM.map.dynamic_plot.draw", wraps=draw) as mock_draw:
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(out_gif), window=3, spf=0.05)

    assert out_gif.exists()
    assert mock_draw.call_count == n_frames


def test_draw_dynamic_fixes_thickness_scale_across_windows(tmp_path: Path) -> None:
    n_frames = 6
    _write_dimensions_csv(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)
    _write_thickness_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot.draw", wraps=draw) as mock_draw:
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(tmp_path / "out.gif"), window=3, spf=0.05)

    thickness_minmax_per_call = [call.kwargs["thickness_minmax"] for call in mock_draw.call_args_list]
    assert all(tm is not None for tm in thickness_minmax_per_call)
    assert len({tuple(tm) for tm in thickness_minmax_per_call}) == 1


def _write_principal_frames(d: Path, n_frames: int, gridsize: int = 10) -> None:
    rng = np.random.default_rng(2)
    for i in range(n_frames):
        np.save(d / f"{i}_principal_curvatures.npy", rng.uniform(-0.1, 0.1, size=(6, gridsize, gridsize)))
        vecs = rng.normal(size=(6, gridsize, gridsize, 3))
        vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
        np.save(d / f"{i}_principal_dirs.npy", vecs)


def test_draw_dynamic_uses_each_windows_own_frame_for_vectors(tmp_path: Path) -> None:
    # The vector overlay must reflect that video frame's own instantaneous
    # directions, never the rolling window used for the curvature background.
    n_frames = 6
    _write_dimensions_csv(tmp_path, n_frames, 100.0, 80.0)
    _write_principal_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot.draw", wraps=draw) as mock_draw:
        draw_dynamic(str(tmp_path), mode="principal", out_gif=str(tmp_path / "out.gif"), window=3, spf=0.05)

    vector_frames = [call.kwargs["vector_frame"] for call in mock_draw.call_args_list]
    assert vector_frames == list(range(n_frames))


def test_draw_dynamic_raises_for_missing_mode_files(tmp_path: Path) -> None:
    _write_dimensions_csv(tmp_path, 2, 100.0, 80.0)
    with pytest.raises(FileNotFoundError):
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(tmp_path / "out.gif"))
