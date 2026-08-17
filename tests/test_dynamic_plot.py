"""Tests for map/dynamic_plot.py:
- _windows: full-size rolling-window construction, including strided frame
  numbers and shared boundary windows at the sequence's edges
- draw()'s frame_numbers filter (in map/plot.py), as consumed by
  draw_dynamic: correct subset when given, unchanged (all-frames) behavior
  when omitted
- draw_dynamic end-to-end: one output GIF frame per available trajectory frame
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from PIL import Image

from CALM.core.fourier_sft import SFT
from CALM.map import plot as plot_module
from CALM.map.dynamic_plot import (
    _draw_frame_isolated,
    _find_ffmpeg,
    _run_ffmpeg,
    _stitch_gif_in_memory,
    _stitch_gif_with_ffmpeg,
    _windowed_minmax,
    _windows,
    draw_dynamic,
)
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


def _write_sft(d: Path, n_frames: int, Lx: float, Ly: float, Nx: int = 3, Ny: int = 3) -> None:
    """A minimal non-rotated SFT - just enough for draw()'s own box_size read."""
    rng = np.random.default_rng(0)
    s = SFT()
    s.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    M, N = 2 * Nx + 1, 2 * Ny + 1
    m = np.where(np.arange(M) > M // 2, np.arange(M) - M, np.arange(M))
    n = np.where(np.arange(N) > N // 2, np.arange(N) - N, np.arange(N))
    qx, qy = np.meshgrid(2 * np.pi * m / Lx, 2 * np.pi * n / Ly, indexing="ij")
    s.q_mn = np.stack([np.stack([qx, qy])] * n_frames).astype(np.float32)  # theta=0 -> no rotation
    s.frame_indices = np.arange(n_frames)
    s.dimensions = np.tile([Lx, Ly, 60.0], (n_frames, 1))
    s.write(d)


def _write_mean_curvature_frames(d: Path, n_frames: int, gridsize: int = 10) -> None:
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        np.save(d / f"{i}_mean_curvature.npy", rng.uniform(-0.1, 0.1, size=(3, gridsize, gridsize)))


def _write_thickness_frames(d: Path, n_frames: int, gridsize: int = 10) -> None:
    rng = np.random.default_rng(1)
    for i in range(n_frames):
        np.save(d / f"{i}_thickness.npy", rng.uniform(3.0, 5.0, size=(gridsize, gridsize)))


def test_windowed_minmax_percentile_zero_matches_the_plain_min_max(tmp_path: Path) -> None:
    n_frames = 5
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_thickness_frames(tmp_path, n_frames)
    # One deliberate outlier, planted after the fact so it's a known, exact value.
    outlier_frame = np.load(tmp_path / "2_thickness.npy")
    outlier_frame[0, 0] = 100.0
    np.save(tmp_path / "2_thickness.npy", outlier_frame)

    windows = _windows(list(range(n_frames)), 1)  # window=1: each window is one frame's own raw data
    Minimum, Maximum = _windowed_minmax(str(tmp_path), "thickness", windows, percentile=0.0)

    assert Maximum == 100.0


def test_windowed_minmax_percentile_trims_a_rare_outlier(tmp_path: Path) -> None:
    n_frames = 5
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_thickness_frames(tmp_path, n_frames)
    outlier_frame = np.load(tmp_path / "2_thickness.npy")
    outlier_frame[0, 0] = 100.0  # 1 of 500 pooled points - well under a 2.5% tail
    np.save(tmp_path / "2_thickness.npy", outlier_frame)

    windows = _windows(list(range(n_frames)), 1)  # window=1: each window is one frame's own raw data
    Minimum, Maximum = _windowed_minmax(str(tmp_path), "thickness", windows, percentile=5.0)

    assert Maximum < 6.0  # comfortably below the outlier, within the real 3.0-5.0 data range


def test_draw_frame_numbers_filters_files_passed_to_load_and_mask(tmp_path: Path) -> None:
    _write_sft(tmp_path, 3, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, 3)

    with patch.object(plot_module, "_load_and_mask", wraps=plot_module._load_and_mask) as mock_load:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"), frame_numbers=[0, 2])

    files_arg = mock_load.call_args_list[0][0][0]
    frame_numbers_used = sorted(int(Path(f).stem.split("_")[0]) for f in files_arg)
    assert frame_numbers_used == [0, 2]


def test_draw_frame_numbers_omitted_uses_every_file(tmp_path: Path) -> None:
    _write_sft(tmp_path, 3, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, 3)

    with patch.object(plot_module, "_load_and_mask", wraps=plot_module._load_and_mask) as mock_load:
        draw(str(tmp_path), mode="mean", filename=str(tmp_path / "out.png"))

    files_arg = mock_load.call_args_list[0][0][0]
    frame_numbers_used = sorted(int(Path(f).stem.split("_")[0]) for f in files_arg)
    assert frame_numbers_used == [0, 1, 2]


def _draw_in_process(**kwargs) -> None:
    """Forward straight to the real `draw()` in the test's own process.

    Stands in for `_draw_frame_isolated` in tests that need to inspect the
    kwargs each frame was drawn with (via the `Mock` recording them) while
    still producing real output files - `_draw_frame_isolated`'s own real
    subprocess isolation is covered separately, in
    `test_draw_frame_isolated_runs_draw_in_a_subprocess` below.
    """
    draw(**kwargs)


def test_draw_dynamic_renders_one_frame_per_available_frame(tmp_path: Path) -> None:
    n_frames = 4
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)

    out_gif = tmp_path / "out.gif"
    with patch("CALM.map.dynamic_plot._draw_frame_isolated", side_effect=_draw_in_process) as mock_draw:
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(out_gif), window=3, spf=0.05)

    assert out_gif.exists()
    assert mock_draw.call_count == n_frames


def test_draw_dynamic_fixes_thickness_scale_across_windows(tmp_path: Path) -> None:
    n_frames = 6
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)
    _write_thickness_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot._draw_frame_isolated", side_effect=_draw_in_process) as mock_draw:
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
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_principal_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot._draw_frame_isolated", side_effect=_draw_in_process) as mock_draw:
        draw_dynamic(str(tmp_path), mode="principal", out_gif=str(tmp_path / "out.gif"), window=3, spf=0.05)

    vector_frames = [call.kwargs["vector_frame"] for call in mock_draw.call_args_list]
    assert vector_frames == list(range(n_frames))


def test_draw_dynamic_raises_for_missing_mode_files(tmp_path: Path) -> None:
    _write_sft(tmp_path, 2, 100.0, 80.0)
    with pytest.raises(FileNotFoundError):
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(tmp_path / "out.gif"))


def test_draw_frame_isolated_runs_draw_in_a_subprocess(tmp_path: Path) -> None:
    _write_sft(tmp_path, 1, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, 1)

    out_png = tmp_path / "frame.png"
    _draw_frame_isolated(
        Dir=str(tmp_path), mode="mean", minmax=[-0.1, 0.1], thickness_minmax=None,
        filename=str(out_png), show_vectors=True, frame_numbers=[0], vector_frame=0, histogram=False,
    )

    assert out_png.exists()


def test_draw_frame_isolated_raises_when_the_subprocess_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="subprocess"):
        _draw_frame_isolated(
            Dir=str(tmp_path / "does_not_exist"), mode="mean", minmax=[-0.1, 0.1], thickness_minmax=None,
            filename=str(tmp_path / "frame.png"), show_vectors=True, frame_numbers=[0], vector_frame=0,
            histogram=False,
        )


def test_find_ffmpeg_prefers_an_executable_already_on_path() -> None:
    with patch("CALM.map.dynamic_plot.shutil.which", return_value="/usr/bin/ffmpeg") as mock_which:
        assert _find_ffmpeg() == "/usr/bin/ffmpeg"
    mock_which.assert_called_once_with("ffmpeg")


def test_find_ffmpeg_falls_back_to_imageio_ffmpeg_when_nothing_is_on_path() -> None:
    fake_imageio_ffmpeg = MagicMock()
    fake_imageio_ffmpeg.get_ffmpeg_exe.return_value = "/fake/bundled/ffmpeg"
    with patch("CALM.map.dynamic_plot.shutil.which", return_value=None), \
         patch.dict("sys.modules", {"imageio_ffmpeg": fake_imageio_ffmpeg}):
        assert _find_ffmpeg() == "/fake/bundled/ffmpeg"


def test_run_ffmpeg_raises_with_stderr_on_a_nonzero_exit() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        _run_ffmpeg(["python3", "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"])


def _write_solid_frames(d: Path, n: int) -> list[str]:
    paths = []
    for i in range(n):
        p = d / f"frame_{i:06d}.png"
        Image.new("RGB", (20, 16), color=(i * 30 % 256, 10, 200)).save(p)
        paths.append(str(p))
    return paths


def test_stitch_gif_with_ffmpeg_produces_a_valid_multi_frame_gif(tmp_path: Path) -> None:
    n = 6
    _write_solid_frames(tmp_path, n)

    out_gif = tmp_path / "out.gif"
    _stitch_gif_with_ffmpeg(str(tmp_path), str(out_gif), duration_ms=50)

    assert out_gif.exists()
    with Image.open(out_gif) as gif:
        assert gif.n_frames == n


def test_stitch_gif_with_ffmpeg_passes_an_absolute_output_path(tmp_path: Path) -> None:
    # A relative, dash-prefixed filename must reach ffmpeg's argv as an
    # absolute path, so ffmpeg's own parser reads it as the output file
    # rather than mistaking it for a flag.
    with patch("CALM.map.dynamic_plot._run_ffmpeg") as mock_run:
        _stitch_gif_with_ffmpeg(str(tmp_path), "-weird.gif", duration_ms=50)

    paletteuse_cmd = mock_run.call_args_list[1][0][0]
    out_arg = paletteuse_cmd[-1]
    assert os.path.isabs(out_arg)
    assert out_arg.endswith("-weird.gif")


def test_stitch_gif_in_memory_produces_a_valid_multi_frame_gif(tmp_path: Path) -> None:
    n = 4
    frame_paths = _write_solid_frames(tmp_path, n)

    out_gif = tmp_path / "out.gif"
    _stitch_gif_in_memory(frame_paths, str(out_gif), duration_ms=50)

    assert out_gif.exists()
    with Image.open(out_gif) as gif:
        assert gif.n_frames == n


def test_draw_dynamic_in_memory_flag_uses_the_pillow_path(tmp_path: Path) -> None:
    n_frames = 3
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot._stitch_gif_in_memory") as mock_in_memory, \
         patch("CALM.map.dynamic_plot._stitch_gif_with_ffmpeg") as mock_ffmpeg:
        draw_dynamic(
            str(tmp_path), mode="mean", out_gif=str(tmp_path / "out.gif"),
            window=3, spf=0.05, in_memory=True,
        )

    mock_in_memory.assert_called_once()
    mock_ffmpeg.assert_not_called()


def test_draw_dynamic_default_uses_the_ffmpeg_path(tmp_path: Path) -> None:
    n_frames = 3
    _write_sft(tmp_path, n_frames, 100.0, 80.0)
    _write_mean_curvature_frames(tmp_path, n_frames)

    with patch("CALM.map.dynamic_plot._stitch_gif_in_memory") as mock_in_memory, \
         patch("CALM.map.dynamic_plot._stitch_gif_with_ffmpeg") as mock_ffmpeg:
        draw_dynamic(str(tmp_path), mode="mean", out_gif=str(tmp_path / "out.gif"), window=3, spf=0.05)

    mock_ffmpeg.assert_called_once()
    mock_in_memory.assert_not_called()
