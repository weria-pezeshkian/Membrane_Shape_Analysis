"""Tests for SFT.write / SFT.from_directory (the --sft <dir> load path)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from CALM.core.fourier_sft import SFT, _check_hole_mask_matches_frames


def make_sft(seed: int = 0, n_frames: int = 4) -> SFT:
    rng = np.random.default_rng(seed)
    s = SFT()
    s.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 5, 5))
    s.q_mn = rng.uniform(-1, 1, size=(n_frames, 2, 5, 5))
    s.frame_indices = np.arange(0, n_frames * 2, 2)  # e.g. Step=2 sampling
    s.dimensions = rng.uniform(50, 100, size=(n_frames, 3))
    return s


def test_write_from_directory_round_trip(tmp_path: Path) -> None:
    original = make_sft()
    original.write(tmp_path)

    loaded = SFT.from_directory(tmp_path)

    assert np.allclose(loaded.A_mn, original.A_mn)
    assert np.allclose(loaded.q_mn, original.q_mn)
    assert np.array_equal(loaded.frame_indices, original.frame_indices)
    assert np.allclose(loaded.dimensions, original.dimensions)


def test_from_directory_missing_all_files_raises_descriptive_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        SFT.from_directory(tmp_path)
    message = str(exc_info.value)
    assert "Amn.npy" in message
    assert "qmn.npy" in message
    assert "dimensions.npy" in message


def test_from_directory_missing_one_file_raises_descriptive_error(tmp_path: Path) -> None:
    original = make_sft()
    original.write(tmp_path)
    (tmp_path / "qmn.npy").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        SFT.from_directory(tmp_path)
    message = str(exc_info.value)
    assert "missing qmn.npy" in message


def test_hole_mask_round_trips_when_present(tmp_path: Path) -> None:
    original = make_sft()
    rng = np.random.default_rng(1)
    original.hole_mask = rng.uniform(size=(4, 2, 5, 5)) > 0.5  # (n_frames, upper/lower, gridsize, gridsize)
    original.write(tmp_path)

    assert (tmp_path / "holemask.npy").exists()

    loaded = SFT.from_directory(tmp_path)
    assert np.array_equal(loaded.hole_mask, original.hole_mask)


def test_hole_mask_absent_when_not_set(tmp_path: Path) -> None:
    original = make_sft()  # hole_mask left as None (default)
    original.write(tmp_path)

    assert not (tmp_path / "holemask.npy").exists()

    loaded = SFT.from_directory(tmp_path)
    assert loaded.hole_mask is None


def test_check_hole_mask_matches_frames_passes_when_identical() -> None:
    _check_hole_mask_matches_frames(np.array([0, 2, 4]), np.array([0, 2, 4]), "irrelevant")


def test_check_hole_mask_matches_frames_raises_naming_missing_and_stray_frames() -> None:
    # A_mn covers frames 0/2/4/6; hole_mask (stale, from an earlier,
    # differently-scoped build) covers 0/2/8 - 4/6 are missing a hole_mask
    # file, 8 is a stray one with no matching A_mn frame at all.
    with pytest.raises(ValueError) as exc_info:
        _check_hole_mask_matches_frames(np.array([0, 2, 4, 6]), np.array([0, 2, 8]), "some_dir")
    message = str(exc_info.value)
    assert "some_dir" in message
    assert "covers 3 frame(s)" in message and "covers 4" in message
    assert "Frames missing a hole_mask file: 4, 6" in message
    assert "Stray hole_mask file(s) with no matching frame: 8" in message


def test_build_raises_when_raw_sft_has_stale_hole_mask_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates raw_sft/ left over from an earlier, differently-scoped
    # --Remove-TMD build (e.g. interrupted after 1 frame) that a later,
    # complete rebuild of this same --out didn't clean out - calc_fourier
    # is mocked out since this only exercises SFT.build's own
    # post-calc_fourier consistency check, not the fit itself.
    raw_sft = tmp_path / "raw_sft"
    raw_sft.mkdir()
    for frame in (0, 1, 2, 3):
        np.save(raw_sft / f"{frame:04d}_A_mn.npy", np.zeros((3, 5, 5)))
        np.save(raw_sft / f"{frame:04d}_q_mn.npy", np.zeros((2, 5, 5)))
        np.save(raw_sft / f"{frame:04d}_dimensions.npy", np.array([80.0, 80.0, 100.0]))
    np.save(raw_sft / "0000_hole_mask.npy", np.zeros((2, 5, 5), dtype=bool))  # only frame 0, not 1/2/3

    monkeypatch.setattr("CALM.core.fourier_sft.calc_fourier", lambda args, universe: None)
    args = argparse.Namespace(out=str(tmp_path), regularize=False)

    with pytest.raises(ValueError) as exc_info:
        SFT().build(args, universe=None)  # type: ignore[arg-type]
    assert "hole_mask" in str(exc_info.value)


def test_from_directory_mismatched_holemask_row_count_raises(tmp_path: Path) -> None:
    # Simulates a stale holemask.npy (e.g. left over from an earlier,
    # differently-scoped build) sitting next to a freshly-written,
    # differently-sized Amn.npy/qmn.npy/dimensions.npy - write() itself
    # always keeps these aligned, so this bypasses it to write the mismatch
    # directly, the way a stale leftover file actually would occur.
    original = make_sft(n_frames=4)
    original.write(tmp_path)
    np.save(tmp_path / "holemask.npy", np.zeros((3, 2, 5, 5), dtype=bool))  # 3 frames, not 4

    with pytest.raises(ValueError) as exc_info:
        SFT.from_directory(tmp_path)
    message = str(exc_info.value)
    assert "holemask.npy" in message
    assert "3 frame(s)" in message
    assert "dimensions.npy" in message


def test_regularized_flag_round_trips_when_set(tmp_path: Path) -> None:
    original = make_sft()
    original.regularized = True
    original.write(tmp_path)

    assert (tmp_path / "regularized.npy").exists()

    loaded = SFT.from_directory(tmp_path)
    assert loaded.regularized is True


def test_regularized_flag_false_round_trips(tmp_path: Path) -> None:
    original = make_sft()
    original.regularized = False
    original.write(tmp_path)

    loaded = SFT.from_directory(tmp_path)
    assert loaded.regularized is False


def test_regularized_flag_absent_when_not_set(tmp_path: Path) -> None:
    original = make_sft()  # regularized left as None (default)
    original.write(tmp_path)

    assert not (tmp_path / "regularized.npy").exists()

    loaded = SFT.from_directory(tmp_path)
    assert loaded.regularized is None
