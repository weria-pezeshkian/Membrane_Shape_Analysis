"""Tests for SFT.write / SFT.from_directory (the --sft <dir> load path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CALM.core.fourier_sft import SFT


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
