"""Tests for calibrate/enter.py::Calibrate: the CLI scaffold that loads a
built SFT and hands off to calibrate/calibrate.py::calibrate (not yet
implemented).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CALM.calibrate.enter import Calibrate
from CALM.core.fourier_sft import SFT


def _write_sft(d: Path, n_frames: int = 2, Nx: int = 2, Ny: int = 2) -> None:
    rng = np.random.default_rng(0)
    sft = SFT()
    sft.A_mn = rng.uniform(-1, 1, size=(n_frames, 3, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    sft.q_mn = rng.uniform(-1, 1, size=(n_frames, 2, 2 * Nx + 1, 2 * Ny + 1)).astype(np.float32)
    sft.frame_indices = np.arange(n_frames)
    sft.dimensions = np.tile([100.0, 80.0, 60.0], (n_frames, 1))
    sft.regularize = False
    sft.write(d)


def test_calibrate_loads_sft_then_raises_not_implemented(tmp_path: Path) -> None:
    _write_sft(tmp_path)
    with pytest.raises(NotImplementedError):
        Calibrate(["-i", str(tmp_path), "--radius", "5.0", "-o", str(tmp_path / "out.json")])


def test_calibrate_missing_sft_directory_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Calibrate(["-i", str(missing), "--radius", "5.0", "-o", str(tmp_path / "out.json")])


def test_calibrate_requires_all_three_flags() -> None:
    with pytest.raises(SystemExit):
        Calibrate(["--radius", "5.0"])
