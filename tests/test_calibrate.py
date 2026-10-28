"""Tests for calibrate/enter.py::Calibrate: the CLI entry point that loads a
built SFT and hands off to calibrate/calibrate.py::calibrate.
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
    sft.regularized = False
    sft.write(d)


def test_calibrate_loads_sft_and_writes_calibration_output(tmp_path: Path) -> None:
    _write_sft(tmp_path)
    out_base = tmp_path / "out"
    Calibrate([
        "-i", str(tmp_path), "--radius", "5.0",
        "--startframe", "0", "--endframe", "2",
        "-o", str(out_base),
    ])
    assert out_base.with_suffix(".npy").exists()
    assert out_base.with_suffix(".txt").exists()


def test_calibrate_missing_sft_directory_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Calibrate([
            "-i", str(missing), "--radius", "5.0",
            "--startframe", "0", "--endframe", "2",
            "-o", str(tmp_path / "out"),
        ])


def test_calibrate_requires_its_other_flags() -> None:
    with pytest.raises(SystemExit):
        Calibrate(["--radius", "5.0"])
