"""Tests for calibrate/lambda_sweep.py::lambda_sweep: the 'CALM calibrate lambda' CLI entry
point - argument parsing/validation and the printed report, built on top of
calibrate/lambda_calibration.py's already-covered pure logic (see test_lambda_calibration.py).
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from CALM.calibrate.lambda_sweep import lambda_sweep


def _write_two_leaflet_structure(
    path: Path, Lx: float = 1000.0, Ly: float = 1000.0, Lz: float = 100.0, spacing: float = 50.0,
) -> int:
    """Writes a .gro with two flat leaflets of 'P' beads; returns atoms-per-leaflet."""
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    n = xy.shape[0]
    positions = np.vstack([
        np.column_stack([xy, np.full(n, 70.0)]),
        np.column_stack([xy, np.full(n, 30.0)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
    u.atoms.write(str(path))
    return n


def _write_ndx(path: Path, n_per_leaflet: int) -> None:
    with open(path, "w") as f:
        f.write("[ Upper ]\n")
        f.write(" ".join(str(i) for i in range(1, n_per_leaflet + 1)) + "\n")
        f.write("[ Lower ]\n")
        f.write(" ".join(str(i) for i in range(n_per_leaflet + 1, 2 * n_per_leaflet + 1)) + "\n")


def test_structure_only_runs_and_recommends_a_lambda(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro)
    ndx = tmp_path / "index.ndx"
    _write_ndx(ndx, n)

    lambda_sweep(["-s", str(gro), "--index-file", str(ndx)])

    out = capsys.readouterr().out
    assert "Lambda calibration for" in out
    assert "no -f/--trajectory given" in out
    assert "lambda_x = lambda_y =" in out


def test_trajectory_with_varying_box_size_and_explicit_sampling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    Lx = Ly = 600.0
    spacing = 40.0
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    n = xy.shape[0]
    positions = np.vstack([
        np.column_stack([xy, np.full(n, 70.0)]),
        np.column_stack([xy, np.full(n, 30.0)]),
    ])

    n_frames = 6
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True, n_frames=n_frames)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    rng = np.random.default_rng(1)
    for i, ts in enumerate(u.trajectory):
        ts.positions = positions
        drift = rng.uniform(-20.0, 20.0)
        ts.dimensions = [Lx + drift, Ly + drift, 100.0, 90.0, 90.0, 90.0]

    gro = tmp_path / "structure.gro"
    xtc = tmp_path / "traj.xtc"
    u.atoms.write(str(gro))
    with mda.Writer(str(xtc), n_atoms=u.atoms.n_atoms) as w:
        for ts in u.trajectory:
            w.write(u.atoms)

    ndx = tmp_path / "index.ndx"
    _write_ndx(ndx, n)

    lambda_sweep([
        "-s", str(gro), "-f", str(xtc), "--index-file", str(ndx),
        "--n-frames", "3", "--seed", "0",
    ])

    out = capsys.readouterr().out
    assert "Sampled up to 3 frame(s)" in out
    assert "lambda_x = lambda_y =" in out


def test_lambda_candidates_override_takes_precedence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro)
    ndx = tmp_path / "index.ndx"
    _write_ndx(ndx, n)

    lambda_sweep([
        "-s", str(gro), "--index-file", str(ndx),
        "--lambda-candidates", "3.0", "6.0", "12.0",
        "--n-lambda", "5",  # a non-default grid flag, alongside --lambda-candidates
    ])

    out = capsys.readouterr().out
    assert "--lambda-candidates was given" in out

    # Every reported lambda value must come from the explicit candidate list
    # (possibly deduped, never anything from the default grid, which would
    # have included values well below 3.0 given --lambda-min=1.0).
    table_rows = [
        line for line in out.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    assert table_rows, "expected at least one candidate row in the report"
    for row in table_rows:
        lambda_value = float(row.split()[0])
        assert lambda_value in {3.0, 6.0, 12.0}


def test_missing_index_and_index_file_raises_system_exit(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    _write_two_leaflet_structure(gro)

    with pytest.raises(SystemExit):
        lambda_sweep(["-s", str(gro)])


def test_missing_structure_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        lambda_sweep(["-n", "name P"])
