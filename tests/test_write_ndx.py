"""Tests for utilize/write_ndx.py::write(): the CLI/file-I/O wrapper around
core/leaflet.py's get_components() (see tests/test_leaflet.py for the
algorithm itself). Covers Upper/Lower labeling and the excluded-atom warning.
"""

from __future__ import annotations

import logging

import MDAnalysis as mda
import numpy as np
import pytest

from CALM.utilize.write_ndx import write


def _two_leaflet_universe_with_bridge(n_per_leaflet: int = 10, bridge: bool = True) -> mda.Universe:
    # xy positions kept tightly clustered (scale=3) around a shared center
    # so intra-leaflet distances stay small relative to the 40 A z-gap
    # between leaflets, representative of a real bilayer.
    rng = np.random.default_rng(2)
    Lx = Ly = Lz = 100.0
    upper = rng.normal(loc=[50.0, 50.0], scale=3.0, size=(n_per_leaflet, 2))
    lower = rng.normal(loc=[50.0, 50.0], scale=3.0, size=(n_per_leaflet, 2))
    positions = np.vstack([
        np.column_stack([upper, np.full(n_per_leaflet, 70.0)]),
        np.column_stack([lower, np.full(n_per_leaflet, 30.0)]),
    ])
    if bridge:
        positions = np.vstack([positions, [[50.0, 50.0, 50.0]]])

    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
    return u


def test_write_labels_upper_and_lower_by_z() -> None:
    u = _two_leaflet_universe_with_bridge(bridge=False)
    ndx, upper_index, lower_index = write(u, "all", write=False)
    assert len(upper_index) == 10
    assert len(lower_index) == 10
    # upper_index/lower_index (write=False path) are already 0-based global
    # atom indices - no -1 adjustment (that's only correct for the +1'd
    # 1-based indices written to a .ndx file, see write()'s `if write:` branch).
    upper_z = u.atoms[upper_index].positions[:, 2]
    lower_z = u.atoms[lower_index].positions[:, 2]
    assert upper_z.mean() > lower_z.mean()


def test_write_warns_and_excludes_bridging_atom(caplog: pytest.LogCaptureFixture) -> None:
    u = _two_leaflet_universe_with_bridge(bridge=True)
    with caplog.at_level(logging.WARNING, logger="CALM.utilize.write_ndx"):
        ndx, upper_index, lower_index = write(u, "all", write=False)

    assert len(upper_index) == 10
    assert len(lower_index) == 10
    assert any("excluded from both" in rec.message for rec in caplog.records)
