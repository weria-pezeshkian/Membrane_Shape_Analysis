"""Tests for map/lipids_plot.py: a reserved entry point (registered and
reachable, rendering not yet implemented).
"""

from __future__ import annotations

import pytest

from CALM.map.lipids_plot import lipids_plot


def test_lipids_plot_raises_not_implemented(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        lipids_plot(["-i", str(tmp_path), "-o", str(tmp_path / "out.png")])
