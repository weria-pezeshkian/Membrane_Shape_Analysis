"""Tests for analyze/analyze.py's _thickness_root: the per-point brentq
root search used to compute bilayer thickness.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from CALM.analyze.analyze import _thickness_root


def _flat_interp(Lx: float, Ly: float, z: float) -> RectBivariateSpline:
    y = np.linspace(0, Ly, 5)
    x = np.linspace(0, Lx, 5)
    return RectBivariateSpline(y, x, np.full((5, 5), z))


def test_thickness_root_finds_a_root_within_t_max_base() -> None:
    Lx = Ly = 100.0
    interp = _flat_interp(Lx, Ly, z=25.0)  # surface at z=25

    root = _thickness_root(
        interp, mx=50.0, my=50.0, mz=20.0, nx=0.0, ny=0.0, nz=1.0,
        Lx=Lx, Ly=Ly, t_max_base=10.0, upper=True,
    )

    assert root is not None
    assert np.isclose(root, 5.0)  # 20 + 5*1 = 25


def test_thickness_root_rejects_a_root_only_found_via_widening() -> None:
    # The true root sits at t=30, well past t_max_base=5 - only the
    # widened brackets (10, 20, 40) can bracket it. It must be rejected
    # (None) rather than returned, since it's farther than a genuine
    # nearby leaflet intersection should be.
    Lx = Ly = 100.0
    interp = _flat_interp(Lx, Ly, z=50.0)

    root = _thickness_root(
        interp, mx=50.0, my=50.0, mz=20.0, nx=0.0, ny=0.0, nz=1.0,
        Lx=Lx, Ly=Ly, t_max_base=5.0, upper=True,
    )

    assert root is None


def test_thickness_root_returns_none_when_no_root_exists_at_any_width() -> None:
    Lx = Ly = 100.0
    interp = _flat_interp(Lx, Ly, z=5000.0)  # unreachable even after 3 widenings

    root = _thickness_root(
        interp, mx=50.0, my=50.0, mz=20.0, nx=0.0, ny=0.0, nz=1.0,
        Lx=Lx, Ly=Ly, t_max_base=5.0, upper=True,
    )

    assert root is None


def test_thickness_root_lower_branch_searches_negative_t() -> None:
    Lx = Ly = 100.0
    interp = _flat_interp(Lx, Ly, z=15.0)

    root = _thickness_root(
        interp, mx=50.0, my=50.0, mz=20.0, nx=0.0, ny=0.0, nz=1.0,
        Lx=Lx, Ly=Ly, t_max_base=10.0, upper=False,
    )

    assert root is not None
    assert np.isclose(root, -5.0)  # 20 + (-5)*1 = 15
