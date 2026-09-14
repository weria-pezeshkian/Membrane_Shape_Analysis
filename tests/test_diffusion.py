"""Tests for core/diffusion.py: the surface-projection, segment-breaking,
multi-tau MSD, and diffusion-coefficient-fitting primitives behind
`CALM analyze diffusion`.
"""

from __future__ import annotations

import numpy as np

from CALM.core.diffusion import (
    _break_into_segments,
    _fit_diffusion_coefficient,
    _multi_tau_msd,
    _project_onto_surface,
    _selection_centers,
    _SurfaceAsInterp,
)
from CALM.core.fourier_core import Fourier_Series_Function


def _sinusoidal_surface(Lx: float = 100.0, Ly: float = 100.0, amplitude: float = 5.0) -> Fourier_Series_Function:
    f = Fourier_Series_Function(Lx, Ly, Nx=1, Ny=1)
    Anm = np.zeros(f.Anm.shape)
    Anm[2, 1] = amplitude  # m=1, n=0 term (cosine along x)
    f.setAnm(Anm)
    return f


def test_project_onto_surface_flat_surface_leaves_xy_unchanged() -> None:
    f = Fourier_Series_Function(100.0, 100.0, Nx=1, Ny=1)  # Anm all zero: Z == 0 everywhere
    interp = _SurfaceAsInterp(f)

    x_proj, y_proj, converged = _project_onto_surface(30.0, 40.0, 5.0, f, interp)

    assert converged
    assert np.isclose(x_proj, 30.0)
    assert np.isclose(y_proj, 40.0)


def test_project_onto_surface_recovers_known_point_from_above() -> None:
    f = _sinusoidal_surface()
    interp = _SurfaceAsInterp(f)

    x0, y0 = 20.0, 35.0
    z0 = float(f.Z(x0, y0))
    fx, fy = float(f.Zx(x0, y0)), float(f.Zy(x0, y0))
    n = np.array([-fx, -fy, 1.0])
    n /= np.linalg.norm(n)

    t0 = 2.5  # above the surface
    px, py, pz = x0 + t0 * n[0], y0 + t0 * n[1], z0 + t0 * n[2]

    x_proj, y_proj, converged = _project_onto_surface(px, py, pz, f, interp)

    assert converged
    assert abs(x_proj - x0) < 5e-2
    assert abs(y_proj - y0) < 5e-2


def test_project_onto_surface_recovers_known_point_from_below() -> None:
    f = _sinusoidal_surface()
    interp = _SurfaceAsInterp(f)

    x0, y0 = 60.0, 15.0
    z0 = float(f.Z(x0, y0))
    fx, fy = float(f.Zx(x0, y0)), float(f.Zy(x0, y0))
    n = np.array([-fx, -fy, 1.0])
    n /= np.linalg.norm(n)

    t0 = -2.5  # below the surface
    px, py, pz = x0 + t0 * n[0], y0 + t0 * n[1], z0 + t0 * n[2]

    x_proj, y_proj, converged = _project_onto_surface(px, py, pz, f, interp)

    assert converged
    assert abs(x_proj - x0) < 5e-2
    assert abs(y_proj - y0) < 5e-2


def test_break_into_segments_single_run_when_fully_assigned() -> None:
    leaflet = np.array([1, 1, 1, 1, 1])
    in_hole = np.zeros(5, dtype=bool)
    assert _break_into_segments(leaflet, in_hole) == [(0, 5)]


def test_break_into_segments_splits_at_leaflet_flip() -> None:
    leaflet = np.array([1, 1, 1, -1, -1])
    in_hole = np.zeros(5, dtype=bool)
    assert _break_into_segments(leaflet, in_hole) == [(0, 3), (3, 5)]


def test_break_into_segments_splits_at_hole_status_change() -> None:
    leaflet = np.array([1, 1, 1, 1, 1])
    in_hole = np.array([False, False, True, True, True])
    assert _break_into_segments(leaflet, in_hole) == [(0, 2), (2, 5)]


def test_break_into_segments_excludes_unassigned_frames() -> None:
    leaflet = np.array([1, 1, 0, 1, 1])
    in_hole = np.zeros(5, dtype=bool)
    assert _break_into_segments(leaflet, in_hole) == [(0, 2), (3, 5)]


def test_break_into_segments_empty_when_never_assigned() -> None:
    leaflet = np.zeros(4, dtype=int)
    in_hole = np.zeros(4, dtype=bool)
    assert _break_into_segments(leaflet, in_hole) == []


def test_multi_tau_msd_pools_across_segments() -> None:
    seg1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # 1 unit/frame in x
    seg2 = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])  # 1 unit/frame in y

    tau, msd, n_samples = _multi_tau_msd([seg1, seg2], dt=1.0, max_tau_fraction=1.0)

    assert np.allclose(tau, [1.0, 2.0])
    assert np.allclose(msd, [1.0, 4.0])
    assert np.array_equal(n_samples, [4, 2])


def test_multi_tau_msd_empty_segments_returns_empty_arrays() -> None:
    tau, msd, n_samples = _multi_tau_msd([], dt=1.0, max_tau_fraction=0.25)
    assert len(tau) == 0
    assert len(msd) == 0
    assert len(n_samples) == 0


def test_fit_diffusion_coefficient_recovers_known_D_from_random_walk() -> None:
    rng = np.random.default_rng(42)
    D_true = 5.0
    dt = 1.0
    n_steps = 2000
    sigma = np.sqrt(2 * D_true * dt)

    steps = rng.normal(0.0, sigma, size=(n_steps, 2))
    trajectory = np.cumsum(steps, axis=0)

    tau, msd, _ = _multi_tau_msd([trajectory], dt=dt, max_tau_fraction=0.1)
    D, D_stderr, r2, loglog_slope = _fit_diffusion_coefficient(
        tau, msd, fit_tau_min_fraction=0.1, fit_tau_max_fraction=0.8
    )

    assert abs(D - D_true) / D_true < 0.15
    assert r2 > 0.99
    assert 0.8 < loglog_slope < 1.2


def test_fit_diffusion_coefficient_flags_ballistic_motion_via_loglog_slope() -> None:
    tau = np.linspace(1.0, 10.0, 20)
    msd = tau ** 2  # ballistic: MSD ~ tau^2, not tau^1

    _, _, _, loglog_slope = _fit_diffusion_coefficient(tau, msd, fit_tau_min_fraction=0.0, fit_tau_max_fraction=1.0)

    assert loglog_slope > 1.8


def test_fit_diffusion_coefficient_empty_tau_returns_nans() -> None:
    D, D_stderr, r2, loglog_slope = _fit_diffusion_coefficient(
        np.empty(0), np.empty(0), fit_tau_min_fraction=0.1, fit_tau_max_fraction=0.5
    )
    assert np.isnan(D)
    assert np.isnan(D_stderr)
    assert np.isnan(r2)
    assert np.isnan(loglog_slope)


def _two_residue_universe():
    import MDAnalysis as mda

    positions = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],  # residue 0: COG (1, 0, 0)
        [0.0, 4.0, 0.0], [0.0, 6.0, 0.0],  # residue 1: COG (0, 5, 0)
    ])
    u = mda.Universe.empty(n_atoms=4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    u.add_TopologyAttr("name", ["A", "B", "A", "B"])
    u.atoms.positions = positions
    u.add_bonds([(0, 1), (2, 3)])
    return u


def test_selection_centers_returns_per_fragment_center_of_geometry() -> None:
    u = _two_residue_universe()
    xy, z = _selection_centers(u.atoms)

    assert xy.shape == (2, 2)
    assert np.allclose(xy[0], [1.0, 0.0])
    assert np.allclose(xy[1], [0.0, 5.0])
    assert np.allclose(z, [0.0, 0.0])


def test_selection_centers_empty_atomgroup_returns_empty_arrays() -> None:
    u = _two_residue_universe()
    xy, z = _selection_centers(u.atoms[:0])
    assert xy.shape == (0, 2)
    assert z.shape == (0,)
