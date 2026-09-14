"""Tests for shape_operator_curvatures, and f/_thickness_root: the per-point
brentq root search for a ray-surface intersection along a local normal
(used by analyze/analyze.py to compute bilayer thickness).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from CALM.core.curvature import _thickness_root, nearest_surface_intersection, shape_operator_curvatures, surface_normal
from CALM.core.fourier_core import Fourier_Series_Function


def make_surface(
    Nx: int = 2, Ny: int = 2, Lx: float = 100.0, Ly: float = 80.0, seed: int = 3
) -> Fourier_Series_Function:
    rng = np.random.default_rng(seed)
    f = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    f.setAnm(rng.uniform(-1.0, 1.0, size=f.Anm.shape))
    return f


def test_shape_operator_curvature_self_consistency() -> None:
    f = make_surface()
    x = np.linspace(10, f.Lx - 10, 4)
    y = np.linspace(10, f.Ly - 10, 4)
    X, Y = np.meshgrid(x, y)

    H, K, k1, k2, dirs1, dirs2 = shape_operator_curvatures(f, X, Y)

    assert np.all(np.isfinite(H))
    assert np.allclose(H, 0.5 * (k1 + k2))
    assert np.allclose(K, k1 * k2)
    assert np.all(k1 >= k2 - 1e-12)  # convention: k1 is the larger eigenvalue
    assert dirs1.shape == X.shape + (3,)
    assert dirs2.shape == X.shape + (3,)


def test_shape_operator_curvature_directions_are_unit_tangent_vectors() -> None:
    f = make_surface()
    x = np.linspace(10, f.Lx - 10, 4)
    y = np.linspace(10, f.Ly - 10, 4)
    X, Y = np.meshgrid(x, y)

    _, _, _, _, dirs1, dirs2 = shape_operator_curvatures(f, X, Y)
    fx = f.Zx(X, Y)
    fy = f.Zy(X, Y)

    for dirs in (dirs1, dirs2):
        assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)
        # Tangency: (dx, dy, dz) must lie in the plane spanned by the surface's
        # own local tangent basis (1, 0, fx) and (0, 1, fy), i.e. dz == fx*dx + fy*dy.
        # Linear in (dx, dy, dz), so it holds after the unit-length renormalization too.
        assert np.allclose(dirs[..., 2], fx * dirs[..., 0] + fy * dirs[..., 1])


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
    # The true root sits at t=30, past t_max_base=5 - only the widened
    # brackets (10, 20, 40) can bracket it. A root found only via widening
    # is rejected (None): it's farther than a genuine nearby intersection
    # should be.
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


def _flat_fourier(Lx: float, Ly: float, z: float) -> Fourier_Series_Function:
    f = Fourier_Series_Function(Lx, Ly, 0, 0)  # Nx=Ny=0 -> Anm is just the DC (constant) term
    f.setAnm(np.array([[z]]))
    return f


def test_surface_normal_is_unit_length() -> None:
    f = make_surface()
    n = surface_normal(f, 30.0, 40.0)
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_surface_normal_matches_the_gradient_formula() -> None:
    f = make_surface()
    x, y = 30.0, 40.0
    n = surface_normal(f, x, y)
    zx = float(f.Zx(np.asarray(x), np.asarray(y)))
    zy = float(f.Zy(np.asarray(x), np.asarray(y)))
    expected = np.array([-zx, -zy, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(n, expected)


def test_surface_normal_flat_surface_points_straight_up() -> None:
    f = _flat_fourier(100.0, 80.0, z=25.0)
    assert np.allclose(surface_normal(f, 30.0, 40.0), [0.0, 0.0, 1.0])


def test_nearest_surface_intersection_finds_a_root_above_the_query_point() -> None:
    Lx = Ly = 100.0
    f = _flat_fourier(Lx, Ly, z=25.0)
    t = nearest_surface_intersection(f, 50.0, 50.0, 20.0, 0.0, 0.0, 1.0, Lx, Ly, t_max_base=10.0)
    assert t is not None
    assert np.isclose(t, 5.0)  # 20 + 5*1 = 25


def test_nearest_surface_intersection_finds_a_root_below_the_query_point() -> None:
    # Unlike _thickness_root (which needs an `upper` flag to know which
    # direction to search), this tries both directions itself - the same
    # (0.0, 0.0, 1.0) normal finds a root whether the surface is above or
    # below the query point.
    Lx = Ly = 100.0
    f = _flat_fourier(Lx, Ly, z=15.0)
    t = nearest_surface_intersection(f, 50.0, 50.0, 20.0, 0.0, 0.0, 1.0, Lx, Ly, t_max_base=10.0)
    assert t is not None
    assert np.isclose(t, -5.0)  # 20 + (-5)*1 = 15


def test_nearest_surface_intersection_returns_none_when_unreachable() -> None:
    Lx = Ly = 100.0
    f = _flat_fourier(Lx, Ly, z=5000.0)
    t = nearest_surface_intersection(f, 50.0, 50.0, 20.0, 0.0, 0.0, 1.0, Lx, Ly, t_max_base=5.0)
    assert t is None
