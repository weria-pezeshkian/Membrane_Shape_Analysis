"""Tests for CALM analyze lipids: per-species lipid-composition assignment
(analyze/lipids.py's _assign_nearest_leaflet/_lipid_voronoi_fractions/
_true_surface_area/_one_lipid_frame, and core/headgroup.py's
_headgroup_centers), and the final trajectory-averaged area_per_lipid.csv
(_write_area_per_lipid_csv). Headgroup-detection internals
(_contract_rings/_terminal_arms/_headgroup_atoms_from_graph/_require_bonds)
have their own tests in test_headgroup.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np

from CALM.analyze import lipids as lipids_module
from CALM.analyze.lipids import (
    _assign_nearest_leaflet,
    _lipid_voronoi_fractions,
    _true_surface_area,
    _write_area_per_lipid_csv,
)
from CALM.core import fourier_build as fb
from CALM.core.fourier_core import Fourier_Series_Function


def test_assign_nearest_leaflet_splits_by_closer_fitted_surface() -> None:
    f_upper = Fourier_Series_Function(100.0, 100.0, 1, 1)
    f_upper.setAnm(np.zeros(f_upper.Anm.shape))  # flat surface z=0
    f_lower = Fourier_Series_Function(100.0, 100.0, 1, 1)
    f_lower.setAnm(np.zeros(f_lower.Anm.shape))

    # Both fitted surfaces are flat at z=0 (Anm all zero); a point above is
    # equidistant unless we offset the surfaces, so shift lower down instead.
    # The (i=0, j=0) constant term lives at Anm[Nx, Ny] (Anm is indexed by
    # i+Nx, j+Ny), not Anm[0, 0].
    f_lower.Anm[f_lower.Nx, f_lower.Ny] = -20.0  # constant term shifts the flat surface to z=-20

    xy = np.array([[10.0, 10.0], [10.0, 10.0]])
    z = np.array([5.0, -18.0])  # first is near upper (z=0), second near lower (z=-20)

    assignment = _assign_nearest_leaflet(xy, z, f_upper, f_lower)
    assert list(assignment) == [1, -1]


def _flat_fourier(Lx: float, Ly: float) -> Fourier_Series_Function:
    f = Fourier_Series_Function(Lx, Ly, 1, 1)
    f.setAnm(np.zeros(f.Anm.shape))
    return f


def test_lipid_voronoi_fractions_pure_species_point_gets_fraction_one() -> None:
    Lx = Ly = 100.0
    x = np.linspace(0, Lx, 40, endpoint=False)
    y = np.linspace(0, Ly, 40, endpoint=False)
    X, Y = np.meshgrid(x, y)
    f = _flat_fourier(Lx, Ly)

    # Dense POPC cluster around (25, 25), dense POPE cluster around (75, 75).
    popc_xy = np.array([[25.0 + dx, 25.0 + dy] for dx in range(-3, 4) for dy in range(-3, 4)], dtype=float)
    pope_xy = np.array([[75.0 + dx, 75.0 + dy] for dx in range(-3, 4) for dy in range(-3, 4)], dtype=float)

    fractions = _lipid_voronoi_fractions([popc_xy, pope_xy], X, Y, f, Lx, Ly)

    # A grid point deep in POPC's own territory: pure POPC.
    i = np.argmin(np.abs(x - 25.0))
    j = np.argmin(np.abs(y - 25.0))
    assert fractions[0, j, i] == 1.0
    assert fractions[1, j, i] == 0.0


def test_lipid_voronoi_fractions_boundary_point_assigned_to_nearer_species() -> None:
    Lx = Ly = 100.0
    x = np.linspace(0, Lx, 20, endpoint=False)
    y = np.linspace(0, Ly, 20, endpoint=False)
    X, Y = np.meshgrid(x, y)
    f = _flat_fourier(Lx, Ly)

    popc_xy = np.array([[40.0, 10.0 + i] for i in range(0, 40, 5)])
    pope_xy = np.array([[60.0, 10.0 + i] for i in range(0, 40, 5)])

    fractions = _lipid_voronoi_fractions([popc_xy, pope_xy], X, Y, f, Lx, Ly)

    # Points on either side of the x=50 midline: each fully assigned to the
    # nearer species (not a blend), and every point's own fractions sum to
    # exactly 1, since every grid point has some nearest lipid.
    j = np.argmin(np.abs(y - 20.0))
    i_left = np.argmin(np.abs(x - 45.0))
    i_right = np.argmin(np.abs(x - 55.0))
    assert fractions[0, j, i_left] == 1.0 and fractions[1, j, i_left] == 0.0
    assert fractions[1, j, i_right] == 1.0 and fractions[0, j, i_right] == 0.0
    assert fractions[:, j, i_left].sum() == 1.0


def test_lipid_voronoi_fractions_accounts_for_surface_curvature() -> None:
    Lx = Ly = 100.0
    x = np.linspace(0, Lx, 40, endpoint=False)
    y = np.linspace(0, Ly, 40, endpoint=False)
    X, Y = np.meshgrid(x, y)
    f = Fourier_Series_Function(Lx, Ly, 1, 1)
    Anm = np.zeros(f.Anm.shape)
    Anm[f.Nx + 1, f.Ny] = 15.0  # a genuine x-dependent wave, not a flat offset
    f.setAnm(Anm)

    # Two lipids equidistant from (50, 50) in flat XY but on opposite sides
    # in x, where the fitted surface's own height differs because of the
    # wave: a query point at its own fitted height sits closer, in 3D chord
    # terms, to whichever lipid's fitted height is nearer its own.
    query_z = f.Z(np.array([50.0]), np.array([50.0]))[0]
    left_z = f.Z(np.array([40.0]), np.array([50.0]))[0]
    right_z = f.Z(np.array([60.0]), np.array([50.0]))[0]
    assert left_z != right_z  # sanity: the wave genuinely differs left vs right

    species_left = np.array([[40.0, 50.0]])
    species_right = np.array([[60.0, 50.0]])

    fractions = _lipid_voronoi_fractions([species_left, species_right], X, Y, f, Lx, Ly)

    i = np.argmin(np.abs(x - 50.0))
    j = np.argmin(np.abs(y - 50.0))
    nearer_species = 0 if abs(left_z - query_z) < abs(right_z - query_z) else 1
    assert fractions[nearer_species, j, i] == 1.0
    assert fractions[1 - nearer_species, j, i] == 0.0


def test_true_surface_area_matches_flat_cell_area_for_a_flat_surface() -> None:
    f = Fourier_Series_Function(100.0, 100.0, 1, 1)
    f.setAnm(np.zeros(f.Anm.shape))  # perfectly flat: Zx = Zy = 0 everywhere

    x = np.linspace(0, 100.0, 10, endpoint=False)
    y = np.linspace(0, 100.0, 10, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cell_area = 10.0 * 10.0

    area = _true_surface_area(f, X, Y, cell_area)
    assert np.allclose(area, cell_area)


def _flat_lipid_universe() -> tuple[mda.Universe, mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """A flat membrane: upper fit selection at z=70, lower at z=30, POPC
    residues on the left half of the upper leaflet, POPE on the right half,
    and an unlisted CHOL residue in between - nothing assigned to the lower
    leaflet (leaflet-splitting itself is covered separately by
    test_assign_nearest_leaflet_...).
    """
    Lx = Ly = 100.0
    xs = np.linspace(2, 98, 10)
    ys = np.linspace(2, 98, 10)
    XX, YY = np.meshgrid(xs, ys)
    fit_xy = np.column_stack([XX.ravel(), YY.ravel()])
    fit_upper_pos = np.column_stack([fit_xy, np.full(len(fit_xy), 70.0)])
    fit_lower_pos = np.column_stack([fit_xy, np.full(len(fit_xy), 30.0)])

    popc_centers = np.array([[10.0, 20.0, 70.0], [10.0, 80.0, 70.0], [20.0, 50.0, 70.0]])
    pope_centers = np.array([[90.0, 20.0, 70.0], [90.0, 80.0, 70.0]])
    chol_centers = np.array([[50.0, 50.0, 70.0]])

    def two_atom_residues(centers: np.ndarray) -> np.ndarray:
        pos = []
        for c in centers:
            pos.append(c + np.array([0.5, 0.0, 0.0]))
            pos.append(c - np.array([0.5, 0.0, 0.0]))
        return np.array(pos)

    popc_pos = two_atom_residues(popc_centers)
    pope_pos = two_atom_residues(pope_centers)
    chol_pos = two_atom_residues(chol_centers)

    n_fit_upper, n_fit_lower = len(fit_upper_pos), len(fit_lower_pos)
    n_popc_res, n_pope_res, n_chol_res = len(popc_centers), len(pope_centers), len(chol_centers)

    # Each fit atom is its own residue (identity irrelevant); each lipid is
    # exactly one residue (its own 2 atoms).
    resindex = list(range(n_fit_upper + n_fit_lower))
    next_res = n_fit_upper + n_fit_lower
    for _ in range(n_popc_res):
        resindex += [next_res, next_res]
        next_res += 1
    for _ in range(n_pope_res):
        resindex += [next_res, next_res]
        next_res += 1
    for _ in range(n_chol_res):
        resindex += [next_res, next_res]
        next_res += 1

    resnames = (
        ["FITRES"] * (n_fit_upper + n_fit_lower)
        + ["POPC"] * n_popc_res + ["POPE"] * n_pope_res + ["CHOL"] * n_chol_res
    )
    names = (
        ["PO4"] * (n_fit_upper + n_fit_lower)
        + ["C1"] * len(popc_pos) + ["C2"] * len(pope_pos) + ["C3"] * len(chol_pos)
    )
    all_pos = np.vstack([fit_upper_pos, fit_lower_pos, popc_pos, pope_pos, chol_pos])

    u = mda.Universe.empty(
        n_atoms=len(all_pos), n_residues=next_res, atom_resindex=np.array(resindex), trajectory=True
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("masses", [72.0] * len(all_pos))
    u.atoms.positions = all_pos
    u.dimensions = [Lx, Ly, 60.0, 90.0, 90.0, 90.0]

    # Each lipid's own 2 atoms are bonded to each other (_headgroup_centers
    # requires bonds); fit-selection atoms need none, they never go through it.
    lipid_start = n_fit_upper + n_fit_lower
    n_lipid_atoms = len(popc_pos) + len(pope_pos) + len(chol_pos)
    u.add_bonds([(lipid_start + 2 * i, lipid_start + 2 * i + 1) for i in range(n_lipid_atoms // 2)])

    layer_group = u.atoms[:n_fit_upper]
    layer_group_2 = u.atoms[n_fit_upper:n_fit_upper + n_fit_lower]
    return u, layer_group, layer_group_2


def test_one_lipid_frame_flat_surface_area_per_lipid(tmp_path: Path) -> None:
    u, layer_group, layer_group_2 = _flat_lipid_universe()

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = layer_group
    fb._worker_state["layer_group_2"] = layer_group_2
    fb._worker_state["rotation_and_center"] = None
    try:
        result = lipids_module._one_lipid_frame(
            0,
            out_dir=str(tmp_path),
            species=["POPC", "POPE", "CHOL"],
            dynamic_select=False,
            dynamic_leaflets=None,
            until=1,
            Nx=5.0, Ny=5.0,
            sqrt_n_atoms=40,
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    assert result["frame"] == 0
    fractions = np.load(tmp_path / "0_lipid_fractions.npy")
    area_per_lipid = np.load(tmp_path / "0_area_per_lipid.npy")
    counts = np.load(tmp_path / "0_lipid_counts.npy")

    # species order: POPC=0, POPE=1, CHOL=2; leaflet order: upper=0, lower=1
    assert counts[0, 0] == 3  # 3 POPC residues, all upper
    assert counts[1, 0] == 2  # 2 POPE residues, all upper
    assert counts[2, 0] == 1  # 1 CHOL residue, upper - present in counts even though not "requested" beyond being listed
    assert (counts[:, 1] == 0).all()  # nothing assigned to the (empty) lower leaflet

    # On a flat surface, true (curved) area-per-lipid equals the flat area-per-lipid.
    assert np.allclose(area_per_lipid[:, 0, 0], area_per_lipid[:, 0, 1], atol=1e-6)

    # Total assigned area (summed fractions * cell area) should roughly match
    # count * area-per-lipid by construction (self-consistency, not a hand
    # count): every valid grid point's fractions sum to <=1 and area-per-lipid
    # is exactly that quotient, so re-multiplying must recover the same sum.
    cell_area = (100.0 / 40) * (100.0 / 40)
    total_popc_area = fractions[0, 0].sum() * cell_area
    assert np.isclose(total_popc_area, area_per_lipid[0, 0, 0] * counts[0, 0], atol=1e-3)


def _rotatable_lipid_universe() -> tuple[mda.Universe, mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """Like `_flat_lipid_universe`, plus a 2-frame trajectory and a CTR/DIR
    atom pair for --center/--rotation-direction: DIR sits due east of CTR
    in frame 0 and due north of CTR in frame 1, a clean, known 90-degree
    rotation between frames. CTR sits at the box center in both frames, so
    _center()'s own recentering shift is always exactly zero, isolating
    the rotation from any translation.
    """
    u, layer_group, layer_group_2 = _flat_lipid_universe()
    n = len(u.atoms)

    u2 = mda.Universe.empty(
        n_atoms=n + 2, n_residues=len(u.residues) + 2,
        atom_resindex=np.concatenate([u.atoms.resindices, [len(u.residues), len(u.residues) + 1]]),
        trajectory=True,
    )
    u2.add_TopologyAttr("name", list(u.atoms.names) + ["CTR", "DIR"])
    u2.add_TopologyAttr("resname", list(u.residues.resnames) + ["CTR", "DIR"])
    u2.add_TopologyAttr("masses", list(u.atoms.masses) + [72.0, 72.0])

    frame0 = np.vstack([u.atoms.positions, [[50.0, 50.0, 70.0], [60.0, 50.0, 70.0]]])  # DIR due east of CTR
    frame1 = np.vstack([u.atoms.positions, [[50.0, 50.0, 70.0], [50.0, 60.0, 70.0]]])  # DIR due north of CTR
    u2.load_new(np.stack([frame0, frame1]), order="fac")
    # Lz=100, not _flat_lipid_universe's own 60: _center() (called whenever a
    # real tracker exists, even before this test's baseline-vs-rotate
    # comparison gets to --rotate itself) wraps every atom into the box, and
    # z=70 positions need real headroom above Lx/Ly's own 100 to not wrap
    # around a shorter Lz.
    for ts in u2.trajectory:
        ts.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]

    if u.bonds:
        u2.add_bonds([(a.index, b.index) for a, b in ((bond.atoms[0], bond.atoms[1]) for bond in u.bonds)])

    layer_group_u2 = u2.atoms[layer_group.indices]
    layer_group_2_u2 = u2.atoms[layer_group_2.indices]
    return u2, layer_group_u2, layer_group_2_u2


def test_one_lipid_frame_rotate_leaves_area_per_lipid_unchanged_but_rotates_fractions(tmp_path: Path) -> None:
    u, layer_group, layer_group_2 = _rotatable_lipid_universe()

    common = dict(
        out_dir=str(tmp_path), species=["POPC", "POPE", "CHOL"], dynamic_select=False,
        dynamic_leaflets=None, until=2, Nx=5.0, Ny=5.0, sqrt_n_atoms=40, regularize=False,
    )

    # Baseline: a real tracker, but rotate=False - isolates --rotate's own
    # effect from _center()'s (always-on whenever --center is given at all).
    u.trajectory[0]
    tracker_plain = fb.Rotation_and_Center_tracker(u, "resname CTR", "resname DIR", rotate=False)
    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = layer_group
    fb._worker_state["layer_group_2"] = layer_group_2
    fb._worker_state["rotation_and_center"] = tracker_plain
    try:
        lipids_module._one_lipid_frame(1, **common)
    finally:
        fb._worker_state.clear()
    fractions_plain = np.load(tmp_path / "1_lipid_fractions.npy")
    area_plain = np.load(tmp_path / "1_area_per_lipid.npy")
    counts_plain = np.load(tmp_path / "1_lipid_counts.npy")

    # Rotating: same universe, a real Rotation_and_Center_tracker built
    # while the universe sits on frame 0 (its own base direction), then
    # _one_lipid_frame(1, ...) recomputes the current direction from
    # frame 1's own DIR/CTR positions - a genuine 90-degree difference.
    u.trajectory[0]
    tracker = fb.Rotation_and_Center_tracker(u, "resname CTR", "resname DIR", rotate=True)
    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = layer_group
    fb._worker_state["layer_group_2"] = layer_group_2
    fb._worker_state["rotation_and_center"] = tracker
    try:
        lipids_module._one_lipid_frame(1, **common)
    finally:
        fb._worker_state.clear()
    fractions_rotated = np.load(tmp_path / "1_lipid_fractions.npy")
    area_rotated = np.load(tmp_path / "1_area_per_lipid.npy")
    counts_rotated = np.load(tmp_path / "1_lipid_counts.npy")

    # area_per_lipid/counts: identical whether or not --rotate is used -
    # always built from each frame's own raw, unrotated grid.
    np.testing.assert_array_equal(area_plain, area_rotated)
    np.testing.assert_array_equal(counts_plain, counts_rotated)

    # lipid_fractions.npy: genuinely different - the saved spatial map is
    # queried at a rotated grid, not the plain one.
    assert not np.array_equal(fractions_plain, fractions_rotated)


def test_write_area_per_lipid_csv_averages_across_frames(tmp_path: Path) -> None:
    species = ["POPC", "POPE"]
    # frame 0: area 1.0/2.0 flat, 1.5/2.5 curved; counts 10/20
    np.save(tmp_path / "0_area_per_lipid.npy", np.array([[[1.0, 1.5], [0.0, 0.0]], [[2.0, 2.5], [0.0, 0.0]]]))
    np.save(tmp_path / "0_lipid_counts.npy", np.array([[10.0, 0.0], [20.0, 0.0]]))
    # frame 1: area 3.0/3.5 flat, 4.0/4.5 curved; counts 12/18
    np.save(tmp_path / "1_area_per_lipid.npy", np.array([[[3.0, 4.0], [0.0, 0.0]], [[3.5, 4.5], [0.0, 0.0]]]))
    np.save(tmp_path / "1_lipid_counts.npy", np.array([[12.0, 0.0], [18.0, 0.0]]))

    _write_area_per_lipid_csv(str(tmp_path), species, [0, 1], until=2)

    content = (tmp_path / "area_per_lipid.csv").read_text()
    lines = content.strip().splitlines()
    assert lines[0] == "leaflet,species,area_per_lipid_flat,area_per_lipid_curved,mean_count"
    # upper/POPC: flat mean(1.0,3.0)=2.0, curved mean(1.5,4.0)=2.75, count mean(10,12)=11.0
    assert "upper,POPC,2.000000,2.750000,11.00" in content
    # upper/POPE: flat mean(2.0,3.5)=2.75, curved mean(2.5,4.5)=3.5, count mean(20,18)=19.0
    assert "upper,POPE,2.750000,3.500000,19.00" in content
