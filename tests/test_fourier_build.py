"""Tests for the --Remove-TMD hole-detection helpers in core/fourier_build.py:
- _tmd_threshold: the Nyquist (half-wavelength) threshold formula
- _hole_mask_for_layer: periodic nearest-atom distance masking
- _close_enclosed_gaps: periodic-boundary-aware enclosed-region closing
- _one_frame's dynamic_select path
- _one_frame's remove_tmd selection resolution: bare (True) falls back to
  --center's selection, a string uses that selection directly with no
  --center required
- _fetch_dynamic_positions / _track_dynamic_leaflets: the two-phase dynamic
  leaflet detection pipeline (parallel position-fetch + sequential,
  history-aware leaflet tracking)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np

from CALM.core import fourier_build as fb
from CALM.core.fourier_build import (
    _close_enclosed_gaps,
    _hole_mask_for_layer,
    _tmd_protein_atoms_xy,
    _tmd_threshold,
    _track_dynamic_leaflets,
)
from CALM.core.fourier_core import Fourier_Series_Function


def test_tmd_threshold_is_half_the_shortest_wavelength() -> None:
    # Lx/Nx = 300/30 = 10 A (the fit's shortest representable wavelength);
    # Nyquist spacing is half that.
    assert _tmd_threshold(Lx=300.0, Ly=300.0, Nx=30, Ny=30) == 5.0


def test_tmd_threshold_uses_the_finer_of_x_and_y() -> None:
    # x is coarser (Lx/Nx=20 -> half=10), y is finer (Ly/Ny=8 -> half=4):
    # threshold should be the tighter (smaller) of the two half-wavelengths.
    assert _tmd_threshold(Lx=200.0, Ly=80.0, Nx=10, Ny=10) == 4.0


def test_close_enclosed_gaps_fills_an_interior_island() -> None:
    hole = np.zeros((10, 10), dtype=bool)
    hole[2:8, 2:8] = True
    hole[4:6, 4:6] = False  # non-hole island, fully surrounded

    result = _close_enclosed_gaps(hole)
    assert result[4:6, 4:6].all()


def test_close_enclosed_gaps_fills_an_island_touching_the_array_edge() -> None:
    # The island touches (0,0); its periodic neighbor across the wraparound
    # is also hole, so it is enclosed all the way around. Sized to close
    # fully at the conservative default of 1 iteration (a single cell or
    # small gap; a wider one may only close partially at this setting).
    hole = np.ones((10, 10), dtype=bool)
    hole[0:2, 0:2] = False

    result = _close_enclosed_gaps(hole)
    assert result[0:2, 0:2].all()


def test_close_enclosed_gaps_wider_gap_needs_more_iterations() -> None:
    # A 3x3 gap only partially closes at the conservative default of 1
    # iteration, but fully closes once given enough iterations to bridge it.
    hole = np.ones((10, 10), dtype=bool)
    hole[0:3, 0:3] = False

    assert not _close_enclosed_gaps(hole, iterations=1)[0:3, 0:3].all()
    assert _close_enclosed_gaps(hole, iterations=3)[0:3, 0:3].all()


def test_close_enclosed_gaps_leaves_a_periodically_open_strip_alone() -> None:
    # Non-hole columns at both the left and right edges are periodically
    # adjacent, forming one connected strip that reaches outside any single
    # period - not enclosed.
    hole = np.ones((10, 10), dtype=bool)
    hole[:, 0:2] = False
    hole[:, 8:10] = False

    result = _close_enclosed_gaps(hole)
    assert not result[:, 0:2].any()
    assert not result[:, 8:10].any()


def _flat_surface(Lx: float, Ly: float, z: float) -> Fourier_Series_Function:
    f = Fourier_Series_Function(Lx, Ly, 0, 0)  # Nx=Ny=0 -> Anm is just the DC (constant) term
    f.setAnm(np.array([[z]]))
    return f


def test_tmd_protein_atoms_xy_keeps_only_atoms_between_the_two_surfaces() -> None:
    Lx = Ly = 100.0
    upper = _flat_surface(Lx, Ly, 70.0)
    lower = _flat_surface(Lx, Ly, 30.0)

    u = mda.Universe.empty(n_atoms=3, trajectory=True)
    u.add_TopologyAttr("name", ["BB", "BB", "BB"])
    u.atoms.positions = [
        [50.0, 50.0, 50.0],  # between the leaflets -> in the TMD
        [50.0, 50.0, 90.0],  # above the upper leaflet -> soluble domain, excluded
        [50.0, 50.0, 10.0],  # below the lower leaflet -> soluble domain, excluded
    ]

    xy = _tmd_protein_atoms_xy("name BB", u, upper, lower)
    assert xy.shape == (1, 2)
    assert np.allclose(xy[0], [50.0, 50.0])


def test_tmd_protein_atoms_xy_empty_selection_returns_empty_array() -> None:
    Lx = Ly = 100.0
    upper = _flat_surface(Lx, Ly, 70.0)
    lower = _flat_surface(Lx, Ly, 30.0)

    u = mda.Universe.empty(n_atoms=1, trajectory=True)
    u.add_TopologyAttr("name", ["P"])
    u.atoms.positions = [[50.0, 50.0, 50.0]]

    xy = _tmd_protein_atoms_xy("name BB", u, upper, lower)  # no atom named BB
    assert xy.shape == (0, 2)


def test_one_frame_catches_tmd_gap_at_coarse_lambda_via_spacing_floor(tmp_path: Path) -> None:
    # Nyquist alone (lambda=5nm -> Nx=6 -> half-wavelength 25 A) is too loose
    # to flag a modest 15 A-radius gap; the spacing-based floor (tied to the
    # ~8 A real lipid grid spacing here) is what catches it. Lower leaflet is
    # fully dense (no gap) as a control. Positions are jittered, since a
    # perfectly regular grid has an inherent "half the cells above their own
    # median" pattern from grid geometry alone that a k*median threshold
    # would react to (see test_one_frame_does_not_flag_fully_dense_disordered_leaflet_as_holes
    # for the disordered-but-dense control). A handful of "protein" atoms sit
    # inside the gap, embedded in the membrane (z between the two leaflets),
    # required for the gate (--Remove-TMD requires --center; a hole must be
    # both lipid-unsupported AND spatially plausible as protein-displaced).
    Lx = Ly = Lz = 300.0
    rng = np.random.default_rng(1)
    spacing = 8.0
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    pts = np.column_stack([xs.ravel(), ys.ravel()]) + rng.normal(0, 1.0, size=(xs.size, 2))
    pts = np.mod(pts, [Lx, Ly])
    center = np.array([Lx / 2, Ly / 2])
    keep = np.linalg.norm(pts - center, axis=1) > 15.0

    upper_xy = pts[keep]  # gap present
    lower_xy = pts  # fully dense, no gap
    protein_xy = center + np.array([[0.0, 0.0], [3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    protein_z = np.full(len(protein_xy), 50.0)  # between upper (70) and lower (30)

    positions = np.vstack([
        np.column_stack([upper_xy, np.full(len(upper_xy), 70.0)]),
        np.column_stack([lower_xy, np.full(len(lower_xy), 30.0)]),
        np.column_stack([protein_xy, protein_z]),
    ])
    names = ["P"] * (len(upper_xy) + len(lower_xy)) + ["BB"] * len(protein_xy)
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", names)
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = u.atoms[: len(upper_xy)]
    fb._worker_state["layer_group_2"] = u.atoms[len(upper_xy):len(upper_xy) + len(lower_xy)]
    fb._worker_state["rotation_and_center"] = SimpleNamespace(
        sel1="name BB", rotate=False, _center=lambda: None
    )
    try:
        result = fb._one_frame(
            0,
            out_dir=str(tmp_path),
            dynamic_select=False,
            dynamic_leaflets=None,
            until=1,
            Nx=5.0, Ny=5.0,  # lambda_x/lambda_y (nm) -> Nx=Ny=6 for Lx=Ly=300 A
            sqrt_n_atoms=60,
            remove_tmd=True,
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    hole_mask = np.load(tmp_path / "raw_sft" / "0_hole_mask.npy")
    grid = np.linspace(0, Lx, 60, endpoint=False)
    center_idx = np.argmin(np.abs(grid - Lx / 2))

    assert hole_mask[0, center_idx, center_idx]  # upper: gap correctly flagged

    # Returned hole_stats give calc_fourier something to log; totals must
    # match what actually got saved.
    stats = result["hole_stats"]
    assert stats["n_grid"] == hole_mask[0].size
    assert stats["upper"]["total"] == int(hole_mask[0].sum())
    assert stats["lower"]["total"] == int(hole_mask[1].sum())
    # lower: fully dense at the same xy as upper's gap and protein, so a
    # hole flagged there is a false positive.
    far_from_gap = np.hypot(*(np.meshgrid(grid, grid) - np.array([Lx / 2, Ly / 2])[:, None, None])) > 60
    assert not hole_mask[1][far_from_gap].any()


def test_one_frame_uses_its_own_tmd_selection_with_no_center_at_all(tmp_path: Path) -> None:
    # --Remove-TMD given its own selection (a string, not True) identifies
    # protein atoms from that selection directly and works with
    # rotation_and_center being None, decoupling the gate from --center -
    # useful for multiple, disconnected transmembrane regions where a
    # single --center selection would be ambiguous to center on.
    Lx = Ly = Lz = 300.0
    rng = np.random.default_rng(1)
    spacing = 8.0
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    pts = np.column_stack([xs.ravel(), ys.ravel()]) + rng.normal(0, 1.0, size=(xs.size, 2))
    pts = np.mod(pts, [Lx, Ly])
    center = np.array([Lx / 2, Ly / 2])
    keep = np.linalg.norm(pts - center, axis=1) > 15.0

    upper_xy = pts[keep]
    lower_xy = pts
    protein_xy = center + np.array([[0.0, 0.0], [3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    protein_z = np.full(len(protein_xy), 50.0)

    positions = np.vstack([
        np.column_stack([upper_xy, np.full(len(upper_xy), 70.0)]),
        np.column_stack([lower_xy, np.full(len(lower_xy), 30.0)]),
        np.column_stack([protein_xy, protein_z]),
    ])
    names = ["P"] * (len(upper_xy) + len(lower_xy)) + ["BB"] * len(protein_xy)
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", names)
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = u.atoms[: len(upper_xy)]
    fb._worker_state["layer_group_2"] = u.atoms[len(upper_xy):len(upper_xy) + len(lower_xy)]
    fb._worker_state["rotation_and_center"] = None
    try:
        result = fb._one_frame(
            0,
            out_dir=str(tmp_path),
            dynamic_select=False,
            dynamic_leaflets=None,
            until=1,
            Nx=5.0, Ny=5.0,
            sqrt_n_atoms=60,
            remove_tmd="name BB",
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    hole_mask = np.load(tmp_path / "raw_sft" / "0_hole_mask.npy")
    grid = np.linspace(0, Lx, 60, endpoint=False)
    center_idx = np.argmin(np.abs(grid - Lx / 2))

    assert hole_mask[0, center_idx, center_idx]  # gap correctly flagged via the separate selection
    stats = result["hole_stats"]
    assert stats["upper"]["total"] == int(hole_mask[0].sum())


def test_one_frame_does_not_flag_fully_dense_disordered_leaflet_as_holes(tmp_path: Path) -> None:
    # A fully dense, disordered (jittered, not perfectly regular) leaflet
    # with no real gap must not be over-flagged: calibrating the spacing
    # threshold against the wrong distance kind (atom-to-atom instead of
    # grid-point-to-nearest-atom) would over-flag normal packing, since
    # grid points can legitimately fall in the gaps between tightly packed
    # atoms even with no hole present.
    Lx = Ly = Lz = 300.0
    rng = np.random.default_rng(0)
    spacing = 8.0
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    pts = np.column_stack([xs.ravel(), ys.ravel()]) + rng.normal(0, 1.0, size=(xs.size, 2))
    pts = np.mod(pts, [Lx, Ly])

    positions = np.vstack([
        np.column_stack([pts, np.full(len(pts), 70.0)]),
        np.column_stack([pts, np.full(len(pts), 30.0)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = u.atoms[: len(pts)]
    fb._worker_state["layer_group_2"] = u.atoms[len(pts):]
    # No protein atoms at all: with the gate in place, an empty --center
    # selection means nothing can ever pass the "spatially plausible as
    # protein-displaced" half of the test, so this also exercises that an
    # empty _tmd_protein_atoms_xy match doesn't crash.
    fb._worker_state["rotation_and_center"] = SimpleNamespace(
        sel1="name BB", rotate=False, _center=lambda: None
    )
    try:
        fb._one_frame(
            0,
            out_dir=str(tmp_path),
            dynamic_select=False,
            dynamic_leaflets=None,
            until=1,
            Nx=5.0, Ny=5.0,
            sqrt_n_atoms=60,
            remove_tmd=True,
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    hole_mask = np.load(tmp_path / "raw_sft" / "0_hole_mask.npy")
    # Fully dense, no real gap anywhere: flagged fraction should be small,
    # nowhere close to "almost everything".
    assert hole_mask[0].mean() < 0.05
    assert hole_mask[1].mean() < 0.05


def test_one_frame_flags_a_gap_far_enough_from_lipid_with_no_protein_present(tmp_path: Path) -> None:
    # A void much larger than normal packing irregularity gets flagged even
    # with no protein atoms anywhere in the system, via the far_threshold
    # fallback (dist > far_threshold), independent of the near-protein gate.
    Lx = Ly = Lz = 300.0
    rng = np.random.default_rng(2)
    spacing = 8.0
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    pts = np.column_stack([xs.ravel(), ys.ravel()]) + rng.normal(0, 1.0, size=(xs.size, 2))
    pts = np.mod(pts, [Lx, Ly])
    center = np.array([Lx / 2, Ly / 2])
    keep = np.linalg.norm(pts - center, axis=1) > 40.0

    upper_xy = pts[keep]
    lower_xy = pts

    positions = np.vstack([
        np.column_stack([upper_xy, np.full(len(upper_xy), 70.0)]),
        np.column_stack([lower_xy, np.full(len(lower_xy), 30.0)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = u.atoms[: len(upper_xy)]
    fb._worker_state["layer_group_2"] = u.atoms[len(upper_xy):]
    fb._worker_state["rotation_and_center"] = SimpleNamespace(
        sel1="name BB", rotate=False, _center=lambda: None
    )
    try:
        fb._one_frame(
            0,
            out_dir=str(tmp_path),
            dynamic_select=False,
            dynamic_leaflets=None,
            until=1,
            Nx=5.0, Ny=5.0,
            sqrt_n_atoms=60,
            remove_tmd=True,
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    hole_mask = np.load(tmp_path / "raw_sft" / "0_hole_mask.npy")
    grid = np.linspace(0, Lx, 60, endpoint=False)
    center_idx = np.argmin(np.abs(grid - Lx / 2))

    assert hole_mask[0, center_idx, center_idx]  # gap flagged with no protein anywhere
    assert hole_mask[0].mean() < 0.05  # rest of the leaflet stays clean


def _atoms(positions: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(positions=np.asarray(positions, dtype=float))


def test_hole_mask_flags_region_with_no_nearby_atoms() -> None:
    Lx = Ly = 100.0
    # Dense grid of atoms covering the box, spaced 5 A apart, except a
    # circular gap of radius 20 A around the center, mimicking a TMD.
    xs, ys = np.meshgrid(np.arange(0, Lx, 5.0), np.arange(0, Ly, 5.0))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    center = np.array([Lx / 2, Ly / 2])
    keep = np.linalg.norm(pts - center, axis=1) > 20.0
    layer_group = _atoms(np.column_stack([pts[keep], np.zeros(keep.sum())]))

    grid = np.linspace(0, Lx, 21, endpoint=False)
    X, Y = np.meshgrid(grid, grid)

    # Threshold well below the real inter-atom spacing (5 A -> half-diagonal
    # ~3.5 A) but well above what any point strictly inside the gap sees.
    mask = _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold=4.0)

    center_idx = np.argmin(np.abs(grid - Lx / 2))
    assert mask[center_idx, center_idx]  # inside the gap -> hole
    assert not mask[0, 0]  # densely covered corner -> not a hole


def test_hole_mask_respects_periodic_boundary() -> None:
    Lx = Ly = 100.0
    # Single atom right at the corner (0,0). A query point near the OPPOSITE
    # corner (99,99) is close to (0,0) only through periodic wraparound.
    layer_group = _atoms([[0.5, 0.5, 0.0]])

    X = np.array([[99.0]])
    Y = np.array([[99.0]])

    # True (non-periodic) distance from (99,99) to (0.5,0.5) would be ~139 A;
    # periodic distance is ~2.1 A.
    mask_tight = _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold=5.0)
    mask_loose_but_still_periodic = _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold=1.0)

    assert not mask_tight[0, 0]  # within periodic distance -> not a hole
    assert mask_loose_but_still_periodic[0, 0]  # periodic distance exceeds this tighter threshold


# ---- _init_worker: the one real-file-backed entry point other tests bypass ----

def test_init_worker_builds_static_leaflets_from_a_real_ndx_group(tmp_path: Path) -> None:
    # Every other test sets _worker_state directly and never calls
    # _init_worker, so this is the only coverage for its static-ndx branch
    # (assert ndx_groups is not None) - and the only test that round-trips
    # through a real on-disk structure file rather than an in-memory Universe.
    u = mda.Universe.empty(n_atoms=2, trajectory=True)
    u.add_TopologyAttr("name", ["P", "P"])
    u.atoms.positions = [[10.0, 10.0, 70.0], [20.0, 20.0, 30.0]]
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    gro_path = tmp_path / "structure.gro"
    u.atoms.write(str(gro_path))

    try:
        fb._init_worker(
            structure=str(gro_path),
            trajectory=str(gro_path),
            ndx_groups={"Upper": [1], "Lower": [2]},  # 1-based, as in a real .ndx file
            dynamic_select=False,
            center=None,
            rotation_direction=None,
            rotate=False,
        )
        layer_group = fb._worker_state["layer_group"]
        layer_group_2 = fb._worker_state["layer_group_2"]
        assert layer_group.positions[0, 2] == 70.0
        assert layer_group_2.positions[0, 2] == 30.0
    finally:
        fb._worker_state.clear()


def test_init_worker_leaves_leaflets_unset_for_dynamic_selection(tmp_path: Path) -> None:
    u = mda.Universe.empty(n_atoms=2, trajectory=True)
    u.add_TopologyAttr("name", ["P", "P"])
    u.atoms.positions = [[10.0, 10.0, 70.0], [20.0, 20.0, 30.0]]
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    gro_path = tmp_path / "structure.gro"
    u.atoms.write(str(gro_path))

    try:
        fb._init_worker(
            structure=str(gro_path),
            trajectory=str(gro_path),
            ndx_groups=None,
            dynamic_select=True,
            center=None,
            rotation_direction=None,
            rotate=False,
        )
        assert fb._worker_state["layer_group"] is None
        assert fb._worker_state["layer_group_2"] is None
    finally:
        fb._worker_state.clear()


def test_calc_fourier_runs_end_to_end_with_a_dynamic_selection(tmp_path: Path) -> None:
    # calc_fourier itself has no other test coverage - this exercises its
    # real ProcessPoolExecutor path end to end, including the
    # dynamic-selection branch's assert dynamic_selection is not None.
    xs, ys = np.meshgrid(np.arange(0, 100.0, 10.0), np.arange(0, 100.0, 10.0))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    n = xy.shape[0]
    positions = np.vstack([
        np.column_stack([xy, np.full(n, 70.0)]),
        np.column_stack([xy, np.full(n, 30.0)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]

    gro_path = tmp_path / "structure.gro"
    u.atoms.write(str(gro_path))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    args = SimpleNamespace(
        Until=None, index="name P", From=0, Step=1, Workers=1,
        structure=str(gro_path), trajectory=str(gro_path),
        center=None, rotation_direction=None, rotate=False,
        out=str(out_dir), lambda_x=None, lambda_y=None, gridsize=10,
        remove_tmd=False, regularize=False, min_balance=0.6, margin=2.0,
    )

    fb.calc_fourier(args, mda.Universe(str(gro_path)))

    assert (out_dir / "raw_sft" / "0_A_mn.npy").exists()
    assert (out_dir / "raw_sft" / "0_q_mn.npy").exists()
    assert (out_dir / "raw_sft" / "0_dimensions.npy").exists()


# ---- _one_frame's dynamic_select path ----

def _two_leaflet_universe(
    Lx: float = 100.0, Ly: float = 100.0, Lz: float = 100.0,
    upper_z: float = 70.0, lower_z: float = 30.0, spacing: float = 10.0,
) -> tuple[mda.Universe, int]:
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    n = xy.shape[0]
    positions = np.vstack([
        np.column_stack([xy, np.full(n, upper_z)]),
        np.column_stack([xy, np.full(n, lower_z)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
    return u, n


def test_one_frame_dynamic_select_runs_without_error(tmp_path: Path) -> None:
    # Exercises _one_frame's consumption of a precomputed dynamic_leaflets
    # dict (as calc_fourier's sequential pre-pass produces): the indices
    # there are already 0-based global atom indices, used directly (no -1
    # adjustment, unlike the static ndx-file path).
    universe, n_per_leaflet = _two_leaflet_universe()
    upper_index = list(range(n_per_leaflet))
    lower_index = list(range(n_per_leaflet, 2 * n_per_leaflet))

    fb._worker_state["universe"] = universe
    fb._worker_state["layer_group"] = None
    fb._worker_state["layer_group_2"] = None
    fb._worker_state["rotation_and_center"] = None
    try:
        fb._one_frame(
            0,
            out_dir=str(tmp_path),
            dynamic_select=True,
            dynamic_leaflets={0: (upper_index, lower_index)},
            until=1,
            Nx=8.0, Ny=8.0,  # lambda_x/lambda_y (nm) -> Nx=Ny=1 for Lx=Ly=100 A
            sqrt_n_atoms=10,
            remove_tmd=False,
            regularize=False,
        )
    finally:
        fb._worker_state.clear()

    raw_dir = tmp_path / "raw_sft"
    assert (raw_dir / "0_A_mn.npy").exists()
    assert (raw_dir / "0_q_mn.npy").exists()
    A_mn = np.load(raw_dir / "0_A_mn.npy")
    assert A_mn.shape[0] == 3  # upper, lower, middle


# ---- two-phase dynamic leaflet detection pipeline ----

def test_fetch_dynamic_positions_returns_this_frames_positions_and_box(tmp_path: Path) -> None:
    universe, n_per_leaflet = _two_leaflet_universe()
    fb._worker_state["universe"] = universe
    try:
        frame, positions, dimensions = fb._fetch_dynamic_positions(0, dynamic_selection="all")
    finally:
        fb._worker_state.clear()

    assert frame == 0
    assert positions.shape == (2 * n_per_leaflet, 3)
    assert np.allclose(dimensions[:3], [100.0, 100.0, 100.0])


def test_track_dynamic_leaflets_follows_a_flip_flopping_atom_across_frames() -> None:
    # Frame 0: two 3x3 grids (spacing 1 A, densely/redundantly connected so
    # no single atom is load-bearing for its own group's connectivity), one
    # at z=70 (upper, atoms 0-8), one at z=30 (lower, atoms 9-17).
    # Frame 1: atom 0 (an upper corner) moves down to join the lower grid -
    # a flip-flop. Tracking should move it to lower without disturbing
    # anyone else's assignment.
    grid = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
    upper_xy = grid + [50.0, 50.0]
    lower_xy = grid + [50.0, 50.0]

    frame0_positions = np.vstack([
        np.column_stack([upper_xy, np.full(9, 70.0)]),
        np.column_stack([lower_xy, np.full(9, 30.0)]),
    ])

    frame1_positions = frame0_positions.copy()
    frame1_positions[0] = [lower_xy[0, 0], lower_xy[0, 1], 30.0]  # atom 0 flip-flops to lower

    dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    ordered_results = [(0, frame0_positions, dimensions), (1, frame1_positions, dimensions)]
    selection_global_indices = np.arange(18)  # local == global here

    leaflets = _track_dynamic_leaflets(ordered_results, selection_global_indices, min_balance=0.6)

    upper0, lower0 = leaflets[0]
    assert sorted(upper0) == list(range(9))
    assert sorted(lower0) == list(range(9, 18))

    upper1, lower1 = leaflets[1]
    assert sorted(upper1) == list(range(1, 9))  # atom 0 dropped out
    assert sorted(lower1) == [0] + list(range(9, 18))  # and rejoined the other leaflet
