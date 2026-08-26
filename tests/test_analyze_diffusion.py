"""Tests for CALM analyze diffusion: the tracked-point roster
(_track_blocks/_resolve_tracked_points), the per-frame worker
(_one_diffusion_frame), the PBC-aware whole-and-continuous position
extraction (_extract_whole_continuous_positions), CLI validation, and a
full calc_diffusion end-to-end run recovering a known diffusion
coefficient from a synthetic random walk.
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from CALM.analyze.diffusion import (
    _extract_whole_continuous_positions,
    _one_diffusion_frame,
    _resolve_tracked_points,
    _track_blocks,
    calc_diffusion,
    diffusion,
)
from CALM.core import fourier_build as fb


def _flat_diffusion_universe(
    n_lipids_upper: int = 2, n_lipids_lower: int = 1, bonded: bool = True
) -> tuple[mda.Universe, mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """A flat membrane: upper fit selection at z=70, lower at z=30, with `n_lipids_upper`
    POPC residues near the upper surface and `n_lipids_lower` near the lower one, each a
    bonded 2-atom residue (unless `bonded=False`).
    """
    Lx = Ly = 100.0
    xs = np.linspace(2, 98, 10)
    ys = np.linspace(2, 98, 10)
    XX, YY = np.meshgrid(xs, ys)
    fit_xy = np.column_stack([XX.ravel(), YY.ravel()])
    fit_upper_pos = np.column_stack([fit_xy, np.full(len(fit_xy), 70.0)])
    fit_lower_pos = np.column_stack([fit_xy, np.full(len(fit_xy), 30.0)])

    upper_centers = np.array([[20.0, 20.0, 70.0], [60.0, 60.0, 70.0]])[:n_lipids_upper]
    lower_centers = np.array([[40.0, 40.0, 30.0]])[:n_lipids_lower]

    def two_atom_residues(centers: np.ndarray) -> np.ndarray:
        pos = []
        for c in centers:
            pos.append(c + np.array([0.5, 0.0, 0.0]))
            pos.append(c - np.array([0.5, 0.0, 0.0]))
        return np.array(pos)

    lipid_pos = two_atom_residues(np.vstack([upper_centers, lower_centers]))
    n_fit_upper, n_fit_lower = len(fit_upper_pos), len(fit_lower_pos)
    n_lipids = n_lipids_upper + n_lipids_lower

    resindex = list(range(n_fit_upper + n_fit_lower))
    next_res = n_fit_upper + n_fit_lower
    for _ in range(n_lipids):
        resindex += [next_res, next_res]
        next_res += 1

    resnames = ["FITRES"] * (n_fit_upper + n_fit_lower) + ["POPC"] * n_lipids
    names = ["PO4"] * (n_fit_upper + n_fit_lower) + ["C1", "C2"] * n_lipids
    all_pos = np.vstack([fit_upper_pos, fit_lower_pos, lipid_pos])

    u = mda.Universe.empty(
        n_atoms=len(all_pos), n_residues=next_res, atom_resindex=np.array(resindex), trajectory=True
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("masses", [72.0] * len(all_pos))
    u.atoms.positions = all_pos
    u.dimensions = [Lx, Ly, 60.0, 90.0, 90.0, 90.0]

    if bonded:
        lipid_start = n_fit_upper + n_fit_lower
        u.add_bonds([(lipid_start + 2 * i, lipid_start + 2 * i + 1) for i in range(n_lipids)])

    layer_group = u.atoms[:n_fit_upper]
    layer_group_2 = u.atoms[n_fit_upper:n_fit_upper + n_fit_lower]
    return u, layer_group, layer_group_2


def test_track_blocks_builds_one_block_per_species_and_one_for_select() -> None:
    blocks = _track_blocks(["POPC", "POPE"], "protein", False, "prot")
    assert [b.label for b in blocks] == ["POPC", "POPE", "prot"]
    assert [b.kind for b in blocks] == ["lipids", "lipids", "select"]
    assert blocks[0].select_string == "resname POPC"
    assert blocks[2].select_string == "protein"
    assert blocks[2].whole is False


def test_track_blocks_without_select_omits_the_select_block() -> None:
    blocks = _track_blocks(["POPC"], None, False, "select")
    assert [b.kind for b in blocks] == ["lipids"]


def test_resolve_tracked_points_orders_rows_by_block_and_residue() -> None:
    u, _, _ = _flat_diffusion_universe(n_lipids_upper=2, n_lipids_lower=1)
    blocks = _track_blocks(["POPC"], None, False, "select")

    tracked = _resolve_tracked_points(u, blocks)

    assert len(tracked) == 3
    assert list(tracked["index"]) == [0, 1, 2]
    assert set(tracked["label"]) == {"POPC"}
    assert list(tracked["kind"]) == ["lipids", "lipids", "lipids"]
    assert list(tracked["resindex"]) == sorted(tracked["resindex"])


def test_resolve_tracked_points_select_whole_gives_single_row_with_sentinel_resindex() -> None:
    u, _, _ = _flat_diffusion_universe(n_lipids_upper=2, n_lipids_lower=1)
    blocks = _track_blocks([], "resname POPC", True, "popc_group")

    tracked = _resolve_tracked_points(u, blocks)

    assert len(tracked) == 1
    assert tracked["label"][0] == "popc_group"
    assert tracked["resindex"][0] == -1


def test_one_diffusion_frame_assigns_leaflet_and_saves_surface(tmp_path: Path) -> None:
    u, layer_group, layer_group_2 = _flat_diffusion_universe(n_lipids_upper=2, n_lipids_lower=1)
    blocks = _track_blocks(["POPC"], None, False, "select")

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = layer_group
    fb._worker_state["layer_group_2"] = layer_group_2
    fb._worker_state["rotation_and_center"] = None
    try:
        result = _one_diffusion_frame(
            0, out_dir=str(tmp_path), blocks=blocks, headgroup_override={},
            dynamic_select=False, until=1,
            Nx=5.0, Ny=5.0, regularize=False, remove_tmd=False, gridsize=20,
        )
    finally:
        fb._worker_state.clear()

    assert result["frame"] == 0
    meta = np.load(tmp_path / "0_diffusion_meta.npy")
    assert meta.shape == (3, 2)
    assert sorted(meta[:, 0].tolist()) == [-1, 1, 1]  # 2 upper (1), 1 lower (-1)
    assert (meta[:, 1] == 0).all()  # no --Remove-TMD given: never in a hole

    surface = np.load(tmp_path / "0_diffusion_surface.npy")
    assert surface.shape[0] == 2  # [Anm_upper, Anm_lower]
    assert surface.shape[1] == surface.shape[2] == 2 * 2 + 1  # Nx=Ny=2 from lambda=5.0nm on a 100A box


def test_one_diffusion_frame_with_remove_tmd_writes_boolean_hole_column(tmp_path: Path) -> None:
    u, layer_group, layer_group_2 = _flat_diffusion_universe(n_lipids_upper=2, n_lipids_lower=1)
    blocks = _track_blocks(["POPC"], None, False, "select")

    fb._worker_state["universe"] = u
    fb._worker_state["layer_group"] = layer_group
    fb._worker_state["layer_group_2"] = layer_group_2
    fb._worker_state["rotation_and_center"] = None
    try:
        _one_diffusion_frame(
            0, out_dir=str(tmp_path), blocks=blocks, headgroup_override={},
            dynamic_select=False, until=1,
            Nx=5.0, Ny=5.0, regularize=False, remove_tmd="resname POPC", gridsize=20,
        )
    finally:
        fb._worker_state.clear()

    meta = np.load(tmp_path / "0_diffusion_meta.npy")
    assert set(np.unique(meta[:, 1])).issubset({0, 1})


def _write_pbc_fixture(
    tmp_path: Path, n_frames: int, lipid_xy: np.ndarray, Lx: float = 100.0, Ly: float = 100.0
) -> tuple[Path, Path]:
    """Writes a PDB (bonds via CONECT) + XTC pair for one bonded 2-atom POPC residue whose
    per-frame COG follows `lipid_xy` (shape (n_frames, 2)), wrapped into [0, Lx) x [0, Ly)."""
    z0 = 70.0
    n_atoms = 2
    u = mda.Universe.empty(n_atoms=n_atoms, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["C1", "C2"])
    u.add_TopologyAttr("resname", ["POPC"])
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("masses", [72.0, 72.0])
    u.add_TopologyAttr("elements", ["C", "C"])
    u.add_bonds([(0, 1)])

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for t in range(n_frames):
        c = np.mod(lipid_xy[t], [Lx, Ly])
        coords[t, 0] = [c[0] + 3.0, c[1], z0]  # 6 A apart in x - wider than half a small box, exercises unwrap
        coords[t, 1] = [c[0] - 3.0, c[1], z0]

    u.trajectory = mda.coordinates.memory.MemoryReader(
        coords, dimensions=np.tile([Lx, Ly, 60.0, 90, 90, 90], (n_frames, 1)), dt=1.0
    )

    struct_path = tmp_path / "struct.pdb"
    traj_path = tmp_path / "traj.xtc"
    u.atoms.write(str(struct_path), bonds="all")
    with mda.Writer(str(traj_path), n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return struct_path, traj_path


def test_extract_whole_continuous_positions_unwraps_within_frame_split(tmp_path: Path) -> None:
    # A single frame where the residue's own two atoms straddle x=0/x=Lx (6 A apart,
    # centered near the boundary): the raw, wrapped COG would be pulled toward the box
    # center; the unwrapped one recovers the true position near the edge.
    Lx = 20.0
    lipid_xy = np.array([[1.0, 10.0]])  # atoms end up at x=4.0 and x=-2.0 -> wraps to x=18.0
    struct_path, traj_path = _write_pbc_fixture(tmp_path, n_frames=1, lipid_xy=lipid_xy, Lx=Lx, Ly=Lx)

    blocks = _track_blocks(["POPC"], None, False, "select")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _extract_whole_continuous_positions(
        str(struct_path), str(traj_path), blocks, frames=[0], out_dir=str(out_dir), until=1
    )

    positions = np.load(out_dir / "0_diffusion_positions.npy")
    assert positions.shape == (1, 3)
    # The true COG is at x=1.0 (possibly shifted by a whole box length by unwrap's
    # own choice of image) - what matters is the two atoms are NOT averaged as
    # if 6 A apart on the same side, which a naive wrapped COG would do.
    assert min(abs(positions[0, 0] - 1.0), abs(positions[0, 0] - 1.0 - Lx), abs(positions[0, 0] - 1.0 + Lx)) < 1e-3


def test_extract_whole_continuous_positions_stays_continuous_across_a_pbc_wrap(tmp_path: Path) -> None:
    Lx = 100.0
    # A residue drifting steadily in +x, crossing the box edge between frames 4 and 5.
    lipid_xy = np.column_stack([np.linspace(90.0, 110.0, 10), np.full(10, 50.0)])
    struct_path, traj_path = _write_pbc_fixture(tmp_path, n_frames=10, lipid_xy=lipid_xy, Lx=Lx, Ly=Lx)

    blocks = _track_blocks(["POPC"], None, False, "select")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    frames = list(range(10))
    _extract_whole_continuous_positions(
        str(struct_path), str(traj_path), blocks, frames=frames, out_dir=str(out_dir), until=10
    )

    num_digits = len(str(10))
    x = np.array([np.load(out_dir / f"{f:0{num_digits}d}_diffusion_positions.npy")[0, 0] for f in frames])
    # No jump of ~Lx anywhere in the continuous trajectory.
    assert np.all(np.abs(np.diff(x)) < 5.0)


def test_diffusion_cli_requires_at_least_one_of_lipids_or_select(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        diffusion([
            "-f", str(tmp_path / "missing.xtc"), "-s", str(tmp_path / "missing.pdb"),
            "-n", "protein", "-o", str(tmp_path / "out"),
        ])


def test_diffusion_cli_rejects_bare_remove_tmd(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        diffusion([
            "-f", str(tmp_path / "missing.xtc"), "-s", str(tmp_path / "missing.pdb"),
            "-n", "protein", "-o", str(tmp_path / "out"), "--lipids", "POPC", "--Remove-TMD",
        ])


def test_calc_diffusion_rejects_bond_free_structure(tmp_path: Path) -> None:
    u, _, _ = _flat_diffusion_universe(n_lipids_upper=2, n_lipids_lower=1, bonded=False)
    struct_path = tmp_path / "struct.gro"
    traj_path = tmp_path / "traj.xtc"
    u.atoms.write(str(struct_path))
    with mda.Writer(str(traj_path), len(u.atoms)) as w:
        w.write(u.atoms)

    ns = type("Namespace", (), {})()
    ns.lipids = ["POPC"]
    ns.select = None
    ns.select_whole = False
    ns.select_label = "select"
    ns.out = str(tmp_path / "out")
    ns.structure = str(struct_path)
    ns.trajectory = str(traj_path)
    ns.index = "resname FITRES"
    ns.From = 0
    ns.Until = 1
    ns.Step = 1
    ns.lambda_x = None
    ns.lambda_y = None
    ns.gridsize = 10
    ns.regularize = False
    ns.remove_tmd = False
    ns.min_balance = 0.6
    ns.margin = 2.0
    ns.Workers = 1
    ns.min_segment_fraction = 0.1
    ns.max_tau_fraction = 0.25
    ns.fit_tau_min_fraction = 0.1
    ns.fit_tau_max_fraction = 0.5

    (tmp_path / "out").mkdir()
    u2 = mda.Universe(str(struct_path), str(traj_path))
    with pytest.raises(SystemExit):
        calc_diffusion(ns, u2)


def test_calc_diffusion_recovers_known_diffusion_coefficient(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    Lx = Ly = 100.0
    n_frames = 40
    n_lipids = 4
    D_true = 5.0
    dt = 1.0
    sigma = np.sqrt(2 * D_true * dt)

    xs = np.linspace(2, 98, 10)
    ys = np.linspace(2, 98, 10)
    XX, YY = np.meshgrid(xs, ys)
    fit_xy = np.column_stack([XX.ravel(), YY.ravel()])
    n_fit = len(fit_xy)

    start_xy = np.column_stack([rng.uniform(20, 80, n_lipids), rng.uniform(20, 80, n_lipids)])
    lipid_xy = np.zeros((n_frames, n_lipids, 2))
    lipid_xy[0] = start_xy
    for t in range(1, n_frames):
        lipid_xy[t] = np.mod(lipid_xy[t - 1] + rng.normal(0, sigma, size=(n_lipids, 2)), [Lx, Ly])

    z0 = 70.0
    n_atoms_lipid = n_lipids * 2
    n_atoms = n_fit * 2 + n_atoms_lipid  # upper + lower fit grids, both flat, plus the lipids

    resindex = list(range(2 * n_fit))
    next_res = 2 * n_fit
    for _ in range(n_lipids):
        resindex += [next_res, next_res]
        next_res += 1

    resnames = ["FITRES"] * (2 * n_fit) + ["POPC"] * n_lipids
    names = ["PO4"] * (2 * n_fit) + ["C1", "C2"] * n_lipids

    u = mda.Universe.empty(n_atoms=n_atoms, n_residues=next_res, atom_resindex=np.array(resindex), trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", list(range(1, next_res + 1)))
    u.add_TopologyAttr("masses", [72.0] * n_atoms)
    u.add_TopologyAttr("elements", ["C"] * n_atoms)
    u.add_bonds([(2 * n_fit + 2 * i, 2 * n_fit + 2 * i + 1) for i in range(n_lipids)])

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for t in range(n_frames):
        coords[t, :n_fit, :2] = fit_xy
        coords[t, :n_fit, 2] = 70.0
        coords[t, n_fit:2 * n_fit, :2] = fit_xy
        coords[t, n_fit:2 * n_fit, 2] = 30.0
        for i in range(n_lipids):
            c = lipid_xy[t, i]
            coords[t, 2 * n_fit + 2 * i] = [c[0] + 0.5, c[1], z0]
            coords[t, 2 * n_fit + 2 * i + 1] = [c[0] - 0.5, c[1], z0]

    u.trajectory = mda.coordinates.memory.MemoryReader(
        coords, dimensions=np.tile([Lx, Ly, 60.0, 90, 90, 90], (n_frames, 1)), dt=dt
    )

    struct_path = tmp_path / "struct.pdb"
    traj_path = tmp_path / "traj.xtc"
    u.atoms.write(str(struct_path), bonds="all")
    with mda.Writer(str(traj_path), n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)

    ndx_path = tmp_path / "index.ndx"
    ndx_path.write_text(
        "[ Upper ]\n" + " ".join(str(i + 1) for i in range(n_fit)) + "\n"
        "[ Lower ]\n" + " ".join(str(i + 1) for i in range(n_fit + 1, 2 * n_fit + 1)) + "\n"
    )

    out_dir = tmp_path / "out"
    diffusion([
        "-f", str(traj_path), "-s", str(struct_path), "-n", str(ndx_path), "-o", str(out_dir),
        "--lipids", "POPC", "--gridsize", "20", "--min-segment-fraction", "0.05",
    ])

    result = np.load(out_dir / "diffusion.npy")
    both_row = result[result["leaflet"] == "both"][0]
    assert abs(both_row["D_cm2_s"] - D_true * 1e-4) / (D_true * 1e-4) < 0.6
    assert both_row["fit_r2"] > 0.9
