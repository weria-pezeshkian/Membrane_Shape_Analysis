"""Tests for CALM's forcefield-agnostic headgroup detection
(core/headgroup.py's _require_bonds/_contract_rings/_terminal_arms/
_headgroup_atoms_from_graph/_headgroup_centers): identifying a lipid's
headgroup structurally from its bond graph, rather than by atom name or
by any single atom's own distance to the interface. Also covers the
--lipids RESNAME:NAME1,NAME2 opt-in override (_parse_lipids_argument/
_validate_headgroup_override/_named_headgroup_centers) that lets a user
bypass the bond graph for specific species and name the headgroup atom(s)
directly instead.
"""

from __future__ import annotations

import logging

import MDAnalysis as mda
import networkx as nx
import numpy as np
import pytest

from CALM.core.headgroup import (
    _contract_rings,
    _headgroup_atoms_from_graph,
    _headgroup_centers,
    _named_headgroup_centers,
    _parse_lipids_argument,
    _require_bonds,
    _terminal_arms,
    _validate_headgroup_override,
    _validate_species_exist,
)
from CALM.core.fourier_core import Fourier_Series_Function


def _flat_leaflet_surfaces(Lx: float, Ly: float, z_upper: float, z_lower: float) -> tuple[
    Fourier_Series_Function, Fourier_Series_Function
]:
    f_upper = Fourier_Series_Function(Lx, Ly, 0, 0)
    f_upper.setAnm(np.array([[z_upper]]))
    f_lower = Fourier_Series_Function(Lx, Ly, 0, 0)
    f_lower.setAnm(np.array([[z_lower]]))
    return f_upper, f_lower


def _bonded_universe(
    positions: np.ndarray, bonds: list[tuple[int, int]], masses: list[float] | None = None
) -> mda.core.groups.AtomGroup:
    n = len(positions)
    u = mda.Universe.empty(n_atoms=n, n_residues=1, atom_resindex=[0] * n, trajectory=True)
    u.add_TopologyAttr("name", [f"A{i}" for i in range(n)])
    u.add_TopologyAttr("masses", masses if masses is not None else [72.0] * n)
    u.atoms.positions = positions
    u.add_bonds(bonds)
    return u.atoms


def test_require_bonds_exits_when_structure_has_no_bonds() -> None:
    u = mda.Universe.empty(n_atoms=1, trajectory=True)
    u.add_TopologyAttr("name", ["C1"])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(SystemExit):
        _require_bonds(u)


def test_require_bonds_passes_when_bonds_present() -> None:
    u = mda.Universe.empty(n_atoms=2, n_residues=1, atom_resindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["C1", "C2"])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    u.add_bonds([(0, 1)])

    _require_bonds(u)  # must not raise


def test_contract_rings_merges_a_real_cycle_but_not_a_plain_edge() -> None:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])  # 0-1-2 triangle, plus a plain edge 2-3

    tree, node_atoms = _contract_rings(g)

    assert set(tree.nodes) == {0, 3}  # triangle collapses to its lowest-numbered member
    assert node_atoms[0] == {0, 1, 2}
    assert node_atoms[3] == {3}
    assert set(tree.edges) == {(0, 3)}


def _trivial_node_atoms(t: "nx.Graph") -> dict[int, set[int]]:
    return {n: {n} for n in t.nodes}


def test_terminal_arms_splits_a_star_into_arms_and_interior() -> None:
    t = nx.Graph()
    t.add_edges_from([(0, 1), (0, 2), (0, 3), (3, 4)])  # hub 0, two direct leaves, one longer arm

    arms, interior = _terminal_arms(t, _trivial_node_atoms(t))

    assert sorted(sorted(a) for a in arms) == [[1], [2], [3, 4]]
    assert interior == [0]


def test_terminal_arms_splits_a_plain_path_at_its_midpoint() -> None:
    t = nx.Graph()
    t.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])  # no branch point anywhere, no ring either

    arms, interior = _terminal_arms(t, _trivial_node_atoms(t))

    assert interior == []
    assert len(arms) == 2
    assert sorted(sum(arms, [])) == [0, 1, 2, 3, 4, 5]


def test_terminal_arms_treats_a_bridging_ring_as_interior_even_at_degree_two() -> None:
    # A contracted ring (node 0, representing 3 original atoms) bridging
    # exactly two arms has tree-degree 2 - topologically indistinguishable
    # from an ordinary chain link by degree alone, but it must still count
    # as interior, since it represents real physical extent (a genuine
    # junction), not a single pass-through atom.
    t = nx.Graph()
    t.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 4)])
    node_atoms = {0: {0, 5, 6}, 1: {1}, 2: {2}, 3: {3}, 4: {4}}

    arms, interior = _terminal_arms(t, node_atoms)

    assert interior == [0]
    assert sorted(sorted(a) for a in arms) == [[1, 2], [3, 4]]


def test_terminal_arms_treats_a_dangling_ring_as_an_ordinary_arm() -> None:
    # A ring with only one tree edge (node 0, representing 2 original
    # atoms, attached only to hub 1) is not a junction - it is a single
    # candidate arm, judged on distance the same as any other pendant
    # group (e.g. a dangling sugar headgroup), not automatically kept.
    t = nx.Graph()
    t.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])
    node_atoms = {0: {0, 5}, 1: {1}, 2: {2}, 3: {3}, 4: {4}}

    arms, interior = _terminal_arms(t, node_atoms)

    assert interior == [1]
    assert sorted(sorted(a) for a in arms) == [[0], [2], [3, 4]]


def test_headgroup_atoms_from_graph_drops_the_farther_arm() -> None:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])  # pendant 0 - hub 1 - {short arm 2, longer arm 3-4}
    d_min = np.array([0.5, 1.0, 15.0, 20.0, 25.0])  # node 0 near the surface; 2, 3, 4 buried

    kept = _headgroup_atoms_from_graph(g, d_min)

    assert kept == [{0, 1}]  # one hub, one group


def test_headgroup_atoms_from_graph_drops_a_ring_with_only_two_external_arms() -> None:
    # A ring (0, 1, 2) bridging exactly two arms - one a normal buried tail
    # (1-3-4), one an appendage protruding outward past the surface into
    # solvent (2-5-6, e.g. a bound tag/linker on the same residue). Before
    # the fix, a ring with only two external connections had tree-degree 2
    # and was missed as a branch point entirely, so this whole residue fell
    # into the plain-path midpoint-split fallback instead - which let the
    # appendage leak into the kept group depending on where the split
    # landed. d_min doesn't encode direction (it's a plain absolute
    # distance to the nearer surface), so both the buried tail and the
    # outward appendage must be dropped the same way.
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (2, 5), (5, 6)])
    d_min = np.array([1.0, 1.0, 1.0, 15.0, 20.0, 15.0, 20.0])

    kept = _headgroup_atoms_from_graph(g, d_min)

    assert kept == [{0, 1, 2}]


def test_headgroup_atoms_from_graph_splits_two_hubs_into_separate_groups() -> None:
    # Two hubs (1, 6) joined by a linker (0), each also carrying two tails
    # - mirrors cardiolipin's real bond topology (two phosphate/glycerol
    # rings joined by a central glycerol, four tails). The linker breaks
    # ties toward the lower-numbered hub; what matters is that the two
    # hubs end up as separate groups rather than one averaged position.
    g = nx.Graph()
    g.add_edges_from([
        (0, 1), (1, 2), (2, 3), (1, 4), (4, 5),   # linker-hub1, hub1's two tails
        (0, 6), (6, 7), (7, 8), (6, 9), (9, 10),  # linker-hub2, hub2's two tails
    ])
    d_min = np.array([1.0, 1.0, 20.0, 20.0, 20.0, 20.0, 1.0, 20.0, 20.0, 20.0, 20.0])

    kept = _headgroup_atoms_from_graph(g, d_min)

    assert len(kept) == 2
    assert {0, 1} in kept
    assert {6} in kept


def test_headgroup_centers_finds_pendant_head_and_ring_not_diluted_by_tails() -> None:
    f_upper, f_lower = _flat_leaflet_surfaces(100.0, 100.0, 70.0, 30.0)
    # NC3(pendant head) - PO4 - GL1 - GL2 (ring: PO4-GL1, PO4-GL2, GL1-GL2)
    # - two tails hanging off GL1/GL2 - mirrors MARTINI 3's real POPC bond
    # topology (verified against the actual .tpr earlier in this session).
    positions = np.array([
        [40.0, 50.0, 70.0],  # NC3
        [41.0, 50.0, 69.0],  # PO4
        [40.0, 51.0, 68.0],  # GL1
        [41.0, 51.0, 68.0],  # GL2
        [40.0, 51.0, 60.0],  # tail A bead 1
        [40.0, 51.0, 50.0],  # tail A bead 2
        [41.0, 51.0, 60.0],  # tail B bead 1
        [41.0, 51.0, 50.0],  # tail B bead 2
    ])
    bonds = [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 6), (6, 7)]
    atoms = _bonded_universe(positions, bonds)

    xy, z, hub_xy = _headgroup_centers(atoms, f_upper, f_lower)

    headgroup_idx = [0, 1, 2, 3]  # NC3, PO4, GL1, GL2
    expected_xy = positions[headgroup_idx, :2].mean(axis=0)
    assert np.allclose(xy[0], expected_xy)
    assert np.allclose(z[0], positions[headgroup_idx, 2].mean())
    assert hub_xy[0].shape == (1, 2)  # single hub, one point
    assert np.allclose(hub_xy[0][0], expected_xy)


def test_headgroup_centers_prunes_light_atoms_before_branch_detection() -> None:
    f_upper, f_lower = _flat_leaflet_surfaces(100.0, 100.0, 70.0, 30.0)
    # An all-atom-style fixture: a plain heavy-atom chain C1-C2-C3, each
    # carrying light "hydrogen" substituents. Counting hydrogens toward
    # degree would make C1 (and C2, C3) look like branch points; pruning
    # them first leaves a plain 3-node path with no branch point, split at
    # its own midpoint the same way any other unbranched chain would be.
    masses = [12.0, 1.0, 1.0, 12.0, 1.0, 1.0, 12.0, 1.0, 1.0]
    positions = np.array([
        [40.0, 50.0, 70.0],  # C1 (near upper surface)
        [39.0, 50.0, 70.5],  # H1a
        [41.0, 50.0, 70.5],  # H1b
        [40.0, 51.0, 60.0],  # C2 (buried)
        [39.0, 51.0, 60.5],  # H2a
        [41.0, 51.0, 60.5],  # H2b
        [40.0, 52.0, 50.0],  # C3 (buried further)
        [39.0, 52.0, 50.5],  # H3a
        [41.0, 52.0, 50.5],  # H3b
    ])
    bonds = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6), (6, 7), (6, 8)]
    atoms = _bonded_universe(positions, bonds, masses=masses)

    xy, z, hub_xy = _headgroup_centers(atoms, f_upper, f_lower)

    assert np.allclose(xy[0], positions[0, :2])
    assert np.allclose(z[0], positions[0, 2])
    assert hub_xy[0].shape == (1, 2)


def test_headgroup_centers_gives_multiple_hub_points_for_a_double_headgroup_lipid() -> None:
    f_upper, f_lower = _flat_leaflet_surfaces(100.0, 100.0, 70.0, 30.0)
    # Mirrors cardiolipin's real bond topology: two phosphate/glycerol
    # rings (one near x=30, one near x=45), each with two tails, joined by
    # a bridging glycerol (GLC) - verified against the actual .tpr earlier
    # in this session. The point of this test: the two rings must NOT be
    # averaged into one midpoint between them (that structurally caps a
    # double-headgroup lipid's Voronoi footprint at whatever fits around a
    # single point) - each should compete separately.
    positions = np.array([
        [35.0, 50.0, 69.0],  # GLC (bridge)
        [29.0, 50.0, 70.0],  # P1
        [30.0, 51.0, 69.0],  # G1a
        [31.0, 51.0, 69.0],  # G1b
        [30.0, 51.0, 60.0],  # tail A bead 1
        [30.0, 51.0, 50.0],  # tail A bead 2
        [31.0, 51.0, 60.0],  # tail B bead 1
        [31.0, 51.0, 50.0],  # tail B bead 2
        [46.0, 50.0, 70.0],  # P2
        [45.0, 51.0, 69.0],  # G2a
        [44.0, 51.0, 69.0],  # G2b
        [45.0, 51.0, 60.0],  # tail C bead 1
        [45.0, 51.0, 50.0],  # tail C bead 2
        [44.0, 51.0, 60.0],  # tail D bead 1
        [44.0, 51.0, 50.0],  # tail D bead 2
    ])
    bonds = [
        (0, 1), (1, 2), (2, 3), (1, 3), (2, 4), (4, 5), (3, 6), (6, 7),
        (0, 8), (8, 9), (9, 10), (8, 10), (9, 11), (11, 12), (10, 13), (13, 14),
    ]
    atoms = _bonded_universe(positions, bonds)

    xy, z, hub_xy = _headgroup_centers(atoms, f_upper, f_lower)

    assert hub_xy[0].shape[0] == 2  # two separate hub points, not one averaged midpoint
    xs = sorted(hub_xy[0][:, 0])
    assert xs[0] < 35.0 < xs[1]  # one point near ring1 (x~30), one near ring2 (x~45)


def _named_universe(
    resnames: list[str], names: list[str], positions: np.ndarray, resindex: list[int] | None = None
) -> mda.Universe:
    n = len(names)
    if resindex is None:
        resindex = list(range(n))  # one atom per residue, by default
    n_residues = max(resindex) + 1
    u = mda.Universe.empty(n_atoms=n, n_residues=n_residues, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("name", names)
    u.atoms.positions = positions
    return u


def test_parse_lipids_argument_separates_species_and_override() -> None:
    species, override = _parse_lipids_argument(["POPC:PO4", "TCL1:PO41,PO42", "SAPE24"])

    assert species == ["POPC", "TCL1", "SAPE24"]
    assert override == {"POPC": ["PO4"], "TCL1": ["PO41", "PO42"]}


def test_named_headgroup_centers_uses_the_single_named_atom_directly() -> None:
    # Two POPC residues, each with a PO4 (the named headgroup atom) and an
    # unrelated tail atom that must be ignored since it isn't named.
    u = _named_universe(
        ["POPC", "POPC"],
        ["PO4", "C1A", "PO4", "C1A"],
        np.array([[10.0, 20.0, 70.0], [10.0, 20.0, 40.0], [30.0, 40.0, 70.0], [30.0, 40.0, 40.0]]),
        resindex=[0, 0, 1, 1],
    )

    xy, z, hub_xy = _named_headgroup_centers(u.select_atoms("resname POPC"), ["PO4"])

    assert np.allclose(xy, [[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(z, [70.0, 70.0])
    assert [h.shape for h in hub_xy] == [(1, 2), (1, 2)]


def test_named_headgroup_centers_gives_one_point_per_named_atom() -> None:
    # A single cardiolipin-like residue with two named phosphates.
    u = _named_universe(
        ["TCL1"],
        ["PO41", "PO42", "C1A1"],
        np.array([[10.0, 20.0, 70.0], [20.0, 20.0, 69.0], [15.0, 20.0, 40.0]]),
        resindex=[0, 0, 0],
    )

    xy, z, hub_xy = _named_headgroup_centers(u.select_atoms("resname TCL1"), ["PO41", "PO42"])

    assert hub_xy[0].shape == (2, 2)
    assert np.allclose(sorted(hub_xy[0][:, 0].tolist()), [10.0, 20.0])
    assert np.allclose(xy[0], [15.0, 20.0])  # xy_avg is still the mean across both


def test_validate_headgroup_override_exits_when_named_atom_matches_nothing() -> None:
    u = _named_universe(["POPC"], ["PO4"], np.array([[0.0, 0.0, 70.0]]))

    with pytest.raises(SystemExit):
        _validate_headgroup_override(u, ["POPC"], {"POPC": ["NOPE"]})


def test_validate_headgroup_override_warns_when_only_some_species_are_covered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    u = _named_universe(["POPC", "POPE"], ["PO4", "PO4"], np.array([[0.0, 0.0, 70.0], [10.0, 0.0, 70.0]]))

    with caplog.at_level(logging.WARNING, logger="CALM.core.headgroup"):
        _validate_headgroup_override(u, ["POPC", "POPE"], {"POPC": ["PO4"]})

    assert any("POPE" in r.message and "automatic" in r.message for r in caplog.records)


def test_validate_headgroup_override_does_not_warn_when_fully_covered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    u = _named_universe(["POPC", "POPE"], ["PO4", "PO4"], np.array([[0.0, 0.0, 70.0], [10.0, 0.0, 70.0]]))

    with caplog.at_level(logging.WARNING, logger="CALM.core.headgroup"):
        _validate_headgroup_override(u, ["POPC", "POPE"], {"POPC": ["PO4"], "POPE": ["PO4"]})

    assert caplog.records == []


def test_validate_headgroup_override_does_not_warn_when_empty(caplog: pytest.LogCaptureFixture) -> None:
    u = _named_universe(["POPC"], ["PO4"], np.array([[0.0, 0.0, 70.0]]))

    with caplog.at_level(logging.WARNING, logger="CALM.core.headgroup"):
        _validate_headgroup_override(u, ["POPC"], {})

    assert caplog.records == []


def test_validate_species_exist_exits_on_a_resname_with_no_atoms() -> None:
    u = _named_universe(["POPC"], ["PO4"], np.array([[0.0, 0.0, 70.0]]))

    with pytest.raises(SystemExit, match="POPE"):
        _validate_species_exist(u, ["POPC", "POPE"])


def test_validate_species_exist_passes_when_every_species_has_atoms() -> None:
    u = _named_universe(["POPC", "POPE"], ["PO4", "PO4"], np.array([[0.0, 0.0, 70.0], [10.0, 0.0, 70.0]]))

    _validate_species_exist(u, ["POPC", "POPE"])  # does not raise
