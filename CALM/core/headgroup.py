from __future__ import annotations

import sys

import MDAnalysis as mda
import networkx as nx
import numpy as np

from ..core.fourier_core import Fourier_Series_Function


def _require_bonds(universe: mda.Universe) -> None:
    """Fail early and clearly if `universe` carries no bond information.

    `analyze lipids` identifies each residue's headgroup from its own bond
    graph (`_headgroup_centers`) rather than from an atom-name selection -
    a bare .gro/.pdb has no bonds; a GROMACS .tpr (or other bonded
    topology MDAnalysis can read) does.
    """
    try:
        universe.atoms.bonds
    except mda.exceptions.NoDataError:
        sys.exit(
            "analyze lipids requires --structure to carry bond information (e.g. a GROMACS .tpr) - "
            "the current --structure has none."
        )


def _contract_rings(graph: "nx.Graph") -> tuple["nx.Graph", dict[int, set[int]]]:
    """(tree, node_atoms): `graph` with every real cycle contracted to one node, guaranteed a tree.

    A biconnected component of exactly 2 nodes is a single bond and is
    left alone; one of 3 or more nodes is an actual ring (such as MARTINI
    3's glycerol backbone) and is merged into its lowest-numbered member.
    `node_atoms` maps each surviving tree node to the set of original
    graph nodes it now represents.
    """
    node_group = {n: n for n in graph.nodes}
    for component in nx.biconnected_components(graph):
        if len(component) >= 3:
            rep = min(component)
            for n in component:
                node_group[n] = rep

    node_atoms: dict[int, set[int]] = {}
    for original, rep in node_group.items():
        node_atoms.setdefault(rep, set()).add(original)

    tree = nx.Graph()
    tree.add_nodes_from(node_atoms.keys())
    for u, v in graph.edges:
        gu, gv = node_group[u], node_group[v]
        if gu != gv:
            tree.add_edge(gu, gv)

    return tree, node_atoms


def _is_hub(tree: "nx.Graph", node_atoms: dict[int, set[int]], n: int) -> bool:
    """A node is a real structural junction if it has degree 3+, or is a ring bridging 2+ tree edges.

    A ring with only one tree edge (a dangling ring, such as a pendant
    sugar headgroup) is not a junction at all - it is a single candidate
    arm, judged on distance the same as any other pendant group.
    """
    return tree.degree(n) >= 3 or (len(node_atoms[n]) > 1 and tree.degree(n) >= 2)


def _terminal_arms(tree: "nx.Graph", node_atoms: dict[int, set[int]]) -> tuple[list[list[int]], list[int]]:
    """(arms, interior): every leaf-to-branch-point path in `tree`, and everything left over.

    Each arm is a maximal run of non-hub nodes starting at a leaf and
    stopping just short of the first hub it reaches (`_is_hub`). If `tree`
    has no hub at all (a plain path, e.g. a single-tailed lipid with no
    ring anywhere), it is instead split into two arms at its midpoint,
    with no separate interior.
    """
    hub_nodes = {n for n in tree.nodes if _is_hub(tree, node_atoms, n)}
    if not hub_nodes:
        leaves = [n for n, d in tree.degree() if d <= 1]
        full_path = nx.shortest_path(tree, leaves[0], leaves[1]) if len(leaves) == 2 else list(tree.nodes)
        mid = len(full_path) // 2
        return [full_path[:mid], full_path[mid:]], []

    arms = []
    visited: set[int] = set()
    for leaf in [n for n, d in tree.degree() if d == 1 and n not in hub_nodes]:
        path = [leaf]
        prev, cur = None, leaf
        while True:
            neighbors = [x for x in tree.neighbors(cur) if x != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt in hub_nodes:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        arms.append(path)
        visited.update(path)
    interior = [n for n in tree.nodes if n not in visited]
    return arms, interior


def _headgroup_atoms_from_graph(graph: "nx.Graph", d_min: np.ndarray) -> list[set[int]]:
    """List of atom-index groups to average for one residue's headgroup position(s), one group per hub.

    Contracts every real ring into a single node (`_contract_rings`), then
    compares each resulting arm's own average `d_min` (distance to the
    nearer fitted leaflet surface) against the tree's remaining interior
    atoms' - an arm farther than the interior, on average, is a tail and is
    dropped; the rest (the interior plus any non-tail arm) is kept.

    The kept atoms are then grouped by which hub (`_is_hub`) they are
    nearest to in the tree, so a lipid with more than one real headgroup
    junction - cardiolipin's two phosphate/glycerol rings, joined by a
    bridging glycerol - returns one group per junction rather than being
    collapsed into a single averaged position between them. A lipid with
    only one hub (the common case) always returns exactly one group.
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_nodes() == 1:
        return [set(graph.nodes)]

    tree, node_atoms = _contract_rings(graph)
    if tree.number_of_nodes() == 1:
        return [set(next(iter(node_atoms.values())))]

    arms, interior = _terminal_arms(tree, node_atoms)
    interior_atoms = {a for n in interior for a in node_atoms[n]}
    interior_distance = float(d_min[sorted(interior_atoms)].mean()) if interior_atoms else float(d_min.mean())

    kept_tree_nodes = set(interior)
    for arm in arms:
        arm_atoms = {a for n in arm for a in node_atoms[n]}
        arm_distance = float(d_min[sorted(arm_atoms)].mean())
        if arm_distance <= interior_distance:
            kept_tree_nodes |= set(arm)

    hub_nodes = sorted(n for n in kept_tree_nodes if _is_hub(tree, node_atoms, n))
    if not hub_nodes:
        return [{a for n in kept_tree_nodes for a in node_atoms[n]}]

    subtree = tree.subgraph(kept_tree_nodes)
    distances = {h: nx.single_source_shortest_path_length(subtree, h) for h in hub_nodes}
    groups: dict[int, set[int]] = {h: set() for h in hub_nodes}
    for n in kept_tree_nodes:
        nearest_hub = min(hub_nodes, key=lambda h: distances[h].get(n, float("inf")))
        groups[nearest_hub] |= node_atoms[n]

    return list(groups.values())


def _headgroup_centers(
    atomgroup: mda.core.groups.AtomGroup, fourier_upper: Fourier_Series_Function, fourier_lower: Fourier_Series_Function,
    min_mass: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(xy_avg, z_avg, hub_xy) per residue in `atomgroup`: a forcefield-agnostic headgroup position, found from bonds.

    A lipid's headgroup is identified structurally rather than by name or
    by any single atom's own distance to the interface. Atoms lighter than
    `min_mass` (hydrogens, which are always monovalent and so can only
    ever be leaves, never real branch points) are excluded from the graph
    before it is built, so an all-atom force field's explicit hydrogens
    can't manufacture spurious branch points the way they would if degree
    were counted directly. `_headgroup_atoms_from_graph` then does the
    structural classification and hub grouping per residue.

    `xy_avg`/`z_avg` are each residue's own single average position across
    every kept headgroup atom, regardless of how many hubs it has - used
    for leaflet assignment, where one coarse position per residue is
    enough. `hub_xy[i]` is that same residue's kept atoms grouped by hub
    instead: one row per structurally distinct headgroup junction, so a
    multi-headgroup lipid (cardiolipin) contributes multiple competing
    points to the Voronoi tessellation rather than being collapsed to one
    averaged midpoint between them.
    """
    residues = atomgroup.residues
    if len(residues) == 0:
        return np.empty((0, 2)), np.empty((0,)), []

    xy_avg = np.zeros((len(residues), 2))
    z_avg = np.zeros(len(residues))
    hub_xy: list[np.ndarray] = []

    for i, res in enumerate(residues):
        atoms = res.atoms
        pos = atoms.positions
        heavy_mask = atoms.masses >= min_mass
        if not heavy_mask.any():
            heavy_mask = np.ones(len(atoms), dtype=bool)
        heavy_local = set(np.flatnonzero(heavy_mask).tolist())

        ix_to_local = {int(a.ix): j for j, a in enumerate(atoms)}
        graph = nx.Graph()
        graph.add_nodes_from(heavy_local)
        for bond in atoms.bonds:
            a0, a1 = (int(x) for x in bond.atoms.ix)
            l0, l1 = ix_to_local[a0], ix_to_local[a1]
            if l0 in heavy_local and l1 in heavy_local:
                graph.add_edge(l0, l1)

        z_upper = fourier_upper.Z(pos[:, 0], pos[:, 1])
        z_lower = fourier_lower.Z(pos[:, 0], pos[:, 1])
        d_min = np.minimum(np.abs(pos[:, 2] - z_upper), np.abs(pos[:, 2] - z_lower))

        groups = _headgroup_atoms_from_graph(graph, d_min)
        group_xy = np.array([pos[sorted(g), :2].mean(axis=0) for g in groups])
        hub_xy.append(group_xy)

        all_kept = sorted({a for g in groups for a in g})
        xy_avg[i] = pos[all_kept, :2].mean(axis=0)
        z_avg[i] = pos[all_kept, 2].mean()

    return xy_avg, z_avg, hub_xy
