from __future__ import annotations

import networkx as nx
import numpy as np
from MDAnalysis.lib.distances import distance_array


def _split_score(sizes: list[int], total: int, min_balance: float) -> float:
    """Score a candidate 2-leaflet split.

    Score is the fraction of `total` covered by the two largest components,
    or -1.0 if there are fewer than 2 components or their size balance
    (1.0 = equal sizes, 0.0 = one is empty) is below `min_balance`.
    """
    if len(sizes) < 2:
        return -1.0
    n0, n1 = sizes[0], sizes[1]
    balance = 1.0 - abs(n0 - n1) / (n0 + n1)
    if balance < min_balance:
        return -1.0
    return (n0 + n1) / total


def get_components(
    matrix: np.ndarray,
    low_percentile: float = 0.01,
    high_percentile: float = 50.0,
    n_steps: int = 40,
    min_balance: float = 0.6,
) -> tuple[list[set[int]], float]:
    """Find the best 2-leaflet split of a periodic pairwise distance matrix.

    Scans distance-percentile cutoffs (log-spaced, `n_steps` steps from
    `low_percentile` to `high_percentile`); at each cutoff, atoms within
    that distance of each other form a graph, and its connected components
    are a candidate split, scored by `_split_score`. Returns the two
    largest components of the best-scoring cutoff and that cutoff's actual
    distance value.

    Raises ValueError if no cutoff yields a split meeting `min_balance`.
    """
    total = matrix.shape[0]
    best_score = -1.0
    best_components = None
    best_cutoff = None

    for p in np.geomspace(low_percentile, high_percentile, n_steps):
        dist_cutoff = np.percentile(matrix, p)
        adj_matrix = np.where(matrix > dist_cutoff, 0, matrix)
        G = nx.from_numpy_array(adj_matrix)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        score = _split_score([len(c) for c in components], total, min_balance)
        if score > best_score:
            best_score = score
            best_components = components[:2]
            best_cutoff = dist_cutoff

    if best_components is None or len(best_components) < 2:
        raise ValueError(
            "Could not find a 2-component split of the selection meeting "
            f"min_balance={min_balance} across percentiles "
            f"{low_percentile}-{high_percentile}."
        )
    assert best_cutoff is not None
    return best_components, best_cutoff


def track_components(
    positions: np.ndarray,
    box: np.ndarray,
    prev_upper: set[int],
    prev_lower: set[int],
    cutoff: float,
) -> tuple[set[int], set[int]]:
    """Update a previous (upper, lower) leaflet assignment for the current frame.

    `positions`, `prev_upper`, `prev_lower`, and the return value all use
    local indices into `positions`. Uses the persisted `cutoff` distance
    (from `get_components`) for both passes:

    - an atom is dropped from its group if no longer within `cutoff` of any
      other atom currently in that group;
    - an unassigned atom joins a group if it is within `cutoff` of exactly
      one group; within `cutoff` of both or neither leaves it unassigned.
    """
    positions = np.asarray(positions)
    n = len(positions)

    def _still_connected(group: set[int]) -> set[int]:
        idx = sorted(group)
        if len(idx) < 2:
            return set()
        d = distance_array(positions[idx], positions[idx], box=box)
        np.fill_diagonal(d, np.inf)
        min_other = d.min(axis=1)
        return {i for i, keep in zip(idx, min_other <= cutoff) if keep}

    upper = _still_connected(prev_upper)
    lower = _still_connected(prev_lower)

    unassigned = sorted(set(range(n)) - upper - lower)
    if unassigned and (upper or lower):
        cand_pos = positions[unassigned]

        def _min_dist_to(group: set[int]) -> np.ndarray:
            if not group:
                return np.full(len(unassigned), np.inf)
            d = distance_array(cand_pos, positions[sorted(group)], box=box)
            return d.min(axis=1)

        close_upper = _min_dist_to(upper) <= cutoff
        close_lower = _min_dist_to(lower) <= cutoff

        for i, cu, cl in zip(unassigned, close_upper, close_lower):
            if cu and not cl:
                upper.add(i)
            elif cl and not cu:
                lower.add(i)

    return upper, lower


def apply_margin_filter(
    positions: np.ndarray,
    box: np.ndarray,
    upper: set[int],
    lower: set[int],
    margin: float = 2.0,
) -> tuple[set[int], set[int]]:
    """Drop atoms that aren't decisively closer to one leaflet than the other.

    An atom is dropped from upper/lower if its distance to the nearest atom
    in the OTHER leaflet is less than `margin` times its distance to the
    nearest atom in its OWN leaflet. `positions`, `upper`, `lower`, and the
    return value use local indices.

    XY-connectivity alone (`get_components`/`track_components`) can miss an
    atom that is well-connected sideways to its own leaflet but squeezed
    toward the mid-plane (e.g. near a protein, or mid flip-flop); this
    checks the inter-leaflet distance directly instead, which genuine
    membrane curvature preserves but a structural anomaly does not.
    """
    positions = np.asarray(positions)
    upper_idx = np.array(sorted(upper), dtype=int)
    lower_idx = np.array(sorted(lower), dtype=int)

    def _own_nearest(idx: np.ndarray) -> np.ndarray:
        if len(idx) < 2:
            return np.full(len(idx), np.inf)
        d = distance_array(positions[idx], positions[idx], box=box)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)

    own_upper = _own_nearest(upper_idx)
    own_lower = _own_nearest(lower_idx)

    if len(upper_idx) and len(lower_idx):
        cross = distance_array(positions[upper_idx], positions[lower_idx], box=box)
        other_for_upper = cross.min(axis=1)
        other_for_lower = cross.min(axis=0)
    else:
        other_for_upper = np.full(len(upper_idx), np.inf)
        other_for_lower = np.full(len(lower_idx), np.inf)

    keep_upper = {i for i, own, other in zip(upper_idx, own_upper, other_for_upper) if other >= margin * own}
    keep_lower = {i for i, own, other in zip(lower_idx, own_lower, other_for_lower) if other >= margin * own}

    return keep_upper, keep_lower


def _label_by_z(
    component_a: set[int],
    component_b: set[int],
    positions: np.ndarray,
) -> tuple[set[int], set[int]]:
    """Return (component_a, component_b) reordered as (upper, lower) by mean z.

    Indices are local into `positions`.
    """
    z_a = np.mean(positions[sorted(component_a), 2])
    z_b = np.mean(positions[sorted(component_b), 2])
    return (component_a, component_b) if z_a > z_b else (component_b, component_a)


if __name__ == "__main__":
    pass
