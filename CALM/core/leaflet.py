import numpy as np
import networkx as nx
from MDAnalysis.lib.distances import distance_array


def _split_score(sizes, total, min_balance):
    """Quality score for a candidate 2-leaflet split: coverage (fraction of
    the whole selection the two largest components account for), gated by
    balance (how close in size those two components are - 1.0 if equal, 0.0
    if one of them is empty).

    Coverage is NOT a reliable signal on its own: a split where every atom
    falls into exactly one of 2 components (e.g. one lipid mid flip-flop
    isolated as its own singleton "leaflet", everything else merged into one
    blob) trivially has coverage=1.0 despite being nonsense - nothing was
    excluded, it's just wildly lopsided. So balance acts as a hard gate,
    not a competing score: any split more lopsided than `min_balance`
    (`balance < min_balance`) is rejected outright (returns -1, sorts below
    every valid candidate), and coverage is only maximized *among* splits
    that pass the gate. This way a bigger, slightly-less-even split is
    always preferred over a smaller, more-perfectly-even one (accounting
    for more of the selection matters more than an extra bit of evenness),
    while a split that's technically "complete" but not a real bilayer split
    at all still gets rejected."""
    if len(sizes) < 2:
        return -1.0
    n0, n1 = sizes[0], sizes[1]
    balance = 1.0 - abs(n0 - n1) / (n0 + n1)
    if balance < min_balance:
        return -1.0
    return (n0 + n1) / total


def get_components(matrix, low_percentile=0.01, high_percentile=50.0, n_steps=40, min_balance=0.6):
    """Find the best 2-leaflet split of `matrix` (a periodic pairwise
    distance matrix over one selection) by scanning distance-percentile
    cutoffs (log-spaced across `n_steps` steps from `low_percentile` to
    `high_percentile`). At each cutoff, atoms within that distance of each
    other form a graph; its connected components are candidate leaflets.

    The cutoff producing exactly 2 connected components is not a reliable
    signal by itself: a lipid sitting between the two leaflets (e.g. mid
    flip-flop) can bridge them, so a cutoff at which it first reconnects to
    whichever leaflet it's nearer to can yield {that one atom} +
    {everything else merged into one blob} - technically 2 components, but a
    degenerate split. Every scanned cutoff is instead scored via
    _split_score (coverage, gated by balance >= min_balance - see its
    docstring), and the best-scoring split wins regardless of how many
    components that particular cutoff produced. Atoms outside the two
    returned components (e.g. a bridging lipid, or anything excluded by the
    balance gate) are implicitly excluded from both leaflets.

    Returns (best_components, best_cutoff) - the winning cutoff (an actual
    distance, not a percentile) is exposed so callers doing frame-to-frame
    leaflet tracking (see track_components) can persist and reuse the same
    connectivity notion across a trajectory instead of re-deriving a
    (potentially different) cutoff every frame.
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
            f"{low_percentile}-{high_percentile} - check the selection "
            "actually contains two distinguishable, reasonably-balanced "
            "leaflets, or lower min_balance."
        )
    return best_components, best_cutoff


def track_components(positions, box, prev_upper, prev_lower, cutoff):
    """Incrementally update a previous (upper, lower) leaflet assignment
    using the current frame's positions, instead of reclustering from
    scratch (which risks the two groups' identity being rebuilt
    inconsistently frame to frame, purely from search noise, even when the
    underlying membrane barely changed). positions/prev_upper/prev_lower/the
    return value are all in terms of LOCAL indices into `positions`
    (0..len(positions)-1) - the caller maps to/from global atom indices.

    Uses the SAME persisted `cutoff` (the winning distance from
    get_components' initial search, kept fixed rather than re-derived every
    frame) for both passes, applying the same single-linkage connectivity
    rule get_components used to build the groups in the first place:

    - Removal: an atom stays in its group only if it is still within
      `cutoff` of at least one OTHER atom currently in that group (a lone
      atom with no remaining group-mate is dropped). This is what catches a
      lipid mid flip-flop.
    - Addition: an unassigned atom joins whichever group it is within
      `cutoff` of - but only if that's true of EXACTLY ONE group. Within
      cutoff of both (ambiguous/bridging) or neither leaves it unassigned,
      matching get_components' own exclude-rather-than-guess philosophy.

    Atoms that were fine and stay fine are never touched by either pass -
    no global reshuffle.

    Returns (upper, lower) as sets of local indices.
    """
    positions = np.asarray(positions)
    n = len(positions)

    def _still_connected(group):
        idx = sorted(group)
        if len(idx) < 2:
            return set()  # no other group-mate left to anchor it
        d = distance_array(positions[idx], positions[idx], box=box)
        np.fill_diagonal(d, np.inf)
        min_other = d.min(axis=1)
        return {i for i, keep in zip(idx, min_other <= cutoff) if keep}

    upper = _still_connected(prev_upper)
    lower = _still_connected(prev_lower)

    unassigned = sorted(set(range(n)) - upper - lower)
    if unassigned and (upper or lower):
        cand_pos = positions[unassigned]

        def _min_dist_to(group):
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
            # both or neither -> stays unassigned

    return upper, lower


def apply_margin_filter(positions, box, upper, lower, margin=2.0):
    """Remove any atom from upper/lower whose distance to the nearest atom
    in the OTHER leaflet isn't at least `margin` times its distance to the
    nearest OTHER atom in its OWN leaflet - i.e. atoms that aren't
    decisively closer to one leaflet than the other. Removed atoms become
    unassigned (excluded from both), same exclude-rather-than-guess
    philosophy as get_components/track_components.

    This catches a failure mode neither XY-connectivity (get_components/
    track_components) nor a leaflet-local Z check can: a lipid squeezed
    toward mid-plane (e.g. by a nearby protein) or mid flip-flop can be
    perfectly well-connected sideways to its own leaflet's neighbors, so
    connectivity alone won't flag it - and a plain Z-vs-neighbors check
    can't distinguish it from genuine sharp local membrane curvature (which
    this tool needs to keep, not suppress: both look like "Z differs from
    same-leaflet neighbors"). The distinguishing signal is inter-leaflet,
    not intra-leaflet: real curvature moves a whole local patch of one
    leaflet together, staying uniformly far from the OTHER leaflet even
    where it bends sharply, while a structural anomaly like a flip-flopping
    lipid sits suspiciously close to the other leaflet's surface in full 3D
    - regardless of how well-connected it looks to its own side.

    margin=2.0 is data-grounded, not arbitrary: on a real system, >99% of
    atoms had other/own ratio > 4 (comfortably decisive), while a known
    problem lipid (squeezed near a protein, confirmed structurally
    anomalous) sat at ratio ~1.9 - a clean, well-separated gap, with 2.0
    sitting inside it.

    positions/upper/lower/return value are LOCAL indices, as elsewhere in
    this module. Returns (upper, lower) as sets.
    """
    positions = np.asarray(positions)
    upper_idx = np.array(sorted(upper), dtype=int)
    lower_idx = np.array(sorted(lower), dtype=int)

    def _own_nearest(idx):
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


def _label_by_z(component_a, component_b, positions):
    """Decide which of two candidate leaflet components is "upper" (higher
    mean z) vs "lower", given LOCAL indices into `positions`. Returns
    (upper, lower) as the same two components, reordered."""
    z_a = np.mean(positions[sorted(component_a), 2])
    z_b = np.mean(positions[sorted(component_b), 2])
    return (component_a, component_b) if z_a > z_b else (component_b, component_a)


if __name__=="__main__":
    pass
