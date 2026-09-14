"""Tests for core/leaflet.py: the leaflet-detection/tracking algorithm.

Covers:
- _split_score: balance gate + coverage-maximization scoring
- get_components: robustness to a bridging atom connecting two leaflets
- track_components / _label_by_z: frame-to-frame leaflet tracking
- apply_margin_filter: inter-leaflet decisiveness check
"""

from __future__ import annotations

import numpy as np
import pytest

from CALM.core.leaflet import _label_by_z, _split_score, apply_margin_filter, get_components, track_components


def test_split_score_rejects_below_min_balance() -> None:
    # 9 vs 1 -> balance = 1 - 8/10 = 0.2, below a 0.6 gate.
    assert _split_score([9, 1], total=10, min_balance=0.6) == -1.0


def test_split_score_accepts_and_returns_coverage_above_min_balance() -> None:
    # 6 vs 4 -> balance = 1 - 2/10 = 0.8, passes a 0.6 gate; coverage = 1.0.
    assert _split_score([6, 4], total=10, min_balance=0.6) == pytest.approx(1.0)


def test_split_score_prefers_more_coverage_over_more_balance_when_both_pass_gate() -> None:
    # Both pass a 0.6 gate: {5,5} out of 1000 (balance=1.0, coverage=0.01)
    # vs {480,520} out of 1000 (balance=0.96, coverage=1.0). The bigger,
    # slightly less balanced split should win once the gate is satisfied.
    tiny_balanced = _split_score([5, 5], total=1000, min_balance=0.6)
    big_slightly_uneven = _split_score([480, 520], total=1000, min_balance=0.6)
    assert big_slightly_uneven > tiny_balanced


def test_split_score_needs_at_least_two_components() -> None:
    assert _split_score([10], total=10, min_balance=0.6) == -1.0


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def test_get_components_excludes_a_bridging_atom_between_two_leaflets() -> None:
    rng = np.random.default_rng(0)
    leaflet_a = rng.normal(loc=[0, 0, 0], scale=0.5, size=(10, 3))
    leaflet_b = rng.normal(loc=[0, 0, 20], scale=0.5, size=(10, 3))
    bridge = np.array([[0.0, 0.0, 10.0]])  # roughly equidistant from both

    points = np.vstack([leaflet_a, leaflet_b, bridge])
    matrix = _pairwise_distances(points)

    components, cutoff = get_components(matrix, min_balance=0.6)
    sizes = sorted(len(c) for c in components)
    assert sizes == [10, 10]  # both leaflets found intact

    bridge_idx = 20
    assert not any(bridge_idx in c for c in components)  # bridge excluded from both


def test_get_components_raises_for_a_single_atom_selection() -> None:
    # A 1-atom selection can never form 2 components at any threshold.
    matrix = np.zeros((1, 1))
    with pytest.raises(ValueError, match="Could not find a 2-component split"):
        get_components(matrix, min_balance=0.6)


# ---- track_components / _label_by_z (frame-to-frame leaflet tracking) ----

def test_label_by_z_orders_higher_mean_z_first() -> None:
    positions = np.array([
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 30.0],
    ])
    upper, lower = _label_by_z({0}, {1}, positions)
    assert upper == {1}
    assert lower == {0}


def test_track_components_removes_atom_that_drifted_away() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 50.0, 50.0],  # atom 2 drifted far away
        [0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0],
    ])
    upper, lower = track_components(positions, None, {0, 1, 2}, {3, 4, 5}, cutoff=3.0)
    assert upper == {0, 1}  # atom 2 dropped - no longer near its own group
    assert lower == {3, 4, 5}
    assert 2 not in upper and 2 not in lower  # also too far from the other group to be re-added


def test_track_components_adds_unambiguous_atom_to_the_group_it_is_near() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],    # group A (0,1)
        [0.0, 0.0, 20.0], [1.0, 0.0, 20.0],  # group B (2,3)
        [0.5, 0.0, 0.5],                      # unassigned, close only to group A
    ])
    upper, lower = track_components(positions, None, {0, 1}, {2, 3}, cutoff=3.0)
    assert upper == {0, 1, 4}
    assert lower == {2, 3}


def test_track_components_leaves_atom_close_to_both_groups_unassigned() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # group A near z=0
        [0.0, 0.0, 6.0], [1.0, 0.0, 6.0],  # group B near z=6
        [0.5, 0.0, 3.0],                    # ~3.04 from each - within cutoff of both
    ])
    upper, lower = track_components(positions, None, {0, 1}, {2, 3}, cutoff=4.0)
    assert upper == {0, 1}
    assert lower == {2, 3}
    assert 4 not in upper and 4 not in lower


def test_track_components_leaves_stable_atoms_untouched() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0],
    ])
    upper, lower = track_components(positions, None, {0, 1, 2}, {3, 4, 5}, cutoff=3.0)
    assert upper == {0, 1, 2}
    assert lower == {3, 4, 5}


# ---- apply_margin_filter (inter-leaflet decisiveness check) ----

def test_apply_margin_filter_keeps_normal_well_separated_atoms() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0],
    ])
    upper, lower = apply_margin_filter(positions, None, {0, 1, 2}, {3, 4, 5}, margin=2.0)
    assert upper == {0, 1, 2}
    assert lower == {3, 4, 5}


def test_apply_margin_filter_removes_atom_squeezed_toward_midplane() -> None:
    # Own-leaflet group is tight (spacing ~1), but this atom sits almost
    # exactly halfway between the two leaflets (z=10, leaflets at 0 and 20):
    # not decisively closer to either side.
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 10.0],  # atom 2: squeezed toward mid-plane
        [0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0],
    ])
    upper, lower = apply_margin_filter(positions, None, {0, 1, 2}, {3, 4, 5}, margin=2.0)
    assert 2 not in upper
    assert upper == {0, 1}
    assert lower == {3, 4, 5}


def test_apply_margin_filter_does_not_flag_genuine_coherent_curvature() -> None:
    # Both leaflets bend toward each other together in a local region (a
    # smooth Gaussian dip/bulge affecting neighboring points together, not
    # one isolated spike), narrowing local thickness from 30 to 14 but
    # staying physically sane. Real curvature must survive this filter.
    grid = np.array([[i, j] for i in range(-3, 4) for j in range(-3, 4)], dtype=float) * 3.0
    r2 = grid[:, 0] ** 2 + grid[:, 1] ** 2
    sigma2 = 5.0 ** 2
    upper_z = 30.0 - 10.0 * np.exp(-r2 / sigma2)
    lower_z = 0.0 + 6.0 * np.exp(-r2 / sigma2)
    assert upper_z.min() - lower_z.max() > 10.0  # sanity: still a real, sane local gap

    positions = np.vstack([
        np.column_stack([grid, upper_z]),
        np.column_stack([grid, lower_z]),
    ])
    n = len(grid)
    upper, lower = apply_margin_filter(positions, None, set(range(n)), set(range(n, 2 * n)), margin=2.0)
    assert len(upper) == n  # nothing removed
    assert len(lower) == n


def test_apply_margin_filter_handles_empty_leaflet() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    upper, lower = apply_margin_filter(positions, None, {0, 1}, set(), margin=2.0)
    assert upper == {0, 1}
    assert lower == set()
