"""Tests for core/packing.py::median_multiple_threshold."""

from __future__ import annotations

import numpy as np

from CALM.core.packing import median_multiple_threshold


def test_median_multiple_threshold_is_k_times_median() -> None:
    distances = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # median=3.0
    assert median_multiple_threshold(distances, k=3.0) == 9.0


def test_median_multiple_threshold_ignores_spread_unlike_a_mad_based_stat() -> None:
    # Same median, different spread: threshold must be the same.
    tight = np.array([4.9, 5.0, 5.1])
    wide = np.array([1.0, 5.0, 9.0])
    assert median_multiple_threshold(tight, k=3.0) == median_multiple_threshold(wide, k=3.0)


def test_median_multiple_threshold_grows_with_k() -> None:
    distances = np.full(10, 5.0)
    assert median_multiple_threshold(distances, k=4.0) > median_multiple_threshold(distances, k=2.0)


def test_median_multiple_threshold_returns_inf_for_fewer_than_two_values() -> None:
    assert median_multiple_threshold(np.array([5.0])) == np.inf
    assert median_multiple_threshold(np.array([])) == np.inf
