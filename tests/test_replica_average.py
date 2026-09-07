"""Tests for map/replica_average.py: add_replica_argument's CLI shape, all_replica_dirs's ordering,
and align_and_average's interpolation/NaN-exclusion/statistics.
"""

from __future__ import annotations

import argparse

import numpy as np

from CALM.map.replica_average import add_replica_argument, align_and_average, all_replica_dirs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--numpys_directory", required=True)
    add_replica_argument(parser)
    return parser


def test_replica_flag_defaults_to_empty_and_is_repeatable() -> None:
    parser = _parser()
    assert parser.parse_args(["-i", "dir0"]).replicas == []
    ns = parser.parse_args(["-i", "dir0", "--replica", "dir1", "--replica", "dir2"])
    assert ns.replicas == ["dir1", "dir2"]


def test_all_replica_dirs_puts_the_primary_directory_first() -> None:
    ns = _parser().parse_args(["-i", "dir0", "--replica", "dir1", "--replica", "dir2"])
    assert all_replica_dirs(ns) == ["dir0", "dir1", "dir2"]


def test_all_replica_dirs_is_a_single_element_list_with_no_replica_given() -> None:
    ns = _parser().parse_args(["-i", "dir0"])
    assert all_replica_dirs(ns) == ["dir0"]


def test_align_and_average_recovers_exact_mean_on_identical_x_grids() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    curves = [(x, np.array([1.0, 2.0, 3.0, 4.0])), (x, np.array([3.0, 4.0, 5.0, 6.0]))]
    x_common, mean, std, n = align_and_average(curves)
    assert np.array_equal(x_common, x)
    assert np.allclose(mean, [2.0, 3.0, 4.0, 5.0])
    assert np.allclose(std, [1.0, 1.0, 1.0, 1.0])
    assert n == 2


def test_align_and_average_uses_the_shortest_range_curve_as_the_common_grid() -> None:
    long_curve = (np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    short_curve = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    x_common, mean, std, n = align_and_average([long_curve, short_curve])
    assert np.array_equal(x_common, short_curve[0])
    assert mean.shape == (3,)


def test_align_and_average_excludes_out_of_range_replicas_via_nan_not_extrapolation() -> None:
    # short_curve is the reference (smaller last-x); a curve that starts
    # LATER than short_curve's own first point must not contribute a
    # flatlined value at those early points - it should be excluded there
    # (nanmean over the other curve(s) alone), not pull the mean toward its
    # own edge value the way plain np.interp's default left/right would.
    short_curve = (np.array([0.0, 1.0, 2.0]), np.array([10.0, 10.0, 10.0]))
    late_starting_curve = (np.array([1.0, 2.0, 3.0]), np.array([100.0, 100.0, 100.0]))
    x_common, mean, std, n = align_and_average([short_curve, late_starting_curve])
    assert np.array_equal(x_common, short_curve[0])
    # At x=0.0, late_starting_curve has no data at all - mean must come
    # from short_curve alone (10.0), not be pulled toward 100.0.
    assert mean[0] == 10.0
    assert std[0] == 0.0
    # At x=1.0 and x=2.0, both curves contribute.
    assert np.allclose(mean[1:], 55.0)


def test_align_and_average_single_curve_is_the_identity() -> None:
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([1.0, 2.0, 3.0])
    x_common, mean, std, n = align_and_average([(x, y)])
    assert np.array_equal(x_common, x)
    assert np.array_equal(mean, y)
    assert np.all(std == 0.0)
    assert n == 1
