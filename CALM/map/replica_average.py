from __future__ import annotations

import argparse

import numpy as np


def add_replica_argument(parser: argparse.ArgumentParser) -> None:
    """Add the repeatable --replica flag: each occurrence names one more 'numpys_directory' to
    average together with -i, treating it as an independent replica of the same system.

    Shared by every 'map' command that supports this (currently
    radial_plot, diffusion_plot - a 2D spatial map or a per-frame video
    doesn't reduce to "one curve" the same simple way a radial or MSD(tau)
    profile does, so those aren't offered it). -i alone (no --replica at
    all) is unchanged from before this flag existed - averaging only ever
    kicks in once there is more than one directory.
    """
    parser.add_argument(
        "--replica", dest="replicas", action="append", default=[], metavar="DIRECTORY",
        help="another 'numpys_directory' to average together with -i (repeatable) - if given at "
             "least once, the plotted curve becomes the mean across -i and every --replica, with "
             "a shaded +/- 1 std band showing how much the replicas actually disagree; -i alone "
             "(the default) is unchanged",
    )


def all_replica_dirs(ns: argparse.Namespace) -> list[str]:
    """[-i's directory, *every --replica directory], in the order given - the group to average
    over. A single-element list (no --replica given) means "no averaging", exactly the behavior
    from before this flag existed."""
    return [ns.numpys_directory, *ns.replicas]


def align_and_average(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(x_common, mean, std, n) across `curves` - a list of (x, y) arrays, one per replica, each
    already sorted by x.

    `x_common` is the shortest-range curve's own x values (the one whose
    last x is smallest) - every other curve is linearly interpolated onto
    it, so nothing is extrapolated past a shorter replica's own range (a
    replica that stops earlier - e.g. a tracked point that left the
    region partway through, or a bigger protein leaving a bigger --Remove-TMD
    hole - shouldn't have its missing tail guessed at). Points outside a
    given curve's own range (`left`/`right` of `np.interp`) are marked NaN
    rather than flatlined to that curve's own endpoint value, and
    `mean`/`std` are taken with `nanmean`/`nanstd`, so a replica that
    simply doesn't reach a particular x is excluded there rather than
    silently pulling the average toward its own edge value. The reference
    curve itself always contributes at every one of its own x_common
    points (interpolating a curve at its own exact x-values is exact, not
    an approximation), so `nanmean`/`nanstd` never see an all-NaN column.

    `std` is the population std (ddof=0) across replicas at each point,
    not the standard error - it answers "how much do independent replicas
    actually disagree", which is what a replica-comparison plot is for;
    with only 2-3 replicas (typical), the standard error can look
    deceptively tight.
    """
    assert curves
    x_common = min(curves, key=lambda c: c[0][-1])[0]
    aligned = np.array([
        np.interp(x_common, x, y, left=np.nan, right=np.nan) for x, y in curves
    ])
    return x_common, np.nanmean(aligned, axis=0), np.nanstd(aligned, axis=0), len(curves)


if __name__ == "__main__":
    pass
