import numpy as np


def median_multiple_threshold(distances, k=3.0):
    """"This counts as unusually far" cutoff for an array of distances:
    k * median(distances).

    Structurally the same idea as core/fourier_build.py's _tmd_threshold
    (Nyquist: 0.5 * the fit's own characteristic wavelength) - a multiple of
    a characteristic length scale - just anchored to the DATA's own
    characteristic spacing (its median) instead of the fit's chosen
    resolution. Physically interpretable regardless of system: "how many
    typical spacings away before this counts as unsupported", the same kind
    of coordination-shell-cutoff reasoning already standard in MD analysis
    (e.g. a first-shell cutoff as ~1.2-1.5x the nearest-neighbor RDF peak).

    Deliberately NOT a spread-based (e.g. median + n*MAD) statistic: spread
    reflects how heterogeneous/disordered a PARTICULAR system's packing
    happens to be, which has nothing to do with holes - the same spread-
    based multiplier can translate to very different absolute distances on
    two different systems, without that difference meaning anything.
    Anchoring purely to the median (a system's typical case) instead of its
    spread keeps k's meaning stable across systems: confirmed empirically
    (k=3.0 gave 0.00% far-field false positives on a real system, matching
    a separately spread-tuned n_mad=6.0's 0.00% - same result via a
    portable mechanism instead of a system-specific tuned spread multiple).

    IMPORTANT: calibrate against the SAME kind of distance you intend to
    test against. E.g. atom-to-atom nearest-neighbor spacing is NOT
    interchangeable with grid-point-to-nearest-atom distance - real
    (sterically-excluded) lipid packing has tightly clustered atom-to-atom
    spacing, while grid points can legitimately fall in the gaps between
    tightly packed atoms even with no hole present, giving a wider
    distribution. Calibrating on the wrong one over-flags normal packing as
    if it were a hole.

    k is the one honestly-irreducible judgment call (analogous to choosing
    a coordination-shell cutoff multiple) - default 3.0.

    Returns np.inf if fewer than 2 distances are given (nothing to
    calibrate against - never the binding constraint against any other
    threshold).
    """
    distances = np.asarray(distances, dtype=float).ravel()
    if len(distances) < 2:
        return np.inf
    return k * np.median(distances)


if __name__=="__main__":
    pass
