from __future__ import annotations

import numpy as np
import numpy.typing as npt


def median_multiple_threshold(distances: npt.ArrayLike, k: float = 3.0) -> float:
    """Return k * median(distances); np.inf if fewer than 2 values are given.

    Calibrate against the same kind of distance being tested against:
    atom-to-atom and grid-to-atom distances have different distributions
    and are not interchangeable.
    """
    values = np.asarray(distances, dtype=float).ravel()
    if len(values) < 2:
        return np.inf
    return float(k * np.median(values))


if __name__ == "__main__":
    pass
