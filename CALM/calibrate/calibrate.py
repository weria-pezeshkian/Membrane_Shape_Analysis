from __future__ import annotations

from ..core.fourier_sft import SFT


def calibrate(sft: SFT, radius: float, out_path: str) -> None:
    """Compute calibrated membrane material parameters from `sft` and write them to `out_path`.

    Not yet implemented - this is where the physics (e.g. kappa/sigma
    extraction from the Anm fluctuation spectrum) plugs in. See TODO.md's
    "Regularized Anm must never feed kappa/sigma calibration" entry for the
    constraint any such implementation must respect (unregularized Anm only).
    """
    raise NotImplementedError("CALM calibrate's physics is not yet implemented.")
