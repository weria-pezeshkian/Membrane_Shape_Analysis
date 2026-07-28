from __future__ import annotations

from ..core.fourier_sft import SFT


def calibrate(sft: SFT, radius: float, out_path: str) -> None:
    """Compute calibrated membrane material parameters from `sft` and write them to `out_path`.

    Not yet implemented - this is where the physics (e.g. kappa/sigma
    extraction from the Anm fluctuation spectrum) plugs in. Any such
    implementation must assert `sft.regularized in (False, None)` first:
    Tikhonov regularization biases Anm toward zero in proportion to
    curvature, which would circularly contaminate a fluctuation-spectrum
    fit built from it.
    """
    raise NotImplementedError("CALM calibrate's physics is not yet implemented.")
