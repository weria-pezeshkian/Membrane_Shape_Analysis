from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def fit_coefficients(
    Data_3M: np.ndarray,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    regularize: bool = False,
    diagnostics: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """Least-squares Fourier coefficients (Anm) reproducing z at given (x, y) points.

    `Data_3M` has shape (3, M): rows are x, y, z. Returns an Anm array of
    shape (2*Nx+1, 2*Ny+1), for `Fourier_Series_Function.setAnm`.

    If `regularize` (default False), adds a Tikhonov penalty weighted by
    each mode's |q|^2 = (q0*i)^2 + (p0*j)^2 (a discretized Helfrich
    bending-energy prior), so long-wavelength modes aren't shrunk as hard
    as short-wavelength ones. The DC mode (i=j=0) always has weight 0.
    Strength auto-scales with how underdetermined the fit is (see `ramp`
    below).

    Regularizing biases Anm toward zero in proportion to curvature: usable
    for single/few-frame visualization, but not for kappa/sigma calibration
    from cross-frame Anm statistics, which needs unbiased per-frame Anm.

    Runs inside a per-frame worker process (via `_fourier_by_layer` /
    `_one_frame`). When `diagnostics` is given, a `(level, message)` pair
    is appended to it for underdetermined/low-redundancy/low-oversampling
    fits; the caller collects these across frames and logs them once back
    in the main process, keeping every write to the replay log
    single-process.
    """
    if Data_3M.shape[0] != 3:
        raise ValueError("Data_3M must have shape (3, M).")

    q0 = 2 * np.pi / Lx
    p0 = 2 * np.pi / Ly

    M = Data_3M.shape[1]
    Length = (2 * Nx + 1) * (2 * Ny + 1)
    if diagnostics is None:
        diagnostics = []

    # Length: number of unknown Anm coefficients. M: number of fit atoms.
    # Least squares needs M >> Length for noise to average out rather than
    # be absorbed into the fit; curvature further amplifies overfit since
    # Zxx/Zyy/Zxy scale each mode by (q0*i)^2 / (p0*j)^2 / (q0*p0*i*j).
    if Length >= M:
        if regularize:
            diagnostics.append(("info", f"Fourier fit is underdetermined ({Length} coefficients for "
                  f"only {M} fit atoms) - auto-scaled curvature-weighted (Tikhonov) "
                  "regularization is being applied to keep the fit physical. This "
                  "biases Anm - do not use this build for kappa/sigma calibration."))
        else:
            diagnostics.append(("warning", f"Fourier fit is underdetermined: {Length} coefficients "
                  f"((2*{Nx}+1)*(2*{Ny}+1)) for only {M} fit atoms, and regularization "
                  "is OFF. The least-squares solution is not unique or noise-robust - "
                  "curvature is likely unphysical. Increase lambda_x/lambda_y, use a "
                  "selection with more atoms, or opt into --regularization (visualization "
                  "use only)."))
    elif M < 3 * Length:
        if regularize:
            diagnostics.append(("info", f"Fourier fit has low redundancy ({Length} coefficients for "
                  f"{M} fit atoms, less than 3x oversampling) - auto-scaled "
                  "regularization is being applied. This biases Anm - do not use this "
                  "build for kappa/sigma calibration."))
        else:
            diagnostics.append(("warning", f"Fourier fit has low redundancy: {Length} coefficients for "
                  f"{M} fit atoms (less than 3x oversampling), and regularization is "
                  "OFF. Curvature may be noisy or overfit - consider increasing "
                  "lambda_x/lambda_y or opting into --regularization (visualization "
                  "use only)."))
    elif M < 10 * Length:
        # Statistical (bias-variance), not numerical-conditioning: even a
        # well-determined fit here can resolve per-lipid thermal protrusions
        # as membrane curvature. The 10x cutoff is an empirical judgment
        # call, not analytically derived, and may need retuning.
        diagnostics.append(("warning", f"Fourier fit has only {M/Length:.1f}x oversampling ({Length} "
              f"coefficients for {M} fit atoms) - below numerical risk, but curvature "
              "may still be inflated beyond a physically reasonable membrane ballpark "
              "by real per-lipid thermal protrusions being resolved as if they were "
              "membrane shape. Consider increasing lambda_x/lambda_y (regularization "
              "does not fix this)."))

    A = np.zeros((M, Length))
    b = Data_3M[2, :]
    mode_weights = np.zeros(Length)

    index = 0
    for i in range(-Nx, Nx + 1):
        for j in range(-Ny, Ny + 1):
            for k in range(M):
                x, y = Data_3M[0, k], Data_3M[1, k]
                A[k, index] = (
                    np.cos(q0 * i * x + p0 * j * y) +
                    np.sin(q0 * i * x + p0 * j * y)
                )
            mode_weights[index] = (q0 * i) ** 2 + (p0 * j) ** 2
            index += 1

    if regularize:
        r = Length / M
        # r/(1-r): ~0 for r << 1, ramps through the M < 3*Length caution
        # zone (r=1/3 -> ramp=0.5), grows sharply as r -> 1 (Length -> M,
        # where plain lstsq would go singular), clipped short of that pole.
        # Scaled by var(b) so the penalty is in the data's own units.
        ramp = r / max(1e-6, 1.0 - min(r, 0.999))
        alpha = np.var(b) * ramp
        gamma = np.sqrt(alpha * mode_weights)
        A_reg = np.vstack([A, np.diag(gamma)])
        b_reg = np.concatenate([b, np.zeros(Length)])
        coeffs, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return coeffs.reshape((2 * Nx + 1, 2 * Ny + 1))


if __name__ == "__main__":
    pass
