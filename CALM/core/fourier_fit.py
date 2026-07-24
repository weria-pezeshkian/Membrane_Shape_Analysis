import numpy as np


def fit_coefficients(Data_3M, Lx, Ly, Nx, Ny, regularize=False):
    """Solve for the least-squares Fourier coefficients (Anm) that best
    reproduce z-values at the given (x, y) points.

    Data_3M must have shape (3, M): rows are x, y, z.
    Returns an Anm array of shape (2*Nx+1, 2*Ny+1), ready for
    Fourier_Series_Function.setAnm.

    If `regularize` (default False - must be explicitly opted into), adds a
    Tikhonov penalty weighted by each mode's |q|^2 = (q0*i)^2 + (p0*j)^2 - a
    discretized Helfrich bending-energy prior (curvature is expensive,
    flatness is free), rather than a uniform ridge that would shrink
    well-supported long-wavelength modes exactly as hard as poorly-supported
    short-wavelength ones. The DC mode (i=j=0, the average height) always has
    weight 0 - it has no curvature and there's no physical reason to bias it
    toward zero. Strength auto-scales from the same Length/M ratio used for
    the warning below: ~0 when M >> Length (well-determined, left untouched),
    ramping up smoothly as Length approaches M, growing sharply as the plain
    least-squares problem approaches singularity - see the `ramp`/`alpha`
    comment further down.

    Off by default deliberately: regularizing biases Anm toward zero in
    proportion to curvature - fine for single/few-frame visualization, but it
    would circularly contaminate any later kappa/sigma (bending
    rigidity/tension) calibration derived from these coefficients' cross-
    frame ensemble statistics, which needs unbiased per-frame Anm to be
    valid. See TODO.md.
    """
    if Data_3M.shape[0] != 3:
        raise ValueError("Data_3M must have shape (3, M).")

    q0 = 2 * np.pi / Lx
    p0 = 2 * np.pi / Ly

    M = Data_3M.shape[1]
    Length = (2 * Nx + 1) * (2 * Ny + 1)

    # Length is the number of unknown Anm coefficients being solved for (one
    # per (i,j) mode); M is the number of equations (one per fit atom). Least
    # squares needs redundancy (M >> Length) to average noise away - as
    # Length approaches M, each coefficient is "voted on" by fewer atoms, so
    # per-atom noise gets absorbed into the fit instead of smoothed out
    # (same phenomenon as fitting a high-degree polynomial to too few
    # points). Curvature then amplifies that overfit further, since Zxx/Zyy/
    # Zxy scale each mode's coefficient by (q0*i)^2 / (p0*j)^2 / (q0*p0*i*j).
    if Length >= M:
        if regularize:
            print(f"NOTE: Fourier fit is underdetermined ({Length} coefficients for "
                  f"only {M} fit atoms) - auto-scaled curvature-weighted (Tikhonov) "
                  "regularization is being applied to keep the fit physical. This "
                  "biases Anm - do not use this build for kappa/sigma calibration.")
        else:
            print(f"WARNING: Fourier fit is underdetermined: {Length} coefficients "
                  f"((2*{Nx}+1)*(2*{Ny}+1)) for only {M} fit atoms, and regularization "
                  "is OFF. The least-squares solution is not unique or noise-robust - "
                  "curvature is likely unphysical. Increase lambda_x/lambda_y, use a "
                  "selection with more atoms, or opt into --regularization (visualization "
                  "use only - see fit_coefficients()'s docstring).")
    elif M < 3 * Length:
        if regularize:
            print(f"NOTE: Fourier fit has low redundancy ({Length} coefficients for "
                  f"{M} fit atoms, less than 3x oversampling) - auto-scaled "
                  "regularization is being applied. This biases Anm - do not use this "
                  "build for kappa/sigma calibration.")
        else:
            print(f"WARNING: Fourier fit has low redundancy: {Length} coefficients for "
                  f"{M} fit atoms (less than 3x oversampling), and regularization is "
                  "OFF. Curvature may be noisy or overfit - consider increasing "
                  "lambda_x/lambda_y or opting into --regularization (visualization "
                  "use only).")
    elif M < 10 * Length:
        # Separate, stricter tier: NOT a numerical-conditioning problem (the
        # 3x check above is calibrated for that), and NOT fixed by
        # regularize=True (which ramps to ~0 well before this point by
        # design - it only responds to the same Length/M ratio, and this
        # tier is specifically the range where that ratio is still "safe"
        # by that measure). This is a statistical (bias-variance) issue:
        # even a numerically well-determined fit can resolve real per-lipid
        # thermal protrusions as if they were membrane curvature. Confirmed
        # empirically across M=150/300/600 with realistic ~1.5 A protrusion
        # noise: curvature stays near a sane membrane ballpark above
        # M/Length~10, and can already be an order of magnitude too high by
        # M/Length~3-5 - this tier's cutoff (10x) is an empirical, not
        # analytically derived, judgment call from that data and may need
        # retuning against real trajectories.
        print(f"WARNING: Fourier fit has only {M/Length:.1f}x oversampling ({Length} "
              f"coefficients for {M} fit atoms) - below numerical risk, but curvature "
              "may still be inflated beyond a physically reasonable membrane ballpark "
              "by real per-lipid thermal protrusions being resolved as if they were "
              "membrane shape. Consider increasing lambda_x/lambda_y (regularization "
              "does not fix this).")

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
        # r/(1-r): ~0 for r << 1 (well-determined, negligible penalty), ramps
        # up through the M < 3*Length caution zone (r=1/3 -> ramp=0.5), and
        # grows sharply as r -> 1 (Length -> M, where plain lstsq would go
        # singular) - clipped short of the r=1 pole itself. Scaled by var(b)
        # so the penalty sits in the natural units of the data's own height
        # fluctuations rather than an absolute number needing per-system
        # retuning.
        ramp = r / max(1e-6, 1.0 - min(r, 0.999))
        alpha = np.var(b) * ramp
        gamma = np.sqrt(alpha * mode_weights)
        A_reg = np.vstack([A, np.diag(gamma)])
        b_reg = np.concatenate([b, np.zeros(Length)])
        coeffs, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return coeffs.reshape((2 * Nx + 1, 2 * Ny + 1))


if __name__=="__main__":
    pass
