import numpy as np


def recover_rotation_angle(q_mn_frame, Lx, Ly, Nx, Ny):
    """Recover the rotation angle baked into this frame's q_mn (see
    core/fourier_build.py::_rotate_q / _fourier_by_layer) by comparing the
    actual (rotated) q at a reference mode against that mode's analytically
    known, unrotated value. Mode (m=1, n=0) - array index [1, 0] - always
    exists and always has a nonzero original q, since Nx/Ny >= 1 always
    (see get_fourier_modes).

    Shared by analyze/analyze.py (to evaluate the fit at a rotated grid) and
    utilize/get_vmd_visualization.py (to generate a VMD rotation script).
    """
    M = 2 * Nx + 1
    N = 2 * Ny + 1
    m = np.arange(M)
    n = np.arange(N)
    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)
    qx = 2 * np.pi * m / Lx
    qy = 2 * np.pi * n / Ly

    ref_i, ref_j = 1, 0
    qx0, qy0 = qx[ref_i], qy[ref_j]
    qx_rot, qy_rot = q_mn_frame[0, ref_i, ref_j], q_mn_frame[1, ref_i, ref_j]

    return np.arctan2(
        qx0 * qy_rot - qy0 * qx_rot,
        qx0 * qx_rot + qy0 * qy_rot,
    )


if __name__ == "__main__":
    pass
