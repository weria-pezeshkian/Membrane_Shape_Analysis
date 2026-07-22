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


# radians; distinguishes "q_mn present but --rotate wasn't used" (theta == 0.0 exactly) from a real rotation
ROTATION_ANGLE_TOLERANCE = 1e-9


def recover_all_rotation_angles(sft):
    """recover_rotation_angle for every frame in an SFT (q_mn/dimensions
    already loaded, e.g. via SFT.from_directory)."""
    n_frames, _, M, N = sft.q_mn.shape
    Nx = (M - 1) // 2
    Ny = (N - 1) // 2
    return np.array([
        recover_rotation_angle(sft.q_mn[i], sft.dimensions[i, 0], sft.dimensions[i, 1], Nx, Ny)
        for i in range(n_frames)
    ])


def rotation_was_used(sft, tolerance=ROTATION_ANGLE_TOLERANCE):
    """Whether --rotate was actually used to build this SFT. q_mn is always
    saved (see core/fourier_build.py::_one_frame) whether or not --rotate was
    given, so file presence alone isn't a reliable signal; this checks
    whether the recovered angle is genuinely nonzero for at least one frame.
    Frame 0 alone is not a reliable check even when rotation was used, since
    it's the reference frame (its own recovered angle is always ~0) - so
    every frame is checked."""
    return bool(np.any(np.abs(recover_all_rotation_angles(sft)) > tolerance))


def fixed_circle_radius(sft):
    """The radius (in sft.dimensions' units) of the largest same-centered
    circle that fits inside every frame's box - see analyze/analyze.py's
    circle_cutter. Box size can drift per frame (NPT), and circle_cutter's
    per-frame radius is min(Lx,Ly)/2 of that frame's own box; since all these
    circles share the same center, the region valid across every frame is
    simply the smallest of these radii."""
    return float(np.min(np.minimum(sft.dimensions[:, 0], sft.dimensions[:, 1])) / 2.0)


if __name__ == "__main__":
    pass
