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


def rotated_grid(X, Y, cx, cy, angle):
    """Coordinates to evaluate the (unrotated-as-fit) surface/data at, so the
    result at output grid point (X, Y) is what it would look like after
    rotating by `angle` around (cx, cy). This is the ONE rotation mechanism
    used throughout CALM: real data (atom positions, Anm, the hole mask) is
    never itself rotated - only the query point is transformed, so nothing
    needs redoing to account for a frame's rotation.

    Only meaningful (no PBC wraparound ambiguity - see TODO.md) for (X, Y)
    within fixed_circle_radius of (cx, cy): rotation preserves distance from
    the pivot exactly, so such points' transformed coordinates never actually
    leave the box under any angle, unlike points nearer the box edges.
    """
    dx = X - cx
    dy = Y - cy
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    x_old = cos_a * dx - sin_a * dy + cx
    y_old = sin_a * dx + cos_a * dy + cy
    return x_old, y_old


def lookup_mask_at_rotated_grid(mask, X, Y, Lx, Ly, cx, cy, angle):
    """Look up a boolean mask (computed on the canonical, unrotated grid -
    e.g. the hole mask from core/fourier_build.py) at the position each
    output grid point (X, Y) corresponds to in the original (unrotated,
    as-measured) frame, via nearest-cell snapping (a mask has no continuous
    interpolation the way Z() does). Only call this for (X, Y) within
    fixed_circle_radius of (cx, cy) - see rotated_grid's docstring."""
    x_old, y_old = rotated_grid(X, Y, cx, cy, angle)
    grid_rows, grid_cols = mask.shape
    col_idx = np.mod(np.round(x_old / Lx * grid_cols).astype(int), grid_cols)
    row_idx = np.mod(np.round(y_old / Ly * grid_rows).astype(int), grid_rows)
    return mask[row_idx, col_idx]


if __name__ == "__main__":
    pass
