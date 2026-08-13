from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fourier_sft import SFT

# Radians; distinguishes "q_mn present but --rotate wasn't used" (theta == 0.0
# exactly) from a real rotation.
ROTATION_ANGLE_TOLERANCE = 1e-9


def recover_rotation_angle(
    q_mn_frame: np.ndarray, Lx: float, Ly: float, Nx: int, Ny: int
) -> float:
    """Recover the rotation angle baked into a frame's q_mn.

    Compares the frame's actual (possibly rotated) q at mode (m=1, n=0)
    against that mode's analytically known, unrotated value. That mode
    always exists with a nonzero unrotated q, since Nx, Ny >= 1 always (see
    `get_fourier_modes`). See `_rotate_q` / `_fourier_by_layer` in
    `fourier_build.py` for where the rotation is applied at build time.
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


def recover_all_rotation_angles(sft: SFT) -> np.ndarray:
    """`recover_rotation_angle` for every frame in an SFT with q_mn/dimensions loaded."""
    assert sft.q_mn is not None
    assert sft.dimensions is not None
    n_frames, _, M, N = sft.q_mn.shape
    Nx = (M - 1) // 2
    Ny = (N - 1) // 2
    return np.array([
        recover_rotation_angle(sft.q_mn[i], sft.dimensions[i, 0], sft.dimensions[i, 1], Nx, Ny)
        for i in range(n_frames)
    ])


def rotation_was_used(sft: SFT, tolerance: float = ROTATION_ANGLE_TOLERANCE) -> bool:
    """Whether --rotate was used to build this SFT.

    q_mn is always saved regardless of --rotate (see `_one_frame` in
    `fourier_build.py`), so file presence alone isn't a reliable signal.
    Checks every frame, since frame 0 is the reference frame and its own
    recovered angle is always ~0 even when rotation was used elsewhere.
    """
    return bool(np.any(np.abs(recover_all_rotation_angles(sft)) > tolerance))


def fixed_circle_radius(dimensions: np.ndarray) -> float:
    """Radius of the largest same-centered circle that fits inside every frame's box.

    `dimensions` is (n_frames, >=2): each row's first two columns are that
    frame's own Lx, Ly (an SFT's own `dimensions`, or dimensions.csv's
    columns for a command with no SFT, e.g. 'CALM analyze lipids'). Box
    size can drift per frame (NPT); each frame's own circle radius is
    min(Lx, Ly)/2 of that frame's box (see `circle_cutter` in
    `analyze/analyze.py`). Since all such circles share the same center,
    the region valid across every frame is the smallest of these radii.
    """
    return float(np.min(np.minimum(dimensions[:, 0], dimensions[:, 1])) / 2.0)


def rotated_grid(
    X: np.ndarray, Y: np.ndarray, cx: float, cy: float, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map output grid point (X, Y) to the as-fit coordinates it corresponds to under rotation.

    This is the one rotation mechanism used throughout CALM: fitted data
    (atom positions, Anm, the hole mask) is never itself rotated, only the
    query point is transformed.

    Only meaningful for (X, Y) within `fixed_circle_radius` of (cx, cy):
    rotation preserves distance from the pivot, so such points' transformed
    coordinates never leave the box under any angle, unlike points nearer
    the box edges.
    """
    dx = X - cx
    dy = Y - cy
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    x_old = cos_a * dx - sin_a * dy + cx
    y_old = sin_a * dx + cos_a * dy + cy
    return x_old, y_old


def lookup_mask_at_rotated_grid(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    cx: float,
    cy: float,
    angle: float,
) -> np.ndarray:
    """Look up a boolean mask (computed on the canonical, unrotated grid) under rotation.

    Uses nearest-cell snapping, since a mask has no continuous
    interpolation the way `Z()` does. Only meaningful for (X, Y) within
    `fixed_circle_radius` of (cx, cy) - see `rotated_grid`.
    """
    x_old, y_old = rotated_grid(X, Y, cx, cy, angle)
    grid_rows, grid_cols = mask.shape
    col_idx = np.mod(np.round(x_old / Lx * grid_cols).astype(int), grid_cols)
    row_idx = np.mod(np.round(y_old / Ly * grid_rows).astype(int), grid_rows)
    return mask[row_idx, col_idx]


if __name__ == "__main__":
    pass
