from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .curvature import _thickness_root

if TYPE_CHECKING:
    import MDAnalysis as mda

    from .fourier_core import Fourier_Series_Function


class _SurfaceAsInterp:
    """Adapts a `Fourier_Series_Function` to `f`/`_thickness_root`'s `interp(y, x, grid=False)` calling convention."""

    def __init__(self, fourier: Fourier_Series_Function) -> None:
        self._fourier = fourier

    def __call__(self, y: np.ndarray, x: np.ndarray, grid: bool = False) -> np.ndarray:
        return self._fourier.Z(np.asarray(x), np.asarray(y))


def _project_onto_surface(
    px: float, py: float, pz: float,
    fourier: Fourier_Series_Function,
    interp: _SurfaceAsInterp,
    t_max_multiple: float = 5.0,
    t_max_floor: float = 2.0,
) -> tuple[float, float, bool]:
    """Project (px, py, pz) onto `fourier`'s own fitted surface along the local normal at (px, py).

    The normal is the surface's own upward unit normal at (px, py):
    `(-Zx, -Zy, 1)` normalized. The root search itself (`_thickness_root`)
    walks along that normal until it crosses `interp`'s surface, exactly
    the way `analyze/analyze.py`'s thickness calculation walks from a
    mid-surface point to find the nearby leaflet - here the starting point
    is the tracked point's own real position instead.

    The search direction is chosen from where the point already sits
    relative to its own surface: `offset = pz - fourier.Z(px, py)`. A
    point already above its own surface needs a negative step to reach
    it, since `f`'s root function is always locally increasing at t=0
    (its derivative there equals the normal's own length, always
    positive, independent of the local slope) - so `upper=(offset < 0)`.

    The returned point stays in the same coordinate frame as the input:
    it is `(px, py)` plus the small projection step, with no wrap into
    `[0, Lx)`, so feeding in a continuous (unwrapped) input keeps the
    output continuous too.

    `t_max_base` scales with the point's own offset from the surface
    (`t_max_multiple` times it, floored at `t_max_floor`), since a real
    headgroup sits close to its own fitted surface. `converged` is True
    when `_thickness_root` finds a root within that search; when it
    doesn't, the point is returned exactly as given.
    """
    fx = float(fourier.Zx(np.asarray(px), np.asarray(py)))
    fy = float(fourier.Zy(np.asarray(px), np.asarray(py)))
    n = np.array([-fx, -fy, 1.0])
    n /= np.linalg.norm(n)

    z_local = float(fourier.Z(np.asarray(px), np.asarray(py)))
    offset = pz - z_local
    t_max_base = max(t_max_multiple * abs(offset), t_max_floor)
    upper = offset < 0.0

    t = _thickness_root(interp, px, py, pz, n[0], n[1], n[2], fourier.Lx, fourier.Ly, t_max_base, upper=upper)
    if t is None:
        return px, py, False
    return px + t * n[0], py + t * n[1], True


def _selection_centers(atomgroup: mda.core.groups.AtomGroup) -> tuple[np.ndarray, np.ndarray]:
    """(xy, z) per fragment in `atomgroup`: each fragment's own center of geometry over its selected atoms."""
    if len(atomgroup) == 0:
        return np.empty((0, 2)), np.empty((0,))
    centers = atomgroup.center_of_geometry(compound="fragments")
    return centers[:, :2], centers[:, 2]


def _break_into_segments(leaflet: np.ndarray, in_hole: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame-index pairs, end exclusive: every maximal run of consecutive frames
    where a tracked point stays assigned to one leaflet (`leaflet != 0`) with a constant hole
    status, starting a new segment at any leaflet flip, hole-status change, or unassigned frame.
    """
    n = len(leaflet)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        assigned = leaflet[i] != 0
        breaks = start is not None and (
            not assigned or leaflet[i] != leaflet[i - 1] or in_hole[i] != in_hole[i - 1]
        )
        if breaks:
            assert start is not None
            segments.append((start, i))
            start = None
        if assigned and start is None:
            start = i
    if start is not None:
        segments.append((start, n))
    return segments


def _multi_tau_msd(
    segments: list[np.ndarray], dt: float, max_tau_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tau, msd, n_samples): ensemble-and-time-averaged mean-squared displacement, pooled
    across every given segment's own (T, 2) xy trajectory.

    Each segment contributes every valid (t, t+tau) pair within its own
    length, for every tau up to `max_tau_fraction` of that segment's own
    length - a short segment only ever contributes to small tau, a long
    one to the full range, and every contribution at a shared tau value
    is pooled into the same average regardless of which segment or
    tracked point it came from.
    """
    if not segments:
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)

    max_len = max(len(seg) for seg in segments)
    max_tau_frames = max(1, int(max_len * max_tau_fraction))

    sums = np.zeros(max_tau_frames)
    counts = np.zeros(max_tau_frames, dtype=int)
    for seg in segments:
        seg_max_tau = max(1, int(len(seg) * max_tau_fraction))
        for tau_frames in range(1, seg_max_tau + 1):
            disp = seg[tau_frames:] - seg[:-tau_frames]
            sq = np.sum(disp ** 2, axis=1)
            sums[tau_frames - 1] += sq.sum()
            counts[tau_frames - 1] += len(sq)

    valid = counts > 0
    tau_frames_valid = np.arange(1, max_tau_frames + 1)[valid]
    msd = sums[valid] / counts[valid]
    n_samples = counts[valid]
    tau = tau_frames_valid * dt
    return tau, msd, n_samples


def _fit_diffusion_coefficient(
    tau: np.ndarray, msd: np.ndarray, fit_tau_min_fraction: float, fit_tau_max_fraction: float,
) -> tuple[float, float, float, float]:
    """(D, D_stderr, r2, loglog_slope): the lateral diffusion coefficient from an MSD(tau) curve.

    D is the slope, divided by 4 (the 2D Einstein relation MSD = 4*D*tau),
    of a linear fit over the window
    `[fit_tau_min_fraction, fit_tau_max_fraction]` of the curve's own max
    tau - the earliest lags (dominated by local rattling, not diffusion)
    and the longest ones (fewest independent samples) are left out of the
    fit. `loglog_slope` is the slope of log(msd) against log(tau) over
    the same window: a value near 1 is the signature of normal diffusion;
    a value near 2 is the signature of ballistic motion over that window.
    """
    if len(tau) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    tau_max = tau[-1]
    window = (tau >= fit_tau_min_fraction * tau_max) & (tau <= fit_tau_max_fraction * tau_max)
    if window.sum() < 2:
        window = np.ones_like(tau, dtype=bool)

    t_fit = tau[window]
    m_fit = msd[window]

    slope, intercept = np.polyfit(t_fit, m_fit, 1)
    D = slope / 4.0

    predicted = slope * t_fit + intercept
    ss_res = np.sum((m_fit - predicted) ** 2)
    ss_tot = np.sum((m_fit - m_fit.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    log_slope, _ = np.polyfit(np.log(t_fit), np.log(m_fit), 1)

    n = len(t_fit)
    if n > 2:
        residual_var = ss_res / (n - 2)
        t_var = np.sum((t_fit - t_fit.mean()) ** 2)
        D_stderr = np.sqrt(residual_var / t_var) / 4.0
    else:
        D_stderr = float("nan")

    return float(D), float(D_stderr), float(r2), float(log_slope)


if __name__ == "__main__":
    pass
