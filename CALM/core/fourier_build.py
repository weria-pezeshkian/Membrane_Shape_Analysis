from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import MDAnalysis as mda
import numpy as np
import threadpoolctl
from MDAnalysis.lib.distances import distance_array
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from ..core import argument_parser as arg_helper
from ..core.curvature import nearest_surface_intersection, surface_normal
from ..core.fourier_core import Fourier_Series_Function, average_coefficients, get_fourier_modes
from ..core.fourier_fit import fit_coefficients
from ..core.leaflet import _label_by_z, apply_margin_filter, get_components, track_components
from ..core.packing import median_multiple_threshold

logger = logging.getLogger(__name__)


class _LayerHoleStats(TypedDict):
    near_protein: int
    far_fallback_only: int
    closed_gap: int
    total: int


class _HoleStats(TypedDict):
    n_grid: int
    upper: _LayerHoleStats
    lower: _LayerHoleStats


class _FrameResult(TypedDict):
    frame: int
    diagnostics: list[tuple[str, str]]
    hole_stats: _HoleStats | None


class Rotation_and_Center_tracker:
    """Recenters a Universe on `sel1` each frame, and optionally tracks its
    rotation relative to frame 0 (by `sel2`'s direction if given, otherwise
    by `sel1`'s gyration-tensor major axis)."""

    def __init__(
        self,
        u: mda.Universe,
        sel1: str = "protein",
        sel2: str | None = None,
        rotate: bool = False,
    ) -> None:
        self.u = u
        self.rotate = rotate
        self.sel1 = sel1
        self.sel2 = sel2
        self.base_rot_vector = np.zeros(3)
        self.current_vector = np.zeros(3)

        self.sel_center = np.zeros(3)

        self._center()
        if self.rotate:
            self._get_vec()

    def _get_vec(self, base: bool = True) -> None:
        if self.sel2 is not None:
            self._rot_by_points(base)
        else:
            self._rot_by_gyration(base)

    def _center(self) -> None:
        sel = self.u.select_atoms(self.sel1)

        box_center = self.u.dimensions[:3] / 2.0
        if hasattr(sel.atoms,"bonds"):
            if len(u.atoms.bonds)>0:
                u.atoms.unwrap(compound="fragments")
        sel_center = sel.center_of_geometry(wrap=True)

        shift = box_center - sel_center
        shift[2] = 0.0

        self.u.atoms.translate(shift)
        self.u.atoms.wrap(compound="atoms")

        self.sel_center = box_center.copy()

    def _rotate(self) -> None:
        angle = np.arctan2(
            self.current_vector[0] * self.base_rot_vector[1]
            - self.current_vector[1] * self.base_rot_vector[0],
            self.current_vector[0] * self.base_rot_vector[0]
            + self.current_vector[1] * self.base_rot_vector[1],
        )

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rot = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0],
        ])

        coords = self.u.atoms.positions
        self.u.atoms.positions = (coords - self.sel_center) @ rot.T + self.sel_center

    def _distance_to_hull_along_vector(
        self, point: np.ndarray, vector: np.ndarray, hull: ConvexHull
    ) -> float:
        distances = []

        for a, b, c in hull.equations:
            denom = a * vector[0] + b * vector[1]

            if denom > 0:
                t = -(a * point[0] + b * point[1] + c) / denom
                distances.append(t)

        return min(distances)

    def _rot_by_gyration(self, base: bool = True) -> None:
        ag = self.u.select_atoms(self.sel1)

        coords = ag.positions[:, :2]
        center = ag.center_of_geometry()[:2]

        rel = coords - center

        gyration = rel.T @ rel / len(rel)

        eigvals, eigvecs = np.linalg.eigh(gyration)
        axis = eigvecs[:, np.argmax(eigvals)]

        hull = ConvexHull(coords)

        d_pos = self._distance_to_hull_along_vector(center, axis, hull)
        d_neg = self._distance_to_hull_along_vector(center, -axis, hull)

        if d_neg > d_pos:
            axis = -axis
        if base:
            self.base_rot_vector = np.array([axis[0], axis[1], 0.0])
            self.current_vector = np.array([axis[0], axis[1], 0.0])
        else:
            self.current_vector = np.array([axis[0], axis[1], 0.0])

    def _rot_by_points(self, base: bool = True) -> None:
        ag1 = self.u.select_atoms(self.sel1)
        ag2 = self.u.select_atoms(self.sel2)

        center1 = ag1.center_of_geometry()[:2]
        center2 = ag2.center_of_geometry()[:2]

        vector = center2 - center1
        vector = vector / np.linalg.norm(vector)

        if base:
            self.base_rot_vector = np.array([vector[0], vector[1], 0.0])
            self.current_vector = np.array([vector[0], vector[1], 0.0])
        else:
            self.current_vector = np.array([vector[0], vector[1], 0.0])

    def _get_rotation_angle(self) -> float:
        return np.arctan2(
            self.current_vector[0] * self.base_rot_vector[1]
            - self.current_vector[1] * self.base_rot_vector[0],
            self.current_vector[0] * self.base_rot_vector[0]
            + self.current_vector[1] * self.base_rot_vector[1],
        )


def _rotate_q(q: np.ndarray, angle: float) -> np.ndarray:
    qx, qy = q

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    qx_rot = cos_a * qx - sin_a * qy
    qy_rot = sin_a * qx + cos_a * qy

    return np.asarray((qx_rot, qy_rot), dtype=np.float32)


def _tmd_threshold(Lx: float, Ly: float, Nx: int, Ny: int) -> float:
    """Nyquist spacing for the fit's shortest representable wavelength (Lx/Nx, Ly/Ny).

    Resolving a feature at wavelength lambda needs support at least every
    lambda/2; below that spacing the fit can't distinguish "lipid here"
    from "no lipid here" at its own resolution anyway.
    """
    return min(Lx / (2 * Nx), Ly / (2 * Ny))


def _grid_to_atom_distances(
    positions_xyz: np.ndarray, X: np.ndarray, Y: np.ndarray, fourier: Fourier_Series_Function, Lx: float, Ly: float
) -> np.ndarray:
    """3D chord distance from each grid point, on `fourier`'s own fitted surface, to its nearest position in `positions_xyz`.

    `positions_xyz` is an (N, 3) array of real atom positions. Each grid
    point takes `fourier.Z(x, y)` as its own height; `positions_xyz` keep
    their real height. Periodic in x/y, open in z, via one cKDTree with
    boxsize=[Lx, Ly, 0].
    """
    positions_xyz = np.asarray(positions_xyz, dtype=float)
    if len(positions_xyz) == 0:
        return np.full(X.shape, np.inf)
    positions_xyz = positions_xyz.copy()
    positions_xyz[:, :2] = np.mod(positions_xyz[:, :2], [Lx, Ly])
    tree = cKDTree(positions_xyz, boxsize=[Lx, Ly, 0])
    grid_points = np.column_stack([X.ravel(), Y.ravel(), fourier.Z(X, Y).ravel()])
    distances, _ = tree.query(grid_points)
    return distances.reshape(X.shape)


def _close_enclosed_gaps(hole: np.ndarray) -> np.ndarray:
    """Fill every non-hole gap in `hole` that is enclosed but for at most a single-cell-wide leak,
    periodic-boundary-aware.

    Tiles `hole` 3x3 to resolve periodic connectivity (a gap that wraps
    across the box edge is still correctly seen as connected to the open
    membrane on the far side), then two steps:

    1. Seal single-cell-wide leaks in the boundary with one fixed iteration
       of `scipy.ndimage.binary_closing`, using a full 3x3 (8-connected)
       structuring element rather than scipy's default 4-connected cross -
       just enough to bridge the kind of one-cell notch real per-grid-point
       noise in the underlying distance field can produce (e.g. a single
       lipid atom sitting right at a hole's edge), independent of
       hole/protein size. The 4-connected default was tried and confirmed
       (empirically) NOT to seal a single missing cell in an otherwise
       solid wall unless the gap happens to face directly toward hole on
       two opposite sides already - a leak punched sideways through a
       1-cell-thick wall (the realistic case) needs the diagonal
       neighbors 8-connectivity adds to actually bridge.
    2. `scipy.ndimage.binary_fill_holes` on the SEALED mask then fills
       whatever is topologically enclosed there, with no size limit - a
       genuinely enclosed gap is filled regardless of how large it is.

    A plain `binary_fill_holes` on the raw (unsealed) mask, with no
    tolerance at all, is fragile to exactly the single-cell leak step (1)
    seals: a real hole boundary, being a per-grid-point threshold on a
    continuous distance field, commonly has that kind of one-cell notch,
    and without sealing it first, a region that should read as enclosed
    registers as "connected to the open membrane" and never gets filled -
    the original un-closed-island bug, back through a different door. The
    seal is small and fixed rather than derived from `far_threshold`, so it
    can only bridge two separate hole regions that are already within
    about 2 grid cells of each other - a resolution-scale tolerance, not
    the unbounded (10+ cell) bridging the old far_threshold-derived
    iteration count could do, which is what let unrelated holes merge into
    one oversized blob.
    """
    H, W = hole.shape
    tiled = np.tile(hole, (3, 3))
    seal_structure = ndimage.generate_binary_structure(2, 2)
    sealed = ndimage.binary_closing(tiled, structure=seal_structure, iterations=1)
    filled = ndimage.binary_fill_holes(sealed)
    return filled[H:2 * H, W:2 * W]


def _assign_tmd_atoms_to_leaflets(
    center_selection: str,
    universe: mda.Universe,
    fourier_upper: Fourier_Series_Function,
    fourier_lower: Fourier_Series_Function,
    dist_upper: np.ndarray,
    dist_lower: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(upper_xyz, lower_xyz): `center_selection`'s own atoms, split by which leaflet each is
    actually closer to - never both, never an average of the two - and kept only if it maps onto
    a genuinely unsupported region of that same leaflet.

    Deliberately NOT "is this atom's z between the fitted upper and lower
    surfaces at its own (x, y)" (the previous approach here): that fit is a
    smooth, low-order, whole-box surface, defined everywhere including
    where there's no real lipid data to constrain it - inside a large
    hole, it just continues the bulk membrane's own typical z-span rather
    than reflecting that there's no real membrane there at all, so nearly
    any atom near that typical span (not just genuine transmembrane atoms)
    would pass. This never asks the fit whether an atom "looks embedded" -
    it asks whether the atom's own real position maps, geometrically, onto
    a part of a real leaflet's surface that itself has no real lipid
    support (`dist_upper`/`dist_lower`, already computed from real
    positions - ground truth, not a fit value).

    For each atom: `nearest_surface_intersection` against upper (using
    upper's own analytic normal at the atom's own (x, y), from `surface_normal`)
    and independently against lower (using lower's own normal) - each
    curvature-correct on its own terms, never a shared/averaged normal.
    Whichever gives the smaller-magnitude signed distance is the leaflet
    this atom actually belongs to - if only one intersection succeeds,
    that one wins; if neither does (both brackets failed even widened),
    the atom is dropped, nothing meaningful to assign.

    How far the atom itself is from that leaflet plays no further part in
    the decision - the hole is a property of the 2D surface (does real
    lipid support exist at this (x, y)), and the atom is only ever used as
    a probe for that; a deep-in-the-bilayer transmembrane atom and a loop
    resting near the interface, at the same (x, y), point at the same
    question about the same patch of surface. So the winning intersection
    point's own grid cell is checked directly against that same leaflet's
    own `dist_upper`/`dist_lower`: if at least 3 of its 4 corners are
    already within `threshold` of real lipid (one outlier corner is
    treated as noise, not a real gap), this atom maps onto a region that's
    genuinely supported regardless of the atom's own presence, so it's
    discarded; otherwise it's kept as a real contributor to that leaflet's
    hole.
    """
    atoms = universe.select_atoms(center_selection)
    gridsize = X.shape[0]
    dx, dy = Lx / gridsize, Ly / gridsize
    t_max_base = float(np.nanmax(np.abs(fourier_upper.Z(X, Y) - fourier_lower.Z(X, Y)))) * 2

    upper_xyz: list[np.ndarray] = []
    lower_xyz: list[np.ndarray] = []
    for pos in atoms.positions:
        ax, ay, az = float(pos[0]), float(pos[1]), float(pos[2])
        n_upper = surface_normal(fourier_upper, ax, ay)
        n_lower = surface_normal(fourier_lower, ax, ay)
        l_upper = nearest_surface_intersection(
            fourier_upper, ax, ay, az, n_upper[0], n_upper[1], n_upper[2], Lx, Ly, t_max_base,
        )
        l_lower = nearest_surface_intersection(
            fourier_lower, ax, ay, az, n_lower[0], n_lower[1], n_lower[2], Lx, Ly, t_max_base,
        )

        if l_upper is None and l_lower is None:
            continue
        if l_lower is None or (l_upper is not None and abs(l_upper) <= abs(l_lower)):
            leaflet, signed_distance, n, dist_field = "upper", l_upper, n_upper, dist_upper
        else:
            leaflet, signed_distance, n, dist_field = "lower", l_lower, n_lower, dist_lower

        xq = (ax + signed_distance * n[0]) % Lx
        yq = (ay + signed_distance * n[1]) % Ly
        i0, j0 = int(xq / dx) % gridsize, int(yq / dy) % gridsize
        i1, j1 = (i0 + 1) % gridsize, (j0 + 1) % gridsize
        corners = (dist_field[j0, i0], dist_field[j0, i1], dist_field[j1, i0], dist_field[j1, i1])
        if sum(c <= threshold for c in corners) >= 3:
            continue  # already genuinely supported there, regardless of this atom

        (upper_xyz if leaflet == "upper" else lower_xyz).append(pos)

    return (
        np.array(upper_xyz) if upper_xyz else np.empty((0, 3)),
        np.array(lower_xyz) if lower_xyz else np.empty((0, 3)),
    )


def _fourier_by_layer(
    layer_group: mda.core.groups.AtomGroup,
    box_size: np.ndarray,
    Nx: int = 3,
    Ny: int = 3,
    regularize: bool = False,
    diagnostics: list[tuple[str, str]] | None = None,
) -> tuple[Fourier_Series_Function, np.ndarray]:
    """Fit `layer_group`'s Fourier surface and return (fourier, q), q being the (qx, qy) meshgrid for its modes."""
    Lx = box_size[0]
    Ly = box_size[1]
    data_3m = layer_group.positions.T
    fourier = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    fourier.setAnm(fit_coefficients(data_3m, Lx, Ly, Nx, Ny, regularize=regularize, diagnostics=diagnostics))

    M = fourier.Anm.shape[0]
    N = fourier.Anm.shape[1]

    m = np.arange(M)
    n = np.arange(N)

    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)

    qx = 2 * np.pi * m / Lx
    qy = 2 * np.pi * n / Ly

    q = np.asarray(np.meshgrid(qx, qy, indexing="ij"))

    return fourier, q


@dataclass
class _DynamicLeafletReference:
    """The fixed split every dynamic-selection frame is matched against.

    Built once, from the run's first selected frame
    (`_build_dynamic_leaflet_reference`), before any worker starts fitting
    frames. Every later frame reproduces its own (upper, lower) split
    against this same reference independently (`_dynamic_leaflet_groups`) -
    never against another frame - so frames carry no dependency on each
    other and can be dispatched to the worker pool in any order.
    """

    selection: str
    first_upper: set[int]
    first_lower: set[int]
    cutoff: float
    selection_global_indices: np.ndarray
    margin: float


# Per-worker-process state, populated once by _init_worker at pool startup
# (not once per frame) so ProcessPoolExecutor never has to pickle the
# Universe/AtomGroups/tracker into _one_frame's arguments.
_worker_state: dict[str, object] = {}


def _init_worker(
    structure: str,
    trajectory: str,
    ndx_groups: dict[str, list[int]] | None,
    dynamic_select: bool,
    dynamic_ref: _DynamicLeafletReference | None,
    center: str | None,
    rotation_direction: str | None,
    rotate: bool,
) -> None:
    """ProcessPoolExecutor initializer: opens this worker's own Universe once.

    Also caps this worker's BLAS thread pool to 1: process-level
    parallelism (one worker per frame) already uses every core, so each
    worker's numpy/scipy calls also trying to use every core oversubscribes
    the machine.
    """
    threadpoolctl.threadpool_limits(1)

    u = mda.Universe(structure, trajectory)

    if not dynamic_select:
        assert ndx_groups is not None
        layer_group = u.atoms[[x - 1 for x in ndx_groups["Upper"]]]#TODO, check if this should be u.atoms.indices
        layer_group_2 = u.atoms[[x - 1 for x in ndx_groups["Lower"]]]
    else:
        layer_group, layer_group_2 = None, None

    ract = None
    if center is not None:
        ract = Rotation_and_Center_tracker(u, center, rotation_direction, rotate)

    _worker_state["universe"] = u
    _worker_state["layer_group"] = layer_group
    _worker_state["layer_group_2"] = layer_group_2
    _worker_state["rotation_and_center"] = ract
    _worker_state["dynamic_ref"] = dynamic_ref


def _bootstrap_dynamic_leaflet_split(
    positions: np.ndarray, dimensions: np.ndarray, min_balance: float, margin: float,
) -> tuple[set[int], set[int], float]:
    """Fresh (upper, lower) split of `positions`, clustered from scratch, plus the cutoff distance used.

    Only ever called once per run, on the first selected frame -
    `_build_dynamic_leaflet_reference` turns the result into the fixed
    reference every later frame is matched against (`_dynamic_leaflet_split`).
    """
    matrix = distance_array(positions, positions, box=dimensions)
    (c0, c1), cutoff = get_components(matrix, min_balance=min_balance)
    upper, lower = _label_by_z(c0, c1, positions)
    upper, lower = apply_margin_filter(positions, dimensions, upper, lower, margin=margin)
    return upper, lower, cutoff


def _dynamic_leaflet_split(
    positions: np.ndarray, dimensions: np.ndarray,
    ref_upper: set[int], ref_lower: set[int], cutoff: float, margin: float,
) -> tuple[set[int], set[int]]:
    """This frame's own (upper, lower) split, re-derived entirely from its own geometry.

    `track_components` only uses `ref_upper`/`ref_lower` as a candidate
    membership seed - every atom's actual group membership is re-tested
    against this frame's own distances (still connected to the rest of its
    seed group; otherwise closer to which group), not against the
    reference's. That makes the result independent of which frame the
    reference came from, as long as leaflet flip-flopping is rare relative
    to how long ago the reference frame was - see `_DynamicLeafletReference`.
    """
    upper, lower = track_components(positions, dimensions, ref_upper, ref_lower, cutoff)
    return apply_margin_filter(positions, dimensions, upper, lower, margin=margin)


def _build_dynamic_leaflet_reference(
    u: mda.Universe, frame0: int, dynamic_selection: str, min_balance: float, margin: float,
) -> _DynamicLeafletReference:
    """Build the fixed reference `_dynamic_leaflet_groups` matches every frame against.

    Seeks `u` to `frame0` (the run's first selected frame) and clusters it
    fresh - a single-frame operation, done once in the main process before
    the worker pool starts, not a pass over the whole trajectory.
    """
    u.trajectory[frame0]
    selection = u.select_atoms(dynamic_selection)
    dimensions = np.array(u.dimensions, dtype=np.float64)
    first_upper, first_lower, cutoff = _bootstrap_dynamic_leaflet_split(
        selection.positions.copy(), dimensions, min_balance, margin
    )
    return _DynamicLeafletReference(
        selection=dynamic_selection,
        first_upper=first_upper,
        first_lower=first_lower,
        cutoff=cutoff,
        selection_global_indices=selection.atoms.indices,
        margin=margin,
    )


def _dynamic_leaflet_groups(
    universe: mda.Universe, dimensions: np.ndarray,
) -> tuple[mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """This frame's own (upper, lower) leaflet AtomGroups, matched against `_worker_state`'s fixed reference.

    `universe`'s trajectory must already be positioned at the frame being
    processed. Independent of every other frame - safe to call from any
    worker, for any frame, in any order.
    """
    ref = cast(_DynamicLeafletReference, _worker_state["dynamic_ref"])
    selection = universe.select_atoms(ref.selection)
    upper_local, lower_local = _dynamic_leaflet_split(
        selection.positions, dimensions, ref.first_upper, ref.first_lower, ref.cutoff, ref.margin
    )
    upper_index = [int(ref.selection_global_indices[i]) for i in sorted(upper_local)]
    lower_index = [int(ref.selection_global_indices[i]) for i in sorted(lower_local)]
    return universe.atoms[upper_index], universe.atoms[lower_index]


def _remove_tmd_hole_mask(
    remove_tmd: bool | str,
    rotation_and_center: "Rotation_and_Center_tracker | None",
    universe: mda.Universe,
    layer_group: mda.core.groups.AtomGroup,
    layer_group_2: mda.core.groups.AtomGroup,
    fourier1: Fourier_Series_Function,
    fourier2: Fourier_Series_Function,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    tmd_far_multiple: float = 5.0,
) -> tuple[np.ndarray, _HoleStats]:
    """(hole_mask, hole_stats) for one frame's --Remove-TMD hole detection.

    One threshold shared by both leaflets. The Nyquist term
    (_tmd_threshold) reflects the fit's chosen resolution; capping it with
    median_multiple_threshold anchors it to how densely lipids are packed,
    using the larger of the two leaflets' own median-based candidates,
    which keeps the protein-proximity gate below equally permissive for
    both leaflets.

    A grid point counts as a hole when it is unsupported by lipids (the
    distance test) and either within `threshold` of a protein atom
    actually assigned to that same leaflet (`_assign_tmd_atoms_to_leaflets`
    - each atom belongs to exactly one leaflet, never both, decided by
    which leaflet's own fitted surface it's geometrically closer to), or
    farther from any lipid than `far_threshold`. Each leaflet's own
    protein distance is measured against that leaflet's own fitted
    surface. The protein selection is `remove_tmd`'s own value when
    --Remove-TMD was given one, and --center's selection
    (`rotation_and_center.sel1`) when --Remove-TMD was given bare
    (`remove_tmd is True`).
    """
    nyquist = _tmd_threshold(Lx, Ly, Nx, Ny)

    dist_upper = _grid_to_atom_distances(layer_group.positions, X, Y, fourier1, Lx, Ly)
    dist_lower = _grid_to_atom_distances(layer_group_2.positions, X, Y, fourier2, Lx, Ly)

    threshold = min(nyquist, max(
        median_multiple_threshold(dist_upper, k=1),
        median_multiple_threshold(dist_lower, k=1),
    ))
    far_threshold = threshold * tmd_far_multiple

    if remove_tmd is True:
        assert rotation_and_center is not None
        tmd_selection = rotation_and_center.sel1
    else:
        tmd_selection = remove_tmd
    tmd_xyz_upper, tmd_xyz_lower = _assign_tmd_atoms_to_leaflets(
        tmd_selection, universe, fourier1, fourier2, dist_upper, dist_lower, X, Y, Lx, Ly, threshold,
    )
    # A protein atom's own z is its real transmembrane depth, not a height
    # on either leaflet's surface, so distance to each leaflet is measured
    # from the atom's (x, y) projected onto that leaflet's own fitted
    # height - both points then live on the same surface, matching how
    # dist_upper/dist_lower measure grid-to-lipid distance.
    tmd_x_upper, tmd_y_upper = tmd_xyz_upper[:, 0], tmd_xyz_upper[:, 1]
    tmd_x_lower, tmd_y_lower = tmd_xyz_lower[:, 0], tmd_xyz_lower[:, 1]
    tmd_on_upper = np.column_stack([tmd_x_upper, tmd_y_upper, fourier1.Z(tmd_x_upper, tmd_y_upper)])
    tmd_on_lower = np.column_stack([tmd_x_lower, tmd_y_lower, fourier2.Z(tmd_x_lower, tmd_y_lower)])
    dist_to_protein_upper = _grid_to_atom_distances(tmd_on_upper, X, Y, fourier1, Lx, Ly)
    dist_to_protein_lower = _grid_to_atom_distances(tmd_on_lower, X, Y, fourier2, Lx, Ly)

    near_protein_upper = dist_to_protein_upper <= threshold
    near_protein_lower = dist_to_protein_lower <= threshold
    hole_upper = (dist_upper > threshold) & (near_protein_upper | (dist_upper > far_threshold))
    hole_lower = (dist_lower > threshold) & (near_protein_lower | (dist_lower > far_threshold))
    n_before_upper = int(hole_upper.sum())
    n_before_lower = int(hole_lower.sum())
    hole_upper = _close_enclosed_gaps(hole_upper)
    hole_lower = _close_enclosed_gaps(hole_lower)
    hole_mask = np.stack((hole_upper, hole_lower), axis=0)

    # Breaks each leaflet's flagged points down by which rule flagged them,
    # and how many more the enclosed-gap-closing pass added, so the caller
    # can log a concrete accounting of what --Remove-TMD did on this frame.
    hole_stats: _HoleStats = {
        "n_grid": int(hole_upper.size),
        "upper": {
            "near_protein": int((near_protein_upper & (dist_upper > threshold)).sum()),
            "far_fallback_only": int(
                ((dist_upper > far_threshold) & ~near_protein_upper & (dist_upper > threshold)).sum()
            ),
            "closed_gap": int(hole_upper.sum()) - n_before_upper,
            "total": int(hole_upper.sum()),
        },
        "lower": {
            "near_protein": int((near_protein_lower & (dist_lower > threshold)).sum()),
            "far_fallback_only": int(
                ((dist_lower > far_threshold) & ~near_protein_lower & (dist_lower > threshold)).sum()
            ),
            "closed_gap": int(hole_lower.sum()) - n_before_lower,
            "total": int(hole_lower.sum()),
        },
    }
    return hole_mask, hole_stats


def _one_frame(
    frame: int,
    *,
    out_dir: str,
    dynamic_select: bool,
    until: int,
    Nx: float | None = 3,
    Ny: float | None = 3,
    sqrt_n_atoms: int = 100,
    remove_tmd: bool | str = False,
    regularize: bool = False,
    tmd_far_multiple: float = 5.0,
) -> _FrameResult:
    """Fit one frame's Fourier surfaces (and optional hole mask) and save its raw_sft output.

    Returns `{"frame": frame, "diagnostics": [(level, message), ...],
    "hole_stats": {...} or None}` for the caller (`calc_fourier`, in the
    main process) to log - this function runs inside a worker process, so
    everything worth logging is collected here and handed back rather than
    logged directly, keeping every write to the replay log single-process.
    """
    diagnostics: list[tuple[str, str]] = []
    universe = cast(mda.Universe, _worker_state["universe"])
    layer_group = cast(mda.core.groups.AtomGroup, _worker_state["layer_group"])
    layer_group_2 = cast(mda.core.groups.AtomGroup, _worker_state["layer_group_2"])
    rotation_and_center = cast(
        "Rotation_and_Center_tracker | None", _worker_state["rotation_and_center"]
    )

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    x = np.linspace(0, dimensions[:3][0], sqrt_n_atoms, endpoint=False)
    y = np.linspace(0, dimensions[:3][1], sqrt_n_atoms, endpoint=False)
    X, Y = np.meshgrid(x, y)

    if dynamic_select:
        layer_group, layer_group_2 = _dynamic_leaflet_groups(universe, dimensions)

    Nx, Ny = get_fourier_modes(dimensions[:3], lambda_x=Nx, lambda_y=Ny, diagnostics=diagnostics)

    q_angle = None

    if rotation_and_center is not None:
        rotation_and_center._center()

        if rotation_and_center.rotate:
            rotation_and_center._get_vec(base=False)
            q_angle = rotation_and_center._get_rotation_angle()

    fourier1, q = _fourier_by_layer(
        layer_group, dimensions[:3], Nx, Ny, regularize=regularize, diagnostics=diagnostics
    )
    fourier2, _ = _fourier_by_layer(
        layer_group_2, dimensions[:3], Nx, Ny, regularize=regularize, diagnostics=diagnostics
    )
    fouriermiddle = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
    fouriermiddle.setAnm(average_coefficients(fourier1.Anm, fourier2.Anm))

    hole_stats: _HoleStats | None = None
    if remove_tmd:
        hole_mask, hole_stats = _remove_tmd_hole_mask(
            remove_tmd, rotation_and_center, universe, layer_group, layer_group_2,
            fourier1, fourier2, X, Y, dimensions[:3][0], dimensions[:3][1], Nx, Ny, tmd_far_multiple,
        )

    if q_angle is not None:
        q = _rotate_q(q, q_angle)
    else:
        q = np.asarray(q, dtype=np.float32)

    SFT_A_mn = np.asarray(np.stack((fourier1.Anm, fourier2.Anm, fouriermiddle.Anm), axis=0), dtype=np.float32)

    raw_dir = Path(out_dir) / "raw_sft"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fileAmn = raw_dir / f"{frame:0{num_digits}d}_A_mn.npy"
    fileqmn = raw_dir / f"{frame:0{num_digits}d}_q_mn.npy"
    filedimensions = raw_dir / f"{frame:0{num_digits}d}_dimensions.npy"

    np.save(fileAmn, SFT_A_mn)
    np.save(fileqmn, q)
    np.save(filedimensions, np.asarray(dimensions[:3], dtype=np.float64))

    if remove_tmd:
        filehole = raw_dir / f"{frame:0{num_digits}d}_hole_mask.npy"
        np.save(filehole, hole_mask)

    return {"frame": frame, "diagnostics": diagnostics, "hole_stats": hole_stats}


def calc_fourier(args: argparse.Namespace, u: mda.Universe) -> None:
    """Fit every selected frame's Fourier surfaces in parallel and save raw_sft output to args.out."""
    Until = args.Until

    if Until is None:
        Until = len(u.trajectory)
    else:
        Until = int(Until)

    ndx_groups, dynamic_selection = arg_helper.resolve_index_source(args)
    if ndx_groups is None and dynamic_selection is None:
        sys.exit("An index selection or file has to be supplied. Exiting.")
    dynamic_select = ndx_groups is None

    frames = list(range(args.From, Until, args.Step))

    dynamic_ref = None
    if dynamic_select:
        # A single-frame bootstrap, not a whole-trajectory pass: every other
        # frame's own leaflet split is matched against this one independently
        # inside its own worker (_dynamic_leaflet_groups), so dispatching
        # frame fits doesn't wait on anything but this.
        assert dynamic_selection is not None
        dynamic_ref = _build_dynamic_leaflet_reference(
            u, frames[0], dynamic_selection, args.min_balance, args.margin
        )

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(
            args.structure, args.trajectory, ndx_groups, dynamic_select, dynamic_ref,
            args.center, args.rotation_direction, args.rotate,
        ),
    ) as ex:
        fn = partial(_one_frame,
                    out_dir=args.out,
                    dynamic_select=dynamic_select,
                    until=Until,
                    Nx=args.lambda_x,
                    Ny=args.lambda_y,
                    sqrt_n_atoms=args.gridsize,
                    remove_tmd=args.remove_tmd,
                    regularize=args.regularize,
                    )

        futures = [ex.submit(fn, x) for x in frames]
        hole_totals: _HoleStats | None = None
        for future in futures:
            result: _FrameResult = future.result()  # surfaces exceptions raised in worker processes
            frame_num = result["frame"]
            for level, message in result["diagnostics"]:
                log_fn = logger.warning if level == "warning" else logger.info
                log_fn(f"frame {frame_num}: {message}")

            stats = result["hole_stats"]
            if stats is None:
                continue
            logger.info(
                f"frame {frame_num}: --Remove-TMD flagged "
                f"{stats['upper']['total']}/{stats['n_grid']} upper and "
                f"{stats['lower']['total']}/{stats['n_grid']} lower grid points as holes "
                f"(near-protein upper/lower: {stats['upper']['near_protein']}/{stats['lower']['near_protein']}, "
                f"far-fallback-only upper/lower: "
                f"{stats['upper']['far_fallback_only']}/{stats['lower']['far_fallback_only']}, "
                f"enclosed gaps closed upper/lower: {stats['upper']['closed_gap']}/{stats['lower']['closed_gap']})"
            )
            if hole_totals is None:
                hole_totals = {
                    "n_grid": 0,
                    "upper": {"near_protein": 0, "far_fallback_only": 0, "closed_gap": 0, "total": 0},
                    "lower": {"near_protein": 0, "far_fallback_only": 0, "closed_gap": 0, "total": 0},
                }
            hole_totals["n_grid"] += stats["n_grid"]
            layer_pairs = ((stats["upper"], hole_totals["upper"]), (stats["lower"], hole_totals["lower"]))
            for stats_layer, totals_layer in layer_pairs:
                totals_layer_dict = cast(dict, totals_layer)
                for key, value in stats_layer.items():
                    totals_layer_dict[key] += value

        if hole_totals is not None:
            logger.info(
                f"--Remove-TMD summary over {len(frames)} frames: "
                f"upper holes {hole_totals['upper']['total']}/{hole_totals['n_grid']} "
                f"({100 * hole_totals['upper']['total'] / hole_totals['n_grid']:.1f}%), "
                f"lower holes {hole_totals['lower']['total']}/{hole_totals['n_grid']} "
                f"({100 * hole_totals['lower']['total'] / hole_totals['n_grid']:.1f}%); "
                f"enclosed gaps closed: upper {hole_totals['upper']['closed_gap']}, "
                f"lower {hole_totals['lower']['closed_gap']}"
            )


if __name__ == "__main__":
    pass
