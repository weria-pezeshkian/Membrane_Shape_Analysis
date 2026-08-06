from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import MDAnalysis as mda
import numpy as np
import threadpoolctl
from MDAnalysis.lib.distances import distance_array
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

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


def _read_ndx(filename: str) -> dict[str, list[int]]:
    # TODO: confirm whether MDAnalysis has a built-in reader for this format
    # now, and replace this with it if so.
    groups: dict[str, list[int]] = {}
    with open(filename) as f:
        group_name = None
        for line in f:
            line = line[:line.find(";")].strip()
            if line.startswith('['):
                group_name = line[1:-1].strip()
                groups[group_name] = []
            elif group_name is not None:
                groups[group_name].extend(map(int, line.split()))
    return groups


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
    positions_xy: np.ndarray, X: np.ndarray, Y: np.ndarray, Lx: float, Ly: float
) -> np.ndarray:
    """Periodic XY distance from each grid point (X, Y) to its nearest position in `positions_xy`.

    `positions_xy` is an (N, 2) array; Z is irrelevant to a membrane-plane
    hole. Used both for lipid fit-input positions (`_hole_mask_for_layer`)
    and for TMD-filtered protein positions (`_one_frame`'s remove_tmd
    block) - see `core/packing.py`'s `median_multiple_threshold` for why
    the distance kind being calibrated against and tested against must
    match.
    """
    positions_xy = np.mod(np.asarray(positions_xy, dtype=float), [Lx, Ly])
    if len(positions_xy) == 0:
        return np.full(X.shape, np.inf)
    tree = cKDTree(positions_xy, boxsize=[Lx, Ly])
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    distances, _ = tree.query(grid_points)
    return distances.reshape(X.shape)


def _close_enclosed_gaps(hole: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Extend `hole` with a periodic-boundary-aware morphological closing (dilate then erode).

    Tiles `hole` 3x3 to resolve periodic connectivity, then applies
    `scipy.ndimage.binary_closing` with `iterations` steps, merging hole
    cells across non-hole gaps up to about that many cells wide.
    """
    H, W = hole.shape
    tiled = np.tile(hole, (3, 3))
    closed = ndimage.binary_closing(tiled, iterations=iterations)
    return closed[H:2 * H, W:2 * W]


def _hole_mask_for_layer(
    layer_group: mda.core.groups.AtomGroup,
    X: np.ndarray,
    Y: np.ndarray,
    Lx: float,
    Ly: float,
    threshold: float,
) -> np.ndarray:
    """True where grid point (X, Y) has no atom of `layer_group` within `threshold` (periodic, XY only)."""
    return _grid_to_atom_distances(layer_group.positions[:, :2], X, Y, Lx, Ly) > threshold


def _tmd_protein_atoms_xy(
    center_selection: str,
    universe: mda.Universe,
    fourier_upper: Fourier_Series_Function,
    fourier_lower: Fourier_Series_Function,
) -> np.ndarray:
    """XY positions of `center_selection` atoms currently embedded in the membrane.

    An atom counts only if its own z falls between the upper and lower
    leaflet surfaces evaluated at that atom's own (x, y) - curvature-aware,
    and with no size/margin tolerance so it stays force-field-independent.
    This discards soluble/extramembrane domains of the same selection
    without needing a separate "just the TMD part" selection string.
    """
    atoms = universe.select_atoms(center_selection)
    xy = atoms.positions[:, :2]
    z = atoms.positions[:, 2]
    z_upper = fourier_upper.Z(xy[:, 0], xy[:, 1])
    z_lower = fourier_lower.Z(xy[:, 0], xy[:, 1])
    in_tmd = (z >= np.minimum(z_upper, z_lower)) & (z <= np.maximum(z_upper, z_lower))
    return xy[in_tmd]


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


# Per-worker-process state, populated once by _init_worker at pool startup
# (not once per frame) so ProcessPoolExecutor never has to pickle the
# Universe/AtomGroups/tracker into _one_frame's arguments.
_worker_state: dict[str, object] = {}


def _init_worker(
    structure: str,
    trajectory: str,
    ndx_groups: dict[str, list[int]] | None,
    dynamic_select: bool,
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


def _fetch_dynamic_positions(
    frame: int, *, dynamic_selection: str
) -> tuple[int, np.ndarray, np.ndarray]:
    """Fetch one frame's dynamic-selection positions and box dimensions.

    Independent per frame, safe to run across workers in any order.
    `calc_fourier` calls this via `ProcessPoolExecutor.map()`, which
    preserves input order regardless of completion order, since the
    sequential tracking pass that consumes the results
    (`_track_dynamic_leaflets`) needs trajectory order.
    """
    universe = cast(mda.Universe, _worker_state["universe"])
    universe.trajectory[frame]
    selection = universe.select_atoms(dynamic_selection)
    return frame, selection.positions.copy(), np.array(universe.dimensions, dtype=np.float64)


def _track_dynamic_leaflets(
    ordered_results: list[tuple[int, np.ndarray, np.ndarray]],
    selection_global_indices: np.ndarray,
    min_balance: float,
    margin: float = 2.0,
) -> dict[int, tuple[list[int], list[int]]]:
    """Sequential leaflet-tracking pass over trajectory-ordered per-frame positions.

    The first frame is clustered fresh via `get_components`; every later
    frame is incrementally updated via `track_components`, using the
    previous frame's (upper, lower) and the cutoff persisted from that
    first clustering, so leaflet identity stays stable across the
    trajectory instead of being independently re-derived (and possibly
    reshuffled) every frame.

    `apply_margin_filter` is then applied to every frame's result -
    XY-connectivity alone can miss an atom that is well-connected sideways
    to its own leaflet but structurally anomalous in 3D.

    `positions`/`track_components`/`get_components`/`apply_margin_filter`
    all use local indices (0..len(selection)-1); `selection_global_indices`
    maps a local index to its 0-based global atom index (assumes the
    selection's atom membership is the same every frame).

    Returns {frame: (upper_global_indices, lower_global_indices)}.
    """
    out: dict[int, tuple[list[int], list[int]]] = {}
    prev_upper = prev_lower = None
    cutoff = None

    for frame, positions, dimensions in ordered_results:
        if prev_upper is None:
            matrix = distance_array(positions, positions, box=dimensions)
            (c0, c1), cutoff = get_components(matrix, min_balance=min_balance)
            upper, lower = _label_by_z(c0, c1, positions)
        else:
            assert prev_lower is not None and cutoff is not None
            upper, lower = track_components(positions, dimensions, prev_upper, prev_lower, cutoff)

        upper, lower = apply_margin_filter(positions, dimensions, upper, lower, margin=margin)

        out[frame] = (
            [int(selection_global_indices[i]) for i in sorted(upper)],
            [int(selection_global_indices[i]) for i in sorted(lower)],
        )
        prev_upper, prev_lower = upper, lower

    return out


def _one_frame(
    frame: int,
    *,
    out_dir: str,
    dynamic_select: bool,
    dynamic_leaflets: dict[int, tuple[list[int], list[int]]] | None,
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

    with open(f"{out_dir}/dimensions.csv", "a", encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str, dimensions[:3]))}\n")

    if dynamic_select:
        # Leaflets for this frame were already determined by the sequential
        # _track_dynamic_leaflets pass in calc_fourier, as 0-based global
        # atom indices.
        assert dynamic_leaflets is not None
        upper_index, lower_index = dynamic_leaflets[frame]
        layer_group = universe.atoms[upper_index]
        layer_group_2 = universe.atoms[lower_index]

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
        # One threshold shared by both leaflets. The Nyquist term
        # (_tmd_threshold) reflects the fit's chosen resolution; capping it
        # with median_multiple_threshold anchors it to how densely lipids
        # are packed, using the larger of the two leaflets' own
        # median-based candidates, which keeps the protein-proximity gate
        # below equally permissive for both leaflets.
        Lx, Ly = dimensions[:3][0], dimensions[:3][1]
        nyquist = _tmd_threshold(Lx, Ly, Nx, Ny)

        dist_upper = _grid_to_atom_distances(layer_group.positions[:, :2], X, Y, Lx, Ly)
        dist_lower = _grid_to_atom_distances(layer_group_2.positions[:, :2], X, Y, Lx, Ly)

        threshold = min(nyquist, max(
            median_multiple_threshold(dist_upper, k=1),
            median_multiple_threshold(dist_lower, k=1),
        ))
        far_threshold = threshold * tmd_far_multiple

        # A grid point counts as a hole when it is unsupported by lipids
        # (the distance test) and either within `threshold` of a protein
        # atom currently embedded in the membrane (_tmd_protein_atoms_xy),
        # or farther from any lipid than `far_threshold`. The protein
        # selection is remove_tmd's own value when --Remove-TMD was given
        # one, and --center's selection (rotation_and_center.sel1) when
        # --Remove-TMD was given bare (remove_tmd is True).
        if remove_tmd is True:
            assert rotation_and_center is not None
            tmd_selection = rotation_and_center.sel1
        else:
            tmd_selection = remove_tmd
        tmd_xy = _tmd_protein_atoms_xy(tmd_selection, universe, fourier1, fourier2)
        dist_to_protein = _grid_to_atom_distances(tmd_xy, X, Y, Lx, Ly)

        near_protein = dist_to_protein <= threshold
        hole_upper = (dist_upper > threshold) & (near_protein | (dist_upper > far_threshold))
        hole_lower = (dist_lower > threshold) & (near_protein | (dist_lower > far_threshold))
        n_before_upper = int(hole_upper.sum())
        n_before_lower = int(hole_lower.sum())
        # `iterations` is in grid cells; `far_threshold` is the physical
        # scale already established as "far enough from any lipid to count
        # as hole regardless of protein proximity" - a non-hole island
        # narrower than that, in physical units, is closed. `threshold`
        # itself is typically too close to one grid cell's own size to
        # bridge realistic islands (see TODO.md).
        cell_size = min(Lx, Ly) / sqrt_n_atoms
        close_iterations = max(1, int(round(far_threshold / cell_size)))
        hole_upper = _close_enclosed_gaps(hole_upper, iterations=close_iterations)
        hole_lower = _close_enclosed_gaps(hole_lower, iterations=close_iterations)
        hole_mask = np.stack((hole_upper, hole_lower), axis=0)

        # Breaks each leaflet's flagged points down by which rule flagged
        # them, and how many more the enclosed-gap-closing pass added, so
        # calc_fourier can log a concrete accounting of what --Remove-TMD
        # did on this frame.
        hole_stats = {
            "n_grid": int(hole_upper.size),
            "upper": {
                "near_protein": int((near_protein & (dist_upper > threshold)).sum()),
                "far_fallback_only": int(
                    ((dist_upper > far_threshold) & ~near_protein & (dist_upper > threshold)).sum()
                ),
                "closed_gap": int(hole_upper.sum()) - n_before_upper,
                "total": int(hole_upper.sum()),
            },
            "lower": {
                "near_protein": int((near_protein & (dist_lower > threshold)).sum()),
                "far_fallback_only": int(
                    ((dist_lower > far_threshold) & ~near_protein & (dist_lower > threshold)).sum()
                ),
                "closed_gap": int(hole_lower.sum()) - n_before_lower,
                "total": int(hole_lower.sum()),
            },
        }

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
    ndx = args.index

    if Until is None:
        Until = len(u.trajectory)
    else:
        Until = int(Until)
    if ndx is None:
        sys.exit("An index selection or file has to be supplied. Exiting.")
    try:
        ndx_groups = _read_ndx(ndx)
        dynamic_select = False
        dynamic_selection = None
    except FileNotFoundError:
        logger.info(
            "No ndx file found at the given --index path; treating it as a dynamic MDAnalysis selection instead."
        )
        dynamic_select = True
        dynamic_selection = ndx
        ndx_groups = None

    dimensions = u.trajectory[0].dimensions
    with open(f"{args.out}/dimensions.csv", "w", encoding="UTF8") as dims:
        dims.write(f"#Box Parameters: {' '.join(map(str, dimensions[3:]))}\n")

    frames = list(range(args.From, Until, args.Step))

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(
            args.structure, args.trajectory, ndx_groups, dynamic_select,
            args.center, args.rotation_direction, args.rotate,
        ),
    ) as ex:
        dynamic_leaflets = None
        if dynamic_select:
            # Parallel phase (order preserved by map()): fetch every frame's
            # selection positions independently. Then a cheap sequential
            # pass in this process turns those into a stable per-frame
            # leaflet assignment - see _track_dynamic_leaflets.
            assert dynamic_selection is not None
            fetch = partial(_fetch_dynamic_positions, dynamic_selection=dynamic_selection)
            ordered_results = list(ex.map(fetch, frames))
            selection_global_indices = u.select_atoms(dynamic_selection).atoms.indices
            dynamic_leaflets = _track_dynamic_leaflets(
                ordered_results, selection_global_indices, args.min_balance, margin=args.margin
            )

        fn = partial(_one_frame,
                    out_dir=args.out,
                    dynamic_select=dynamic_select,
                    dynamic_leaflets=dynamic_leaflets,
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
