import MDAnalysis as mda
import numpy as np
import threadpoolctl
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from ..core.fourier_core import Fourier_Series_Function, get_fourier_modes, average_coefficients
from ..core.fourier_fit import fit_coefficients
from ..core import argument_parser as arg_helper
import os
from ..core.leaflet import get_components, track_components, _label_by_z, apply_margin_filter
from ..core.packing import median_multiple_threshold
from MDAnalysis.lib.distances import distance_array
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import shlex
from pathlib import Path
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, cKDTree

class Rotation_and_Center_tracker:
    def __init__(self,u,sel1="protein", sel2=None,rotate=False):
        self.u = u
        self.rotate=rotate
        self.sel1 = sel1
        self.sel2 = sel2
        self.base_rot_vector = np.zeros(3)
        self.current_vector = np.zeros(3)

        self.sel_center=np.zeros(3)

        self._center()
        if self.rotate:
            self._get_vec()
            

    def _get_vec(self,base=True):
        if self.sel2 is not None:
                self._rot_by_points(base)
        else:
            self._rot_by_gyration(base)


    def _center(self):
        sel = self.u.select_atoms(self.sel1)

        box_center = self.u.dimensions[:3] / 2.0
        sel_center = sel.center_of_geometry(wrap=True)

        shift = box_center - sel_center
        shift[2] = 0.0

        self.u.atoms.translate(shift)
        self.u.atoms.wrap(compound="atoms")

        self.sel_center = box_center.copy()

    def _rotate(self):
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

    def _distance_to_hull_along_vector(self, point, vector, hull):
        distances = []

        for a, b, c in hull.equations:
            denom = a * vector[0] + b * vector[1]

            if denom > 0:
                t = -(a * point[0] + b * point[1] + c) / denom
                distances.append(t)

        return min(distances)

    def _rot_by_gyration(self,base=True):
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
            self.current_vector=np.array([axis[0], axis[1], 0.0])
        else:
            self.current_vector=np.array([axis[0], axis[1], 0.0])

    def _rot_by_points(self,base=True):
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

    def _get_rotation_angle(self):
        return np.arctan2(
            self.current_vector[0] * self.base_rot_vector[1]
            - self.current_vector[1] * self.base_rot_vector[0],
            self.current_vector[0] * self.base_rot_vector[0]
            + self.current_vector[1] * self.base_rot_vector[1],
        )


def _read_ndx(filename): #TODO: Weria said, there is an mda plugin for this now. Must be confirmed and if it exists put in here.
    groups = {}
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

def _rotate_q(q, angle):
    qx, qy = q

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    qx_rot = cos_a * qx - sin_a * qy
    qy_rot = sin_a * qx + cos_a * qy

    return np.asarray((qx_rot, qy_rot), dtype=np.float32)

def _tmd_threshold(Lx, Ly, Nx, Ny):
    """Nyquist spacing for the fit's shortest representable wavelength
    (Lx/Nx, Ly/Ny): resolving a feature at wavelength lambda needs real
    support at least every lambda/2 (points bracketing it from both sides),
    not just one point per full period - below that spacing the fit can't
    distinguish "lipid here" from "no lipid here" at its own chosen
    resolution anyway. No new distance parameter - thresholded against the
    fit's own resolution (lambda_x/lambda_y, via Nx/Ny)."""
    return min(Lx / (2 * Nx), Ly / (2 * Ny))

def _grid_to_atom_distances(positions_xy, X, Y, Lx, Ly):
    """Periodic distance from each grid point (X, Y) to its nearest position
    in `positions_xy` (an (N, 2) array - Z is irrelevant to a membrane-plane
    hole). Uses a KD-tree (periodic via boxsize) rather than a full
    O(n_grid * n_atoms) pairwise distance matrix - the tree has to be
    rebuilt every frame since positions differ frame to frame, but each
    frame's build+query is cheap: O(n_atoms log n_atoms) + O(n_grid log
    n_atoms), instead of paying the full pairwise cost every frame.

    Generic over WHICH atoms - shared by _hole_mask_for_layer (lipid
    fit-input positions) and _one_frame's remove_tmd block (also lipid
    positions, to derive the hole threshold FROM this same distribution -
    see median_multiple_threshold in core/packing.py: calibrating against a
    *different* distance measurement, e.g. atom-to-atom spacing, would
    compare against the wrong scale - AND TMD-filtered protein positions,
    to gate hole detection against where the protein actually is)."""
    positions_xy = np.mod(np.asarray(positions_xy, dtype=float), [Lx, Ly])
    if len(positions_xy) == 0:
        return np.full(X.shape, np.inf)
    tree = cKDTree(positions_xy, boxsize=[Lx, Ly])
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    distances, _ = tree.query(grid_points)
    return distances.reshape(X.shape)


def _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold):
    """Boolean mask, True where grid point (X, Y) has no atom of
    layer_group's fit-input selection within `threshold` (periodic nearest-
    neighbor distance in the XY plane - Z is irrelevant to a membrane-plane
    hole)."""
    return _grid_to_atom_distances(layer_group.positions[:, :2], X, Y, Lx, Ly) > threshold


def _tmd_protein_atoms_xy(center_selection, universe, fourier_upper, fourier_lower):
    """XY positions of `center_selection` atoms that are actually embedded
    in the membrane right now - a STRICT check (no size/margin tolerance,
    keeping this force-field-independent rather than needing a per-bead
    size): an atom counts only if its own z falls exactly between the
    upper and lower leaflet surfaces evaluated at that atom's own (x,y)
    (curvature-aware - uses the local surface height, not a flat global z
    cutoff), discarding soluble/extramembrane domains automatically without
    needing a separate selection string for "just the TMD part".
    """
    atoms = universe.select_atoms(center_selection)
    xy = atoms.positions[:, :2]
    z = atoms.positions[:, 2]
    z_upper = fourier_upper.Z(xy[:, 0], xy[:, 1])
    z_lower = fourier_lower.Z(xy[:, 0], xy[:, 1])
    in_tmd = (z >= np.minimum(z_upper, z_lower)) & (z <= np.maximum(z_upper, z_lower))
    return xy[in_tmd]

def _fourier_by_layer(layer_group, box_size, Nx=3, Ny=3, regularize=False):
    Lx = box_size[0]
    Ly = box_size[1]
    data_3m = layer_group.positions.T
    fourier = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    fourier.setAnm(fit_coefficients(data_3m, Lx, Ly, Nx, Ny, regularize=regularize))

    M=fourier.Anm.shape[0]
    N=fourier.Anm.shape[1]

    m = np.arange(M)
    n = np.arange(N)

    m = np.where(m > M // 2, m - M, m)
    n = np.where(n > N // 2, n - N, n)

    qx = 2 * np.pi * m / Lx
    qy = 2 * np.pi * n / Ly

    q = np.meshgrid(qx, qy, indexing="ij")
    return fourier,q

# Per-worker-process state (populated once by _init_worker, not once per
# frame). Keeping the Universe/AtomGroups/Rotation_and_Center_tracker out of
# _one_frame's arguments means ProcessPoolExecutor never has to pickle them -
# each worker opens its own Universe exactly once instead of re-opening it
# (via unpickling) on every single submitted frame.
_worker_state = {}


def _init_worker(structure, trajectory, ndx_groups, dynamic_select, center, rotation_direction, rotate):
    """ProcessPoolExecutor(initializer=...) target: runs once per worker
    process at pool startup.

    Also caps this worker's BLAS thread pool (OpenBLAS/MKL/etc.) to 1: we're
    already parallel at the process level (one worker per frame), so each
    worker's own numpy/scipy calls trying to *also* use every CPU core would
    massively oversubscribe the machine - N workers x all-cores-per-worker
    fights over the same physical cores and ends up slower than 1 worker.
    Confirmed empirically: this single change took a 200-frame benchmark
    from 70s (8 workers, uncapped) to 3s (8 workers, capped).
    """
    threadpoolctl.threadpool_limits(1)

    u = mda.Universe(structure, trajectory)

    if not dynamic_select:
        layer_group = u.atoms[[x - 1 for x in ndx_groups["Upper"]]]
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


def _fetch_dynamic_positions(frame, *, dynamic_selection):
    """Parallel phase for dynamic (selection-string) leaflet detection:
    fetch this frame's selection positions/box only. No dependency on any
    other frame's result, so safe to run across workers independently/out
    of order - the leaflet-tracking pass that consumes these results (see
    _track_dynamic_leaflets) needs its input in trajectory order, which is
    calc_fourier's job to guarantee (via ProcessPoolExecutor.map(), whose
    results preserve input order regardless of which worker finishes when -
    unlike submit(), whose futures complete in arbitrary order)."""
    universe = _worker_state["universe"]
    universe.trajectory[frame]
    selection = universe.select_atoms(dynamic_selection)
    return frame, selection.positions.copy(), np.array(universe.dimensions, dtype=np.float64)


def _track_dynamic_leaflets(ordered_results, selection_global_indices, min_balance, margin=2.0):
    """Sequential leaflet-tracking pass over `ordered_results` (a
    trajectory-ordered list of (frame, positions, dimensions) from the
    parallel _fetch_dynamic_positions phase). This is cheap relative to the
    per-frame Fourier fit (no LSQ solve here, just distance/graph work) -
    a plain loop, not parallelized, since leaflet identity genuinely
    depends on the previous frame's result: the first frame is clustered
    fresh via get_components(); every later frame is incrementally updated
    via track_components() using the previous frame's (upper, lower) and a
    cutoff persisted from that first clustering - see both functions'
    docstrings in core/leaflet.py for why (stable leaflet identity across
    the trajectory, not an independent re-cluster every frame that could
    reshuffle which physical group counts as "Upper").

    apply_margin_filter is then applied to EVERY frame's result (both the
    frame-0 bootstrap and every tracked frame after it) - XY-connectivity
    alone (get_components/track_components) can miss a lipid that's well-
    connected sideways to its own leaflet's neighbors while structurally
    anomalous in 3D (e.g. squeezed toward mid-plane near a protein, or mid
    flip-flop) - see apply_margin_filter's docstring.

    positions/track_components/get_components/apply_margin_filter all work
    in LOCAL indices (0..len(selection)-1); selection_global_indices maps a
    local index back to its 0-based index in the full Universe (assumes the
    selection's atom membership - not positions - is the same every frame,
    true for a plain "name X"-style selection string).

    Returns {frame: (upper_global_indices, lower_global_indices)} as plain
    lists of 0-based global atom indices, ready for universe.atoms[...].
    """
    out = {}
    prev_upper = prev_lower = None
    cutoff = None

    for frame, positions, dimensions in ordered_results:
        if prev_upper is None:
            matrix = distance_array(positions, positions, box=dimensions)
            (c0, c1), cutoff = get_components(matrix, min_balance=min_balance)
            upper, lower = _label_by_z(c0, c1, positions)
        else:
            upper, lower = track_components(positions, dimensions, prev_upper, prev_lower, cutoff)

        upper, lower = apply_margin_filter(positions, dimensions, upper, lower, margin=margin)

        out[frame] = (
            [int(selection_global_indices[i]) for i in sorted(upper)],
            [int(selection_global_indices[i]) for i in sorted(lower)],
        )
        prev_upper, prev_lower = upper, lower

    return out


def _one_frame(frame, *, out_dir, dynamic_select, dynamic_leaflets, until,Nx=3,Ny=3,sqrt_n_atoms=100,remove_tmd=False,regularize=False):
    universe = _worker_state["universe"]
    layer_group = _worker_state["layer_group"]
    layer_group_2 = _worker_state["layer_group_2"]
    rotation_and_center = _worker_state["rotation_and_center"]

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    x = np.linspace(0, dimensions[:3][0], sqrt_n_atoms, endpoint=False)
    y = np.linspace(0, dimensions[:3][1], sqrt_n_atoms, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Save box dimensions
    with open(f"{out_dir}/dimensions.csv", "a", encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str, dimensions[:3]))}\n")

    # Dynamic selection: leaflets were already determined for every frame by
    # the sequential _track_dynamic_leaflets pass in calc_fourier (before
    # this pool of workers was dispatched) - just look this frame's up.
    # upper/lower are already 0-based global atom indices (matching
    # universe.atoms[...] directly, no +/-1 adjustment needed).
    if dynamic_select:
        upper_index, lower_index = dynamic_leaflets[frame]
        layer_group = universe.atoms[upper_index]
        layer_group_2 = universe.atoms[lower_index]

    Nx, Ny = get_fourier_modes(dimensions[:3],lambda_x=Nx,lambda_y=Ny)
    
    q_angle = None

    if rotation_and_center is not None:
        rotation_and_center._center()

        if rotation_and_center.rotate:
            rotation_and_center._get_vec(base=False)
            q_angle = rotation_and_center._get_rotation_angle()

    # Fourier fits
    fourier1,q = _fourier_by_layer(layer_group, dimensions[:3], Nx, Ny, regularize=regularize)
    fourier2,_ = _fourier_by_layer(layer_group_2, dimensions[:3], Nx, Ny, regularize=regularize)
    fouriermiddle = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
    fouriermiddle.setAnm(average_coefficients(fourier1.Anm, fourier2.Anm))

    if remove_tmd:
        # Combined via min() per leaflet, independently (leaflets can pack
        # differently): the Nyquist term (_tmd_threshold) alone can be too
        # loose whenever a coarse lambda_x/lambda_y is chosen for shape
        # smoothing (it tracks the fit's chosen resolution, not the real
        # lipid density), so a moderate protein footprint could go
        # undetected. The spacing term anchors detection sensitivity to how
        # densely THIS leaflet's real lipids are actually packed, computed
        # from (and applied to) the SAME grid-to-atom distance distribution
        # - see median_multiple_threshold's docstring for why calibrating
        # against a *different* distance measurement (e.g. atom-to-atom
        # spacing) would compare against the wrong scale and over-flag
        # normal packing as if it were a hole.
        # Positions are the same (already centered, never rotated) ones the
        # fit itself used - see TODO.md for why this is computed on the
        # unrotated grid/positions, with rotation-aware lookup left to
        # consumers (map plot/write_xtc).
        Lx, Ly = dimensions[:3][0], dimensions[:3][1]
        nyquist = _tmd_threshold(Lx, Ly, Nx, Ny)

        dist_upper = _grid_to_atom_distances(layer_group.positions[:, :2], X, Y, Lx, Ly)
        dist_lower = _grid_to_atom_distances(layer_group_2.positions[:, :2], X, Y, Lx, Ly)

        threshold_upper = min(nyquist, median_multiple_threshold(dist_upper, k=1.5))
        threshold_lower = min(nyquist, median_multiple_threshold(dist_lower, k=1.5))

        # Gate: a grid point only counts as a hole if it's BOTH unsupported
        # by lipids (the distance test above) AND spatially plausible as
        # protein-displaced - within the same threshold of a center-
        # selection atom that's actually embedded in the membrane right now
        # (see _tmd_protein_atoms_xy). This doesn't replace the lipid-
        # distance criterion (still "no real fit support here", not "here's
        # where we assume the protein is") - it corroborates it, filtering
        # out noise-driven flags that don't spatially coincide with the
        # actual protein, which let k be tightened (more sensitive) without
        # reintroducing far-field false positives. Reuses the SAME
        # threshold for the gate radius - no new free parameter.
        # rotation_and_center is guaranteed not None here - argument_parser
        # requires --center whenever --Remove-TMD is used.
        tmd_xy = _tmd_protein_atoms_xy(rotation_and_center.sel1, universe, fourier1, fourier2)
        dist_to_protein = _grid_to_atom_distances(tmd_xy, X, Y, Lx, Ly)

        hole_upper = (dist_upper > threshold_upper) & (dist_to_protein <= threshold_upper)
        hole_lower = (dist_lower > threshold_lower) & (dist_to_protein <= threshold_lower)
        hole_mask = np.stack((hole_upper, hole_lower), axis=0)

    if q_angle is not None:
        q = _rotate_q(q, q_angle)
    else:
        q = np.asarray(q, dtype=np.float32)

    
    SFT_A_mn = np.asarray(np.stack((fourier1.Anm,fourier2.Anm,fouriermiddle.Anm),axis=0),dtype=np.float32)

    raw_dir = Path(out_dir) / "raw_sft"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fileAmn = raw_dir / f"{frame:0{num_digits}d}_A_mn.npy"
    fileqmn= raw_dir / f"{frame:0{num_digits}d}_q_mn.npy"
    filedimensions = raw_dir / f"{frame:0{num_digits}d}_dimensions.npy"

    np.save(fileAmn, SFT_A_mn)
    np.save(fileqmn, q)
    np.save(filedimensions, np.asarray(dimensions[:3], dtype=np.float64))

    if remove_tmd:
        filehole = raw_dir / f"{frame:0{num_digits}d}_hole_mask.npy"
        np.save(filehole, hole_mask)


def calc_fourier(args,u):
    Until=args.Until
    ndx=args.index

    if Until is None:
        Until = len(u.trajectory)
    else:
        Until=int(Until)
    if ndx is None:
        exit("An index selection or file has to be supplied. Exiting.")
    try:
        ndx_groups = _read_ndx(ndx)
        dynamic_select=False
        dynamic_selection=None
    except FileNotFoundError:
        print("INFO: The ndx file does not exist, it is assumed a selection was provided for dynamic components.")
        dynamic_select=True
        dynamic_selection=ndx
        ndx_groups=None

    dimensions=u.trajectory[0].dimensions
    with open(f"{args.out}/dimensions.csv","w",encoding="UTF8") as dims:
        dims.write(f"#Box Parameters: {' '.join(map(str,dimensions[3:]))}\n")

    frames = list(range(args.From, Until, args.Step))

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(args.structure, args.trajectory, ndx_groups, dynamic_select, args.center, args.rotation_direction, args.rotate),
    ) as ex:
        dynamic_leaflets = None
        if dynamic_select:
            # Parallel phase: fetch every frame's selection positions (order
            # preserved by map(), unlike submit()'s arbitrary completion
            # order) - independent per frame, no leaflet-tracking history
            # needed here. Then a cheap sequential pass (in this main
            # process, not the pool) turns those into a stable per-frame
            # leaflet assignment - see _track_dynamic_leaflets.
            fetch = partial(_fetch_dynamic_positions, dynamic_selection=dynamic_selection)
            ordered_results = list(ex.map(fetch, frames))
            selection_global_indices = u.select_atoms(dynamic_selection).atoms.indices
            dynamic_leaflets = _track_dynamic_leaflets(ordered_results, selection_global_indices, args.min_balance, margin=args.margin)

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
        for future in futures:
            future.result()  # surfaces exceptions raised in worker processes


if __name__=="__main__":
    pass