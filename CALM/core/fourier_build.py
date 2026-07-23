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
from ..utilize.write_ndx import write
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

def _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold):
    """Boolean mask, True where grid point (X, Y) has no atom of
    layer_group's fit-input selection within `threshold` (periodic nearest-
    neighbor distance in the XY plane - Z is irrelevant to a membrane-plane
    hole). Uses a KD-tree (periodic via boxsize) rather than a full
    O(n_grid * n_atoms) pairwise distance matrix - the tree has to be
    rebuilt every frame since atom positions differ frame to frame, but each
    frame's build+query is cheap: O(n_atoms log n_atoms) + O(n_grid log
    n_atoms), instead of paying the full pairwise cost every frame."""
    positions_xy = np.mod(layer_group.positions[:, :2], [Lx, Ly])
    tree = cKDTree(positions_xy, boxsize=[Lx, Ly])
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    distances, _ = tree.query(grid_points)
    return distances.reshape(X.shape) > threshold

def _fourier_by_layer(layer_group, box_size, Nx=3, Ny=3):
    Lx = box_size[0]
    Ly = box_size[1]
    data_3m = layer_group.positions.T
    fourier = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    fourier.setAnm(fit_coefficients(data_3m, Lx, Ly, Nx, Ny))

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


def _one_frame(frame, *, out_dir, dynamic_select, dynamic_selection, until,Nx=3,Ny=3,sqrt_n_atoms=100,remove_tmd=False):
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

    # Dynamic selection if requested
    if dynamic_select:
        _, upper_index, lower_index = write(ts, dynamic_selection, write=False)
        layer_group = ts.atoms[[x - 1 for x in upper_index]]
        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]

    Nx, Ny = get_fourier_modes(dimensions[:3],lambda_x=Nx,lambda_y=Ny)
    
    q_angle = None

    if rotation_and_center is not None:
        rotation_and_center._center()

        if rotation_and_center.rotate:
            rotation_and_center._get_vec(base=False)
            q_angle = rotation_and_center._get_rotation_angle()

    # Fourier fits
    fourier1,q = _fourier_by_layer(layer_group, dimensions[:3], Nx, Ny)
    fourier2,_ = _fourier_by_layer(layer_group_2, dimensions[:3], Nx, Ny)
    fouriermiddle = Fourier_Series_Function(dimensions[:3][0], dimensions[:3][1], Nx, Ny)
    fouriermiddle.setAnm(average_coefficients(fourier1.Anm, fourier2.Anm))

    if remove_tmd:
        # Grid points whose nearest atom in that leaflet's own fit-input
        # selection is farther than the fit's own resolution (Lx/Nx, Ly/Ny) -
        # e.g. a transmembrane protein displacing lipids. Same (already
        # centered, never rotated) positions the fit itself used - see
        # TODO.md for why this is computed on the unrotated grid/positions,
        # with rotation-aware lookup left to consumers (map plot/write_xtc).
        Lx, Ly = dimensions[:3][0], dimensions[:3][1]
        threshold = min(Lx / Nx, Ly / Ny)
        hole_upper = _hole_mask_for_layer(layer_group, X, Y, Lx, Ly, threshold)
        hole_lower = _hole_mask_for_layer(layer_group_2, X, Y, Lx, Ly, threshold)
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

    fn = partial(_one_frame,
                out_dir=args.out,
                dynamic_select=dynamic_select,
                dynamic_selection=dynamic_selection,
                until=Until,
                Nx=args.lambda_x,
                Ny=args.lambda_y,
                sqrt_n_atoms=args.gridsize,
                remove_tmd=args.remove_tmd,
                )

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(args.structure, args.trajectory, ndx_groups, dynamic_select, args.center, args.rotation_direction, args.rotate),
    ) as ex:
        futures = [ex.submit(fn, x) for x in range(args.From, Until, args.Step)]
        for future in futures:
            future.result()  # surfaces exceptions raised in worker processes


if __name__=="__main__":
    pass