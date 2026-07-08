import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from ..core.fourier_core import Fourier_Series_Function
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
from scipy.spatial import ConvexHull

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
            _get_vec()
            

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

def _fourier_by_layer(layer_group, box_size, Nx=3, Ny=3):
    Lx = box_size[0]
    Ly = box_size[1]
    data_3m = layer_group.positions.T
    fourier = Fourier_Series_Function(Lx, Ly, Nx, Ny)
    fourier.Fit(data_3m)

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

def _get_XY(box_size,sqrt_n_atoms):
    x = np.linspace(0, box_size[0], sqrt_n_atoms, endpoint=False)
    y = np.linspace(0, box_size[1], sqrt_n_atoms, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return X, Y

def _one_frame(frame, *, layer_group, layer_group_2, out_dir,
              dynamic_select, dynamic_selection, universe, until, rotation_and_center=None,Nx=2,Ny=2,sqrt_n_atoms=100):

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    box_size = dimensions[:3]

    # Save box dimensions
    with open(f"{out_dir}/dimensions.csv", "a", encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str, box_size))}\n")

    # XY grid
    X, Y = _get_XY(box_size,sqrt_n_atoms)

    # Dynamic selection if requested
    if dynamic_select:
        _, upper_index, lower_index = write(ts, dynamic_selection, write=False)
        layer_group = ts.atoms[[x - 1 for x in upper_index]]
        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]
    
    if rotation_and_center is not None:
        rotation_and_center._center()
        if rotation_and_center.rotate:
            rotation_and_center._get_vec(base=False)
            rotation_and_center._rotate()

    # Fourier fits
    fourier1,q = _fourier_by_layer(layer_group, box_size, Nx, Ny)
    fourier2,_ = _fourier_by_layer(layer_group_2, box_size, Nx, Ny)
    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())
    
    SFT_A_mn = np.asarray(np.stack((fourier1.Anm,fourier2.Anm,fouriermiddle.Anm),axis=0),dtype=np.float32)

    np.save(f"{out_dir}/{frame:0{num_digits}d}_A_mn.npy", SFT_A_mn)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_q_mn.npy", q)


def calc_fourier(args,u):
    #out_dir, u, ndx, From=0, Until=None, Step=1,Workers=1, centering_and_rotating=None, Nx=2,Ny=2,sqrt_n_atoms=100):
    n_atoms=args.gridsize**2
    Until=args.Until
    ndx=args.index

    if Until is None:
        Until = len(u.trajectory)
    else:
        Until=int(Until)
    if ndx is None:
        exit("An index selection or file has to be supplied. Exiting.")
    try:
        ndx = _read_ndx(ndx)
        dynamic_select=False
        dynamic_selection=None
    except FileNotFoundError:
        print("INFO: The ndx file does not exist, it is assumed a selection was provided for dynamic components.")
        dynamic_select=True
        dynamic_selection=ndx

    LayerList = ["Upper", "Lower", "Middle"]

    dimensions=u.trajectory[0].dimensions
    with open(f"{args.out}/dimensions.csv","w",encoding="UTF8") as dims:
        dims.write(f"#Box Parameters: {' '.join(map(str,dimensions[3:]))}\n")


    if not dynamic_select:
        layer_group = u.atoms[[x - 1 for x in ndx["Upper"]]]
        layer_group_2 = u.atoms[[x - 1 for x in ndx["Lower"]]]
    else:
        layer_group, layer_group_2=None,None

    ract=None
    if args.center is not None:
        ract=Rotation_and_Center_tracker(u,args.center,args.rotation_direction,args.rotate)

    fn = partial(_one_frame,
                layer_group=layer_group,
                layer_group_2=layer_group_2,
                out_dir=args.out,
                dynamic_select=dynamic_select,
                dynamic_selection=dynamic_selection,
                universe=u,
                until=Until,
                rotation_and_center=ract,
                Nx=args.Nx,
                Ny=args.Ny,
                sqrt_n_atoms=args.gridsize
                )


    with ProcessPoolExecutor(max_workers=args.Workers) as ex:
        # map yields results in the same order as the input iterable.
        # You don't return anything, but you MUST exhaust the iterator to execute and surface exceptions.
        for x in range(args.From,Until,args.Step):
            ex.submit(fn,x)


if __name__=="__main__":
    pass