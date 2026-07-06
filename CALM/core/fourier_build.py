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



def read_ndx(filename): #TODO: Weria said, there is an mda plugin for this now. Must be confirmed and if it exists put in here.
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

def fourier_by_layer(layer_group, box_size, Nx=3, Ny=3):
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

def get_XY(box_size,sqrt_n_atoms):
    x = np.linspace(0, box_size[0], sqrt_n_atoms, endpoint=False)
    y = np.linspace(0, box_size[1], sqrt_n_atoms, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return X, Y

def one_frame(frame, *, layer_group, layer_group_2, out_dir,
              dynamic_select, dynamic_selection, universe, until, remove_protein=False,Nx=2,Ny=2,sqrt_n_atoms=100):

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    box_size = dimensions[:3]

    # Save box dimensions
    with open(f"{out_dir}/dimensions.csv", "a", encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str, box_size))}\n")

    # XY grid
    X, Y = get_XY(box_size,sqrt_n_atoms)

    # Dynamic selection if requested
    if dynamic_select:
        _, upper_index, lower_index = write(ts, dynamic_selection, write=False)
        layer_group = ts.atoms[[x - 1 for x in upper_index]]
        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]

    # Fourier fits
    fourier1,q = fourier_by_layer(layer_group, box_size, Nx, Ny)
    fourier2,_ = fourier_by_layer(layer_group_2, box_size, Nx, Ny)
    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())
    
    SFT_A_mn = np.asarray(np.stack((fourier1.Anm,fourier2.Anm,fouriermiddle.Anm),axis=0),dtype=np.float32)

    np.save(f"{out_dir}/{frame:0{num_digits}d}_A_mn.npy", SFT_A_mn)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_q_mn.npy", q)


def calc(out_dir, u, ndx, From=0, Until=None, Step=1,Workers=1, remove_protein=False, Nx=2,Ny=2,sqrt_n_atoms=100):
    n_atoms=sqrt_n_atoms**2


    if Until is None:
        Until = len(u.trajectory)
    else:
        Until=int(Until)
    if ndx is None:
        exit("An index selection or file has to be supplied. Exiting.")
    try:
        ndx = read_ndx(ndx)
        dynamic_select=False
        dynamic_selection=None
    except FileNotFoundError:
        print("INFO: The ndx file does not exist, it is assumed a selection was provided for dynamic components.")
        dynamic_select=True
        dynamic_selection=ndx

    LayerList = ["Upper", "Lower", "Middle"]

    dimensions=u.trajectory[0].dimensions
    with open(f"{out_dir}/dimensions.csv","w",encoding="UTF8") as dims:
        dims.write(f"#Box Parameters: {' '.join(map(str,dimensions[3:]))}\n")


    if not dynamic_select:
        layer_group = u.atoms[[x - 1 for x in ndx["Upper"]]]
        layer_group_2 = u.atoms[[x - 1 for x in ndx["Lower"]]]
    else:
        layer_group, layer_group_2=None,None


    fn = partial(one_frame,
                layer_group=layer_group,
                layer_group_2=layer_group_2,
                out_dir=out_dir,
                dynamic_select=dynamic_select,
                dynamic_selection=dynamic_selection,
                universe=u,
                until=Until,
                remove_protein=remove_protein,
                Nx=Nx,
                Ny=Ny,
                sqrt_n_atoms=sqrt_n_atoms
                )


    print(From,Until,Step)
    with ProcessPoolExecutor(max_workers=Workers) as ex:
        # map yields results in the same order as the input iterable.
        # You don't return anything, but you MUST exhaust the iterator to execute and surface exceptions.
        for x in range(From,Until,Step):
            ex.submit(fn,x)


if __name__=="__main__":
    pass