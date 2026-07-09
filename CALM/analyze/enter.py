import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from ..core.fourier_core import Fourier_Series_Function
from ..core.fourier_sft import SFT
from ..core import argument_parser as arg_helper
from ..analyze.analyze import analysis
import os
from ..utilize.write_ndx import write
from MDAnalysis.lib.distances import distance_array
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import shlex
from pathlib import Path
from scipy.ndimage import binary_dilation



logger = logging.getLogger(__name__)

def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100, endpoint=False)
    y = np.linspace(0, box_size[1], 100, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return X, Y

def read_ndx(filename):
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
    return fourier

def h(t,Z_func,x0,y0,z0,nvec):
    x_t = x0 + t * nvec[0]             #Compute the candicate point on the ray at parameter t. 
    y_t = y0 + t * nvec[1]
    z_t = z0 + t * nvec[2]
    z_surf = Z_func(x_t, y_t)[0, 0]   #Evaluate the surface height at the projected (x_t,y_t). 
    diff = z_t - z_surf

    return diff


def intersect_surface(Z_func, t_sign,x0,y0,z0,nvec):
    t=t_sign
    while True:
        diff= h(t,Z_func,x0,y0,z0,nvec)
        if abs(diff) < 1e-8:               #Tolerance 
            break
        t -= diff #/ .95            #update using z-component


    return np.abs(t)


def remove_prot(universe, layer_group, layer_group_2, X, Y, box_size):
    mem_atoms = (layer_group + layer_group_2).unique
    mem_residues = mem_atoms.residues
    protein_atoms = universe.atoms.difference(mem_residues.atoms)
    protein_xy = protein_atoms.positions[:, :2]

    dx = box_size[0] / X.shape[1]
    dy = box_size[1] / Y.shape[0]

    xi = np.floor(protein_xy[:, 0] / dx).astype(int)
    yi = np.floor(protein_xy[:, 1] / dy).astype(int)

    xi = np.clip(xi, 0, X.shape[1] - 1)
    yi = np.clip(yi, 0, X.shape[0] - 1)

    mask = np.zeros(X.shape, dtype=bool)
    mask[yi, xi] = True
    mask = binary_dilation(mask, iterations=4)

    return mask


def get_absolute_distances(ref,grid,mask=None,dimensions=None):
    ref = np.asarray(ref, dtype=float)
    grid = np.asarray(grid, dtype=float)

    N,M = ref.shape[0], grid.shape[0]
    if mask is None:

        dists=distance_array(grid,ref,box=dimensions)
        
        # Define your threshold (adjust as needed)
        threshold = np.percentile(dists, 25)

        # Update mask: keep only pairs within threshold
        mask = dists <= threshold
        min_dists = np.min(dists, axis=1)
    else:
        mask = np.asarray(mask, dtype=bool)

        # Output array of distances per ref point
        min_dists = np.full(M, np.inf, dtype=float)

        # Efficient loop: only compute distances for active pairs
        for i in range(M):
            valid = np.nonzero(mask[i])[0]
            if valid.size == 0:
                continue  # remains np.inf
            # Compute only relevant distances for this row
            diffs = ref[valid] - grid[i]
            dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
            min_dists[i] = np.min(dists)
    return min_dists,mask

##### Testing new calculation method ######

def f(t, interp, mx, my, mz, nx, ny, nz,Lx,Ly):
    xq = mx + t * nx
    yq = my + t * ny
    zq = mz + t * nz

    xq = np.mod(xq, Lx)
    yq = np.mod(yq, Ly)

    return zq - interp(yq, xq, grid=False)[()]

def periodic_gradient(Z, dx, dy, periodic_x=True, periodic_y=True):
    # dZ/dx: axis 1
    if periodic_x:
        dz_dx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * dx)
    else:
        dz_dx = np.gradient(Z, dx, axis=1)

    # dZ/dy: axis 0
    if periodic_y:
        dz_dy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * dy)
    else:
        dz_dy = np.gradient(Z, dy, axis=0)

    return dz_dy, dz_dx


def Analyze(args: List[str]) -> None:
    """Main entry point for Analyzer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a Lipid Bilayer",formatter_class=argparse.RawDescriptionHelpFormatter)

    METHODS = (
        "thickness",
        "Z_fitted",
        "mean",
        "gaussian",
        "principal",
        "principal_directions",
    )

    # Real scientific parameters:
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file (.xtc) ")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file (.tpr)")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'. Alternatively provide a selection for a dynamic calculation of the monolayers, i.e. 'name PO4'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here, default=0")
    parser.add_argument('-U','--Until',default=None,type=arg_helper.none_or_int,help="Discard all frames in the trajectory after to the frame supplied here, default=None")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here, default=1")
    parser.add_argument('--lambda_x', type=float, default=None,help="Fourier wavelength scale in x-direction (nm)")
    parser.add_argument('--lambda_y', type=float, default=None,help="Fourier wavelength scale in y-direction (nm)")
    parser.add_argument('--gridsize',default=100,help="Squareroot of the actual grid size number. Default is 100, which would put 100 points in x, 100 points in y direction, resulting in 10000 gridpoints")
    # Manipulation flags:
    parser.add_argument('-C','--center',default=None,type=str,help="MDAnalysis selection syntax to choose what should be centered")
    parser.add_argument('--rotate',default=False, action="store_true", help="Rotation alignment of each frame")
    parser.add_argument('--rotation-direction',default=None, type=str,help="An MDAnalysis selection (syntax). The center of geometry will be used for the rotation.")
    parser.add_argument('--Remove',default=False, action="store_true",help="Attempts to remove the transmembrane domain.")
    #Early exit and subsequent analysis
    parser.add_argument('--early-abort',default=False,action="store_true",help="exit out after writing A_mn and q_mn, no further analysis")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=METHODS,
        default=None,
        help="Analysis method(s) to run. If omitted, all methods are run.",
    )

    # Replay:
    parser.add_argument("--replay", help="Load args from replay file")
    parser.add_argument("--out-replay", default=None,help="Write replay file (includes defaults) [Optional: Specify Path to replay file]")
    # File and Resource Management:
    parser.add_argument('-W','--Workers',default=1,type=int,help="Number of workers for parallel processing, 1 worker=1 cpu, default=1")
    parser.add_argument('-c','--clear',default=False,action=argparse.BooleanOptionalAction,help="Remove old numpy array in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")

    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    args = parser.parse_args(args)


    if args.center is None:
        if args.rotate:
            parser.error("--rotate requires --center")

        if args.rotation_direction is not None:
            parser.error("--rotation-direction requires --center")

    if args.rotation_direction is not None and not args.rotate:
        parser.error("--rotation-direction requires --rotate")

    logging.basicConfig(level=logging.INFO)

    
    replayed: list[str] = []
    if pre_ns.replay:
        replay_path = Path(pre_ns.replay)
        for line in replay_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            replayed.extend(shlex.split(s))

    # User-provided args should have priority -> they must appear LAST
    combined_argv = replayed + remaining
    args=parser.parse_args(combined_argv)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    replay_path = args.out_replay or arg_helper.default_replay_name(args.out)
    arg_helper.write_replay_file(replay_path, parser, args)

    if args.clear:
        for filename in os.listdir(args.out):
            if filename.endswith('.npy'):
                file_path = os.path.join(args.out, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    try:
        structure=Path(args.structure)
        trajectory=Path(args.trajectory)
        start=time.perf_counter()
        universe=mda.Universe(structure,trajectory)
        sft=SFT()
        sft.build(args,universe)
        sft.write(f"{args.out}/complete")
        if args.early_abort:
            print("Early abort active: Finishing after writing Amn and qmn")
        else:
            active_methods = METHODS if args.method is None else tuple(args.method)
            analysis(universe,sft,active_methods,args)

        print(f"Execution with {args.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise



