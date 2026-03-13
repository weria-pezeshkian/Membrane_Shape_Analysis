import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
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
    z_surf = Z_func(x_t, y_t)[0, 0]   #Evaluate the surface height at the projected (x_t,y_t). Z_func is my RectBiva...Slpine interpolator. calling it with scalar x_t, y_t returns a 2D array with shape (1,1), hence the [0,0] to get the scalar value
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


#def intersect_surface(Z_func, t_sign, x0, y0, z0, nvec):
#    t = t_sign
#    max_iter = 200
#    tol = 1e-8
#
#    for _ in range(max_iter):
#        diff = h(t, Z_func, x0, y0, z0, nvec)
#
#        if abs(diff) < tol:
#            return abs(t)
#
#        t -= diff * 0.5  # damped step
#
#    # If we get here → no convergence
#    return np.nan


#def remove_prot(universe, layer_group, layer_group_2, X, Y, box_size, thickness_map, curvature_all):
#    mem_atoms=layer_group+layer_group_2
#    protein_atoms=universe.atoms.difference(mem_atoms)
#    protein_xy=protein_atoms.positions[:, :2]
#
#    #Grid spacing
#    dx=box_size[0]/X.shape[1]
#    dy=box_size[1]/Y.shape[0]
#
#    #Indices
#    xi = np.floor(protein_xy[:,0] / dx).astype(int)
#    yi = np.floor(protein_xy[:,1] / dy).astype(int)
#    
#    xi = np.clip(xi,0,X.shape[1]-1)
#    yi = np.clip(yi,0,X.shape[0]-1)
#
#    #create mask
#    mask=np.ones(X.shape, dtype=float)
#    mask[yi,xi]=0
#
#    thickness_map=thickness_map * mask
#    curvature_all=curvature_all * mask
#
#    return thickness_map, curvature_all, mask

def remove_prot(universe, layer_group, layer_group_2, X, Y, box_size, thickness_map, curvature_all, radius=1):
    mem_atoms = layer_group + layer_group_2
    sel_layer_group=None
    for item in map(str,layer_group.atoms.indices):
        sel1=universe.select_atoms(f"same residue as index {item}")
        if not sel_layer_group:
            sel_layer_group=sel1
        else:
            sel_layer_group=sel_layer_group+sel1
    sel_layer_group_2=None
    for item in map(str,layer_group_2.atoms.indices):
        sel1=universe.select_atoms(f"same residue as index {item}")
        if not sel_layer_group_2:
            sel_layer_group_2=sel1
        else:
            sel_layer_group_2=sel_layer_group_2+sel1

    print("--------")
    print(sel_layer_group.atoms.n_atoms)
    print(sel_layer_group_2.atoms.n_atoms)
    protein=universe.atoms.difference(sel_layer_group+sel_layer_group_2)
    print(protein.intersection(sel_layer_group).atoms.n_atoms)
    print("--------")
    exit()
    protein_atoms = universe.atoms.difference(mem_atoms)
    protein_xy = protein_atoms.positions[:, :2]  # shape (N_protein, 2)

    # Grid spacing
    dx = box_size[0] / X.shape[1]
    dy = box_size[1] / Y.shape[0]

    # Create meshgrid of bin centers
    xv = (np.arange(X.shape[1]) + 0.1) * dx
    yv = (np.arange(Y.shape[0]) + 0.1) * dy
    X_grid, Y_grid = np.meshgrid(xv, yv)

    # Start with mask of ones
    mask = np.ones_like(X, dtype=float)

    # For each protein atom, mask nearby bins
    for px, py in protein_xy:
        dist2 = (X_grid - px)**2 + (Y_grid - py)**2
        mask[dist2 <= radius**2] = 0

    # Apply mask
    thickness_map = thickness_map * mask
    curvature_all = curvature_all * mask

    return thickness_map, curvature_all, mask


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


def one_frame(frame, *, layer_group, layer_group_2, out_dir,
              dynamic_select, dynamic_selection,universe,until,remove_protein=False):
    num_digits = len(str(abs(until)))
    dimensions=universe.trajectory[frame].dimensions
    box_size = dimensions[:3]
    ts=universe.trajectory[frame]
    with open(f"{out_dir}/dimensions.csv","a",encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str,box_size))}\n")
    X, Y = get_XY(box_size)
    if dynamic_select:
        _,upper_index,lower_index=write(ts,dynamic_selection,write=False)
        layer_group = ts.atoms[[x - 1 for x in upper_index]]
        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]

    Nx, Ny = 3,3 
    fourier1 = fourier_by_layer(layer_group, box_size)
    fourier2 = fourier_by_layer(layer_group_2, box_size)
    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())

    #Upper
    Z_fitted_1 = np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    #Lower
    Z_fitted_2 = np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    #Middle
    Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2 
    
    #Interpolators for leaflet surfaces z=f(x,y)
    interp_upper = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_1)  # x, y
    interp_lower = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_2)  # x, y
    

    ### Thickness calculation ###

    #Compute grid spacing of surface in AA.
    dx = box_size[0] / (X.shape[1] - 1)
    dy = box_size[1] / (Y.shape[0] - 1) 

    #Construct the surface normal vectors from fitted mid plane Z(x,y). Take fitted surafce, construct and normalize local normal vector. 
    dz_dy, dz_dx = np.gradient(Z_fitted_vmd, dy, dx) 
    Nx_arr,Ny_arr = -dz_dx,-dz_dy    #Flip signs so that they point "up". ??????

    Nz_arr = np.ones_like(Z_fitted_vmd)
    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)     #Normalises to unit normal vector

    thickness_map = np.zeros_like(Z_fitted_vmd)        
    l1_map = np.zeros_like(Z_fitted_vmd)
    l2_map = np.zeros_like(Z_fitted_vmd)


    for i in range(X.shape[0]):        
        for j in range(X.shape[1]): 
            x0, y0, z0 = X[i, j], Y[i, j], Z_fitted_vmd[i, j]
            nvec = N[i, j]    #Creates N with the unit normal vectors in each point. 

            #Intersection function, finds intersect between a ray starting at the surface point and going along nvec to another surface (a spline) defined as z=f(x,y). 
                    #Returns the distance along the normal. 
            l1 = intersect_surface(interp_upper, 5.0,x0,y0,z0,nvec)   #upwards
            l2 = intersect_surface(interp_lower, -5.0,x0,y0,z0,nvec)  #downwards

            l1_map[i,j]=l1
            l2_map[i,j]=l2
            thickness_map[i, j] = l1 + l2          
        
    #Save thickness map 
    Z_fitted_middle = thickness_map

    curvature_all = np.stack([fourier1.Curv(X,Y),fourier2.Curv(X,Y),fouriermiddle.Curv(X,Y),], axis=0)
    Z_fitted_all=np.stack([Z_fitted_1,Z_fitted_2,Z_fitted_vmd,], axis=0) 

    if remove_protein:
        thickness_map, curvature_all, mask = remove_prot(universe,layer_group,layer_group_2,X,Y,box_size,thickness_map,curvature_all)

    np.save(f"{out_dir}/{frame:0{num_digits}d}_mean_curvature.npy", curvature_all*10)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_thickness.npy", thickness_map/10)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_Z_fitted.npy",Z_fitted_all/10)

def calc(out_dir, u, ndx, From=0, Until=None, Step=1,Workers=1, remove_protein=False):
    n_atoms=10000


    if Until is None:
        Until = len(u.trajectory)
    else:
        Until=int(Until)
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


    fn = partial(
    one_frame,
    layer_group=layer_group,
    layer_group_2=layer_group_2,
    out_dir=out_dir,
    dynamic_select=dynamic_select,
    dynamic_selection=dynamic_selection,
    universe=u,
    until=Until,
    remove_protein=remove_protein
    )

    #with ProcessPoolExecutor(max_workers=Workers) as ex:
    #    # map yields results in the same order as the input iterable.
    #    # You don't return anything, but you MUST exhaust the iterator to execute and surface exceptions.
    #    for x in range(From,Until,Step):
    #        ex.submit(fn,x)      

    futures = []
    with ProcessPoolExecutor(max_workers=Workers) as ex:
        for x in range(From, Until, Step):
            futures.append(ex.submit(fn, x))

    for f in futures:
        try:
            f.result()  # This will raise any exceptions from the worker
        except Exception as e:
            print("Worker failed:", e)




def Analyze(args: List[str]) -> None:
    """Main entry point for Analyzer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a membrane",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

   
    # Real scientific parameters:
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'. Alternatively provide a selection for a dynamic calculation of the monolayers, i.e. 'name PO4'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=arg_helper.none_or_int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    # Replay:
    parser.add_argument("--replay", help="Load args from replay file")
    parser.add_argument("--out-replay", default=None,help="Write replay file (includes defaults) [Optional: Specify Path to replay file]")
    # File and Resource Management:
    parser.add_argument('-W','--Workers',default=1,type=int,help="Number of workers for parallel processing")
    parser.add_argument('-c','--clear',default=False,action=argparse.BooleanOptionalAction,help="Remove old numpy array in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")
    # Remove protein area
    parser.add_argument('-R','--Remove',action="store_true",help="Remove data from where the protein is located")

    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    args = parser.parse_args(args)
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

    replay_path = args.out_replay or arg_helper.default_replay_name(args.out)
    arg_helper.write_replay_file(replay_path, parser, args)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

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
        calc(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,Workers=args.Workers,remove_protein=args.Remove)
        print(f"Execution with {args.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
