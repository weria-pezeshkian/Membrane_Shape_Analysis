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

#def remove_prot(universe, layer_group, layer_group_2, X, Y, box_size, thickness_map, curvature_all):
#    mem_atoms = (layer_group + layer_group_2).unique
#    mem_residues = mem_atoms.residues
#    protein_atoms = universe.atoms.difference(mem_residues.atoms)
#    protein_xy = protein_atoms.positions[:, :2]
#
#    #Grid spacing
#    dx = box_size[0] / X.shape[1]
#    dy = box_size[1] / Y.shape[0]
#
#    #Indices
#    xi = np.floor(protein_xy[:,0] / dx).astype(int)
#    yi = np.floor(protein_xy[:,1] / dy).astype(int)
#    
#    xi = np.clip(xi, 0, X.shape[1]-1)
#    yi = np.clip(yi, 0, X.shape[0]-1)
#
#    #Create mask with NaN
#    mask = np.zeros(X.shape, dtype=bool)
#    mask[yi, xi] = True
#    mask = binary_dilation(mask, iterations=4)
#    mask = np.where(mask, np.nan, 1.0)
#
#    #Apply mask
#    thickness_map = np.where(np.isnan(mask), np.nan, thickness_map)
#    curvature_all = np.where(np.isnan(mask), np.nan, curvature_all)
#
#    return thickness_map, curvature_all, mask



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

def one_frame(frame, *, layer_group, layer_group_2, out_dir,
              dynamic_select, dynamic_selection, universe, until, remove_protein=False):

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    box_size = dimensions[:3]

    # Save box dimensions
    with open(f"{out_dir}/dimensions.csv", "a", encoding="UTF8") as dims:
        dims.write(f"{frame},{','.join(map(str, box_size))}\n")

    # XY grid
    X, Y = get_XY(box_size)

    # Dynamic selection if requested
    if dynamic_select:
        _, upper_index, lower_index = write(ts, dynamic_selection, write=False)
        layer_group = ts.atoms[[x - 1 for x in upper_index]]
        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]

    Nx, Ny = 3, 3
    # Fourier fits
    fourier1 = fourier_by_layer(layer_group, box_size, Nx, Ny)
    fourier2 = fourier_by_layer(layer_group_2, box_size, Nx, Ny)
    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())

    # Evaluate surfaces
    Z_fitted_1 = np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    Z_fitted_2 = np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2

    # Interpolators for intersections
    interp_upper = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_1)
    interp_lower = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_2)

    # ---- Thickness calculation ---- #
    dx = box_size[0] / (X.shape[1] - 1)
    dy = box_size[1] / (Y.shape[0] - 1)
    dz_dy, dz_dx = np.gradient(Z_fitted_vmd, dy, dx)
    Nx_arr, Ny_arr = -dz_dx, -dz_dy
    Nz_arr = np.ones_like(Z_fitted_vmd)
    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)

    thickness_map = np.zeros_like(Z_fitted_vmd)
    l1_map = np.zeros_like(Z_fitted_vmd)
    l2_map = np.zeros_like(Z_fitted_vmd)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0, z0 = X[i, j], Y[i, j], Z_fitted_vmd[i, j]
            nvec = N[i, j]

            l1 = intersect_surface(interp_upper, 5.0, x0, y0, z0, nvec)
            l2 = intersect_surface(interp_lower, -5.0, x0, y0, z0, nvec)

            l1_map[i, j] = l1
            l2_map[i, j] = l2
            thickness_map[i, j] = l1 + l2

    Z_fitted_middle = thickness_map

    # ---- Curvature calculations using shape operator ---- #
    # Upper leaflet
    H1, K1, k1_1, k2_1, dirs1_1, dirs2_1 = fourier1.ShapeOperatorCurvatures(X, Y)
    # Lower leaflet
    H2, K2, k1_2, k2_2, dirs1_2, dirs2_2 = fourier2.ShapeOperatorCurvatures(X, Y)
    # Middle surface
    Hmid, Kmid, k1_mid, k2_mid, dirs1_mid, dirs2_mid = fouriermiddle.ShapeOperatorCurvatures(X, Y)

    # ---- Apply protein mask if requested ---- #
    if remove_protein:
        mask = remove_prot(universe, layer_group, layer_group_2, X, Y, box_size)

        # Thickness
        thickness_map = np.where(mask, np.nan, thickness_map)

        # Mean curvature
        H1   = np.where(mask, np.nan, H1)
        H2   = np.where(mask, np.nan, H2)
        Hmid = np.where(mask, np.nan, Hmid)

        # Gaussian curvature
        K1   = np.where(mask, np.nan, K1)
        K2   = np.where(mask, np.nan, K2)
        Kmid = np.where(mask, np.nan, Kmid)

        # Principal curvatures
        k1_1   = np.where(mask, np.nan, k1_1)
        k2_1   = np.where(mask, np.nan, k2_1)
        k1_2   = np.where(mask, np.nan, k1_2)
        k2_2   = np.where(mask, np.nan, k2_2)
        k1_mid = np.where(mask, np.nan, k1_mid)
        k2_mid = np.where(mask, np.nan, k2_mid)

        # Principal directions (need broadcasting)
        dirs1_1   = np.where(mask[:, :, None], np.nan, dirs1_1)
        dirs2_1   = np.where(mask[:, :, None], np.nan, dirs2_1)
        dirs1_2   = np.where(mask[:, :, None], np.nan, dirs1_2)
        dirs2_2   = np.where(mask[:, :, None], np.nan, dirs2_2)
        dirs1_mid = np.where(mask[:, :, None], np.nan, dirs1_mid)
        dirs2_mid = np.where(mask[:, :, None], np.nan, dirs2_mid)

    # ---- Save results in separate numpy files ---- #
    np.save(f"{out_dir}/{frame:0{num_digits}d}_thickness.npy", thickness_map / 10)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_Z_fitted.npy", np.stack([Z_fitted_1, Z_fitted_2, Z_fitted_vmd], axis=0) / 10)

    # Mean curvature
    np.save(f"{out_dir}/{frame:0{num_digits}d}_mean_curvature.npy",
            np.stack([H1, H2, Hmid], axis=0) * 10)
    # Gaussian curvature
    np.save(f"{out_dir}/{frame:0{num_digits}d}_gaussian_curvature.npy",
            np.stack([K1, K2, Kmid], axis=0) * 10)
    # Principal curvatures
    np.save(f"{out_dir}/{frame:0{num_digits}d}_principal_curvatures.npy",
            np.stack([k1_1, k2_1, k1_2, k2_2, k1_mid, k2_mid], axis=0) * 10)
    # Principal directions
    np.save(f"{out_dir}/{frame:0{num_digits}d}_principal_dirs.npy",
            np.stack([dirs1_1, dirs2_1, dirs1_2, dirs2_2, dirs1_mid, dirs2_mid], axis=0))


## --------------------------------- ######



#def one_frame(frame, *, layer_group, layer_group_2, out_dir,
#              dynamic_select, dynamic_selection,universe,until,remove_protein=False):
#    num_digits = len(str(abs(until)))
#    dimensions=universe.trajectory[frame].dimensions
#    box_size = dimensions[:3]
#    ts=universe.trajectory[frame]
#    with open(f"{out_dir}/dimensions.csv","a",encoding="UTF8") as dims:
#        dims.write(f"{frame},{','.join(map(str,box_size))}\n")
#    X, Y = get_XY(box_size)
#    if dynamic_select:
#        _,upper_index,lower_index=write(ts,dynamic_selection,write=False)
#        layer_group = ts.atoms[[x - 1 for x in upper_index]]
#        layer_group_2 = ts.atoms[[x - 1 for x in lower_index]]
#
#    Nx, Ny = 3,3 
#    fourier1 = fourier_by_layer(layer_group, box_size)
#    fourier2 = fourier_by_layer(layer_group_2, box_size)
#    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
#    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())
#
#    #Upper
#    Z_fitted_1 = np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
#    #Lower
#    Z_fitted_2 = np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
#    #Middle
#    Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2 
#    
#    #Interpolators for leaflet surfaces z=f(x,y)
#    interp_upper = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_1)  # x, y
#    interp_lower = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_2)  # x, y
#
#    #-----Thickness calculation-----#
#
#    #Compute grid spacing of surface in AA.
#    dx = box_size[0] / (X.shape[1] - 1)
#    dy = box_size[1] / (Y.shape[0] - 1) 
#
#    #Construct the surface normal vectors from fitted mid plane Z(x,y).
#    dz_dy, dz_dx = np.gradient(Z_fitted_vmd, dy, dx) 
#    Nx_arr,Ny_arr = -dz_dx,-dz_dy    #Flip signs so normals point pos Z-direction.
#    Nz_arr = np.ones_like(Z_fitted_vmd)
#    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)
#    N /= np.linalg.norm(N, axis=-1, keepdims=True) 
#
#    #Thickness map 
#    thickness_map = np.zeros_like(Z_fitted_vmd)        
#    l1_map = np.zeros_like(Z_fitted_vmd)
#    l2_map = np.zeros_like(Z_fitted_vmd)
#
#    #Intersection function, returns the distance along the normal. 
#    for i in range(X.shape[0]):        
#        for j in range(X.shape[1]): 
#            x0, y0, z0 = X[i, j], Y[i, j], Z_fitted_vmd[i, j]
#            nvec = N[i, j]    #Creates N with the unit normal vectors in each point. 
#
#            
#            l1 = intersect_surface(interp_upper, 5.0,x0,y0,z0,nvec)   #upwards
#            l2 = intersect_surface(interp_lower, -5.0,x0,y0,z0,nvec)  #downwards
#
#            l1_map[i,j]=l1
#            l2_map[i,j]=l2
#            thickness_map[i, j] = l1 + l2          
#        
#    #Save thickness map 
#    Z_fitted_middle = thickness_map
#    curvature_all = np.stack([fourier1.Curv(X,Y),fourier2.Curv(X,Y),fouriermiddle.Curv(X,Y),], axis=0)
#
#    Z_fitted_all=np.stack([Z_fitted_1,Z_fitted_2,Z_fitted_vmd,], axis=0) 
#
#    if remove_protein:
#        thickness_map, curvature_all, mask = remove_prot(universe,layer_group,layer_group_2,X,Y,box_size,thickness_map,curvature_all)
#
#    np.save(f"{out_dir}/{frame:0{num_digits}d}_mean_curvature.npy", curvature_all*10)
#    np.save(f"{out_dir}/{frame:0{num_digits}d}_thickness.npy", thickness_map/10)
#    np.save(f"{out_dir}/{frame:0{num_digits}d}_Z_fitted.npy",Z_fitted_all/10)

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


    fn = partial(one_frame,layer_group=layer_group,layer_group_2=layer_group_2,out_dir=out_dir,dynamic_select=dynamic_select,dynamic_selection=dynamic_selection,universe=u,until=Until,remove_protein=remove_protein)



    with ProcessPoolExecutor(max_workers=Workers) as ex:
        # map yields results in the same order as the input iterable.
        # You don't return anything, but you MUST exhaust the iterator to execute and surface exceptions.
        for x in range(From,Until,Step):
            ex.submit(fn,x)      


#    with ProcessPoolExecutor(max_workers=Workers) as ex:
#        futures = [ex.submit(fn, x) for x in range(From, Until, Step)]
#
#        for f in tqdm(futures):
#            try:
#                f.result()
#            except Exception as e:
#                print("Worker failed:")
#                raise e

def Analyze(args: List[str]) -> None:
    """Main entry point for Analyzer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a Lipid Bilayer",formatter_class=argparse.RawDescriptionHelpFormatter)

    # Real scientific parameters:
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file (.xtc) ")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file (.tpr)")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'. Alternatively provide a selection for a dynamic calculation of the monolayers, i.e. 'name PO4'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here, default=0")
    parser.add_argument('-U','--Until',default=None,type=arg_helper.none_or_int,help="Discard all frames in the trajectory after to the frame supplied here, default=None")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here, default=1")
    # Replay:
    parser.add_argument("--replay", help="Load args from replay file")
    parser.add_argument("--out-replay", default=None,help="Write replay file (includes defaults) [Optional: Specify Path to replay file]")
    # File and Resource Management:
    parser.add_argument('-W','--Workers',default=1,type=int,help="Number of workers for parallel processing, 1 worker=1 cpu, default=1")
    parser.add_argument('-c','--clear',default=False,action=argparse.BooleanOptionalAction,help="Remove old numpy array in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")
    # Remove protein area
    parser.add_argument('-R','--Remove',default=False, action="store_true",help="Remove data from where the protein is located, default=False")

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



