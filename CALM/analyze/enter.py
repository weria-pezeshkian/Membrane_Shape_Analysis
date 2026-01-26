import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
from ..core.fourier_core import Fourier_Series_Function
import os
from ..utilize.write_ndx import write
from MDAnalysis.lib.distances import distance_array
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

logger = logging.getLogger(__name__)

def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100)
    y = np.linspace(0, box_size[1], 100)
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

def fourier_by_layer(layer_group, box_size, Nx=2, Ny=2):
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
              dynamic_select, dynamic_selection,universe,until):
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

    Nx, Ny = 2, 2
    fourier1 = fourier_by_layer(layer_group, box_size)
    fourier2 = fourier_by_layer(layer_group_2, box_size)
    fouriermiddle = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
    fouriermiddle.Update_coff(fourier1.getAnm(), fourier2.getAnm())


    Z_fitted_1 = np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    #np.save(f"{out_dir}/{frame}_Z_fitted_Upper.npy", Z_fitted_1/10)
    Z_fitted_2 = np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    #np.save(f"{out_dir}/{frame}_Z_fitted_Lower.npy", Z_fitted_2/10)

    Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2  #Mid-plane coordinates
    

    #Interpolators for leaflet surfaces z=f(x,y)
    interp_upper = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_1)
    interp_lower = RectBivariateSpline(X[0, :], Y[:, 0], Z_fitted_2)
    

    ### Normal–intersection thickness calculation ###

    #Compute grid spacing of surface in AA based on the shape of X Y already defined above. 
    dx = box_size[0] / (X.shape[1] - 1)
    dy = box_size[1] / (Y.shape[0] - 1) 

    #Construct the surface normal vectors from fitted mid plane Z(x,y). Take fitted surafce, construct and normalize local normal vector. (Z_fitted_vmd gives surface height at each (x,y))
    dz_dx, dz_dy = np.gradient(Z_fitted_vmd, dx, dy)    #Computes partial derivatives, slopes along x and y-axis and gives 2D arrays with local surface slopes. 
    Nx_arr,Ny_arr = -dz_dx,-dz_dy    #Flip signs so that they point "up".

    Nz_arr = np.ones_like(Z_fitted_vmd)   #Sets all values on Z_fitted_vmd to 1
    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)    #Stack the 3 components into a vector at every grid point → shape (Nx, Ny, 3).
    N /= np.linalg.norm(N, axis=-1, keepdims=True)     #Divide by its length so every normal is a unit vector. (Normalises to unit normal vector)

    thickness_map = np.zeros_like(Z_fitted_vmd)        #Creates thickness_map, makes it the size of Z_fitted_vmd and fills it with zeros. 
    l1_map = np.zeros_like(Z_fitted_vmd)
    l2_map = np.zeros_like(Z_fitted_vmd)


    for i in range(X.shape[0]):        #Takes all the x coordinates of my surface, iterates over all rows (i). 
        for j in range(X.shape[1]):    #Iterates over all columns (j)
            x0, y0, z0 = X[i, j], Y[i, j], Z_fitted_vmd[i, j]    #Extracts coordinates, 3D-point of the surface. 
            nvec = N[i, j]    #Get the normal vector that grid point. N is a 3D array containing the unit normal vector at every grid point. nvec is the direction perpendicular to the surface at that point. 

            #Intersection function, finds intersect between a ray starting at the surface point and going along nvec to another surface (a spline) defined as z=f(x,y). 
                    #Returns the distance along the normal. 
            l1 = intersect_surface(interp_upper, 5.0,x0,y0,z0,nvec)   #upwards
            l2 = intersect_surface(interp_lower, -5.0,x0,y0,z0,nvec)  #downwards
            
            l1_map[i,j]=l1
            l2_map[i,j]=l2
            thickness_map[i, j] = l1 + l2          

    Z_fitted_middle = thickness_map
    #np.save(f"{out_dir}/{frame}_Z_fitted_Middle.npy", Z_fitted_middle/10)
 
    Z_fitted_all=np.stack([Z_fitted_1,Z_fitted_2,Z_fitted_middle,], axis=0)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_Z_fitted.npy",Z_fitted_all/10)

    #for fourier,layer in zip([fourier1,fourier2,fouriermiddle],["Upper","Lower","Middle"]):
    #    curvature = fourier.Curv(X, Y)
    #    np.save(f"{out_dir}/{frame}_curvature_frame_{layer}.npy", curvature*10)
    curvature_all = np.stack([fourier1.Curv(X,Y),fourier2.Curv(X,Y),fouriermiddle.Curv(X,Y),], axis=0)
    np.save(f"{out_dir}/{frame:0{num_digits}d}_mean_curvature.npy", curvature_all*10)

def calc(out_dir, u, ndx, From=0, Until=None, Step=1,Workers=1):
    n_atoms=10000

    if Until is None:
        Until = len(u.trajectory)
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
    until=Until
    )

    with ProcessPoolExecutor(max_workers=Workers) as ex:
        # map yields results in the same order as the input iterable.
        # You don't return anything, but you MUST exhaust the iterator to execute and surface exceptions.
        for x in range(From,Until,Step):
            ex.submit(fn,x)      

        #####End of true normal–intersection calculation 






def Analyze(args: List[str]) -> None:
    """Main entry point for Analyzer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a membrane",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'. Alternatively provide a selection for a dynamic calculation of the monolayers, i.e. 'name PO4'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    parser.add_argument('-W','--Workers',default=1,type=int,help="Number of workers for parallel processing")
    parser.add_argument('-c','--clear',default=False,action='store_true',help="Remove old numpy array in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

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
        start=time.perf_counter()
        universe=mda.Universe(args.structure,args.trajectory)
        calc(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,Workers=args.Workers)
        print(f"Execution with {args.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
