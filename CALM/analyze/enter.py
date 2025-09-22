import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List
from scipy.interpolate import RectBivariateSpline
from ..core.fourier_core import Fourier_Series_Function
import os

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
    while True:                          #Newton iterations
        diff= h(t,Z_func,x0,y0,z0,nvec)
        if abs(diff) < 1e-8:               #Tolerance 
            break
        t -= diff #/ .95            #update using z-component


    return t

def calc(out_dir, u, ndx, From=0, Until=None, Step=1, layer_string="Both"):
    if Until is None:
        Until = len(u.trajectory)
    ndx = read_ndx(ndx)
    box_size = u.trajectory[0].dimensions[:3]
    np.save(file=f"{out_dir}/boxsize.npy", arr=box_size)
    X, Y = get_XY(box_size)

    if layer_string.lower() != "both":
        LayerList = [layer_string]
    else:
        LayerList = ["Upper", "Lower", "Both"]

    for Layer in LayerList:
        if Layer == "Both":
            layer_group = u.atoms[[x - 1 for x in ndx["Upper"]]]
            layer_group_2 = u.atoms[[x - 1 for x in ndx["Lower"]]]
        else:
            layer_group = u.atoms[[x - 1 for x in ndx[Layer]]]

        with mda.coordinates.XTC.XTCWriter(f"{out_dir}/fourier_curvature_fitting_{Layer}.xtc", n_atoms=100000) as writer:
            count = 0
            for t, ts in tqdm(enumerate(u.trajectory[From:Until:Step])):
                count += 1

                if Layer == "Both":
                    Nx, Ny = 2, 2
                    fourier1 = fourier_by_layer(layer_group, box_size)
                    fourier2 = fourier_by_layer(layer_group_2, box_size)
                    fourier = Fourier_Series_Function(box_size[0], box_size[1], Nx, Ny)
                    fourier.Update_coff(fourier1.getAnm(), fourier2.getAnm())

                    Z_fitted_1 = np.array([fourier1.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                    Z_fitted_2 = np.array([fourier2.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
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
                            #print(f"[i={i}, j={j}] l1={l1}, l2={l2}")
                            thickness_map[i, j] = l1 + l2               #add them both together



                    Z_fitted = thickness_map
                    #####End of true normal–intersection calculation 

                else:
                    fourier = fourier_by_layer(layer_group, box_size)
                    Z_fitted = np.array([fourier.Z(xi, yi) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
                    Z_fitted_vmd = Z_fitted[:]

                curvature = fourier.Curv(X, Y)
                coordinates = np.vstack([X.flatten(), Y.flatten(), Z_fitted_vmd.flatten()]).T
                Z_fitted = Z_fitted / 10  # Å → nm
                np.save(f"{out_dir}/Z_fitted_{count}_{Layer}.npy", Z_fitted)

                curvature = curvature * 10  # Å⁻¹ → nm⁻¹
                np.save(f"{out_dir}/curvature_frame_{count}_{Layer}.npy", curvature)

                pseudo_universe = mda.Universe.empty(n_atoms=coordinates.shape[0], trajectory=True)
                pseudo_universe.atoms.positions = coordinates
                pseudo_universe.dimensions = ts.dimensions

                if t == 0:
                    pseudo_universe.atoms.write(f"{out_dir}/pseudo_universe_{Layer}.gro")

                writer.write(pseudo_universe.atoms)



def Analyze(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Calculate the curvature of a membrane",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--index',type=str,help="Specify the path to an index file containing the monolayers. To consider both monolayers, they need to be named 'Upper' and 'Lower'")
    parser.add_argument('-o','--out',type=str,help="Specify a path to a folder to which all calculated numpy arrays are saved")
    parser.add_argument('-F','--From',default=0,type=int,help="Discard all frames in the trajectory prior to the frame supplied here")
    parser.add_argument('-U','--Until',default=None,type=int,help="Discard all frames in the trajectory after to the frame supplied here")
    parser.add_argument('-S','--Step',default=1,type=int,help="Traverse the trajectory with a step length supplied here")
    parser.add_argument('-l','--leaflet',default="Both",help="Choose which membrane leaflet to calculate. Default is Both")
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
        universe=mda.Universe(args.structure,args.trajectory)
        calc(out_dir=args.out,u=universe,ndx=args.index,From=args.From,Until=args.Until,Step=args.Step,layer_string=args.leaflet)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise