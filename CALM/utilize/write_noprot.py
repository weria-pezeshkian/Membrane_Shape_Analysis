import numpy as np
import MDAnalysis as mda
import argparse
import os
from pathlib import Path
import shutil
from scipy.ndimage import binary_dilation


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


##Function for protein masking.

def remove_prot(universe, layer_group, layer_group_2, X, Y, box_size):

    #Combine membrane layers
    mem_atoms = (layer_group + layer_group_2).unique
    mem_residues = mem_atoms.residues

    #Everything that is not membrane is considered protein
    protein_atoms = universe.atoms.difference(mem_residues.atoms)
    protein_xy = protein_atoms.positions[:, :2]

    #Grid spacing
    dx = box_size[0] / X.shape[1]
    dy = box_size[1] / Y.shape[0]

    #Convert protein positions to grid indices
    xi = np.floor(protein_xy[:, 0] / dx).astype(int)
    yi = np.floor(protein_xy[:, 1] / dy).astype(int)

    xi = np.clip(xi, 0, X.shape[1] - 1)
    yi = np.clip(yi, 0, X.shape[0] - 1)

    #Build mask
    mask = np.zeros(X.shape, dtype=bool)
    mask[yi, xi] = True

    #Expand mask region slightly
    mask = binary_dilation(mask, iterations=4)

    return mask


## Main function to write curvature data without protein contributions
#Load a per-frame NumPy array and apply a protein mask. 
def load_and_mask(frame, name, data_dir, mask, num_digits):

    path = f"{data_dir}/{frame:0{num_digits}d}_{name}.npy"
    arr = np.load(path)

    if arr.ndim == 2:
        return np.where(mask, np.nan, arr)

    if arr.ndim == 3:
        return np.where(mask[None, :, :], np.nan, arr)

    if arr.ndim == 4:
        return np.where(mask[None, :, :, None], np.nan, arr)

    raise ValueError(f"Unexpected array shape: {arr.shape}")


def get_mask(frame, universe, ndx, dynamic_select, dynamic_selection, X, Y):
    """
    Generate a protein mask for a single trajectory frame.
    """
    ts = universe.trajectory[frame]
    box_size = ts.dimensions[:3]

    if dynamic_select:
        raise NotImplementedError("If needed, we can support dynamic selection later.")

    layer_group = universe.atoms[[i - 1 for i in ndx["Upper"]]]
    layer_group_2 = universe.atoms[[i - 1 for i in ndx["Lower"]]]

    return remove_prot(universe,layer_group,layer_group_2,X,Y,box_size,)

def write_noprot(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--trajectory", required=True)
    parser.add_argument("-s", "--structure", required=True)
    parser.add_argument("-n", "--index", required=True)
    parser.add_argument("-i", "--input_data_folder", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument('-c','--clear',default=False,action=argparse.BooleanOptionalAction,help="Remove old numpy arrays in out directiory. NO WARNING IS GIVEN AND NO BACKUP IS MADE")

    args = parser.parse_args(args)
    os.makedirs(args.out, exist_ok=True)

    shutil.copy(os.path.join(args.input_data_folder, "dimensions.csv"),os.path.join(args.out, "dimensions.csv"))

    #Load universe and index
    u = mda.Universe(args.structure, args.trajectory)
    ndx = read_ndx(args.index)

    # Storage
    thickness_list = []
    mean_list = []
    gauss_list = []
    principal_list = []
    principal_dirs_list = []

    num_digits = len(str(len(u.trajectory)))

    for frame in range(len(u.trajectory)):
        ts = u.trajectory[frame]
        box_size = ts.dimensions[:3]

        X, Y = get_XY(box_size)
        mask = get_mask(frame, u, ndx, False, None, X, Y)

        thickness_list.append(load_and_mask(frame, "thickness", args.input_data_folder, mask, num_digits))
        mean_list.append(load_and_mask(frame, "mean_curvature", args.input_data_folder, mask, num_digits))
        gauss_list.append(load_and_mask(frame, "gaussian_curvature", args.input_data_folder, mask, num_digits))
        principal_list.append(load_and_mask(frame, "principal_curvatures", args.input_data_folder, mask, num_digits))
        principal_dirs_list.append(load_and_mask(frame,"principal_dirs",args.input_data_folder,mask,num_digits))

    num_digits = len(str(len(u.trajectory)))

    for frame_idx in range(len(u.trajectory)):

        thickness = thickness_list[frame_idx]
        mean = mean_list[frame_idx]
        gauss = gauss_list[frame_idx]
        principal = principal_list[frame_idx]
        principal_dirs = principal_dirs_list[frame_idx]

        np.save(f"{args.out}/{frame_idx:0{num_digits}d}_thickness.npy", thickness)
        np.save(f"{args.out}/{frame_idx:0{num_digits}d}_mean_curvature.npy", mean)
        np.save(f"{args.out}/{frame_idx:0{num_digits}d}_gaussian_curvature.npy", gauss)
        np.save(f"{args.out}/{frame_idx:0{num_digits}d}_principal_curvatures.npy", principal)
        np.save(f"{args.out}/{frame_idx:0{num_digits}d}_principal_dirs.npy", principal_dirs)


if __name__ == "__main__":
    pass