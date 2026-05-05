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

def get_available_frames(data_dir, suffix="_thickness.npy"):
    frames = []
    for fname in os.listdir(data_dir):
        if fname.endswith(suffix):
            try:
                frames.append(int(fname.split("_")[0]))
            except ValueError:
                continue
    return sorted(frames)


def filter_valid_frames(frames, data_dir, num_digits):
    required = ["thickness","mean_curvature","gaussian_curvature","principal_curvatures","principal_dirs"]

    valid = []
    for frame in frames:
        ok = True
        for name in required:
            path = f"{data_dir}/{frame:0{num_digits}d}_{name}.npy"
            if not os.path.exists(path):
                ok = False
                break
        if ok:
            valid.append(frame)
    return valid


def remove_prot(universe,layer_group,layer_group_2,X,Y,box_size):

    mem_atoms=(layer_group+layer_group_2).unique
    mem_residues=mem_atoms.residues

    protein_atoms=universe.atoms.difference(mem_residues.atoms)
    protein_xy=protein_atoms.positions[:, :2]

    dx = box_size[0]/X.shape[1]
    dy = box_size[1]/X.shape[0]

    xi = np.floor(protein_xy[:, 0] / dx).astype(int)
    yi = np.floor(protein_xy[:, 1] / dy).astype(int)

    xi = np.clip(xi, 0, X.shape[1] - 1)
    yi = np.clip(yi, 0, X.shape[0] - 1)

    mask = np.zeros(X.shape, dtype=bool)
    mask[yi, xi] = True

    mask = binary_dilation(mask, iterations=3)

    return mask


def load_and_mask(frame,name,data_dir,mask,num_digits):

    path = f"{data_dir}/{frame:0{num_digits}d}_{name}.npy"
    arr = np.load(path)

    if arr.ndim == 2:
        return np.where(mask, np.nan, arr)

    if arr.ndim == 3:
        return np.where(mask[None, :, :], np.nan, arr)

    if arr.ndim == 4:
        return np.where(mask[None, :, :, None], np.nan, arr)

    raise ValueError(f"Unexpected shape: {arr.shape}")


def get_mask(frame, universe, ndx, X, Y):
    ts = universe.trajectory[frame]
    box_size = ts.dimensions[:3]

    layer_group = universe.atoms[[i - 1 for i in ndx["Upper"]]]
    layer_group_2 = universe.atoms[[i - 1 for i in ndx["Lower"]]]

    return remove_prot(universe, layer_group, layer_group_2, X, Y, box_size)

def write_noprot(raw_args):

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--trajectory", required=True)
    parser.add_argument("-s", "--structure", required=True)
    parser.add_argument("-n", "--index", required=True)
    parser.add_argument("-i", "--input_data_folder", required=True)
    parser.add_argument("-o", "--out", required=True)

    # SAME STYLE AS ANALYZER
    parser.add_argument("-F", "--From", default=0, type=int)
    parser.add_argument("-U", "--Until", default=None, type=int)
    parser.add_argument("-S", "--Step", default=1, type=int)

    parser.add_argument('-c','--clear',default=False,
        action=argparse.BooleanOptionalAction)

    args = parser.parse_args(raw_args)

    os.makedirs(args.out, exist_ok=True)

    shutil.copy(
        os.path.join(args.input_data_folder, "dimensions.csv"),
        os.path.join(args.out, "dimensions.csv")
    )

    #Load system
    u = mda.Universe(args.structure, args.trajectory)
    ndx = read_ndx(args.index)

    #Find availabale frames
    frames = get_available_frames(args.input_data_folder)

    if len(frames) == 0:
        raise RuntimeError("No input .npy files found")

    sample_file = next(f for f in os.listdir(args.input_data_folder) if f.endswith("_thickness.npy"))
    num_digits = len(sample_file.split("_")[0])

    frames = filter_valid_frames(frames, args.input_data_folder, num_digits)

    if args.Until is None:
        args.Until = max(frames) + 1

    frames = [f for f in frames if f >= args.From]
    frames = [f for f in frames if f < args.Until]
    frames = frames[::args.Step]

    if len(frames) == 0:
        raise RuntimeError("No frames left after filtering (-F/-U/-S)")

    #Storage
    thickness_list = []
    mean_list = []
    gauss_list = []
    principal_list = []
    dirs_list = []

    #Compute masks + load
    for frame in frames:

        ts = u.trajectory[frame]
        box_size = ts.dimensions[:3]

        X, Y = get_XY(box_size)
        mask = get_mask(frame, u, ndx, X, Y)

        thickness_list.append(load_and_mask(frame, "thickness", args.input_data_folder, mask, num_digits))
        mean_list.append(load_and_mask(frame, "mean_curvature", args.input_data_folder, mask, num_digits))
        gauss_list.append(load_and_mask(frame, "gaussian_curvature", args.input_data_folder, mask, num_digits))
        principal_list.append(load_and_mask(frame, "principal_curvatures", args.input_data_folder, mask, num_digits))
        dirs_list.append(load_and_mask(frame, "principal_dirs", args.input_data_folder, mask, num_digits))

    # Write output
    for i, frame in enumerate(frames):

        np.save(f"{args.out}/{frame:0{num_digits}d}_thickness.npy",
                thickness_list[i])

        np.save(f"{args.out}/{frame:0{num_digits}d}_mean_curvature.npy",
                mean_list[i])

        np.save(f"{args.out}/{frame:0{num_digits}d}_gaussian_curvature.npy",
                gauss_list[i])

        np.save(f"{args.out}/{frame:0{num_digits}d}_principal_curvatures.npy",
                principal_list[i])

        np.save(f"{args.out}/{frame:0{num_digits}d}_principal_dirs.npy",
                dirs_list[i])

if __name__ == "__main__":
    pass