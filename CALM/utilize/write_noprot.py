import numpy as np
import MDAnalysis as mda
import argparse
import os
import shutil
from scipy.ndimage import binary_dilation


# =========================================================
# 1. FILE DETECTION
# =========================================================

def is_frame_file(fname):
    return "_" in fname and fname.split("_")[0].isdigit()


def detect_mode(data_dir):
    """
    Returns:
        "frame" -> frame-based data exists
        "avg"   -> only avg_surface_* exists
    """
    for f in os.listdir(data_dir):
        if f.startswith("avg_surface_") and f.endswith(".npy"):
            return "avg"
    return "frame"


def get_available_frames(data_dir):
    frames = []
    for fname in os.listdir(data_dir):
        if fname.endswith("_mean_curvature.npy") and is_frame_file(fname):
            try:
                frames.append(int(fname.split("_")[0]))
            except ValueError:
                pass
    return sorted(set(frames))


# =========================================================
# 2. GRID + MASKING
# =========================================================

def get_XY(box_size):
    x = np.linspace(0, box_size[0], 100, endpoint=False)
    y = np.linspace(0, box_size[1], 100, endpoint=False)
    return np.meshgrid(x, y)


def read_ndx(filename):
    groups = {}
    with open(filename) as f:
        group_name = None
        for line in f:
            line = line.split(";")[0].strip()
            if line.startswith('['):
                group_name = line[1:-1].strip()
                groups[group_name] = []
            elif group_name:
                groups[group_name].extend(map(int, line.split()))
    return groups


def remove_prot(universe,layer_group,layer_group_2,X,Y,box_size):

    mem_atoms = (layer_group + layer_group_2).unique
    mem_residues = mem_atoms.residues

    non_mem = universe.atoms.difference(mem_residues.atoms)
    protein_atoms = non_mem.select_atoms("not resname W ION SW WS")

    xy = protein_atoms.positions[:, :2]

    dx = box_size[0] / X.shape[1]
    dy = box_size[1] / X.shape[0]

    xi = np.clip((xy[:, 0] / dx).astype(int), 0, X.shape[1] - 1)
    yi = np.clip((xy[:, 1] / dy).astype(int), 0, X.shape[0] - 1)

    mask = np.zeros(X.shape, dtype=bool)
    mask[yi, xi] = True

    return binary_dilation(mask, iterations=1)


def get_mask(frame, universe, ndx, X, Y):
    ts = universe.trajectory[frame]
    box_size = ts.dimensions[:3]

    up = universe.atoms[[i - 1 for i in ndx["Upper"]]]
    low = universe.atoms[[i - 1 for i in ndx["Lower"]]]

    return remove_prot(universe, up, low, X, Y, box_size)


def load_and_mask(path, mask):
    arr = np.load(path)

    if arr.ndim == 2:
        return np.where(mask, np.nan, arr)
    if arr.ndim == 3:
        return np.where(mask[None, :, :], np.nan, arr)
    if arr.ndim == 4:
        return np.where(mask[None, :, :, None], np.nan, arr)

    raise ValueError(f"Unexpected shape: {arr.shape}")


# =========================================================
# 3. AVERAGED SURFACES
# =========================================================

def process_avg(input_dir, out_dir, mask):

    fields = ["mean_curvature","gaussian_curvature","principal_curvatures"]

    for field in fields:
        path = os.path.join(input_dir, f"avg_surface_{field}.npy")

        if not os.path.exists(path):
            continue

        arr = np.load(path)

        if arr.ndim == 2:
            out = np.where(mask, np.nan, arr)
        elif arr.ndim == 3:
            out = np.where(mask[None, :, :], np.nan, arr)
        else:
            raise ValueError(f"Unexpected shape: {arr.shape}")

        np.save(os.path.join(out_dir, f"avg_surface_{field}.npy"), out)


# =========================================================
# 4. FRAME PROCESSING
# =========================================================

FIELDS = ["mean_curvature","gaussian_curvature","principal_curvatures","principal_dirs"]


def process_frame(frame, u, ndx, data_dir):

    ts = u.trajectory[frame]
    box_size = ts.dimensions[:3]

    X, Y = get_XY(box_size)
    mask = get_mask(frame, u, ndx, X, Y)

    results = {}

    for field in FIELDS:
        path = os.path.join(data_dir, f"{frame:08d}_{field}.npy")
        if not os.path.exists(path):
            continue
        results[field] = load_and_mask(path, mask)

    return results, mask


# =========================================================
# 5. MAIN PIPELINE
# =========================================================

def write_noprot(raw_args):

    parser = argparse.ArgumentParser(description="Remove data from the area of the transmembrane protein")

    parser.add_argument("-f", "--trajectory", required=True)
    parser.add_argument("-s", "--structure", required=True)
    parser.add_argument("-n", "--index", required=True)
    parser.add_argument("-i", "--input_data_folder", required=True)
    parser.add_argument("-o", "--out", required=False, default="noprot")
    parser.add_argument("-F", "--From", default=0, type=int)
    parser.add_argument("-U", "--Until", default=None, type=int)
    parser.add_argument("-S", "--Step", default=1, type=int)

    args = parser.parse_args(raw_args)

    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.input_data_folder, "dimensions.csv"),os.path.join(args.out, "dimensions.csv"))

    u = mda.Universe(args.structure, args.trajectory)
    ndx = read_ndx(args.index)

    mode = detect_mode(args.input_data_folder)

    # =====================================================
    # CASE 1: FRAME-BASED DATA
    # =====================================================
    if mode == "frame":

        frames = get_available_frames(args.input_data_folder)

        if not frames:
            raise RuntimeError("No frame-based data found")

        if args.Until is None:
            args.Until = max(frames) + 1

        frames = [f for f in frames if args.From <= f < args.Until]
        frames = frames[::args.Step]

        if not frames:
            raise RuntimeError("No frames after filtering")

        ref_results = process_frame(frames[0], u, ndx, args.input_data_folder)[1]
        mask = ref_results  # fallback reference (safe enough)

        for frame in frames:

            results, _ = process_frame(frame, u, ndx, args.input_data_folder)

            for key, arr in results.items():
                out_name = f"{frame:08d}_{key}.npy"
                np.save(os.path.join(args.out, out_name), arr)

    # =====================================================
    # CASE 2: AVG SURFACE ONLY
    # =====================================================
    else:

        ts = u.trajectory[0]
        box_size = ts.dimensions[:3]

        X, Y = get_XY(box_size)
        mask = get_mask(0, u, ndx, X, Y)

        process_avg(args.input_data_folder, args.out, mask)


if __name__ == "__main__":
    import sys
    write_noprot(sys.argv[1:])