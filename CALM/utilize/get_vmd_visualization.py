import argparse
import os
import MDAnalysis as mda
import numpy as np
import glob
from typing import List, Optional, Sequence, Dict

def get_vmd_visualisation(curvature_dir: str, out_dir: str):
    """Build pseudo-universe from Z_fitted_*.npy files,
    write GRO (first frame), XTC trajectory, and average GRO."""

    # --- Load box size ---
    dim_file = os.path.join(curvature_dir, "dimensions.csv")
    box_size = np.loadtxt(dim_file, delimiter=",", skiprows=1,
                          max_rows=1, usecols=(1, 2, 3))

    # --- Find all Z_fitted files ---
    z_files = sorted(glob.glob(os.path.join(curvature_dir, "*_Z_fitted.npy")))
    if not z_files:
        raise FileNotFoundError(
            f"No *_Z_fitted*.npy files found in {curvature_dir}"
        )

    print(f"Found {len(z_files)} frames.")

    # --- Initialize from first frame ---
    z_values = np.load(z_files[0]) * 10
    n_layers, Nx, Ny = z_values.shape

    x = np.linspace(0, box_size[0], Nx, endpoint=False)
    y = np.linspace(0, box_size[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    def build_coords(z_array):
        return np.vstack([
            np.column_stack([X.flatten(),
                             Y.flatten(),
                             z_array[l].flatten()])
            for l in range(n_layers)
        ])

    coords = build_coords(z_values)

    resindices = np.repeat(np.arange(n_layers), Nx * Ny)
    u = mda.Universe.empty(
        n_atoms=coords.shape[0],
        n_residues=n_layers,
        atom_resindex=resindices,
        trajectory=True
    )

    for attr in ["name", "resname", "resid"]:
        u.add_TopologyAttr(attr)

    u.atoms.names = ["C"] * coords.shape[0]
    u.residues.resnames = ["up", "low", "mid"][:n_layers]
    u.residues.resids = list(range(1, n_layers + 1))
    u.dimensions = [*box_size, 90.0, 90.0, 90.0]

    # Output paths
    gro_path = os.path.join(out_dir, "first_frame.gro")
    xtc_path = os.path.join(out_dir, "trajectory.xtc")
    avg_gro_path = os.path.join(out_dir, "average_structure.gro")

    # Write first frame GRO
    u.atoms.positions = coords
    u.atoms.write(gro_path)


    # --- Trajectory + averaging ---
    avg_z = np.zeros_like(z_values)

    with mda.coordinates.XTC.XTCWriter(
            xtc_path, n_atoms=u.atoms.n_atoms) as writer:

        for z_file in z_files:
            z_values = np.load(z_file) *10
            avg_z += z_values

            u.atoms.positions = build_coords(z_values)
            writer.write(u.atoms)


    # --- Average structure ---
    avg_z /= len(z_files)
    u.atoms.positions = build_coords(avg_z)
    u.atoms.write(avg_gro_path)


def write_xtc(args: List[str]) -> None:

    parser = argparse.ArgumentParser(description="Create GRO + XTC of fitting from CALM analyze output files",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input",help="Folder containing *_Z_fitted.npy and dimensions.csv")
    parser.add_argument("-o", "--output",help="Folder to store generated GRO and XTC files")
    args = parser.parse_args(args)

    os.makedirs(args.output, exist_ok=True)
    get_vmd_visualisation(args.input, args.output)


if __name__ == "__main__":
    pass





