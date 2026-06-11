import numpy as np
import glob
import argparse
import os
import shutil
from typing import List

from ..core.fourier_core import Fourier_Series_Function


def read_box(input_dir):
    path = os.path.join(input_dir, "dimensions.csv")

    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    last = lines[-1].split(",")
    Lx, Ly, _ = map(float, last[1:4])

    return Lx, Ly

def load_stack(input_dir):
    files = sorted(glob.glob(f"{input_dir}/*_Z_fitted.npy"))
    if not files:
        raise ValueError("No *_Z_fitted.npy files found")

    return np.stack([np.load(f) for f in files])  # (T, 3, Nx, Ny)


def compute(input_dir):

    stack = load_stack(input_dir)

    Z1_avg   = stack[:, 0].mean(axis=0)
    Z2_avg   = stack[:, 1].mean(axis=0)
    Zmid_avg = stack[:, 2].mean(axis=0)

    Lx, Ly = read_box(input_dir)

    Nx, Ny = Z1_avg.shape

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)

    X, Y = np.meshgrid(x, y, indexing="ij")

    p1 = np.vstack([X.ravel(), Y.ravel(), Z1_avg.ravel()])
    p2 = np.vstack([X.ravel(), Y.ravel(), Z2_avg.ravel()])
    pm = np.vstack([X.ravel(), Y.ravel(), Zmid_avg.ravel()])

    f1 = Fourier_Series_Function(Lx, Ly, 3, 3)
    f2 = Fourier_Series_Function(Lx, Ly, 3, 3)
    fm = Fourier_Series_Function(Lx, Ly, 3, 3)

    f1.Fit(p1)
    f2.Fit(p2)
    fm.Fit(pm)

    #curvature of averaged surface
    H1, K1, k1_1, k2_1, _, _ = f1.ShapeOperatorCurvatures(X, Y)
    H2, K2, k1_2, k2_2, _, _ = f2.ShapeOperatorCurvatures(X, Y)
    Hm, Km, k1_m, k2_m, _, _ = fm.ShapeOperatorCurvatures(X, Y)

    return (H1, H2, Hm,K1, K2, Km,k1_1, k2_1,k1_2, k2_2,k1_m, k2_m)


def compute_avg_curv(input_dir):
    return compute(input_dir)


def avg_curv(args: List[str]) -> None:

    parser = argparse.ArgumentParser(description="CALM: curvature of averaged surface only")

    parser.add_argument("-i", "--input", '--numpys_directory', required=True)
    parser.add_argument("-o", "--output", '--output_directory', required=True)

    parsed = parser.parse_args(args)
    os.makedirs(parsed.output, exist_ok=True)

    (H1, H2, Hm,K1, K2, Km,k1_1, k2_1,k1_2, k2_2,k1_m, k2_m) = compute_avg_curv(parsed.input)

    np.save(f"{parsed.output}/avg_surface_mean_curvature.npy",np.stack([H1, H2, Hm]))
    np.save(f"{parsed.output}/avg_surface_gaussian_curvature.npy",np.stack([K1, K2, Km]))
    np.save(f"{parsed.output}/avg_surface_principal_curvatures.npy",np.stack([k1_1, k2_1, k1_2, k2_2, k1_m, k2_m]))


if __name__ == "__main__":
    import sys
    avg_curv(sys.argv[1:])

