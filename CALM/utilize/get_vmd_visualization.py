import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
import networkx as nx
import glob
import os

logger = logging.getLogger(__name__)


def build(dir=directory):
    n_atoms=1000
    box=np.load(f"{dir}/dimensions.npy")
    X, Y = get_XY(box[:3])

    for Layer in ["Upper","Lower","Middle"]:
        files = glob.glob(f"{dir}/*_Z_fitted_{Layer}.npy")

        files_sorted = sorted(files,key=lambda f: int(os.path.basename(f).split("_", 1)[0]))

        with mda.coordinates.XTC.XTCWriter(f"{dir}/fourier_curvature_fitting_{Layer}.xtc", n_atoms=n_atoms) as writer:
            First=True
            for t,file in enumerate(files_sorted):
                Z_fitted=np.load(file)
                coordinates = np.vstack([X.flatten(), Y.flatten(), Z_fitted.flatten()]).T
                pseudo_universe = mda.Universe.empty(n_atoms=coordinates.shape[0], trajectory=True)
                pseudo_universe.atoms.positions = coordinates
                pseudo_universe.dimensions = box
            if First:
                pseudo_universe.atoms.write(f"{dir}/pseudo_universe_{Layer}.gro")
                First=False

            writer.write(pseudo_universe.atoms)
    



def build_visualization(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Write visualization files for the surfaces, gro and xtc from a previously run CALM analyze",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d','--directory',type=str,help="Curvature Directory written by CALM analyze")
    
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        build(dir=args.directory)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
