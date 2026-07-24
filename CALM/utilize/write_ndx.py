import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
from ..core.leaflet import get_components, apply_margin_filter, _label_by_z

logger = logging.getLogger(__name__)


def write(u,selection,out_dir="",flip=False,write=True,min_balance=0.6,margin=2.0):
    ndx={}
    #u.trajectory[t]
    selection=u.select_atoms(selection)
    positions = selection.atoms.positions
    box = u.dimensions
    d_matrix=distance_array(positions, positions, box=box)
    two_components,_=get_components(d_matrix,min_balance=min_balance)
    upper_local, lower_local = _label_by_z(two_components[0], two_components[1], positions)
    # apply_margin_filter: catches structural anomalies (a lipid squeezed
    # toward mid-plane near a protein, or mid flip-flop) that XY-
    # connectivity (get_components) alone can miss, without suppressing
    # genuine sharp membrane curvature - see its docstring in core/leaflet.py.
    upper_local, lower_local = apply_margin_filter(positions, box, upper_local, lower_local, margin=margin)

    n_excluded = len(selection.atoms) - len(upper_local) - len(lower_local)
    if n_excluded > 0:
        logger.warning(
            f"{n_excluded} atom(s) in the selection are not part of either "
            "leaflet's best-scoring component (e.g. a lipid mid flip-flop, "
            "sitting between the two leaflets, or squeezed toward mid-plane "
            "near a protein) and are excluded from both."
        )
    upper_index = [selection.atoms[i].index for i in sorted(upper_local)]
    lower_index = [selection.atoms[i].index for i in sorted(lower_local)]
    upper_z = np.mean(positions[sorted(upper_local), 2]) if upper_local else float("nan")
    lower_z = np.mean(positions[sorted(lower_local), 2]) if lower_local else float("nan")
    if flip:
        firstname="Upper"
        secondname="Lower"
    else:
        firstname="Lower"
        secondname="Upper"
    if write:
        with open(out_dir,"w",encoding="UTF8") as f:
            f.write(f"[ {firstname} ]; Avg. Z: {lower_z}")
            ndx[firstname]=[]
            for i,index in enumerate(lower_index):
                if i%16==0:
                    f.write("\n")
                f.write(f"{index+1} ")
                ndx[firstname].append(index+1)
            f.write("\n")
            f.write(f"[ {secondname} ]; Avg. Z: {upper_z}")
            ndx[secondname]=[]
            for i,index in enumerate(upper_index):
                if i%16==0:
                    f.write("\n")
                f.write(f"{index+1} ")
                ndx[secondname].append(index+1)
            f.write("\n")
            return ndx
    else:
        return ndx, upper_index, lower_index



def write_ndx(args: List[str]) -> None:
    """Main entry point for Domain Placer tool"""
    parser = argparse.ArgumentParser(description="Write an index file to be used for other CALM tasks",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f','--trajectory',type=str,help="Specify the path to the trajectory file")
    parser.add_argument('-s','--structure',type=str,help="Specify the path to the structure file")
    parser.add_argument('-n','--selection',type=str,help="Specify the selection of particles to be considered")
    parser.add_argument('-o','--out',default="monolayers.ndx",type=str,help="Specify a path to the to be written index file")
    parser.add_argument('-F','--flip',default=False,action='store_true',help="flip Upper and Lower index")
    parser.add_argument('--min-balance',dest='min_balance',default=0.6,type=float,help="Minimum acceptable leaflet-size balance (1.0=perfectly equal, 0.0=all-in-one-leaflet) for a candidate 2-leaflet split to be considered valid; among valid splits the one covering the most atoms wins. Default 0.6 (rejects splits more lopsided than ~4:1).")
    parser.add_argument('--margin',dest='margin',default=2.0,type=float,help="An atom is only kept in a leaflet if its distance to the nearest atom in the OTHER leaflet is at least this many times its distance to the nearest atom in its OWN leaflet - catches structural anomalies (e.g. a lipid squeezed toward mid-plane near a protein, or mid flip-flop) that XY-connectivity alone can miss. Default 2.0. See core/leaflet.py's apply_margin_filter() docstring for details.")

    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        write(u=universe,selection=args.selection,out_dir=args.out,flip=args.flip,min_balance=args.min_balance,margin=args.margin)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
