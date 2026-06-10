import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional, Sequence, Dict
import networkx as nx
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff


logger = logging.getLogger(__name__)


def get_components(matrix, threshold):
    init_threshold=np.percentile(matrix,threshold)
    adj_matrix=np.where(matrix>init_threshold,0,matrix)
    G=nx.from_numpy_array(adj_matrix)
    components=list(nx.connected_components(G))
    if len(components) > 2:
        return get_components(matrix,threshold*1.5)
    elif len(components) < 2:
        return get_components(matrix,threshold/2)
    else:
        return components

#def write(u,selection,out_dir="",flip=False,write=True):
#    ndx={}
#    #u.trajectory[t]
#    selection=u.select_atoms(selection)
#    d_matrix=distance_array(selection.atoms.positions, selection.atoms.positions, box=u.dimensions)
#    init_threshold=0.01
#    two_components=get_components(d_matrix,init_threshold)
#    first = [selection.atoms[x].index for x in two_components[0]]
#    #np.save(arr=np.asarray([selection.atoms[x].position for x in two_components[0]]),file="component1.npy")
#    #np.save(arr=np.asarray([selection.atoms[x].position for x in two_components[1]]),file="component2.npy")
#    second = [selection.atoms[x].index for x in two_components[1]]
#    first_z = np.mean(np.asarray([selection.atoms[x].position[2] for x in two_components[0]]))
#    second_z=np.mean(np.asarray([selection.atoms[x].position[2] for x in two_components[1]]))
#    if second_z<first_z:
#        upper_index=first
#        upper_z=first_z
#        lower_index=second
#        lower_z=second_z
#    else:
#        upper_index=second
#        upper_z=second_z
#        lower_index=first
#        lower_z=first_z
#    if flip:
#        firstname="Upper"
#        secondname="Lower"
#    else:
#        firstname="Lower"
#        secondname="Upper"
#    if write:
#        with open(out_dir,"w",encoding="UTF8") as f:
#            f.write(f"[ {firstname} ]; Avg. Z: {lower_z}")
#            ndx[firstname]=[]
#            for i,index in enumerate(lower_index):
#                if i%16==0:
#                    f.write("\n")
#                f.write(f"{index+1} ")
#                ndx[firstname].append(index+1)
#            f.write("\n")
#            f.write(f"[ {secondname} ]; Avg. Z: {upper_z}")
#            ndx[secondname]=[]
#            for i,index in enumerate(upper_index):
#                if i%16==0:
#                    f.write("\n")
#                f.write(f"{index+1} ")
#                ndx[secondname].append(index+1)
#            f.write("\n")
#            return ndx
#    else:
#        return ndx, upper_index, lower_index
    

def write(u, selection, out_dir="", flip=False, write=True):

    ndx = {}

    sel = u.select_atoms(selection)

    if len(sel) == 0:
        raise ValueError("Selection returned no atoms. Check your selection string.")

    # --- Leaflet detection ---
    lf = LeafletFinder(u, sel, cutoff=15.0)
    leaflets = lf.groups()

    # Compute mean Z for each detected cluster
    clusters = []
    for g in leaflets:
        z_mean = np.mean(g.positions[:, 2])
        clusters.append((z_mean, g))

    # Sort clusters by Z
    clusters.sort(key=lambda x: x[0])

    # Find largest Z-gap between adjacent clusters
    z_values = [c[0] for c in clusters]
    gaps = np.diff(z_values)

    if len(gaps) == 0:
        raise RuntimeError("Leaflet detection failed: only one cluster found.")

    split_idx = np.argmax(gaps)

    # Lower leaflet = clusters up to split
    lower_clusters = [c[1] for c in clusters[:split_idx+1]]
    upper_clusters = [c[1] for c in clusters[split_idx+1:]]

    # Merge clusters into two AtomGroups
    lower_group = lower_clusters[0]
    for c in lower_clusters[1:]:
        lower_group = lower_group + c

    upper_group = upper_clusters[0]
    for c in upper_clusters[1:]:
        upper_group = upper_group + c

    print(f"Lower leaflet size: {len(lower_group)} atoms")
    print(f"Upper leaflet size: {len(upper_group)} atoms")


    # Compute Z means for merged groups
    lower_z = np.mean(lower_group.positions[:, 2])
    upper_z = np.mean(upper_group.positions[:, 2])

    # Flip naming if requested
    if flip:
        firstname = "Upper"
        secondname = "Lower"
    else:
        firstname = "Lower"
        secondname = "Upper"

    lower_index = lower_group.indices
    upper_index = upper_group.indices

    # --- Write NDX file ---
    if write:
        with open(out_dir, "w", encoding="UTF8") as f:

            # LOWER
            f.write(f"[ {firstname} ]; Avg. Z: {lower_z}\n")
            ndx[firstname] = []

            for i, idx in enumerate(lower_index):
                if i % 16 == 0:
                    f.write("\n")
                f.write(f"{idx + 1} ")
                ndx[firstname].append(idx + 1)

            f.write("\n")

            # UPPER
            f.write(f"[ {secondname} ]; Avg. Z: {upper_z}\n")
            ndx[secondname] = []

            for i, idx in enumerate(upper_index):
                if i % 16 == 0:
                    f.write("\n")
                f.write(f"{idx + 1} ")
                ndx[secondname].append(idx + 1)

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
    
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe=mda.Universe(args.structure,args.trajectory)
        write(u=universe,selection=args.selection,out_dir=args.out,flip=args.flip)


    except Exception as e:
        logger.error(f"Error: {e}")
        raise



