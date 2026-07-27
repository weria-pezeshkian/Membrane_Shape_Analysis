from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _rel_indices_within_protein(
    protein_ag: mda.core.groups.AtomGroup, target_ag: mda.core.groups.AtomGroup
) -> np.ndarray:
    """Indices of `target_ag` atoms relative to `protein_ag`'s ordering; atoms not in `protein_ag` are ignored."""
    prot_idx = protein_ag.atoms.indices
    tgt_idx = target_ag.atoms.indices
    rel = np.nonzero(np.in1d(prot_idx, tgt_idx))[0].astype(np.int32)
    return np.sort(rel)


def get_rotation_and_protein(
    out_dir: str,
    u: mda.Universe,
    From: int,
    Until: int,
    Step: int,
    sele1: str,
    sele2: str,
) -> None:
    """Save per-frame O/P rotation-reference vectors and protein atom positions to `out_dir`.

    Writes: rotation_vectors_o.npy (F, 2), rotation_vectors_p.npy (F, 2),
    protein_atom_positions_rotation.npy (F, N, 3), highlight_p1.npy (K1,),
    highlight_p2.npy (K2,), protein_atom_indices.npy (N,), selections.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    sel_O = u.select_atoms(sele1)
    sel_P = u.select_atoms(sele2)
    sel_prot = u.select_atoms("protein")

    if sel_O.n_atoms == 0:
        raise ValueError(f"Selection1 is empty: {sele1}")
    if sel_P.n_atoms == 0:
        raise ValueError(f"Selection2 is empty: {sele2}")
    if sel_prot.n_atoms == 0:
        raise ValueError('Selection "protein" is empty.')

    p1_rel = _rel_indices_within_protein(sel_prot, sel_O)
    p2_rel = _rel_indices_within_protein(sel_prot, sel_P)
    if p1_rel.size == 0:
        logger.warning("highlight_p1: selection1 has no atoms inside 'protein' - file will be empty.")
    if p2_rel.size == 0:
        logger.warning("highlight_p2: selection2 has no atoms inside 'protein' - file will be empty.")

    o_list, p_list, prot_frames = [], [], []

    traj_slice = u.trajectory[From:Until:Step]
    n_frames = len(traj_slice)
    if n_frames == 0:
        raise ValueError(f"No frames in slice From={From}, Until={Until}, Step={Step}")

    for _ in tqdm(traj_slice, desc="Sampling trajectory"):
        o_list.append(sel_O.center_of_mass()[:2])
        p_list.append(sel_P.center_of_mass()[:2])
        prot_frames.append(sel_prot.positions.copy())

    o_array = np.asarray(o_list, dtype=np.float64)
    p_array = np.asarray(p_list, dtype=np.float64)
    prot_array = np.asarray(prot_frames, dtype=np.float32)

    np.save(os.path.join(out_dir, "rotation_vectors_o.npy"), o_array)
    np.save(os.path.join(out_dir, "rotation_vectors_p.npy"), p_array)
    np.save(os.path.join(out_dir, "protein_atom_positions_rotation.npy"), prot_array)
    np.save(os.path.join(out_dir, "highlight_p1.npy"), p1_rel)
    np.save(os.path.join(out_dir, "highlight_p2.npy"), p2_rel)
    np.save(os.path.join(out_dir, "protein_atom_indices.npy"), sel_prot.atoms.indices.astype(np.int32))

    meta = {
        "selection1": sele1,
        "selection2": sele2,
        "traj_slice": {"From": From, "Until": Until, "Step": Step},
        "shapes": {
            "rotation_vectors_o": list(o_array.shape),
            "rotation_vectors_p": list(p_array.shape),
            "protein_atom_positions_rotation": list(prot_array.shape),
            "highlight_p1": list(p1_rel.shape),
            "highlight_p2": list(p2_rel.shape),
        },
        "notes": "All coordinates are in MDAnalysis native units for the input files."
    }
    with open(os.path.join(out_dir, "selections.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Saved:\n"
        f"  rotation_vectors_o.npy                {o_array.shape}\n"
        f"  rotation_vectors_p.npy                {p_array.shape}\n"
        f"  protein_atom_positions_rotation.npy   {prot_array.shape}\n"
        f"  highlight_p1.npy                      {p1_rel.shape}\n"
        f"  highlight_p2.npy                      {p2_rel.shape}\n"
        f"  protein_atom_indices.npy              ({sel_prot.n_atoms},)\n"
        f"  selections.json"
    )


def calc_vectors(args: List[str]) -> None:
    """CLI entry: compute O/P vectors, per-frame protein positions, and highlight indices."""
    parser = argparse.ArgumentParser(
        description="Compute per-frame rotation vectors, protein atom positions, and highlight indices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-f', '--trajectory', type=str, required=True,
                        help="Path to trajectory file (e.g., .xtc, .dcd)")
    parser.add_argument('-s', '--structure', type=str, required=True,
                        help="Path to structure/topology file (e.g., .pdb, .psf, .gro)")
    parser.add_argument('-F', '--From', default=0, type=int,
                        help="First frame index (inclusive)")
    parser.add_argument('-U', '--Until', default=None, type=int,
                        help="Stop before this frame index (exclusive); None = end")
    parser.add_argument('-S', '--Step', default=1, type=int,
                        help="Stride between frames")
    parser.add_argument('-o', '--out', default="", type=str,
                        help="Output directory for files (default: current dir)")
    parser.add_argument('-p1', '--selection1', type=str, required=True,
                        help="Atom selection for reference point 1 (O)")
    parser.add_argument('-p2', '--selection2', type=str, required=True,
                        help="Atom selection for reference point 2 (P)")
    ns = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO)

    try:
        u = mda.Universe(ns.structure, ns.trajectory)
        get_rotation_and_protein(
            out_dir=ns.out or "./",
            u=u, From=ns.From, Until=ns.Until, Step=ns.Step,
            sele1=ns.selection1, sele2=ns.selection2,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
