from __future__ import annotations

import argparse
import logging

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array

from ..core.leaflet import _label_by_z, apply_margin_filter, get_components
from ..core.manual import add_manual

logger = logging.getLogger(__name__)


def write(
    u: mda.Universe,
    selection: str,
    out_dir: str = "",
    flip: bool = False,
    write: bool = True,
    min_balance: float = 0.6,
    margin: float = 2.0,
) -> dict[str, list[int]] | tuple[dict[str, list[int]], list[int], list[int]]:
    """Detect the two leaflets in `selection` and either write a GROMACS index file or return the split.

    If `write` is True (default), writes `Upper`/`Lower` groups to
    `out_dir` and returns {name: [1-based indices]}. If False, returns
    (ndx, upper_index, lower_index) without writing, where upper_index/
    lower_index are 0-based global atom indices.
    """
    ndx: dict[str, list[int]] = {}
    atoms = u.select_atoms(selection)
    positions = atoms.atoms.positions
    box = u.dimensions
    d_matrix = distance_array(positions, positions, box=box)
    two_components, _ = get_components(d_matrix, min_balance=min_balance)
    upper_local, lower_local = _label_by_z(two_components[0], two_components[1], positions)
    # apply_margin_filter catches structural anomalies (a lipid squeezed
    # toward mid-plane near a protein, or mid flip-flop) that XY-
    # connectivity (get_components) alone can miss, without suppressing
    # genuine sharp membrane curvature - see its docstring in core/leaflet.py.
    upper_local, lower_local = apply_margin_filter(positions, box, upper_local, lower_local, margin=margin)

    n_excluded = len(atoms.atoms) - len(upper_local) - len(lower_local)
    if n_excluded > 0:
        logger.warning(
            f"{n_excluded} atom(s) in the selection are not part of either "
            "leaflet's best-scoring component (e.g. a lipid mid flip-flop, "
            "sitting between the two leaflets, or squeezed toward mid-plane "
            "near a protein) and are excluded from both."
        )
    upper_index = [atoms.atoms[i].index for i in sorted(upper_local)]
    lower_index = [atoms.atoms[i].index for i in sorted(lower_local)]
    upper_z = np.mean(positions[sorted(upper_local), 2]) if upper_local else float("nan")
    lower_z = np.mean(positions[sorted(lower_local), 2]) if lower_local else float("nan")
    if flip:
        firstname = "Upper"
        secondname = "Lower"
    else:
        firstname = "Lower"
        secondname = "Upper"
    if write:
        with open(out_dir, "w", encoding="UTF8") as f:
            f.write(f"[ {firstname} ]; Avg. Z: {lower_z}")
            ndx[firstname] = []
            for i, index in enumerate(lower_index):
                if i % 16 == 0:
                    f.write("\n")
                f.write(f"{index+1} ")
                ndx[firstname].append(index + 1)
            f.write("\n")
            f.write(f"[ {secondname} ]; Avg. Z: {upper_z}")
            ndx[secondname] = []
            for i, index in enumerate(upper_index):
                if i % 16 == 0:
                    f.write("\n")
                f.write(f"{index+1} ")
                ndx[secondname].append(index + 1)
            f.write("\n")
            return ndx
    else:
        return ndx, upper_index, lower_index


def write_ndx(args: list[str]) -> None:
    """CLI entry: write a leaflet index file from a trajectory and selection."""
    parser = argparse.ArgumentParser(description="Write a leaflet index file")
    parser.add_argument('-f', '--trajectory', type=str, help="trajectory file")
    parser.add_argument('-s', '--structure', type=str, help="structure file")
    parser.add_argument('-n', '--selection', type=str, help="MDAnalysis selection to split into leaflets")
    parser.add_argument(
        '-o', '--out', default="monolayers.ndx", type=str,
        help="output index file (default: monolayers.ndx)",
    )
    parser.add_argument('-F', '--flip', default=False, action='store_true', help="swap Upper/Lower labels")
    parser.add_argument(
        '--min-balance', dest='min_balance', default=0.6, type=float,
        help="leaflet-split balance threshold (default: 0.6, see --man)",
    )
    parser.add_argument(
        '--margin', dest='margin', default=2.0, type=float,
        help="leaflet margin-filter ratio (default: 2.0, see --man)",
    )
    add_manual(parser, "link_write_ndx")

    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        universe = mda.Universe(ns.structure, ns.trajectory)
        write(
            u=universe, selection=ns.selection, out_dir=ns.out,
            flip=ns.flip, min_balance=ns.min_balance, margin=ns.margin,
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
