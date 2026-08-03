from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import MDAnalysis as mda

from ..core import argument_parser as arg_helper
from ..core.fourier_build import calc_lipids
from ..core.manual import add_manual

logger = logging.getLogger(__name__)


def lipids(args: list[str]) -> None:
    """CLI entry: per-species lipid composition and area-per-lipid from a trajectory.

    Always re-fits the leaflet surfaces from a live trajectory (no --sft
    reuse) - lipid identity (resname) is only available from real atoms,
    which a previously-built SFT (Amn/qmn coefficients alone) doesn't
    carry. --rotate/--rotation-direction are accepted for consistency with
    'CALM analyze sft'/'full' but have no effect here: lipid composition
    and area-per-lipid are computed independently per frame in the fit's
    own raw coordinate frame, which doesn't need cross-frame rotational
    alignment. --center still applies (needed by --Remove-TMD).
    """
    parser = argparse.ArgumentParser(
        description="Per-species lipid composition and area-per-lipid from a trajectory.",
    )
    arg_helper.add_build_arguments(parser)
    parser.add_argument(
        "--lipids", nargs="+", required=True,
        help="resnames to treat as distinct lipid species, e.g. --lipids POPC GM1 SAPE24",
    )
    add_manual(parser, "analyze_lipids")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    ns = parser.parse_args(args)
    arg_helper.validate_rotation_args(parser, ns)

    ns = arg_helper.apply_replay(parser, pre_ns, remaining)

    if ns.trajectory is None or ns.structure is None:
        parser.error("Both --trajectory/-f and --structure/-s are required.")

    os.makedirs(ns.out, exist_ok=True)

    replay_path = ns.out_replay or arg_helper.default_replay_name(ns.out)
    arg_helper.write_replay_file(replay_path, parser, ns)
    arg_helper.attach_replay_log_handler(replay_path, logger_name="MDAnalysis")
    arg_helper.attach_replay_log_handler(
        replay_path, logger_name="CALM",
        console_level=logging.INFO if ns.loud else logging.WARNING,
    )

    if ns.clear:
        for filename in os.listdir(ns.out):
            if filename.endswith(".npy"):
                file_path = os.path.join(ns.out, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

    try:
        start = time.perf_counter()
        universe = mda.Universe(Path(ns.structure), Path(ns.trajectory))
        calc_lipids(ns, universe)
        print(f"Execution with {ns.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
