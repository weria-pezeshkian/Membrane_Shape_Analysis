from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import MDAnalysis as mda

from ..core import argument_parser as arg_helper
from ..core.fourier_sft import SFT
from ..core.manual import add_manual

logger = logging.getLogger(__name__)


def build_sft(args: argparse.Namespace, universe: mda.Universe) -> SFT:
    """Build the SFT (A_mn/q_mn/dimensions) from a trajectory and save it into args.out.

    Shared by 'CALM analyze sft' and 'CALM analyze full' (the latter calls
    this when it isn't handed a precomputed SFT via --sft).
    """
    built = SFT()
    built.build(args, universe)
    built.write(args.out)
    return built


def sft(args: List[str]) -> None:
    """CLI entry: build and save the SFT (A_mn/q_mn/dimensions) from a
    trajectory. This is the starting point for 'CALM analyze full' and,
    eventually, 'CALM calibrate'."""
    parser = argparse.ArgumentParser(
        description="Build and save the per-frame Fourier coefficient stack from a trajectory.",
    )
    arg_helper.add_build_arguments(parser)
    add_manual(parser, "analyze_sft")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    ns = parser.parse_args(args)
    arg_helper.validate_rotation_args(parser, ns)

    logging.basicConfig(level=logging.INFO)

    ns = arg_helper.apply_replay(parser, pre_ns, remaining)

    if ns.trajectory is None or ns.structure is None:
        parser.error("Both --trajectory/-f and --structure/-s are required.")

    os.makedirs(ns.out, exist_ok=True)

    replay_path = ns.out_replay or arg_helper.default_replay_name(ns.out)
    arg_helper.write_replay_file(replay_path, parser, ns)
    arg_helper.attach_replay_log_handler(replay_path)

    if ns.clear:
        for filename in os.listdir(ns.out):
            if filename.endswith('.npy'):
                file_path = os.path.join(ns.out, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    try:
        start = time.perf_counter()
        universe = mda.Universe(Path(ns.structure), Path(ns.trajectory))
        build_sft(ns, universe)
        print(f"Execution with {ns.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
