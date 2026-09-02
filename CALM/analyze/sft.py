from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

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


def _build_sft_parser() -> argparse.ArgumentParser:
    """The 'CALM analyze sft' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(
        description="Build and save the per-frame Fourier coefficient stack from a trajectory.",
    )
    arg_helper.add_build_arguments(parser)
    add_manual(parser, "analyze_sft")
    return parser


def sft(args: list[str]) -> None:
    """CLI entry: build and save the SFT (A_mn/q_mn/dimensions) from a
    trajectory. This is the starting point for 'CALM analyze full' and,
    eventually, 'CALM calibrate'."""
    parser = _build_sft_parser()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    # Validation runs on the fully-resolved args (replay file's tokens, if
    # any, merged with the direct CLI) - required= on -f/-s means a
    # replay-only invocation (no -f/-s given directly) must not be
    # validated before the replay file's own values are merged in.
    ns = arg_helper.apply_replay(parser, pre_ns, remaining)
    arg_helper.validate_rotation_args(parser, ns)
    arg_helper.validate_index_arguments(parser, ns, required=True)

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
            if filename.endswith('.npy'):
                file_path = os.path.join(ns.out, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
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
