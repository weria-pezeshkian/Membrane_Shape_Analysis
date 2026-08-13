from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import MDAnalysis as mda

from ..analyze.analyze import analysis
from ..analyze.sft import build_sft
from ..core import argument_parser as arg_helper
from ..core.fourier_sft import SFT
from ..core.manual import add_manual

logger = logging.getLogger(__name__)

METHODS = (
    "thickness",
    "Z_fitted",
    "mean",
    "gaussian",
    "principal",
    "principal_directions",
)


def full(args: list[str]) -> None:
    """CLI entry: run the full geometric analysis pipeline (thickness,
    curvature, ...) that 'CALM map' can plot.

    Either builds the SFT from a trajectory itself (same as 'CALM analyze
    sft'), or - if --sft is given - loads a previously built one and skips
    straight to analysis, in which case --trajectory/--structure aren't
    needed.
    """
    parser = argparse.ArgumentParser(
        description="Run the full geometric analysis pipeline (thickness, curvature).",
    )
    required = parser.add_argument_group(
        "Required arguments",
        "Give either --sft, or all three of -f/--trajectory, -s/--structure, and -n/--index.",
    )
    required.add_argument(
        "--sft", default=None, type=str,
        help="reuse a previously built fit instead of --trajectory/--structure/--index (see --man)",
    )
    arg_helper.add_build_arguments(parser, require_inputs=False, required_group=required)
    parser.add_argument(
        "--method",
        nargs="+",
        choices=METHODS,
        default=None,
        help="analysis method(s) to run (default: all)",
    )
    add_manual(parser, "analyze_full")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    # Validation runs on the fully-resolved args (replay file's tokens, if
    # any, merged with the direct CLI) - a replay-only invocation must not
    # be validated before the replay file's own values are merged in.
    ns = arg_helper.apply_replay(parser, pre_ns, remaining)
    arg_helper.validate_rotation_args(parser, ns)

    using_precomputed_sft = ns.sft is not None

    if not using_precomputed_sft and (ns.trajectory is None or ns.structure is None or ns.index is None):
        parser.error("--trajectory/-f, --structure/-s, and --index/-n are all required unless --sft is supplied.")

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

        universe = None
        if ns.trajectory is not None and ns.structure is not None:
            universe = mda.Universe(Path(ns.structure), Path(ns.trajectory))

        if using_precomputed_sft:
            sft_obj = SFT.from_directory(ns.sft)
        else:
            sft_obj = build_sft(ns, universe)

        active_methods = METHODS if ns.method is None else tuple(ns.method)
        analysis(universe, sft_obj, active_methods, ns)

        print(f"Execution with {ns.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
