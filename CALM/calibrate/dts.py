from __future__ import annotations

import argparse
import logging

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from .calibrate import calibrate

logger = logging.getLogger(__name__)


def _build_dts_parser() -> argparse.ArgumentParser:
    """The 'CALM calibrate dts' parser alone, with no side effects - shared by the CLI entry point below
    and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(
        description="Calibrate membrane material parameters from a built SFT",
        prog="CALM calibrate dts",
    )
    parser.add_argument(
        '-i', '--sft', type=str, required=True,
        help="'CALM analyze sft'/'CALM analyze full' output directory",
    )
    parser.add_argument(
        '--radius', type=float, required=True,
        help="calibration radius (meaning defined by the physics implementation)",
    )
    ##Add the other new arguments:  frame_start: int, frame_end: int,
    parser.add_argument(
        '--startframe', type=int, required=True,
        help="frame to start at, together w. endframe it determines number of user chosen frames for integral calculations ",
    )
    parser.add_argument(
        '--endframe', type=int, required=True,
        help="frame to end at, together w. startframe it determines number of user chosen frames for integral calculations ",
    )

    parser.add_argument('-o', '--out', type=str, required=True, help="output file path")
    add_manual(parser, "calibrate_dts")

    parser.add_argument(
    "--surface", type=str,choices=["top", "middle", "bottom"],default="top",
    help="Surface to use for calibration: top, middle, or bottom (default: top)."
    )
    return parser


def dts(args: list[str]) -> None:
    """CLI entry: calibrate membrane material parameters (kappa, delta_kappa, delta_kappa_g, C_0) from a built SFT."""
    parser = _build_dts_parser()

    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        sft_obj = SFT.from_directory(ns.sft)
        calibrate(sft_obj, ns.radius,ns.startframe,ns.endframe, ns.out,ns.surface)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
