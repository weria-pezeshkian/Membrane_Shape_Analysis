from __future__ import annotations

import argparse
import logging

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from .calibrate import calibrate

logger = logging.getLogger(__name__)


def _build_calibrate_parser() -> argparse.ArgumentParser:
    """The 'CALM calibrate' parser alone, with no side effects - shared by the CLI entry point below
    and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(
        description="Calibrate membrane material parameters from a built SFT",
        prog="CALM calibrate",
    )
    parser.add_argument(
        '-i', '--sft', type=str, required=True,
        help="'CALM analyze sft'/'CALM analyze full' output directory",
    )
    parser.add_argument(
        '--radius', type=float, required=True,
        help="calibration radius (meaning defined by the physics implementation)",
    )
    parser.add_argument('-o', '--out', type=str, required=True, help="output file path")
    add_manual(parser, "calibrate")
    return parser


def Calibrate(args: list[str]) -> None:
    """CLI entry: calibrate membrane material parameters from a built SFT. Has no submodules."""
    parser = _build_calibrate_parser()

    ns = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)

    try:
        sft_obj = SFT.from_directory(ns.sft)
        calibrate(sft_obj, ns.radius, ns.out)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
