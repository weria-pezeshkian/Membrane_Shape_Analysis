from __future__ import annotations

import argparse
import logging
from typing import List

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from .calibrate import calibrate

logger = logging.getLogger(__name__)


def Calibrate(args: List[str]) -> None:
    """CLI entry: calibrate membrane material parameters from a built SFT. Has no submodules."""
    parser = argparse.ArgumentParser(
        description="Calibrate membrane material parameters from a built SFT",
        prog="CALM calibrate",
    )
    parser.add_argument('-i', '--sft', type=str, required=True, help="'CALM analyze sft'/'CALM analyze full' output directory")
    parser.add_argument('--radius', type=float, required=True, help="calibration radius (meaning defined by the physics implementation)")
    parser.add_argument('-o', '--out', type=str, required=True, help="output file path")
    add_manual(parser, "calibrate")

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
