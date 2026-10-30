from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

from ..core.manual import add_manual
from .dts import dts
from .lambda_sweep import lambda_sweep

MODULES = ["dts", "lambda"]


def run_module(module_name: str, args: list[str]) -> None:
    """Dispatch to the named calibrate command's entry point."""
    module_name = module_name.lower()
    if module_name == "dts":
        dts(args)
    elif module_name == "lambda":
        lambda_sweep(args)
    else:
        print(f"Unknown module: {module_name}")


def Calibrate(args: list[str]) -> None:
    """CLI entry point: `CALM calibrate <command> <args...>`."""
    sys.argv = [""] + args

    parser = argparse.ArgumentParser(
        description="CALM calibrate: membrane material parameter calibration and the "
                     "Fourier-fit lambda_x/lambda_y advisor.",
        prog="CALM",
    )
    parser.add_argument("module", choices=MODULES, help="command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the chosen command")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('CALM')}")
    add_manual(parser, "calibrate")

    ns = parser.parse_args()
    run_module(ns.module, ns.args)


if __name__ == "__main__":
    pass
