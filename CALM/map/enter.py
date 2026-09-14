from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

from ..core.manual import add_manual
from .diffusion_plot import diffusion_plot
from .dynamic_plot import dynamic_plot
from .lipids_plot import lipids_plot
from .plot import plot
from .radial_plot import radial_plot

MODULES = ["plot", "dynamic_plot", "radial_plot", "lipids_plot", "diffusion_plot"]


def run_module(module_name: str, args: list[str]) -> None:
    """Dispatch to the named map command's entry point."""
    module_name = module_name.lower()
    if module_name == "plot":
        plot(args)
    elif module_name == "dynamic_plot":
        dynamic_plot(args)
    elif module_name == "radial_plot":
        radial_plot(args)
    elif module_name == "lipids_plot":
        lipids_plot(args)
    elif module_name == "diffusion_plot":
        diffusion_plot(args)
    else:
        print(f"Unknown module: {module_name}")


def Map(args: list[str]) -> None:
    """CLI entry point: `CALM map <command> <args...>`."""
    sys.argv = [""] + args

    parser = argparse.ArgumentParser(
        description="CALM map: plots, videos, and rotation tracking from analysis output.",
        prog="CALM",
    )
    parser.add_argument("module", choices=MODULES, help="command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the chosen command")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('CALM')}")
    add_manual(parser, "map")

    ns = parser.parse_args()
    run_module(ns.module, ns.args)


if __name__ == "__main__":
    pass
