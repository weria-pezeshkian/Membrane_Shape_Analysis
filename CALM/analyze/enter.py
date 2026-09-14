from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

from ..core.manual import add_manual
from .diffusion import diffusion
from .full import full
from .lipids import lipids
from .sft import sft

MODULES = ["sft", "full", "lipids", "diffusion"]


def run_module(module_name: str, args: list[str]) -> None:
    """Dispatch to the named analyze command's entry point."""
    module_name = module_name.lower()
    if module_name == "sft":
        sft(args)
    elif module_name == "full":
        full(args)
    elif module_name == "lipids":
        lipids(args)
    elif module_name == "diffusion":
        diffusion(args)
    else:
        print(f"Unknown module: {module_name}")


def Analyze(args: list[str]) -> None:
    """CLI entry point: `CALM analyze <command> <args...>`."""
    sys.argv = [""] + args

    parser = argparse.ArgumentParser(
        description="CALM analyze: fit the Fourier surface and/or run the geometric analysis pipeline.",
        prog="CALM",
    )
    parser.add_argument("module", choices=MODULES, help="command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the chosen command")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('CALM')}")
    add_manual(parser, "analyze")

    ns = parser.parse_args()
    run_module(ns.module, ns.args)


if __name__ == "__main__":
    pass
