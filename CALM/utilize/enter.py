from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

from CALM.utilize.write_ndx import write_ndx

from ..core.manual import add_manual
from .vmd_vectors import vmd_vectors
from .vmd_xtc import vmd_xtc

MODULES = ["write_ndx", "vmd_xtc", "vmd_vectors"]


def run_module(module_name: str, args: list[str]) -> None:
    """Dispatch to the named link command's entry point."""
    module_name = module_name.lower()
    if module_name == "write_ndx":
        write_ndx(args)
    elif module_name == "vmd_xtc":
        vmd_xtc(args)
    elif module_name == "vmd_vectors":
        vmd_vectors(args)
    else:
        print(f"Unknown module: {module_name}")


def Link(args: list[str]) -> None:
    """CLI entry point: `CALM link <command> <args...>`."""
    sys.argv = [""] + args

    parser = argparse.ArgumentParser(
        description="CALM link: leaflet index files and VMD export.",
        prog="CALM",
    )
    parser.add_argument("module", choices=MODULES, help="command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the chosen command")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('CALM')}")
    add_manual(parser, "link")

    ns = parser.parse_args()
    run_module(ns.module, ns.args)


if __name__ == "__main__":
    pass
