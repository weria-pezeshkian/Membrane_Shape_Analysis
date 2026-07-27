from __future__ import annotations

import argparse
from importlib.metadata import version

from CALM.analyze.enter import Analyze
from CALM.calibrate.enter import Calibrate
from CALM.map.enter import Map
from CALM.utilize.enter import Link
from CALM.core.manual import add_manual

PYTHON_MODULES = ["calibrate", "analyze", "link", "map"]


def run_module(module_name: str, args: list[str]) -> None:
    """Dispatch to the named top-level module's entry point."""
    module_name = module_name.lower()
    if module_name == "calibrate":
        Calibrate(args)
    elif module_name == "analyze":
        Analyze(args)
    elif module_name == "link":
        Link(args)
    elif module_name == "map":
        Map(args)
    else:
        print(f"Unknown Python module: {module_name}")


def main() -> None:
    """CLI entry point: `CALM <module> <args...>`."""
    parser = argparse.ArgumentParser(
        description="CALM: Calibrate and Analyze Lipid Membranes.",
        prog="CALM",
    )

    parser.add_argument("module", choices=PYTHON_MODULES, help="module to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the chosen module")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version('CALM')}")
    add_manual(parser, "calm")

    ns = parser.parse_args()
    run_module(ns.module, ns.args)


if __name__ == "__main__":
    main()
