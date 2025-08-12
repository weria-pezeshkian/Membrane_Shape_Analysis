import argparse
from importlib.metadata import version
from CALM.analyze.enter import Analyze
from CALM.calibrate.enter import Calibrate
from CALM.plot.enter import Map
from CALM.utilize.enter import Link



from typing import List, Union, Optional
import os


def run_module(module_name, args):
    """
    run the specified python module with given arguments.
    """
    module_name=module_name.lower()
    if module_name == 'calibrate':
        Calibrate(args)
    elif module_name == 'analyze':
        Analyze(args)
    elif module_name == 'link':
        Link(args)
    elif module_name == 'map':
        Map(args)
    else:
        print(f"Unknown Python module: {module_name}")


def main():
    """
    main entry point for the TS2CG command-line interface.
    """

    # define the python based modules
    python_modules = ['calibrate','analyze','link','map']

    # parse arguments before a calling module
    parser = argparse.ArgumentParser(
        description='TS2CG: converts triangulated surfaces (ts) to coarse-grained membrane models',
        prog='TS2CG',
    )

    parser.add_argument(
        'module',
        choices=python_modules,
        help='choice of which module to run'
    )

    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='arguments for the chosen module'
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f'%(prog)s {version("ts2cg")}'
    )

    args = parser.parse_args()

    # call the right subroutine based on the module type
    if args.module in python_modules:
        run_module(args.module, args.args)

if __name__ == '__main__':
    main()
