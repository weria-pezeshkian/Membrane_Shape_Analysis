from .sft import sft
from .full import full
import sys
import logging
import argparse
from importlib.metadata import version


logger = logging.getLogger(__name__)

def run_module(module_name, args):
    """
    run the specified python module with given arguments.
    """
    module_name=module_name.lower()
    if module_name == 'sft':
        sft(args)
    elif module_name == 'full':
        full(args)
    else:
        print(f"Unknown module: {module_name}")

def Analyze(args):
    """
    main entry point for the CALM analyze command-line interface.
    """

    modules=['sft','full']

    sys.argv=[""]+args
    # call the right subroutine based on the module type

    # parse arguments before a calling module
    parser = argparse.ArgumentParser(
    description='CALM analyze: build the per-frame Fourier coefficient stack and/or run the full geometric analysis pipeline.',
    prog='CALM',
    formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
    'module',
    choices=modules,
    help='choice of which module to run, run CALM analyze <module> -h for detailed help\n\
    <sft> builds and saves Amn.npy/qmn.npy/dimensions.npy from a trajectory - the starting point for "calibrate"\n\
    <full> runs the full analysis pipeline (thickness, curvature, ...) that "map" can plot; builds the SFT itself unless --sft is supplied'
    )

    parser.add_argument(
    'args',
    nargs=argparse.REMAINDER,
    help='arguments for the chosen module'
    )

    parser.add_argument(
    "-v", "--version",
    action="version",
    version=f'%(prog)s {version("CALM")}'
    )

    args = parser.parse_args()

    # call the right subroutine based on the module type
    run_module(args.module, args.args)


if __name__ == '__main__':
    pass
