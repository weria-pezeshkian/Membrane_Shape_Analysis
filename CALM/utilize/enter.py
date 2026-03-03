from CALM.utilize.write_ndx import write_ndx
from .get_vmd_visualization import write_xtc
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
    if module_name == 'write_ndx':
        write_ndx(args)
    elif module_name == 'write_xtc':
        write_xtc(args) 
    #elif module_name == '<new module>': # remember to import module function up top
    #    plot_height(args)
    else:
        print(f"Unknown module: {module_name}")

def Link(args):
    """
    main entry point for the CALM link command-line interface.
    """

    modules=['write_ndx','write_xtc']

    sys.argv=[""]+args
    # call the right subroutine based on the module type

    # parse arguments before a calling module
    parser = argparse.ArgumentParser(
    description='CALM link: link contains utility or preparation functions that are not mandatory, but nice. For instance, the automatic writing of an index file to split the bilayer into monolayers.',
    prog='CALM',
    formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
    'module',
    choices=modules,
    help='choice of which module to run, run CALM link <module> -h for detailed help\n\
    <write_ndx> prepares an index file\n\
    <calculate> calculates the curvature\n\
    <plot> plots the curvature data based on a directory with the files from curvature'
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