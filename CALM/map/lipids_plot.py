from __future__ import annotations

import argparse

from ..core.manual import add_manual


def draw(numpys_directory: str, filename: str) -> None:
    """Render `CALM analyze lipids` output (composition/density maps).

    Not yet implemented - this is where per-species composition/density
    rendering plugs in, reading `lipid_species.txt` (species order) and
    `{frame}_lipid_fractions.npy` (species x [upper, lower] x grid) from
    `numpys_directory`.
    """
    raise NotImplementedError("CALM map lipids_plot's rendering is not yet implemented.")


def lipids_plot(argv: list[str]) -> None:
    """CLI entry: render `CALM analyze lipids` output. Reserved entry point - see `draw`."""
    parser = argparse.ArgumentParser(description="Render CALM analyze lipids output (composition/density maps)")
    parser.add_argument(
        '-i', '--numpys_directory', type=str, required=True,
        help="'CALM analyze lipids' output directory",
    )
    parser.add_argument(
        '-o', '--outfile', type=str, default="lipids.png",
        help="output image path (default: lipids.png)",
    )
    add_manual(parser, "map_lipids_plot")

    ns = parser.parse_args(argv)
    draw(numpys_directory=ns.numpys_directory, filename=ns.outfile)


if __name__ == "__main__":
    pass
