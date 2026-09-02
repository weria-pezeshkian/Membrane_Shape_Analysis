from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.manual import add_manual

plt.rcParams["font.family"] = "serif"

_LEAFLET_STYLE = {"upper": "-", "middle": "-", "lower": "--", "both": ":"}


def _display_label(species: str, per_instance: bool) -> str:
    """The legend's own species text: `species` as-is, or - with `per_instance` - its base label and
    fragindex split back apart (`"select#3"` -> `"select #3"`) for readability."""
    if per_instance and "#" in species:
        base, fragindex = species.rsplit("#", 1)
        return f"{base} #{fragindex}"
    return species


def draw(Dir: str, filename: str = "diffusion.png", scale: str = "loglog", per_instance: bool = False) -> None:
    """Render every `(species, leaflet)` pool's own MSD(tau) curve from 'CALM analyze diffusion' output.

    `per_instance` reads `diffusion_per_instance.npy`/`msd_curves_per_instance.npy`
    instead of the pooled `diffusion.npy`/`msd_curves.npy` - one curve per
    individual tracked point (e.g. every protein matched by the same
    `--select`) instead of one curve combining all of them.

    Curves are colored by species (a fixed color per species/instance,
    assigned in sorted order so it stays the same across a re-plot), and
    styled by leaflet (solid for upper/middle, dashed for lower, dotted
    for the pooled "both") - so two curves from the same species but
    different leaflets share a color but not a line style. Each curve's
    own legend entry gives its already-fitted `D +/- stderr` directly, so
    the fit does not need to be re-derived visually from the plot; the
    curve itself is for judging the fit's own quality (a straight line in
    the plotted window) and for comparing shapes between species - e.g. a
    markedly shallower curve for a larger species is direct evidence of
    its slower diffusion, size-linked drag.
    """
    out = Path(Dir)
    suffix = "_per_instance" if per_instance else ""
    diffusion_rows = np.load(out / f"diffusion{suffix}.npy")
    msd_rows = np.load(out / f"msd_curves{suffix}.npy")

    species_order = sorted({str(s) for s in diffusion_rows["species"]})
    color_by_species = {name: f"C{i}" for i, name in enumerate(species_order)}

    fontsize = 22
    linewidth = 3
    fig, ax = plt.subplots(figsize=(10, 8))

    guide_anchor: tuple[float, float] | None = None
    for row in diffusion_rows:
        leaflet_key, species = str(row["leaflet"]), str(row["species"])
        curve = msd_rows[(msd_rows["leaflet"] == leaflet_key) & (msd_rows["species"] == species)]
        if len(curve) == 0:
            continue
        order = np.argsort(curve["tau_ps"])
        tau_sorted, msd_sorted = curve["tau_ps"][order], curve["msd_A2"][order]
        ax.plot(
            tau_sorted, msd_sorted,
            color=color_by_species[species], linestyle=_LEAFLET_STYLE.get(leaflet_key, "-."),
            linewidth=linewidth,
            label=f"{_display_label(species, per_instance)} ({leaflet_key}): D={row['D_cm2_s']:.3g}"
                  f"±{row['D_stderr_cm2_s']:.1g} cm$^2$/s",
        )
        if guide_anchor is None or tau_sorted[0] < guide_anchor[0]:
            guide_anchor = (float(tau_sorted[0]), float(msd_sorted[0]))

    if scale == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")
        if guide_anchor is not None:
            # A slope-1 reference line (MSD directly proportional to tau -
            # normal diffusion), anchored at the earliest plotted point so
            # every curve's own departure from it is visible at a glance:
            # a curve falling below this line at longer tau is subdiffusive.
            tau0, msd0 = guide_anchor
            tau_guide = np.array([tau0, ax.get_xlim()[1]])
            msd_guide = msd0 * (tau_guide / tau0)
            ax.plot(
                tau_guide, msd_guide, linestyle="--", color="gray", linewidth=linewidth * 0.6,
                label="slope 1 (normal diffusion)", zorder=0,
            )
    ax.set_xlabel("Lag time (ps)", fontsize=fontsize)
    ax.set_ylabel(r"MSD ($\mathrm{\AA}^2$)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize, width=linewidth * 1.5, length=3 * linewidth)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.legend(fontsize=fontsize * 0.5)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _build_diffusion_plot_parser() -> argparse.ArgumentParser:
    """The 'CALM map diffusion_plot' parser alone, with no side effects - shared by the CLI entry
    point below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(description="Render CALM analyze diffusion output (MSD(tau) curves)")
    parser.add_argument(
        '-i', '--numpys_directory', type=str, required=True,
        help="'CALM analyze diffusion' output directory",
    )
    parser.add_argument(
        '-o', '--outfile', type=str, default="diffusion.png",
        help="output image path (default: diffusion.png)",
    )
    parser.add_argument(
        '--scale', choices=["loglog", "linear"], default="loglog",
        help="axis scale: log-log (default, shows the diffusive exponent directly) or linear",
    )
    parser.add_argument(
        '--per-instance', dest="per_instance", default=False, action="store_true",
        help="one curve per individual tracked point instead of one pooled curve per species/label "
             "(reads diffusion_per_instance.npy/msd_curves_per_instance.npy)",
    )
    add_manual(parser, "map_diffusion_plot")
    return parser


def diffusion_plot(argv: list[str]) -> None:
    """CLI entry: render every tracked species/leaflet's own MSD(tau) curve from 'CALM analyze diffusion' output."""
    parser = _build_diffusion_plot_parser()

    ns = parser.parse_args(argv)
    draw(Dir=ns.numpys_directory, filename=ns.outfile, scale=ns.scale, per_instance=ns.per_instance)


if __name__ == "__main__":
    pass
