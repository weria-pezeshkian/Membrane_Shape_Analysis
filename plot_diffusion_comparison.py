#!/usr/bin/env python3
"""Standalone comparison plot: c99 vs fl protein diffusion, one line per replica.

Reads per-instance MSD curves straight from each replica's own
'CALM analyze diffusion --per-instance' output:

    {c99,fl}/{1P,3P,10P}/{rep1,rep2,rep3}/CALM_diffuse/msd_curves_per_instance.npy

One line per (system, count, replica) - up to 18 lines. Each line is the
mean MSD, at each lag time, across that replica's own individual proteins
(1, 3, or 10 of them); the shaded band is +/- 1 std across those same
proteins at each lag time - a wide band means the proteins in that
replica are behaving very differently from each other (consistent with
some clustering together while others stay free), not sampling noise
from pooling everything into one curve. A 1P replica has only one
protein, so its band is zero width by construction.

c99 lines share one color family (Blues), fl lines another (Oranges);
protein count sets the shade within each family (lighter = fewer
proteins); replica sets the line style, so same-count replicas of the
same system are distinguishable from each other. No CALM import - this
only needs numpy and matplotlib, reading the .npy files CALM already
wrote.

Adjust the constants below (BASE_DIR, SYSTEMS, COUNTS, REPS) to match a
different directory layout or run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(".")
SYSTEMS = ["c99", "fl"]
COUNTS = ["1P", "3P", "10P"]
REPS = ["rep1", "rep2", "rep3"]
OUT_SUBDIR = "CALM_diffuse"
OUTFILE = "diffusion_comparison.png"

# Which leaflet row to read - "middle" for a --force-middle run (the
# expected case here, tracking proteins), falling back to "both" or
# whatever single leaflet value is present otherwise.
LEAFLET_PREFERENCE = ("middle", "both")

CMAPS = {"c99": plt.get_cmap("Blues"), "fl": plt.get_cmap("Oranges")}
SHADES = {"1P": 0.4, "3P": 0.65, "10P": 0.9}  # position along each colormap, light -> dark
REP_STYLE = {"rep1": "-", "rep2": "--", "rep3": ":"}

plt.rcParams["font.family"] = "serif"


def _pick_leaflet(msd_rows: np.ndarray) -> str:
    present = set(msd_rows["leaflet"].tolist())
    for candidate in LEAFLET_PREFERENCE:
        if candidate in present:
            return candidate
    return sorted(present)[0]


def _replica_mean_and_spread(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """(tau, mean_msd, std_msd, n_proteins) for one replica: the mean and across-protein spread
    of its own individual proteins' MSD curves, each protein's own curve read from
    msd_curves_per_instance.npy and aligned onto a common tau grid first."""
    msd_path = directory / OUT_SUBDIR / "msd_curves_per_instance.npy"
    if not msd_path.exists():
        print(f"skipping missing {msd_path}")
        return None
    msd_rows = np.load(msd_path)
    leaflet = _pick_leaflet(msd_rows)
    rows = msd_rows[msd_rows["leaflet"] == leaflet]
    if len(rows) == 0:
        print(f"no rows at any leaflet in {msd_path}")
        return None

    protein_curves = []
    for species in np.unique(rows["species"]):
        one = rows[rows["species"] == species]
        order = np.argsort(one["tau_ps"])
        protein_curves.append((one["tau_ps"][order], one["msd_A2"][order]))
    if not protein_curves:
        return None

    # Common tau grid: the shortest protein's own tau values in this
    # replica - avoids extrapolating past a shorter-lived point's range
    # (e.g. one that spent part of the trajectory unassigned/in a hole).
    common_tau = min(protein_curves, key=lambda c: c[0][-1])[0]
    aligned = np.array([np.interp(common_tau, tau, msd) for tau, msd in protein_curves])
    return common_tau, aligned.mean(axis=0), aligned.std(axis=0), len(protein_curves)


fig, ax = plt.subplots(figsize=(10, 8))
guide_anchor: tuple[float, float] | None = None

for system in SYSTEMS:
    cmap = CMAPS[system]
    for count in COUNTS:
        color = cmap(SHADES[count])
        for rep in REPS:
            result = _replica_mean_and_spread(BASE_DIR / system / count / rep)
            if result is None:
                continue
            tau, mean_msd, std_msd, n_proteins = result

            ax.plot(
                tau, mean_msd, color=color, linestyle=REP_STYLE[rep], linewidth=3,
                label=f"{system} {count} {rep} (n={n_proteins} proteins)",
            )
            if n_proteins > 1:
                ax.fill_between(tau, mean_msd - std_msd, mean_msd + std_msd, color=color, alpha=0.2, linewidth=0)

            if guide_anchor is None or tau[0] < guide_anchor[0]:
                guide_anchor = (float(tau[0]), float(mean_msd[0]))

ax.set_xscale("log")
ax.set_yscale("log")
if guide_anchor is not None:
    tau0, msd0 = guide_anchor
    tau_guide = np.array([tau0, ax.get_xlim()[1]])
    ax.plot(
        tau_guide, msd0 * (tau_guide / tau0), linestyle="--", color="gray", linewidth=2,
        label="slope 1 (normal diffusion)", zorder=0,
    )

fontsize = 22
ax.set_xlabel("Lag time (ps)", fontsize=fontsize)
ax.set_ylabel(r"MSD ($\mathrm{\AA}^2$)", fontsize=fontsize)
ax.tick_params(labelsize=fontsize, width=4.5, length=9)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
ax.legend(fontsize=fontsize * 0.4, ncol=2)

fig.tight_layout()
fig.savefig(OUTFILE, dpi=150)
plt.close(fig)
print(f"wrote {OUTFILE}")
