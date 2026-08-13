# CALM map lipids_plot

Render each leaflet's own dominant-species lipid composition map from
`CALM analyze lipids` output.

## Usage

```
CALM map lipids_plot -i lipids_out_dir -o lipids.png
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze lipids` output
  directory.
- `-o`, `--outfile` (default `lipids.png`) - output image path.
- `--purity-floor` (default 0.35) - minimum color opacity for a grid point
  contested among several species (see "How it's rendered" below).

## What's plotted

Two panels, Upper and Lower leaflet. Every `{frame}_lipid_fractions.npy`
(species x [upper, lower] x grid) is a hard per-point species assignment -
1 for that frame's own nearest species at that point, 0 for the rest (see
`CALM analyze lipids --man`). Averaging these across every frame in
`--numpys_directory` turns that into each point's own occupancy
*frequency*: 1.0 where one species always wins there, lower where the
winner changes trajectory frame to frame (e.g. right at a boundary between
two species' territory). Each panel shows, at every grid point, whichever
species has the highest such frequency there.

## How it's rendered

- A fixed color per species (`matplotlib`'s `tab10`/`tab20` categorical
  palette, in `lipid_species.txt` order - `tab20` once there are more than
  10 species; beyond 20, colors repeat and a warning is logged).
- A point's own color opacity scales linearly with its winning species'
  occupancy frequency there, from `--purity-floor` up to fully opaque -
  contested boundary regions read visually faded/blended, stable
  single-species territory reads solid.
- A point with no lipid of any requested species assigned to it in any
  averaged frame (e.g. a typo'd `--lipids` leaving a leaflet with nothing
  selected at all) is left plain white.
- Each panel's own legend gives that leaflet's real mean residue count per
  species too (from `area_per_lipid.csv`'s `mean_count` column, if that
  file is present alongside the `.npy` output - omitted otherwise).

## Known limitation

Unlike `CALM map plot`, `--Remove-TMD` grid points are **not** excluded
here: `CALM analyze lipids` never saves a hole mask (no
`Amn.npy`/`qmn.npy`/`holemask.npy`, only the per-frame composition/area
arrays), so a hole under a protein still shows whichever requested species
happens to be nearest in the raw Voronoi tessellation there.
