# CALM map lipids_plot

Render every species' own continuous occupancy-frequency map from `CALM
analyze lipids` output, per leaflet.

## Usage

```
CALM map lipids_plot -i lipids_out_dir -o lipids.png
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze lipids` output
  directory.
- `-o`, `--outfile` (default `lipids.png`) - output image path. Also
  names every other file this command writes (see Output).

## What's plotted

Every `{frame}_lipid_fractions.npy` (species x [upper, lower] x grid) is a
hard per-point species assignment - 1 for that frame's own nearest species
at that point, 0 for the rest (see `CALM analyze lipids --man`). Averaging
these across every frame in `--numpys_directory` turns that into each
point's own occupancy *frequency*: 1.0 where one species always wins
there, lower where the winner changes trajectory frame to frame (e.g.
right at a boundary between two species' territory).

Each species gets its own panel showing that continuous value directly -
no collapsing species against each other into a single "dominant species
here" map. That was tried and rejected: it throws away the graded
competition CALM already computes, and with only two species it
degenerates into a plain binary map that shows less than the two
continuous fields it was built from.

A point held out by `--Remove-TMD` in even one averaged frame is left
unfilled (same all-or-nothing convention `CALM map plot` uses for its own
holes) - see `{frame}_hole_mask.npy` in `CALM analyze lipids --man`.

## Rotation

If the `CALM analyze lipids` run used `--rotate`, rendering is restricted
to the largest box-centered circle that stays valid across every averaged
frame - the same restriction `CALM map plot` applies to rotated
curvature/thickness output, and for the same reason (`rotated_grid` in
`core/rotation.py`: only points within that circle have a meaningful
rotated lookup). Detected automatically from `rotated.npy` in
`--numpys_directory`; nothing to pass here.

## Output

- `{outfile}` - one combined overview: every species' own row (Upper/Lower
  columns), one shared colorbar. Good for a quick at-a-glance summary.
- `{stem}_{species}{suffix}` per species (e.g. `lipids_POPC.png`) - that
  species' own two-panel figure (Upper, Lower), its own colorbar, for
  closer inspection. Each panel's title also gives that leaflet's real
  mean residue count for that species (from `area_per_lipid.csv`'s
  `mean_count` column, if that file is present alongside the `.npy`
  output - omitted otherwise).
