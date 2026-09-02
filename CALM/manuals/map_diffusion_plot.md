# CALM map diffusion_plot

Plot every tracked species/leaflet's own MSD(tau) curve from a `CALM
analyze diffusion` output directory.

## Usage

```
CALM map diffusion_plot -i diffusion_out_dir -o diffusion.png
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze diffusion` output
  directory.
- `-o`, `--outfile` (default `diffusion.png`) - output image path.
- `--scale` (default `loglog`) - `loglog` or `linear` axis scale.
  `loglog` shows the diffusive exponent directly: a straight line of
  slope 1 is normal diffusion, slope < 1 is subdiffusive (e.g. caged or
  dragged motion), slope near 2 is ballistic.
- `--per-instance` - one curve per individual tracked point instead of
  one pooled curve per species/label, e.g. every one of ten proteins
  matched by the same `--select` gets its own curve instead of all ten
  being combined into one. Reads `diffusion_per_instance.npy`/
  `msd_curves_per_instance.npy` (always written by `CALM analyze
  diffusion`, alongside the pooled files) instead of `diffusion.npy`/
  `msd_curves.npy` - no re-run of `CALM analyze diffusion` needed to
  switch between the two views.

## What's plotted

One curve per `(leaflet, species)` row in `diffusion.npy`, read straight
from `msd_curves.npy` - no re-fitting or re-aggregation happens here, the
curve is exactly what `CALM analyze diffusion` already computed.

- Color is fixed per species (assigned in sorted order, so it stays the
  same across a re-plot); line style is fixed per leaflet - solid for
  `upper`/`middle`, dashed for `lower`, dotted for the pooled `both`. Two
  curves from the same species in different leaflets share a color but
  not a line style. With `--per-instance`, each individual point gets its
  own color and its own legend entry (`"select #3 (middle): D=..."`, from
  `diffusion_per_instance.npy`'s `"select#3"` species field) rather than
  colors being shared across every point with the same label.
- Each curve's legend entry gives its already-fitted `D +/- stderr`
  directly (in cm^2/s) - the plot isn't where the fit comes from, it's
  for judging whether that fit looks right (roughly straight over the
  window it was taken from) and for comparing curve *shapes* between
  species: a species that's both slower and more strongly curved
  (sub-diffusive, slope < 1 on `loglog`) than another is direct evidence
  of size-linked drag, not just a lower `D`.
- On `--scale loglog`, a gray dashed slope-1 reference line is drawn
  through the earliest plotted point of whichever curve starts at the
  smallest lag time, extending to the right edge of the plot - the
  signature of normal diffusion. A curve that falls below this line as
  lag time grows is subdiffusive; one that tracks it closely is diffusing
  normally over that range. Not drawn on `--scale linear`, where a
  straight reference line wouldn't mean the same thing.

## Notes

- Every `(leaflet, species)` combination that has at least one kept
  segment gets its own curve - nothing here decides what to track or how
  to segment it (that's `CALM analyze diffusion`'s own `--min-segment-fraction`
  and leaflet/hole-status rules); this command only plots what already
  made it into `diffusion.npy`/`msd_curves.npy`.
- A `--force-middle` run's rows are labeled `middle` and always get a
  solid line, the same style `upper` gets - and never include a `both`
  row at all (`CALM analyze diffusion` skips writing it in that mode,
  since it would just duplicate `middle`), so there's nothing to filter
  out here.
