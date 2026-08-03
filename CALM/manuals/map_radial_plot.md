# CALM map radial_plot

Plot upper and lower mean curvature or fitted height, radially averaged
outward from the box center, from a `CALM analyze full` output directory.

## Usage

```
CALM map radial_plot -i full_out_dir -o radial.png
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze full` output
  directory.
- `-o`, `--outfile` (default `radial.png`) - output image path.
- `--quantity` (default `mean`) - `mean` (mean curvature) or `height`
  (fitted surface height).
- `--minimum`, `--maximum` - fix the y-axis lower/upper bound. Both must
  be given together; otherwise the axis auto-scales to the data.

## Notes

- Uses the same whole-trajectory-averaged data (and the same
  hole-masking/rotation-awareness) `CALM map plot` already loads:
  `mean_curvature.npy` for `--quantity mean`, `Z_fitted.npy` for
  `--quantity height`.
- `--quantity height` plots each leaflet's height relative to the
  mid-surface's own height at that same point - the membrane's distance
  from its own mid-surface is physical, regardless of where the box
  places the membrane. The mid-surface reference is loaded at its raw
  value everywhere: it's only a subtraction reference, so each leaflet's
  relative-height value stays tied to its own `--Remove-TMD` hole only.
- Center is the box center (`Lx/2`, `Ly/2`) - the same pivot `--rotate`'s
  fixed-circle masking already uses. The plotted range extends out to the
  fixed circle radius under `--rotate` (the region that's actually valid
  data there) or the box's own inscribed-circle radius otherwise. The
  x-axis itself always starts at 0, whether or not either leaflet has
  data there.
- Each leaflet's own curve starts at the largest circle centered on the
  box center that contains only `NaN` points for that leaflet (e.g. where
  `--Remove-TMD` masked a protein sitting at the center) - no value is
  plotted inside it. From there outward, bin edges are quantiles of that
  leaflet's own valid points' radii, computed independently for upper and
  lower, giving each point on a curve roughly the same number of
  contributing grid points from that same leaflet - a locally denser
  region in one leaflet doesn't narrow the other leaflet's bins in the
  same radius range.
- A wedge from the y-axis to the line through the two curves' own
  starting points, spanning the full y-axis range, is shaded gray and
  labeled "No value" in the legend.
- Only upper and lower are plotted; middle is loaded (as a `--quantity
  height` subtraction reference, or unused for `--quantity mean`) but not
  drawn. Both are plotted exactly as computed.
