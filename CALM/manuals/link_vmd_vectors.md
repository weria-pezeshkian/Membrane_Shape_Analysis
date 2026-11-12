# CALM link vmd_vectors

Write VMD TCL scripts that draw principal-curvature direction arrows in real
space, tangent to the fitted surface, from `CALM analyze full --method
principal_directions` output.

## Usage

```
CALM link vmd_vectors -i full_out_dir -o vmd_out_dir
```

## Arguments

- `-i`, `--input` - `CALM analyze full` output directory, containing
  `*_principal_dirs.npy`, `*_Z_fitted.npy`, and
  `Amn.npy`/`qmn.npy`/`dimensions.npy`.
- `-o`, `--output` - directory to write the generated files to.
- `--which` (default `both`) - one of `k1`, `k2`, `both`: which direction
  family to draw, red (`k1`) or blue (`k2`). See Notes for how each point's
  two directions are assigned to the families.
- `--layer` (default all three) - one or more of `upper`, `lower`, `middle`:
  which layer(s) to draw.
- `--step` (default 5) - grid subsampling stride: only every Nth grid point
  along both axes is drawn, cutting the arrow count by roughly `N^2`. The
  dynamic script (see below) redraws its arrows on every VMD frame change, so
  an undownsampled 100x100 grid would be tens of thousands of `draw`
  primitives rebuilt every time the frame slider moves.
- `--dynamic-length` - scale each arrow's length by the magnitude of the
  principal curvature along that arrow's own direction instead of using a
  fixed length. Requires `*_principal_curvatures.npy`
  (`--method principal` in the analyze run).
- `--scale` (default 10) - multiplier on arrow length. The base lengths (a
  fixed length, or `10 * |k|` with `--dynamic-length`) are on the
  same nm scale as CALM's other reported quantities; the default of 10
  converts that to VMD's Angstrom coordinate system. Raise or lower it to
  make arrows more visible against the loaded structure (e.g. bead radii).
- `--draw-all-frames` - also write the per-frame animated script (see
  `principal_vectors_dynamic.tcl` below). Off by default: building it
  reads every frame's own `*_principal_dirs.npy`/`*_Z_fitted.npy`, which
  the static-only default skips.

## Output

- `principal_vectors_static.tcl` - always written. One arrow per grid
  point, drawn once, no frame-tracking. Directions are the trajectory-mean
  (nematic-tensor averaged - see Notes) of `principal_dirs.npy`; positions
  use the plain mean of `Z_fitted.npy`, the same average
  `average_structure.gro` (from `CALM link vmd_xtc`) is built from. Source
  this script against `average_structure.gro`.
- `principal_vectors_dynamic.tcl` - written only with `--draw-all-frames`.
  Each fit-frame's own directions and heights. Registers a `vmd_frame`
  trace so the arrows are deleted and redrawn every time the displayed
  frame changes (slider, `animate goto`, play), and draws frame 0
  immediately. Source this script against `trajectory.xtc` (also from
  `CALM link vmd_xtc`) - it warns if the loaded molecule's frame count
  doesn't match.

Arrows are colored red for the k1 family and blue for the k2 family (see
Notes). Grid points flagged as holes
(`--Remove-TMD`) in that frame, or `NaN` (e.g. outside the `--rotate` circle),
are skipped.

## Notes

- `principal_dirs.npy` labels k1 as the larger signed principal curvature
  at each point on its own. On an undulating membrane that puts k1 across
  the undulation in the troughs and along it on the crests, so drawn by
  that label alone both colors turn 90 degrees at every inflection line.
  The arrows are instead drawn as two families that continue smoothly
  across the grid. Neighboring points whose directions clearly continue
  each other (axes within about 32 degrees) form regions that keep their
  stored labels. Where two regions meet with their k1 and k2 continuing
  each other swapped, as at an inflection line, one region's labels are
  swapped, decided by the summed agreement along the clearest borders
  first. The largest region of each connected group keeps its stored
  labels. The dynamic script also swaps a whole frame's families when that
  continues the previous frame's colors better, so colors stay put while
  stepping through frames. Red therefore marks a direction family, not the
  larger curvature at every point; with `--dynamic-length` each arrow's
  length still comes from the curvature along its own direction.
- An eigenvector's sign is arbitrary per point per frame, so averaging
  `principal_dirs.npy` across the whole trajectory with a plain vector mean
  can cancel a direction wherever frames disagree on sign. The static script
  averages the outer product of each frame's direction with itself
  (sign-invariant) and takes the dominant eigenvector of that summed tensor -
  the same fix `CALM map plot --mode principal` uses for its own
  whole-trajectory view.
- Each direction is a unit 3D vector tangent to the fitted surface, including
  the z-tilt from the surface's local slope. The 2D `CALM map plot --mode
  principal --vectors` overlay uses only the in-plane (x, y) components.
- Before drawing, each (layer, k) direction field is sign-aligned so
  neighboring arrows point consistently rather than flipping between the two
  directions of the same axis: a point is first flipped, if needed, to point
  toward lower z wherever its z-component has a definite sign, then any
  remaining locally-flat point (z-component too close to zero to disambiguate)
  is flipped to match the nearest already-aligned neighbor, spreading outward
  across the grid. A point with no aligned neighbor reachable this way (an
  entirely flat, isolated region) keeps its original sign.
