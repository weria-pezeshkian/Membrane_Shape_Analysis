# CALM analyze diffusion

Curvature-aware lateral diffusion coefficient, per lipid species and/or an
arbitrary MDAnalysis selection, computed frame by frame from a live
trajectory. Each tracked point's real position is projected onto its own
assigned leaflet's fitted surface before displacement is measured, so
membrane undulation is separated from lateral motion.

## Usage

```
CALM analyze diffusion -f traj.xtc -s structure.tpr -o out_dir -n "name PO4" --lipids POPC POPE [options]
```

## Required arguments

- `-f`, `--trajectory` - path to the trajectory file (e.g. `.xtc`).
- `-s`, `--structure` - path to the structure file. Must carry bond
  information (e.g. a GROMACS `.tpr`) - a bare `.gro`/`.pdb` with no bonds
  is rejected immediately. Bonds are needed unconditionally here, for two
  independent reasons: the automatic headgroup detection walks each
  residue's own bond graph, and the PBC-aware position extraction needs
  bonds to keep a tracked residue's own atoms from splitting across a
  periodic boundary within one frame.
- `-o`, `--out` - output directory for the saved arrays.
- `-n`, `--index` - leaflet selection used to fit the surfaces (same as
  `CALM analyze sft`): either a GROMACS-style `.ndx` file with `Upper`/
  `Lower` groups, or an MDAnalysis dynamic selection string.
- At least one of `--lipids`/`--select` (below).

## What gets tracked

- `--lipids` - resnames to track as distinct lipid species, e.g.
  `--lipids POPC GM1 SAPE24`. Each token is either a bare resname
  (automatic bond-graph headgroup detection) or
  `RESNAME:NAME1,NAME2,...` to give that species' own headgroup atom
  name(s) explicitly instead - the same syntax and the same detection
  method `CALM analyze lipids` uses.
- `--select` - an MDAnalysis selection string to track, tracked one point
  per residue in the match (each residue's own center of geometry).
- `--select-whole` - with `--select`, track the entire match as a single
  combined point (its own center of geometry).
- `--select-label` - label for `--select`'s rows in the output
  (default: `select`).

## Leaflet, hole status, and segments

Every tracked point gets a fresh leaflet assignment every frame (whichever
fitted surface it's closer to, or excluded if implausibly far from both -
the same rule `CALM analyze lipids` uses). If `--Remove-TMD` is given, a
hole mask is rebuilt fresh every frame too, and each tracked point's own
hole status is looked up from it directly.

A tracked point's trajectory is broken into a new segment whenever its
leaflet assignment flips, its hole status changes, or it becomes
unassigned - a lipid that genuinely flip-flops or transiently wanders
under a protein contributes only its stable stretches to the diffusion
fit, not one contaminated trajectory spanning the whole run. A segment is
kept only if it reaches both `--min-segment-fraction` of the analyzed
frame range and a fixed minimum frame count; short segments are recorded
in `segments.npy` with the reason they were excluded, not silently
dropped.

- `--min-segment-fraction` - segments shorter than this fraction of the
  analyzed range are excluded from the fit (default: `0.1`).

`--Remove-TMD` must always be given its own explicit selection here (e.g.
`--Remove-TMD 'name BB SC1'`) - this command has no `--center` to fall
back to (see below), so the bare `--Remove-TMD` form is rejected with an
error naming the requirement.

## Projection and PBC handling

Positions come from two passes joined by frame:

- Each frame's leaflet surfaces are fit once (live, no `--sft` reuse - a
  live fit is fast enough that reusing a precomputed one buys nothing
  here), the same way `CALM analyze lipids` fits them.
- Each tracked point's own real position is extracted from a second,
  sequential pass that chains two MDAnalysis transformations:
  `unwrap` keeps a tracked residue's own atoms from splitting across a
  periodic boundary within one frame (which would otherwise corrupt its
  center-of-geometry average), and `NoJump` keeps that point's position
  continuous from one analyzed frame to the next, absorbing any periodic
  boundary crossing between them.

Each tracked point is then projected onto its own assigned leaflet's
fitted surface along the local normal at its own (x, y): the same
ray-surface-intersection search `CALM analyze full`'s bilayer thickness
uses, reused here to find where the normal through the tracked point's
own real position crosses the fitted surface. The projected trajectory
stays in the same continuous coordinate frame the PBC-aware extraction
built, so it is never re-wrapped.

## Diffusion coefficient

The mean-squared displacement is multi-tau: every valid `(t, t+tau)` pair
within each kept segment's own length contributes to that tau's pooled
average, up to `--max-tau-fraction` of that segment's own length - a short
segment only ever contributes to small tau, a long one to the full range.
Segments are pooled by species and leaflet (and once more, combined across
leaflets, as `"both"`).

D is a linear fit of MSD against tau (`MSD = 4*D*tau`, the 2D Einstein
relation) over a window of the pooled curve's own tau range:

- `--max-tau-fraction` - caps tau at this fraction of each segment's own
  length (default: `0.25`).
- `--fit-tau-min-fraction` / `--fit-tau-max-fraction` - the window, as a
  fraction of the pooled curve's own max tau, used for the linear D fit
  (default: `0.1` / `0.5`).

The fit also reports R^2 and the log-log slope of MSD against tau over the
same window - a slope near 1 is the signature of normal diffusion, and a
slope near 2 is the signature of ballistic motion over that window, a
sign the fit window or segment selection needs a closer look before
trusting D.

## Other build arguments

Shares `-F`/`-U`/`-S`/`--lambda_x`/`--lambda_y`/`--gridsize`/
`--Remove-TMD`/`--regularization`/`-W`/`-c`/`--loud`/`--replay`/
`--out-replay` with `CALM analyze sft` - see `CALM analyze sft --man` for
those. There is no `--rotate`/`--rotation-direction` here (nothing in this
command's output depends on rotational alignment) and no `--center`
(centering on a selection that overlaps or moves together with what's
being tracked would remove real signal from the very displacement being
measured).

## Output

- `tracked_points.npy` - the fixed tracked-point roster, written once:
  one row per physical tracked point, `index, label, resindex, kind`.
- `{frame}_dimensions.npy` - that frame's own `(Lx, Ly, Lz)`.
- `{frame}_diffusion_meta.npy` - shape `(n_tracked, 2)`:
  `[leaflet, in_hole]` per tracked point, from the live per-frame fit.
- `{frame}_diffusion_surface.npy` - shape `(2, 2*Nx+1, 2*Ny+1)`:
  `[Anm_upper, Anm_lower]`, the fitted surfaces that frame's leaflet
  assignment and hole lookup used, reused later for projection.
- `{frame}_diffusion_positions.npy` - shape `(n_tracked, 3)`: each tracked
  point's own whole, continuous raw position for that frame.
- `{frame}_hole_mask.npy` - written only if `--Remove-TMD` was given.
- `diffusion.npy` - one row per `(leaflet, species)`, including the
  pooled `"both"` row: `leaflet, species, D_cm2_s, D_stderr_cm2_s,
  n_segments, n_points_pooled, tau_min_ps, tau_max_ps, fit_r2,
  fit_loglog_slope, n_segments_discarded_short`.
- `msd_curves.npy` - one row per `(leaflet, species, tau)`:
  `leaflet, species, tau_ps, msd_A2, n_samples` - for inspecting or
  plotting the diffusive-regime fit directly.
- `segments.npy` - one row per candidate segment, kept or excluded:
  `label, leaflet, resindex, start_frame, end_frame, length_frames,
  length_fraction, kept, discard_reason`.
