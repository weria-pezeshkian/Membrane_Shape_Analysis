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
  independent reasons: every tracked point is one bonded fragment (not one
  residue - see below), and the PBC-aware position extraction needs bonds
  to keep a tracked fragment's own atoms from splitting across a periodic
  boundary within one frame.
- `-o`, `--out` - output directory for the saved arrays.
- Leaflet selection used to fit the surfaces (same as `CALM analyze sft`),
  one of - if both are given, `--index-file` takes precedence and
  `-n`/`--index` is ignored, with a warning:
  - `-n`, `--index` - an MDAnalysis dynamic selection string.
  - `--index-file` - a GROMACS-style `.ndx` file with `Upper`/`Lower`
    groups.
- At least one of `--lipids`/`--select` (below).

## What gets tracked

- `--lipids` - resnames to track as distinct lipid species, e.g.
  `--lipids POPC GM1 SAPE24`. Each token is either a bare resname
  (automatic bond-graph headgroup detection) or
  `RESNAME:NAME1,NAME2,...` to give that species' own headgroup atom
  name(s) explicitly instead - the same syntax and the same detection
  method `CALM analyze lipids` uses, but grouped by bonded fragment
  rather than by residue (see below).
- `--select` - an MDAnalysis selection string to track, tracked one point
  per bonded fragment in the match (each fragment's own center of
  geometry).
- `--select-whole` - with `--select`, track the entire match as a single
  combined point (its own center of geometry).
- `--select-label` - label for `--select`'s rows in the output
  (default: `select`).

Every tracked point is one bonded fragment (a connected component of the
bond graph), not one residue - a topology's residue boundaries aren't
guaranteed to match a molecule's actual bonded extent, while a fragment
always does. For an ordinary lipid selection the two coincide (one residue
is one bonded molecule), so this only matters when they don't.

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
- `--force-middle` - track against the middle surface (the upper/lower
  average) instead of a real leaflet assignment. Every point is `1`
  (embedded, always projected onto the middle surface) or `0`
  (implausibly far from the middle surface entirely, same far-multiple
  rule as ordinary leaflet assignment) - never `-1`, so a leaflet flip can
  never happen and segments only ever break on a hole-status change or
  going unassigned. Meant for something that straddles both leaflets by
  construction, e.g. a transmembrane protein: its own position sits near
  the mid-plane, so a real upper/lower assignment would flip between them
  on thermal noise alone and fragment its trajectory into short, spurious
  segments. Output rows for a `--force-middle` run are labeled `middle`
  instead of `upper`/`lower`, and the pooled `"both"` row is skipped
  entirely - every segment's own leaflet is already `middle`, so `both`
  would just duplicate it rather than being a genuine second pool.
  `--Remove-TMD` hole status still comes from the real upper leaflet's
  mask in this mode (there is no separate middle-surface hole detection).

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
  `unwrap` keeps a tracked fragment's own atoms from splitting across a
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
  one row per physical tracked point, `index, label, fragindex, kind`.
- `{frame}_dimensions.npy` - that frame's own `(Lx, Ly, Lz)`.
- `{frame}_diffusion_meta.npy` - shape `(n_tracked, 2)`:
  `[leaflet, in_hole]` per tracked point, from the live per-frame fit.
- `{frame}_diffusion_surface.npy` - shape `(2, 2*Nx+1, 2*Ny+1)`:
  `[Anm_upper, Anm_lower]`, the fitted surfaces that frame's leaflet
  assignment and hole lookup used, reused later for projection.
- `{frame}_diffusion_positions.npy` - shape `(n_tracked, 3)`: each tracked
  point's own whole, continuous raw position for that frame.
- `{frame}_hole_mask.npy` - written only if `--Remove-TMD` was given.
- `diffusion.npy` - one row per `(leaflet, species)` pooled by
  species/label, including the pooled `"both"` row: `leaflet, species,
  D_cm2_s, D_stderr_cm2_s, n_segments, n_points_pooled, tau_min_ps,
  tau_max_ps, fit_r2, fit_loglog_slope, n_segments_discarded_short`.
- `diffusion.csv` - the same rows as `diffusion.npy`, human-readable.
- `msd_curves.npy` - one row per `(leaflet, species, tau)`:
  `leaflet, species, tau_ps, msd_A2, n_samples` - for inspecting or
  plotting the diffusive-regime fit directly.
- `diffusion_per_instance.npy`, `diffusion_per_instance.csv`,
  `msd_curves_per_instance.npy` - the same three files, but pooled per
  individual tracked point instead of per species/label: `species` reads
  `"<label>#<fragindex>"` (e.g. every one of ten proteins matched by the
  same `--select` gets its own row, `select#0` through `select#9`,
  instead of all ten being combined into one `select` row). Computed from
  the same segments as the pooled files, at negligible extra cost - both
  groupings are always written, so choosing between them (e.g. in `CALM
  map diffusion_plot --per-instance`) needs no re-run.
- `segments.npy` - one row per candidate segment, kept or excluded, at
  the finest (per-point) granularity regardless of pooling: `label,
  leaflet, fragindex, start_frame, end_frame, length_frames,
  length_fraction, kept, discard_reason`.
