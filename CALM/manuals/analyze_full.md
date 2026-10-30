# CALM analyze full

Run the full geometric analysis pipeline (thickness, curvature, ...) that
`CALM map` can plot. Builds the Fourier fit itself unless `--sft` points to
output already produced by `CALM analyze sft`.

## Usage

```
CALM analyze full -f traj.xtc -s structure.tpr -o out_dir -n "name PO4" [options]
CALM analyze full --sft sft_dir -o out_dir [options]
```

## Required arguments

- `-o`, `--out` - output directory for the saved arrays.
- Either all of `-f`/`--trajectory`, `-s`/`--structure`, and one of
  `-n`/`--index` or `--index-file`, or `--sft`.

## Reusing a precomputed fit

- `--sft` - directory containing a previously built fit (`Amn.npy`,
  `qmn.npy`, `dimensions.npy`, as written by `CALM analyze sft`). If given,
  `-f`/`-s`/`-n`/`--index-file` are not required and no fitting is redone.
- `--method` - space-separated list of analysis methods to run: `thickness`,
  `Z_fitted`, `mean`, `gaussian`, `principal`, `principal_directions`. If
  omitted, all methods run.

## Leaflet selection

- One of these (part of the required set above when not using `--sft`;
  if both are given, `--index-file` takes precedence and `-n`/`--index` is
  ignored, with a warning):
  - `-n`, `--index` - an MDAnalysis selection string (e.g. `"name PO4"`)
    for per-frame dynamic leaflet detection.
  - `--index-file` - a GROMACS-style index (.ndx) file with two groups
    named `Upper` and `Lower` (e.g. as written by `CALM link write_ndx`),
    for static leaflet membership instead.
- `--min-balance` (default 0.6) - only used with a dynamic `-n` selection.
  Minimum acceptable leaflet-size balance (1.0 = perfectly equal, 0.0 =
  all atoms in one leaflet) for a candidate split to be accepted; among
  valid splits, the one covering the most atoms wins. 0.6 accepts splits
  down to about 7:3 and rejects anything more lopsided - looser than the
  one directly comparable precedent found in the wider literature
  (MDAnalysis's `LeafletFinder.optimize_cutoff`, `max_imbalance=0.2`, i.e.
  ~6:4), chosen deliberately to tolerate real protein-induced leaflet-size
  asymmetry across many trajectory frames rather than a single offline
  cutoff optimization.
- `--margin` (default 2.0) - only used with a dynamic `-n` selection. An
  atom is kept in a leaflet only if its distance to the nearest atom in the
  OTHER leaflet is at least this many times its distance to the nearest
  atom in its OWN leaflet. Catches atoms XY-connectivity alone would
  misclassify (e.g. squeezed toward the mid-plane near a protein, or mid
  flip-flop) without suppressing genuine sharp curvature, since curvature
  preserves inter-leaflet distance. No comparable ratio-based convention
  was found in the literature for this kind of margin (the nearest analog
  in other tools, an absolute-distance midplane cutoff applied only to
  sterols, solves a different, narrower problem) - this stays an
  empirical, CALM-specific choice.

## Frame range

Only used when building the fit (not with `--sft`).

- `-F`, `--From` (default 0) - first frame, inclusive.
- `-U`, `--Until` (default: end of trajectory) - last frame, exclusive.
- `-S`, `--Step` (default 1) - stride between frames.

## Fourier fit

Only used when building the fit (not with `--sft`).

- `--lambda_x`, `--lambda_y` - Fourier wavelength scale in x/y (nm). If
  neither is given, the finest resolution `CALM calibrate lambda` itself
  calls safe for this exact structure is used instead of a fixed value -
  logged at INFO level. Giving exactly one of the two leaves the other at
  `get_fourier_modes`' own fixed default (no auto-selection for a partial
  override).
- `--gridsize` (default 100) - square root of the output grid's point
  count (100 gives a 100x100 grid). If not given, raised automatically
  past 100 when needed so the output grid is fine enough to represent
  `lambda_x`/`lambda_y`'s own finest mode without aliasing (Nyquist:
  `gridsize >= 2*max(Nx, Ny)`) - an explicitly given value is never
  raised.
- `--regularization` - enable Tikhonov (`|q|^2`-weighted,
  Helfrich-bending-energy-like) regularization as a leaflet's coefficient
  count approaches its atom count. Off by default: it biases per-frame
  `Anm` toward zero in proportion to curvature, which would circularly
  contaminate any later kappa/sigma calibration derived from these
  coefficients' cross-frame statistics. Only enable it for single/few-frame
  curvature visualization, where per-frame overfitting is the bigger risk.

## Centering and rotation

Only used when building the fit (not with `--sft`).

- `-C`, `--center` - MDAnalysis selection to center each frame on.
  Required by `--rotate`. Also used by `--Remove-TMD` to identify protein
  atoms, unless `--Remove-TMD` is given its own selection.
- `--rotate` - apply per-frame rotation alignment. Requires `--center`.
- `--rotation-direction` - MDAnalysis selection whose center of geometry
  defines the reference direction for rotation. Requires `--rotate`.

## Hole detection

Only used when building the fit (not with `--sft`).

- `--Remove-TMD` - flag grid points unsupported by nearby lipids as holes.
  Takes an optional MDAnalysis selection identifying protein atoms, e.g.
  `--Remove-TMD 'name BB SC1'`. Given bare, `--center`'s selection is used
  for this instead, and `--center` is then required. A selection of its own
  is useful when there are multiple, disconnected transmembrane regions and
  a single `--center` selection would be ambiguous to center on.

  A grid point is flagged as a hole in a leaflet when it's farther from
  that leaflet's own atoms than a shared threshold, and either:
  1. it's within reach of a protein atom (from the selection above)
     assigned to that same leaflet; or
  2. it's farther than 5x the threshold from that leaflet's own atoms.

  A protein atom is assigned to exactly one leaflet - whichever its own
  real position is actually closer to, measured by projecting it onto
  each leaflet's own fitted surface along that leaflet's own local normal
  (never both leaflets, never an average of the two). It's then only kept
  as a hole contributor for that one leaflet if the grid cell it maps
  onto isn't already well-supported by real lipid there (at least 3 of
  the cell's own 4 corners within threshold of real lipid counts as
  supported - one outlier corner is treated as noise, not a real gap).

  Each kept atom's own reach is its own distance to the nearest real
  lipid, capped at 5x the threshold - not the shared threshold itself. A
  narrow, single-helix TMD's own atoms already sit within the shared
  threshold of real lipid, so this makes no difference there; a wider
  channel or pore described by only a thin, central line of atoms needs
  each atom's own reach to grow to match the size of the real gap it's
  actually sitting in the middle of - otherwise the hole would only ever
  cover the immediate neighborhood of that central line, not the gap's
  own full extent.

  The threshold is shared by both leaflets: a multiple of lipid spacing
  (the larger of the two leaflets' own typical spacing), capped by the
  fit's Nyquist resolution. After both leaflets' hole masks are built, any
  non-hole grid point fully enclosed by hole cells - accounting for the
  box's periodic boundary - is folded into the hole as well.

  The result is saved as `holemask.npy` alongside
  `Amn.npy`/`qmn.npy`/`dimensions.npy`. This run's own numeric analysis
  (thickness, curvature, ...) doesn't use it; it only takes effect later,
  when `CALM map plot`/`dynamic_plot` or `CALM link vmd_xtc` reads this
  output and masks or excludes the flagged points.

## Replay

- `--replay` - load arguments from a previously written replay file; CLI
  arguments given alongside `--replay` still take priority. Works the same
  whether or not `--sft` is given - it always records the full effective
  configuration, including `--sft`/`--method` themselves. The replay file
  also records a sha256 checksum of the `--trajectory`/`--structure` files
  it was built from (when given); if either file on disk no longer
  matches its recorded checksum, replaying prints a warning and asks for
  interactive confirmation (y/N) before continuing, so a silently changed
  input file can't produce a result that looks like a faithful replay but
  isn't. With no way to answer (e.g. a non-interactive batch job), the
  run aborts.
- `--out-replay` - path to write this run's replay file (records the full
  effective configuration, including defaults). Defaults to a timestamped
  name inside `--out`.

The replay file doubles as this run's log: every warning and info-level
message from the run (e.g. a corrected `--lambda_x`/`--lambda_y`, an
underdetermined or low-redundancy Fourier fit, a per-frame `--Remove-TMD`
breakdown of how many grid points were flagged and why, thickness failing
to compute for a frame) is appended to it as `#`-prefixed comment lines, so
it stays replayable while still keeping a full record of what happened -
whether or not `--sft` is given.

## Other

- `-W`, `--Workers` (default 1) - number of parallel workers.
- `-c`, `--clear` - remove existing `.npy` files in `--out` before running,
  including its `raw_sft/` subdirectory. No warning, no backup.
- `--loud` - also print info-level log messages to the console as the run
  progresses. Warnings and errors always print; by default, info-level
  messages (e.g. per-frame `--Remove-TMD` breakdowns) go to the replay log
  only.
