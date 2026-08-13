# CALM analyze sft

Build and save the per-frame Fourier coefficient stack from a trajectory:
`Amn.npy` (coefficients), `qmn.npy` (wavevectors), `dimensions.npy` (box
size per frame). This is the starting point for `CALM calibrate` and can be
reused later by `CALM analyze full --sft`.

## Usage

```
CALM analyze sft -f traj.xtc -s structure.tpr -o out_dir -n "name PO4" [options]
```

## Required arguments

- `-f`, `--trajectory` - path to the trajectory file (e.g. `.xtc`).
- `-s`, `--structure` - path to the structure file (e.g. `.tpr`).
- `-n`, `--index` - either a path to a GROMACS-style index (.ndx) file
  with two groups named `Upper` and `Lower` (e.g. as written by `CALM link
  write_ndx`), or an MDAnalysis selection string (e.g. `"name PO4"`) for
  per-frame dynamic leaflet detection.
- `-o`, `--out` - output directory for the saved arrays.

## Leaflet selection tuning

- `--min-balance` (default 0.6) - only used with a dynamic `-n` selection.
  Minimum acceptable leaflet-size balance (1.0 = perfectly equal, 0.0 =
  all atoms in one leaflet) for a candidate split to be accepted; among
  valid splits, the one covering the most atoms wins. 0.6 rejects splits
  more lopsided than roughly 4:1.
- `--margin` (default 2.0) - only used with a dynamic `-n` selection. An
  atom is kept in a leaflet only if its distance to the nearest atom in the
  OTHER leaflet is at least this many times its distance to the nearest
  atom in its OWN leaflet. Catches atoms XY-connectivity alone would
  misclassify (e.g. squeezed toward the mid-plane near a protein, or mid
  flip-flop) without suppressing genuine sharp curvature, since curvature
  preserves inter-leaflet distance.

## Frame range

- `-F`, `--From` (default 0) - first frame, inclusive.
- `-U`, `--Until` (default: end of trajectory) - last frame, exclusive.
- `-S`, `--Step` (default 1) - stride between frames.

## Fourier fit

- `--lambda_x`, `--lambda_y` - Fourier wavelength scale in x/y (nm).
- `--gridsize` (default 100) - square root of the output grid's point
  count (100 gives a 100x100 grid).
- `--regularization` - enable Tikhonov (`|q|^2`-weighted,
  Helfrich-bending-energy-like) regularization as a leaflet's coefficient
  count approaches its atom count. Off by default: it biases per-frame
  `Anm` toward zero in proportion to curvature, which would circularly
  contaminate any later kappa/sigma calibration derived from these
  coefficients' cross-frame statistics. Only enable it for single/few-frame
  curvature visualization, where per-frame overfitting is the bigger risk.

## Centering and rotation

- `-C`, `--center` - MDAnalysis selection to center each frame on.
  Required by `--rotate`. Also used by `--Remove-TMD` to identify protein
  atoms, unless `--Remove-TMD` is given its own selection.
- `--rotate` - apply per-frame rotation alignment. Requires `--center`.
- `--rotation-direction` - MDAnalysis selection whose center of geometry
  defines the reference direction for rotation. Requires `--rotate`.

## Hole detection

- `--Remove-TMD` - flag grid points unsupported by nearby lipids as holes.
  Takes an optional MDAnalysis selection identifying protein atoms, e.g.
  `--Remove-TMD 'name BB SC1'`. Given bare, `--center`'s selection is used
  for this instead, and `--center` is then required. A selection of its own
  is useful when there are multiple, disconnected transmembrane regions and
  a single `--center` selection would be ambiguous to center on.

  A grid point is flagged as a hole in a leaflet when it's farther from
  that leaflet's own atoms than a shared threshold, and either:
  1. it's within that same threshold of a protein atom (from the selection
     above) currently embedded in the membrane (z between the fitted Upper
     and Lower surfaces at that atom's own x, y); or
  2. it's farther than 5x the threshold from that leaflet's own atoms.

  The threshold is shared by both leaflets: a multiple of lipid spacing
  (the larger of the two leaflets' own typical spacing), capped by the
  fit's Nyquist resolution. After both leaflets' hole masks are built, any
  non-hole grid point fully enclosed by hole cells - accounting for the
  box's periodic boundary - is folded into the hole as well.

  The result is saved as `holemask.npy` alongside
  `Amn.npy`/`qmn.npy`/`dimensions.npy`; this command's own output doesn't
  otherwise use it. It only takes effect later, when a `CALM analyze full`,
  `CALM map plot`/`dynamic_plot`, or `CALM link vmd_xtc` run reads this
  output (directly or via `--sft`) and masks or excludes the flagged
  points.

## Replay

- `--replay` - load arguments from a previously written replay file; CLI
  arguments given alongside `--replay` still take priority. The replay
  file records a sha256 checksum of the `--trajectory`/`--structure` files
  it was built from; if either file on disk no longer matches its
  recorded checksum, replaying prints a warning and asks for interactive
  confirmation (y/N) before continuing, so a silently changed input file
  can't produce a result that looks like a faithful replay but isn't. With
  no way to answer (e.g. a non-interactive batch job), the run aborts.
- `--out-replay` - path to write this run's replay file (records the full
  effective configuration, including defaults). Defaults to a timestamped
  name inside `--out`.

The replay file doubles as this run's log: every warning and info-level
message from the run (e.g. a corrected `--lambda_x`/`--lambda_y`, an
underdetermined or low-redundancy Fourier fit, a per-frame `--Remove-TMD`
breakdown of how many grid points were flagged and why) is appended to it
as `#`-prefixed comment lines, so it stays replayable while still keeping a
full record of what happened.

## Other

- `-W`, `--Workers` (default 1) - number of parallel workers.
- `-c`, `--clear` - remove existing `.npy` files in `--out` before running.
  No warning, no backup.
- `--loud` - also print info-level log messages to the console as the run
  progresses. Warnings and errors always print; by default, info-level
  messages (e.g. per-frame `--Remove-TMD` breakdowns) go to the replay log
  only.
