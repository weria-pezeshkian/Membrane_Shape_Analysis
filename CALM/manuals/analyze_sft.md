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
- `-o`, `--out` - output directory for the saved arrays.

## Leaflet selection

- `-n`, `--index` - either a path to an index file with two groups named
  `Upper` and `Lower`, or an MDAnalysis selection string (e.g. `"name
  PO4"`) for per-frame dynamic leaflet detection.
- `--min-balance` (default 0.6) - only used with a dynamic `-n` selection.
  Minimum acceptable leaflet-size balance (1.0 = perfectly equal, 0.0 =
  all atoms in one leaflet) for a candidate split to be accepted; among
  valid splits, the one covering the most atoms wins. 0.6 rejects splits
  more lopsided than roughly 4:1. See `core/leaflet.py`'s `get_components`.
- `--margin` (default 2.0) - only used with a dynamic `-n` selection. An
  atom is kept in a leaflet only if its distance to the nearest atom in the
  OTHER leaflet is at least this many times its distance to the nearest
  atom in its OWN leaflet. Catches atoms XY-connectivity alone would
  misclassify (e.g. squeezed toward the mid-plane near a protein, or mid
  flip-flop) without suppressing genuine sharp curvature, since curvature
  preserves inter-leaflet distance. See `core/leaflet.py`'s
  `apply_margin_filter`.

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
  Required by `--rotate` and `--Remove-TMD`.
- `--rotate` - apply per-frame rotation alignment. Requires `--center`.
- `--rotation-direction` - MDAnalysis selection whose center of geometry
  defines the reference direction for rotation. Requires `--rotate`.

## Hole detection

- `--Remove-TMD` - flag grid points unsupported by nearby lipids as holes.
  A grid point is flagged only if both hold:
  1. its distance to the nearest atom in the fit selection (Upper or
     Lower) exceeds a data-driven threshold: a multiple of that leaflet's
     own typical lipid spacing, capped by the fit's Nyquist resolution;
  2. it is within that same threshold of a `--center` atom that is
     currently embedded in the membrane (its z lies between the fitted
     Upper and Lower surfaces at its own x, y) - not a soluble or
     extramembrane domain of the same selection.

  Requires `--center`. See `core/packing.py`'s `median_multiple_threshold`
  and `core/fourier_build.py`'s `_tmd_protein_atoms_xy`.

## Replay

- `--replay` - load arguments from a previously written replay file; CLI
  arguments given alongside `--replay` still take priority.
- `--out-replay` - path to write this run's replay file (records the full
  effective configuration, including defaults). Defaults to a timestamped
  name inside `--out`.

## Other

- `-W`, `--Workers` (default 1) - number of parallel workers.
- `-c`, `--clear` - remove existing `.npy` files in `--out` before running.
  No warning, no backup.
