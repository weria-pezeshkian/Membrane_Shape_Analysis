# CALM link write_ndx

Detect the two leaflets in a selection and write a GROMACS index file with
groups named `Upper` and `Lower`.

## Usage

```
CALM link write_ndx -f traj.xtc -s structure.tpr -n "name PO4" -o monolayers.ndx
```

## Arguments

- `-f`, `--trajectory` - path to the trajectory file.
- `-s`, `--structure` - path to the structure file.
- `-n`, `--selection` - MDAnalysis selection of the particles to split
  into leaflets (e.g. `"name PO4"`).
- `-o`, `--out` (default `monolayers.ndx`) - path to the index file to
  write.
- `-F`, `--flip` - swap which detected leaflet is labelled `Upper` vs.
  `Lower`.
- `--min-balance` (default 0.6) - minimum acceptable leaflet-size balance
  (1.0 = perfectly equal, 0.0 = all atoms in one leaflet) for a candidate
  split to be accepted; among valid splits, the one covering the most
  atoms wins. 0.6 rejects splits more lopsided than roughly 4:1.
- `--margin` (default 2.0) - an atom is kept in a leaflet only if its
  distance to the nearest atom in the OTHER leaflet is at least this many
  times its distance to the nearest atom in its OWN leaflet. Catches atoms
  XY-connectivity alone would misclassify (e.g. squeezed toward the
  mid-plane near a protein, or mid flip-flop) without suppressing genuine
  sharp curvature, since curvature preserves inter-leaflet distance.

Atoms belonging to neither leaflet's best-scoring component are excluded
from both, with a warning giving the count excluded.
