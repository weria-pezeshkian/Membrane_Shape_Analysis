# CALM analyze lipids

Per-species lipid composition and area-per-lipid, computed frame by frame
from a live trajectory. Always re-fits the leaflet surfaces itself - there
is no `--sft` reuse, since lipid identity (`resname`) only exists on real
atoms, and a previously-built fit (`Amn`/`qmn` coefficients alone) doesn't
carry that.

## Usage

```
CALM analyze lipids -f traj.xtc -s structure.tpr -o out_dir -n "name PO4" --lipids POPC GM1 SAPE24 [options]
```

## Required arguments

- `-f`, `--trajectory` - path to the trajectory file (e.g. `.xtc`).
- `-s`, `--structure` - path to the structure file (e.g. `.tpr`).
- `-o`, `--out` - output directory for the saved arrays.
- `-n`, `--index` - leaflet selection used to fit the surfaces (same as
  `CALM analyze sft`): either a GROMACS-style `.ndx` file with `Upper`/
  `Lower` groups, or an MDAnalysis dynamic selection string.
- `--lipids` - resnames to treat as distinct lipid species, e.g.
  `--lipids POPC GM1 SAPE24`. Required: CALM has no way to decide what
  counts as "a lipid" in the system on its own.

## How composition is assigned

The `-n` selection (e.g. one phosphate atom per phospholipid) only exists
to fit the leaflet *surface* - it can miss lipid types with a different
headgroup structure entirely (e.g. a glycolipid with no phosphate). So
lipid composition uses its own, separate selection: for each `--lipids`
name, every atom of that resname is selected and grouped into residues:
each residue's (x, y) is its own atoms' center of geometry, and its
leaflet is whichever of the two fitted surfaces its own z is closer to.

At each grid point, every nearby residue (within a bandwidth set by the
typical lipid spacing in that leaflet, the same convention `--Remove-TMD`
uses) contributes a Gaussian-weighted vote by species, normalized so all
species' fractions sum to 1 at that point. A point with only one species
nearby gets fraction 1.0 for it; a point between two species' territory
gets a fractional split (e.g. 0.7 POPC / 0.3 POPE).

Area-per-lipid, for each species, sums that fraction times the grid
cell's area over every grid point, then divides by the species' real
residue count in that leaflet. Two area conventions are reported: the
flat (projected) area, the usual literature quantity, and the true
(curvature-corrected) area, using `sqrt(1 + Zx^2 + Zy^2)` per cell from
the freshly fit surface.

## Other build arguments

Shares the rest of `CALM analyze sft`'s arguments
(`-F`/`-U`/`-S`/`--lambda_x`/`--lambda_y`/`--gridsize`/`-C`/
`--Remove-TMD`/`--regularization`/`-W`/`-c`/`--loud`/`--replay`/
`--out-replay`) - see `CALM analyze sft --man` for those. `--Remove-TMD`
grid points are excluded from the area-per-lipid sums the same way they're
excluded elsewhere. `--rotate`/`--rotation-direction` are accepted for
consistency but have no effect here: composition and area-per-lipid are
computed independently per frame in the fit's own raw coordinate frame,
which doesn't need cross-frame rotational alignment. `--center` still
applies (needed by `--Remove-TMD`).

## Output

- `{frame}_lipid_fractions.npy` - shape `(n_species, 2, gridsize, gridsize)`
  (species x [upper, lower] x grid): the per-point composition map for
  every frame.
- `{frame}_area_per_lipid.npy` - shape `(n_species, 2, 2)` (species x
  [upper, lower] x [flat, curved]): that frame's own area-per-lipid,
  not yet averaged.
- `{frame}_lipid_counts.npy` - shape `(n_species, 2)` (species x [upper,
  lower]): the real residue count found for each species in each leaflet.
- `lipid_species.txt` - the `--lipids` list, in the fixed order indexing
  all three arrays above, written once.
- `area_per_lipid.csv` - written once, after every frame is processed: the
  trajectory mean of every frame's `area_per_lipid`/`lipid_counts`, one row
  per (leaflet, species): `leaflet,species,area_per_lipid_flat,
  area_per_lipid_curved,mean_count`.
- `dimensions.csv` - box size per frame, same convention as `analyze sft`.

Leaflet composition is assumed static across the trajectory (no
flip-flop) - a lipid's leaflet is decided fresh each frame from its own
position, but nothing tracks an individual lipid switching leaflets over
time.
