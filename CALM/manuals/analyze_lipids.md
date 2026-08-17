# CALM analyze lipids

Per-species lipid composition and area-per-lipid, computed frame by frame
from a live trajectory: fits both leaflet surfaces itself every frame, since
lipid identity (`resname`) only exists on real atoms, which a previously
built fit's `Amn`/`qmn` coefficients alone don't carry.

## Usage

```
CALM analyze lipids -f traj.xtc -s structure.tpr -o out_dir -n "name PO4" --lipids POPC GM1 SAPE24 [options]
```

## Required arguments

- `--lipids` - resnames to treat as distinct lipid species, e.g.
  `--lipids POPC GM1 SAPE24`. CALM has no way to decide what counts as "a
  lipid" in the system on its own.

  Each token is either a bare resname (automatic headgroup detection, see
  below) or `RESNAME:NAME1,NAME2,...` to give that species' own headgroup
  atom name(s) explicitly instead, e.g.
  `--lipids POPC:PO4 TCL1:PO41,PO42 SAPE24` - POPC and TCL1 use the named
  atoms directly; SAPE24 still auto-detects. Warns loudly if only some
  species in the run get explicit names, since mixing the two methods
  makes species-to-species comparisons methodologically inconsistent.
  Every resname (bare or named) must match at least one atom in
  `--structure`, checked before any fitting starts - a typo, a case
  mismatch, or a `.gro` file's 4-character resname truncation exits
  immediately with a descriptive error, rather than silently producing
  zero area/composition for that species everywhere in the output.
- `-f`, `--trajectory` - path to the trajectory file (e.g. `.xtc`).
- `-s`, `--structure` - path to the structure file. Must carry bonds for
  any species using automatic headgroup detection - a bare `.gro` has
  none; a GROMACS `.tpr` does.
- `-o`, `--out` - output directory for the saved arrays.
- `-n`, `--index` - leaflet selection used to fit the surfaces (same as
  `CALM analyze sft`): either a GROMACS-style `.ndx` file with `Upper`/
  `Lower` groups, or an MDAnalysis dynamic selection string.

## How composition is assigned

The `-n` selection (e.g. one phosphate atom per phospholipid) only exists
to fit the leaflet *surface* - it can miss lipid types with a different
headgroup structure entirely (e.g. a glycolipid with no phosphate). So
lipid composition uses its own, separate selection: for each `--lipids`
name, every atom of that resname is selected and grouped into residues.

Each residue's own reference position is its headgroup, found one of two
ways:

- **Automatic (default)**: identified structurally from the bond graph,
  forcefield-agnostic. Real cycles (e.g. MARTINI's glycerol backbone, which
  is bonded as a ring, not a chain) are first contracted to one node each.
  A node then counts as a real branch point ("hub") if it has three or
  more connections in that contracted tree, or if it's a ring bridging two
  or more of them - a ring with only *one* connection back into the rest
  of the molecule (dangling, e.g. a pendant sugar) is not by itself a hub;
  it's judged the same way as any other branch. Every leaf-to-hub path is
  a candidate branch; whatever's left over (the hubs themselves, plus
  anything strung between more than one of them) is the interior. Each
  branch's own average distance to the nearer fitted surface is compared
  against the interior's - farther away on average than the interior means
  "tail" and it's dropped, otherwise it's kept as headgroup. A lipid with
  more than one real hub (e.g. cardiolipin's two phosphate/glycerol
  junctions) contributes one competing point per hub to the Voronoi step
  below, rather than being collapsed to one averaged midpoint between
  them. A lipid with no hub anywhere (a plain single-tailed chain, no ring
  at all) has no dedicated interior, so it's instead split into two arms
  at its own midpoint and each is measured against the whole molecule's
  own average distance instead.

  This comparison can misjudge a dangling branch that extends further
  toward the water than the interior it's compared against (rather than
  toward the tails) - both just look like "farther than the interior" to
  the classifier. Known case: some MARTINI phosphatidylinositol models
  represent the inositol sugar as its own ring hanging off the
  glycerophosphate ring by a single bond, and the sugar ring can be
  dropped as if it were a tail. Give that species an explicit
  `--lipids RESNAME:NAME1,NAME2,...` if this matters for your system.
- **Named** (`RESNAME:NAME1,NAME2,...`): the given atom name(s) are used
  directly, one point per named atom, no bond-graph classification at all.

A residue's leaflet is decided fresh every frame, from that frame's own
headgroup position: whichever fitted surface it's closer to, or excluded
from both if it's implausibly far from both (same convention as
`--Remove-TMD`'s own far-fallback rule). This is independent of how `-n`
selects atoms for the surface *fit* itself - even a static `.ndx` file
only fixes which atoms define the Upper/Lower surfaces; it plays no part
in each lipid's own per-frame leaflet composition assignment. Nothing
carries a residue's leaflet assignment over from the previous frame
either, so a lipid that genuinely flip-flops across the trajectory is
picked up correctly without any special-casing - what isn't tracked is
any individual lipid's own identity across frames (composition output is
aggregate counts/fractions per leaflet, not a per-molecule trajectory).

At each grid point, hard nearest-neighbor assignment (a rasterized Voronoi
tessellation over every assigned species' headgroup points, projected onto
the same fitted surface) gives that point's full weight to whichever
lipid is closest - no blending, no bandwidth parameter.

Area-per-lipid, for each species, sums its grid cells' area over every
point assigned to it, then divides by the species' real residue count in
that leaflet. Two area conventions are reported: the flat (projected)
area, the usual literature quantity, and the true (curvature-corrected)
area, using `sqrt(1 + Zx^2 + Zy^2)` per cell from the freshly fit surface.

## Other build arguments

Shares the rest of `CALM analyze sft`'s arguments
(`-F`/`-U`/`-S`/`--lambda_x`/`--lambda_y`/`--gridsize`/`-C`/`--rotate`/
`--rotation-direction`/`--Remove-TMD`/`--regularization`/`-W`/`-c`/
`--loud`/`--replay`/`--out-replay`) - see `CALM analyze sft --man` for
those.

`--rotate` only ever affects the *saved spatial output*
(`{frame}_lipid_fractions.npy`/`{frame}_hole_mask.npy`), never
`area_per_lipid.csv`. `area_per_lipid.npy` is always built from each
frame's own raw, unrotated grid - it's a whole-leaflet sum, and which
physical position each grid index happens to query doesn't change that
sum. The saved spatial arrays are different: with `--rotate`, real lipid
and protein positions are never moved (same "only the query point is
transformed" mechanism `sft`/`full` use for curvature/thickness - see
`core/rotation.py`), but the *grid queried* to build them is rotationally
aligned to a fixed reference direction across the trajectory, so a real,
protein-relative composition pattern (e.g. one species enriching near a
particular face of a transmembrane region) survives frame-to-frame
averaging instead of washing out as the protein itself rotates. Use this
if you actually want to detect that kind of pattern; skip it if you only
want `area_per_lipid.csv`, which is unaffected either way. `map
lipids_plot` restricts its own rendering to the largest circle that
stays valid across every frame once `--rotate` is used, the same way
`map plot` does for rotated curvature/thickness.

`--Remove-TMD` grid points are excluded from the area-per-lipid sums the
same way they're excluded elsewhere, and (if `--rotate` is also given)
from the saved spatial output too. `--center` still applies (needed by
`--Remove-TMD` and by `--rotate` itself).

## Output

- `{frame}_lipid_fractions.npy` - shape `(n_species, 2, gridsize, gridsize)`
  (species x [upper, lower] x grid): the per-point composition map for
  every frame (1 for the assigned species at that point, 0 for the rest).
- `{frame}_area_per_lipid.npy` - shape `(n_species, 2, 2)` (species x
  [upper, lower] x [flat, curved]): that frame's own area-per-lipid,
  not yet averaged.
- `{frame}_lipid_counts.npy` - shape `(n_species, 2)` (species x [upper,
  lower]): the real residue count found for each species in each leaflet.
- `{frame}_hole_mask.npy` - shape `(2, gridsize, gridsize)` (upper,
  lower): written only if `--Remove-TMD` was given.
- `lipid_species.txt` - the `--lipids` list, in the fixed order indexing
  all the arrays above, written once.
- `rotated.npy` - a single boolean, whether `--rotate` was used, written
  once. Read by `map lipids_plot` to decide whether to restrict its own
  rendering to the fixed circle.
- `area_per_lipid.csv` - written once, after every frame is processed: the
  trajectory mean of every frame's `area_per_lipid`/`lipid_counts`, one row
  per (leaflet, species): `leaflet,species,area_per_lipid_flat,
  area_per_lipid_curved,mean_count`.
- `{frame}_dimensions.npy` - that frame's own `(Lx, Ly, Lz)`, one file per
  frame (not a single shared file - each worker process owns its own,
  avoiding the unlocked-concurrent-append problem a single shared file
  would have). Read by `map lipids_plot` for box size and, with
  `--rotate`, the fixed-circle radius.
