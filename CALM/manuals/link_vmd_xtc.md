# CALM link vmd_xtc

Build a GRO + XTC trajectory of the fitted surface from `CALM analyze full`
output, so it can be loaded in VMD alongside (or instead of) the original
simulation trajectory.

## Usage

```
CALM link vmd_xtc -i full_out_dir -o vmd_out_dir
```

## Arguments

- `-i`, `--input` - directory containing `*_Z_fitted.npy` and
  `dimensions.csv`, as written by `CALM analyze full`.
- `-o`, `--output` - directory to write the generated files to.

## Output

- `first_frame.gro`, `trajectory.xtc`, `average_structure.gro` - one
  pseudo-atom per grid point per layer (Upper/Lower/Middle), positioned at
  the fitted surface height. Grid points that are `NaN` in any frame (e.g.
  outside the fixed circle used when `--rotate` was applied) are dropped
  from the atom count entirely, not written as NaN coordinates.
- If the input was built with `--Remove-TMD`, grid points with no real
  fitting support in any frame are kept but renamed from atom name `C` to
  `S`, so they can be filtered out in VMD with `not name S` without
  changing the atom count across frames.
- `rotate_and_select.tcl` - written only if the input was built with
  `--rotate`. A VMD script that, once sourced against the ORIGINAL
  trajectory (loaded with the same `-F`/`-S`/`-U` stride used to build the
  fit), rotates every atom per frame to match the alignment used during
  the fit, and restricts the display representation to the region that
  stays meaningful across every frame (a fixed-radius circle around each
  frame's own box center).

## See also

`CALM link vmd_vectors` builds on this command's output, drawing
principal-direction arrows against `average_structure.gro`/`trajectory.xtc`.
