# CALM analyze

Fit a 2D Fourier series to each leaflet's surface, frame by frame, and
optionally compute curvature and thickness from that fit.

## Usage

```
CALM analyze <command> [args...]
```

## Commands

- `sft` - build and save the per-frame Fourier coefficient stack
  (`Amn.npy`, `qmn.npy`, `dimensions.npy`) from a trajectory. This is the
  starting point for `CALM calibrate` and for `full` when reusing a
  precomputed fit.
- `full` - run the full geometric analysis pipeline (thickness, curvature)
  that `CALM map` can plot. Builds the fit itself unless `--sft` points to
  output already produced by `sft`.
- `lipids` - per-species lipid composition, area-per-lipid, and preferred
  (spontaneous) curvature, computed frame by frame from a live trajectory.
  Always re-fits the surfaces itself (no `--sft` reuse), since lipid
  identity needs real atoms.
- `diffusion` - curvature-aware lateral diffusion coefficient per lipid
  species and/or an arbitrary MDAnalysis selection, projecting each
  tracked point onto its own leaflet's fitted surface every frame before
  measuring displacement. Always re-fits the surfaces itself, for the
  same reason `lipids` does.

Run `CALM analyze <command> --man` for a command's full manual.
