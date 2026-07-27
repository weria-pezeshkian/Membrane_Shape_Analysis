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

Run `CALM analyze <command> --man` for a command's full manual.
