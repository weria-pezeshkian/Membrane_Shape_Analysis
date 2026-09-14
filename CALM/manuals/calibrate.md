# CALM calibrate

Calibrate membrane material parameters (e.g. bending rigidity kappa,
tension sigma) from a previously built Fourier coefficient stack. Has no
submodules - unlike `analyze`/`map`/`link`, `CALM calibrate` takes its
flags directly.

This command is currently a scaffold: it parses arguments and loads the
SFT, but the physics itself is not yet implemented.

## Usage

```
CALM calibrate -i sft_dir --radius 5.0 -o calibration.json
```

## Arguments

- `-i`, `--sft` (required) - directory containing a previously built SFT
  (`Amn.npy`, `qmn.npy`, `dimensions.npy`), as written by `CALM analyze
  sft` or `CALM analyze full`.
- `--radius` (required) - calibration radius. Units and physical meaning
  are defined by the physics implementation, not yet fixed.
- `-o`, `--out` (required) - output file path.

## Notes

- Do not feed an SFT built with `--regularization` into a kappa/sigma
  calibration: Tikhonov regularization biases each mode's Anm toward zero
  in proportion to curvature, which would circularly contaminate any
  fluctuation-spectrum-based fit. Whether a given build used it is recorded
  in `regularized.npy` alongside the SFT's other output files.
