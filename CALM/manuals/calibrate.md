# CALM calibrate

Membrane material parameter calibration, and the Fourier-fit `lambda_x`/
`lambda_y` advisor.

## Usage

```
CALM calibrate <command> [args...]
```

## Commands

- `dts` - calibrate membrane material parameters (bending rigidity `kappa`,
  `delta_kappa`, `delta_kappa_g`, spontaneous curvature `C_0`) from a
  previously built Fourier coefficient stack.
- `lambda` - sweep candidate `lambda_x`/`lambda_y` values against a
  structure (optionally sampling a trajectory) and recommend the finest
  resolution that would not trigger `CALM analyze`'s own
  underdetermined/low-redundancy/low-oversampling Fourier fit warnings. Run
  this before `CALM analyze sft`/`full`/`lipids`/`diffusion` if you aren't
  sure what `lambda_x`/`lambda_y` to use.

Run `CALM calibrate <command> --man` for a command's full manual.
