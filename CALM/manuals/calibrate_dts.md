# CALM calibrate dts

Calibrate membrane material parameters (bending rigidity `kappa`,
`delta_kappa`, `delta_kappa_g`, spontaneous curvature `C_0`) from a
previously built Fourier coefficient stack, by fitting the fluctuation
spectrum of `Anm` across a frame range.

This command's physics is preliminary: it runs end to end and writes real
numbers, but its output units/values haven't yet been independently
validated against a known system - treat results with care until that's
done.

## Usage

```
CALM calibrate dts -i sft_dir --radius 5.0 --startframe 0 --endframe 5000 -o calibration
```

## Arguments

- `-i`, `--sft` (required) - directory containing a previously built SFT
  (`Amn.npy`, `qmn.npy`, `dimensions.npy`), as written by CALM analyze
  sft or CALM analyze full.
- `--radius` (required) - calibration radius: the area under the protein
  (or region of interest) that the spectral integrals are evaluated over.
- `--startframe`, `--endframe` (both required) - the frame range (as
  indices into the SFT's own stored frames, not raw trajectory frame
  numbers) to average the per-mode integrals over.
- `-o`, `--out` (required) - output file path. A `.npy` extension is
  appended if not already present; a sibling human-readable `.txt` file
  (same basename) is also written.
- `--surface` (default `top`) - which of the SFT's three stored surfaces
  (`top`, `middle`, `bottom`) to calibrate against.

## Notes

- Do not feed an SFT built with `--regularization` into this: Tikhonov
  regularization biases each mode's Anm toward zero in proportion to
  curvature, which would circularly contaminate this fluctuation-spectrum
  fit. Whether a given build used it is recorded in `regularized.npy`
  alongside the SFT's other output files, and checked automatically before
  calibrating.
