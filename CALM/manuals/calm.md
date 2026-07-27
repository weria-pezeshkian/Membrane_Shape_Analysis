# CALM

Calibrate and Analyze Lipid Membranes - geometric analysis of lipid
membranes (curvature, thickness) from MD trajectories, fit with a 2D
Fourier-series surface. Output feeds FreeDTS.

## Usage

```
CALM <module> [args...]
```

## Modules

- `analyze` - build the Fourier coefficient stack from a trajectory and/or
  run the full geometric analysis pipeline (thickness, curvature).
- `map` - turn analysis output into plots, videos, and index files for
  visualization.
- `link` - utility commands: write a leaflet index file, or export an
  MDAnalysis-readable trajectory of the fitted surface for VMD.
- `calibrate` - calibrate membrane material parameters (kappa, sigma) from
  a built Fourier coefficient stack. Has no submodules; run `CALM
  calibrate --man` directly for its full manual.

Run `CALM <module> --man` for the commands within each module, and
`CALM <module> <command> --man` for a specific command's full manual.
