# CALM link

Utility commands that are not part of the core fit/analyze pipeline but
support it: writing a leaflet index file, and exporting the fitted surface
as an MDAnalysis-readable trajectory for VMD.

## Usage

```
CALM link <command> [args...]
```

## Commands

- `write_ndx` - detect the two leaflets in a selection and write a GROMACS
  index file (`Upper`/`Lower` groups).
- `write_xtc` - build a GRO + XTC trajectory of the fitted surface from
  `CALM analyze full` output, for visualization in VMD.

Run `CALM link <command> --man` for a command's full manual.
