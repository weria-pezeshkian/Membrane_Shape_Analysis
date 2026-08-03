# CALM map lipids_plot

Render `CALM analyze lipids` output (per-species composition/density
maps). This command is a reserved entry point - it is registered and
reachable, but rendering is not yet implemented, so running it raises
`NotImplementedError`.

## Usage

```
CALM map lipids_plot -i lipids_out_dir -o lipids.png
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze lipids` output
  directory.
- `-o`, `--outfile` (default `lipids.png`) - output image path.

## Input it will eventually read

- `lipid_species.txt` - the species names, in the fixed order indexing
  the array below.
- `{frame}_lipid_fractions.npy` - shape `(n_species, 2, gridsize,
  gridsize)` (species x [upper, lower] x grid), the per-point composition
  map for each frame.

See `CALM analyze lipids --man` for how these are produced.
