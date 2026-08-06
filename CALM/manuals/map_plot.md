# CALM map plot

Plot mean curvature or thickness from a `CALM analyze full` output
directory, averaged over all saved frames.

## Usage

```
CALM map plot -i full_out_dir --mode mean -o mean.png
```

## Arguments

- `-i`, `--numpys_directory` - directory containing `CALM analyze full`
  output (`dimensions.csv`, `*_thickness.npy`, `*_mean_curvature.npy`,
  etc.).
- `--mode` (default `mean`) - one of `mean`, `gaussian`, `principal`,
  `thickness`.
- `-o`, `--outfile` (default `mean.png`) - path to save the figure to.
- `--minimum`, `--maximum` - fix the color scale's lower/upper bound.
  Both must be given together; otherwise the scale is computed from the
  data.
- `--vectors` - overlay principal-direction vectors (`--mode principal`
  only).
- `--histogram` - add a distribution strip beside each colorbar, showing
  how this plot's own data spreads across the color scale (bars), with
  the colorbar's own tick lines drawn across it.

## Averaging

Frames are averaged with a plain mean, not a NaN-aware one: a single NaN at
a given grid point in any one frame (from the fixed-circle mask under
`--rotate`, or from a hole flagged by `--Remove-TMD`) makes that point NaN
in the average too, rather than silently averaging over whichever frames
happened to have data there. This means an under-tuned `--Remove-TMD`
threshold at fit time can poison a widely-shared point across a long
trajectory; the fix belongs in that per-frame threshold (see `CALM analyze
sft --man`), not in how this command averages.
