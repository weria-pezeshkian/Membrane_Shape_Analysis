# CALM map plot

Plot mean curvature or thickness from a `CALM analyze full` output
directory, averaged over all saved frames.

## Usage

```
CALM map plot -i full_out_dir --mode mean -o mean.png
```

## Arguments

- `-i`, `--numpys_directory` - directory containing `CALM analyze full`
  output (`Amn.npy`/`qmn.npy`/`dimensions.npy`, `*_thickness.npy`,
  `*_mean_curvature.npy`, etc.).
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
- `--alpha-TMD` - fade a point toward white by how much of it is real
  data instead of blanking it outright on a single missing frame (see
  Averaging below). Off by default.

## Averaging

By default (`--alpha-TMD` off), frames are averaged with a plain mean, not
a NaN-aware one: a single NaN at a given grid point in any one frame (from
the fixed-circle mask under `--rotate`, or from a hole flagged by
`--Remove-TMD`) makes that point NaN in the average too, rather than
silently averaging over whichever frames happened to have data there. This
means an under-tuned `--Remove-TMD` threshold at fit time can poison a
widely-shared point across a long trajectory; the fix belongs in that
per-frame threshold (see `CALM analyze sft --man`), not in how this
command averages.

`--alpha-TMD` changes this: a point's own color comes from the mean of
just the frames that DID have real data there (so a point missing in only
a few frames out of many still shows a real, representative value), and
that point is then faded toward white in proportion to how many of the
averaged frames were actually missing there - a point with data in 99 of
100 frames renders at 99% opacity, one with data in 40 of 100 at 40%
opacity, and so on. A point with no real data in ANY averaged frame still
renders fully white either way, on or off. This applies uniformly to
every panel this command draws (thickness, every curvature mode, every
leaflet), and to whichever NaN source produced the gap - a
`--Remove-TMD` hole or the `--rotate` fixed-circle mask alike, since the
fade only ever looks at how many frames were NaN, not why.

The fade is drawn as a flat, semi-transparent white layer on top of the
normal filled-contour colors, at the same grid resolution the underlying
data has (no smoothing) - it changes how a partially-covered point
*looks*, not the color scale or how `--minimum`/`--maximum` are computed.
