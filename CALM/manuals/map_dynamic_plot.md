# CALM map dynamic_plot

Render a GIF of mean curvature or thickness evolving over the trajectory,
from a `CALM analyze full` output directory. One video frame per trajectory
frame; each frame is a rolling-window average of nearby frames rather than
the whole-trajectory average `CALM map plot` produces, so transitions are
smoothed without erasing real changes over time.

## Usage

```
CALM map dynamic_plot -i full_out_dir -o dynamic.gif
```

## Arguments

- `-i`, `--numpys_directory` (required) - `CALM analyze full` output
  directory.
- `--mode` (default `mean`) - one of `mean`, `gaussian`, `principal`,
  `thickness`.
- `-o`, `--outfile` (default `dynamic.gif`) - output GIF path.
- `--window` (default 5) - rolling window size in frames. Each output video
  frame averages that many of the nearest frames that actually have saved
  output, centered on it (shrinking near the start/end of the trajectory).
  Window size is counted by position in the sequence of available frames,
  not by raw frame number, so a strided run (e.g. `--Step 5` when building
  the fit) still averages the N nearest computed frames. `--window 1`
  disables smoothing (each video frame is that one frame, unaveraged).
- `--spf` (default 0.2) - seconds per video frame.
- `--minimum`, `--maximum` - fix the color scale's lower/upper bound. Both
  must be given together; otherwise the scale spans every rolling window's
  own averaged data (not just the full-trajectory average, which is
  narrower) and is held fixed across every video frame so it doesn't
  flicker.
- `--vectors` - overlay principal-direction vectors (`--mode principal`
  only).

## Notes

- Rotation-awareness and `--Remove-TMD` hole-mask handling are identical to
  `CALM map plot`, applied per window.
- In `--mode mean` with thickness data present, the thickness panel's color
  scale is not fixed across frames (matplotlib auto-ranges it from each
  window's own data, the same as `CALM map plot` does for a single image) -
  its brightness may shift slightly between frames.
