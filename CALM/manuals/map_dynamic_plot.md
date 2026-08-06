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
- `--in-memory` - assemble the GIF by holding every frame open via Pillow
  instead of streaming through ffmpeg. The default (ffmpeg) keeps peak
  memory bounded by a single frame's own render, however many frames the
  video has; `--in-memory` is a fallback for a machine with no ffmpeg
  available at all (system or the bundled `imageio-ffmpeg`).
- `--histogram` - add a live per-frame distribution strip beside each
  colorbar, showing how this frame's own data spreads across the fixed
  color scale (bars) with the colorbar's own tick lines drawn across it.
  Since the scale itself is fixed for the whole video, this is the one
  part of each frame that keeps showing how the data is moving over time.
- `--percentile` (default 0) - trim this much (0-100) off the color
  scale's two tails combined, split evenly between them, when computing
  the fixed scale - e.g. `--percentile 5` keeps the 2.5th-97.5th
  percentile range instead of the plain min/max. Guards against a single
  spurious point in a single window (e.g. thickness's brentq root search
  landing on a physically implausible root - see TODO.md) setting the
  scale for the whole video by itself. The default, 0, is exactly the
  plain min/max.

## Notes

- Rotation-awareness and `--Remove-TMD` hole-mask handling are identical to
  `CALM map plot`, applied per window.
- In `--mode mean` with thickness data present, the thickness panel gets its
  own fixed color scale (spanning every window's thickness data), computed
  and held fixed the same way as the curvature scale - and `--percentile`
  applies to both scales the same way.
- In `--mode principal` with `--vectors`, the arrows always show that video
  frame's own instantaneous principal directions; the curvature color field
  behind them is window-averaged.
