# CALM map

Turn `CALM analyze` output into plots and videos, and prepare rotation
tracking data for a protein-centered view.

## Usage

```
CALM map <command> [args...]
```

## Commands

- `rot_plot` - compute per-frame rotation vectors and protein positions,
  then render recentred/rotated curvature plots or a binned video.
- `plot` - plot mean curvature or thickness from a `CALM analyze full`
  output directory.
- `dynamic_plot` - render a rolling-window-averaged curvature/thickness
  video (GIF) from a `CALM analyze full` output directory.

Run `CALM map <command> --man` for a command's full manual.
