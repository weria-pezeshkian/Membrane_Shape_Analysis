# CALM map

Turn `CALM analyze` output into plots and videos.

## Usage

```
CALM map <command> [args...]
```

## Commands

- `plot` - plot mean curvature or thickness from a `CALM analyze full`
  output directory.
- `dynamic_plot` - render a rolling-window-averaged curvature/thickness
  video (GIF) from a `CALM analyze full` output directory.
- `radial_plot` - plot mean curvature (upper/lower) radially averaged
  outward from the box center, from a `CALM analyze full` output
  directory.
- `lipids_plot` - render `CALM analyze lipids` output (composition/density
  maps). Reserved entry point; rendering isn't implemented yet.

Run `CALM map <command> --man` for a command's full manual.
