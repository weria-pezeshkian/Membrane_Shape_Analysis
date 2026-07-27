# CALM map rot_plot

Compute per-frame rotation vectors and protein atom positions from a
trajectory, then render curvature plots or a video recentred on one
reference point and rotated to align with a second, using `CALM analyze
full` output.

## Usage

```
CALM map rot_plot -d numpys_dir -f traj.xtc -s structure.tpr \
    -p1 "protein and resid 10" -p2 "protein and resid 50" --np-dir vec_dir
```

## Trajectory and reference points

These arguments compute the O -> P rotation vector: each frame is recentred
on O's center of mass and rotated so P points toward the top-center of the
box.

- `-f`, `--trajectory` - path to the trajectory file.
- `-s`, `--structure` - path to the structure file.
- `-p1`, `--selection1` - MDAnalysis selection for reference point O.
- `-p2`, `--selection2` - MDAnalysis selection for reference point P.
- `-F`, `--From` (default 0) - first frame, inclusive.
- `-U`, `--Until` (default: end of trajectory) - last frame, exclusive.
- `-S`, `--Step` (default 1) - stride between frames.
- `--np-dir` (default: current directory) - directory to write the
  computed rotation vectors and protein positions to (`rotation_vectors_o.npy`,
  `rotation_vectors_p.npy`, `protein_atom_positions_rotation.npy`,
  `boxsize.npy`, and related index files).

## Plotting

- `-d`, `--numpys_directory` - directory containing `CALM analyze full`
  output (curvature/thickness/`Z_fitted` `.npy` files).
- `-o`, `--outfile` - path to save the output image/video to.
- `-v`, `--video` - one of `Upper`, `Lower`, `Both`. If given, render a
  binned GIF for that layer instead of the default static 2x2 mean-field
  image.
- `--dual` - in video mode, show recentred (left) and rotated (right)
  side by side; otherwise only the rotated view is shown.
- `--spf` (default 3.0) - seconds per frame in the output video.
- `--bins-video` - number of temporal bins for the video (each bin is
  averaged to one output frame); defaults to one bin per input frame.
- `--bins-image` - if given, render a single PNG with this many temporal
  bins for `--bins-layer` instead of producing the default output, then
  exit.
- `--bins-layer` (default `Upper`) - layer to use with `--bins-image`:
  one of `Upper`, `Lower`, `Both`, `Zfit`.
