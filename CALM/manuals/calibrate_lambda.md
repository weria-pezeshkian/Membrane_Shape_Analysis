# CALM calibrate lambda

Sweep a grid of candidate `lambda_x`/`lambda_y` values against a structure
(optionally sampling a few frames of a trajectory) and recommend the
finest resolution that would not trigger `CALM analyze`'s own Fourier-fit
warnings. Choosing `lambda_x`/`lambda_y` by hand needs prior experience -
too small and the fit is underdetermined/overfit, too large and real
membrane features go unresolved - this estimates a good starting point
without running a real build first.

This is purely advisory: it never fits a real Fourier surface, only counts
atoms and compares them to the mode count a given lambda would require, so
it runs in seconds even on a large system.

## Usage

```
CALM calibrate lambda -s structure.tpr -n "name PO4" [-f trajectory.xtc] [options]
```

## Required arguments

- `-s`, `--structure` - path to the structure file (e.g. `.tpr`/`.gro`).
- One of:
  - `-n`, `--index` - an MDAnalysis selection string (e.g. `"name PO4"`)
    for per-frame dynamic leaflet detection.
  - `--index-file` - a GROMACS-style index (.ndx) file with two groups
    named `Upper` and `Lower` (e.g. as written by `CALM link write_ndx`),
    for static leaflet membership instead.

  If both are given, `--index-file` takes precedence and `-n`/`--index` is
  ignored, with a warning.

## Frame sampling

- `-f`, `--trajectory` - optional. Without it, only `-s`/`--structure`'s
  own single coordinate set is evaluated.
- `-F`, `--From` (default 0), `-U`, `--Until` (default: end), `-S`,
  `--Step` (default 1) - restrict the eligible-frame pool before sampling.
- `--n-frames` (default 5) - how many frames to randomly sample from that
  pool (clipped to the pool's own size if smaller). Ignored, with a
  printed note, if `-f`/`--trajectory` isn't given.
- `--seed` (default 0) - random seed for frame sampling; the same seed
  reproduces the same recommendation.

## Candidate lambda grid

- `--lambda-min` (default 1.0 nm), `--lambda-max` (default: derived from
  the sampled frames' own largest box dimension, converted to nm - sweeping
  past that point only adds candidates that collapse into the same
  coarsest-possible configuration), `--n-lambda` (default 20) - the default
  grid is `np.geomspace(lambda_min, lambda_max, n_lambda)`.
- `--lambda-candidates` - an explicit list of candidate lambda values (nm),
  overriding `--lambda-min`/`--lambda-max`/`--n-lambda`.

## Leaflet selection tuning

- `--min-balance` (default 0.6), `--margin` (default 2.0) - only used with
  a dynamic `-n` selection; same meaning and defaults as CALM analyze
  sft's own flags of the same name.

## Method

For each candidate lambda, `core/fourier_core.py::get_fourier_modes`
converts it (together with that frame's own box size) into a Fourier mode
count `Nx`/`Ny`, giving a coefficient count
`Length = (2*Nx+1)*(2*Ny+1)`. This is compared against `M`, the number of
real lipid atoms available to fit against (separately for the upper and
lower leaflet, for each sampled frame), as the oversampling ratio
`M/Length`. That ratio is classified into the exact same four tiers
`core/fourier_fit.py::fit_coefficients` already warns about during a real
build:

- `Length >= M` (oversampling below 1x) - underdetermined.
- `M < 3*Length` (1x-3x) - low redundancy.
- `M < 10*Length` (3x-10x) - low oversampling (the 10x cutoff is an
  empirical judgment call, not analytically derived).
- otherwise (10x or above) - safe.

A candidate's own score is the *worst case* of this ratio - the minimum -
across every sampled frame and both leaflets, not an average: a lambda
choice should hold up reliably everywhere it might be used, not just
happen to look good on one lucky frame or leaflet.

Many different lambda values floor (via `int()`) to the identical `Nx`/`Ny`
on every sampled frame - those are numerically indistinguishable to
evaluate, so they are grouped first and only a single representative value
per group (the one with the fewest decimal digits, ties broken by the
smallest value) is scored and reported; the printed table shows one row
per distinct configuration, not one row per raw candidate.

Each row also reports `lipids/mode`: the same worst-case `M`, but divided
by `Nx*Ny` instead of `Length` - the average number of fit atoms within
one finest-resolved real-space patch (`Lx/Nx` by `Ly/Ny`). This is a
second, independent check on the same physical risk the 10x tier is
guarding against (real per-lipid thermal protrusions being mistaken for
membrane curvature), but grounded in an actual measurement rather than a
coefficient-count proxy: Lindahl & Edholm (2000, Biophys. J. 79:426-433)
found that, fitting a DPPC bilayer's own spectral decomposition, modes
involving more than ~50 lipids showed genuine collective (undulatory)
behavior, while modes involving fewer than ~25 lipids were dominated by
independent per-lipid protrusions instead. 50 is used here as the
conservative (genuine-undulation) end of that crossover band.

The recommendation is the smallest lambda (finest resolution, most Fourier
modes) among the distinct configurations that clear *both* checks: tier
`safe` and `lipids/mode >= 50`. A candidate can be `safe` by the M/Length
proxy without actually clearing the directly-measured lipid-count
threshold (and vice versa for some configurations) - requiring both means
a candidate that only clears one of them is passed over for a slightly
coarser one that clears both. If no candidate clears both, the tool still
always returns an answer: it falls back to whichever candidate has the
best worst-case ratio, and says so explicitly ("best available (no
candidate clears both the safe tier and the 50-lipids-per-mode
threshold)") rather than presenting a fallback pick with the same
confidence as a genuine recommendation.
