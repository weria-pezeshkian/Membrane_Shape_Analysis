# CALM (Calibrate and Analyze Lipid Membranes)

CALM analyzes the large-scale geometry of a lipid membrane from an MD
trajectory: it fits each leaflet's surface with a 2D Fourier series, then
derives curvature and thickness from that fit. Built for near-flat
membranes, e.g. to quantify the curvature a protein induces. Also
calibrates membrane material parameters (bending rigidity, tension) from
that fit, and exports output for
[FreeDTS](https://github.com/weria-pezeshkian/FreeDTS).

## Installation

### Prerequisites

Python >= 3.9.

### Install CALM

#### Directly from GitHub
```console
pip3 install git+https://github.com/weria-pezeshkian/Membrane_Shape_Analysis
```

#### From source
```console
git clone https://github.com/weria-pezeshkian/Membrane_Shape_Analysis
cd Membrane_Shape_Analysis
python3 -m venv venv && source venv/bin/activate  # not required, but convenient
pip3 install .
```

## Usage

CALM is organized into modules, each with its own subcommands:

```console
CALM -h
CALM {calibrate,analyze,link,map} -h
```

### GUI

```console
CALM-gui
```

A thin, cross-platform (Tkinter) interface over the exact same commands
above - every tab's form is generated directly from that command's own
CLI flags, so it never drifts out of sync with the CLI itself. It builds
and runs the real `CALM` command as a subprocess (the Output box mirrors
its terminal output live), never re-implements any argument logic. Each
tab also has a Manual button (opens that command's manual, rendered, in
your browser) and a Load replay button (repopulates the form from a
previously-written `*_calm_replay.log`). Needs Tkinter, which ships with
the official Windows/macOS Python installers; on Linux it's usually a
separate system package (e.g. `python3-tk` on Debian/Ubuntu).

Every command has a full manual: rendered on the command line via
`--man`, or readable directly as Markdown. Start from
[`CALM/manuals/calm.md`](CALM/manuals/calm.md) for the full module
overview, or jump straight to a command below.

- [`calibrate`](CALM/manuals/calibrate.md) - calibrate membrane material
  parameters (kappa, sigma) from a built Fourier coefficient stack.
- [`analyze`](CALM/manuals/analyze.md) - build the Fourier coefficient
  stack from a trajectory and run the geometric analysis pipeline
  (thickness, curvature).
  - [`analyze sft`](CALM/manuals/analyze_sft.md) - build and save the
    per-frame Fourier fit.
  - [`analyze full`](CALM/manuals/analyze_full.md) - run the full
    analysis pipeline, building the fit itself or reusing one from `sft`.
  - [`analyze lipids`](CALM/manuals/analyze_lipids.md) - per-species lipid
    composition, area-per-lipid, and preferred (spontaneous) curvature,
    computed frame by frame.
  - [`analyze diffusion`](CALM/manuals/analyze_diffusion.md) -
    curvature-aware lateral diffusion coefficient per lipid species and/or
    an MDAnalysis selection.
- [`link`](CALM/manuals/link.md) - utility commands supporting the
  pipeline.
  - [`link write_ndx`](CALM/manuals/link_write_ndx.md) - detect the two
    leaflets in a selection and write a GROMACS index file.
  - [`link vmd_xtc`](CALM/manuals/link_vmd_xtc.md) - export the
    fitted surface as a GRO + XTC trajectory for VMD.
  - [`link vmd_vectors`](CALM/manuals/link_vmd_vectors.md) - write VMD TCL
    scripts drawing principal-direction arrows in real space.
- [`map`](CALM/manuals/map.md) - turn analysis output into plots and
  videos.
  - [`map plot`](CALM/manuals/map_plot.md) - plot mean curvature or
    thickness, averaged over the trajectory.
  - [`map dynamic_plot`](CALM/manuals/map_dynamic_plot.md) - render a
    rolling-window-averaged curvature/thickness video.
  - [`map radial_plot`](CALM/manuals/map_radial_plot.md) - plot mean
    curvature radially averaged outward from the box center.
  - [`map lipids_plot`](CALM/manuals/map_lipids_plot.md) - render
    `analyze lipids` output: per-species, per-leaflet occupancy maps.
  - [`map diffusion_plot`](CALM/manuals/map_diffusion_plot.md) - render
    `analyze diffusion` output: per-species, per-leaflet MSD(tau) curves.

## Tests

```console
pip3 install .[dev]
pytest
```

See [`tests/README.md`](tests/README.md) for what's covered.

## Development

```console
ruff check CALM/
mypy CALM/ --ignore-missing-imports
```

To run these automatically before every commit:

```console
pre-commit install
```

Run it with your virtual environment active - the mypy hook runs against
your own installed environment (not an isolated one), since it needs
CALM's real dependencies (MDAnalysis, numpy, scipy, ...) available to type
against.

## License

GPLv3, see [LICENSE](LICENSE).
