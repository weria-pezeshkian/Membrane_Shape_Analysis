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
- [`link`](CALM/manuals/link.md) - utility commands supporting the
  pipeline.
  - [`link write_ndx`](CALM/manuals/link_write_ndx.md) - detect the two
    leaflets in a selection and write a GROMACS index file.
  - [`link write_xtc`](CALM/manuals/link_write_xtc.md) - export the
    fitted surface as a GRO + XTC trajectory for VMD.
- [`map`](CALM/manuals/map.md) - turn analysis output into plots and
  videos.
  - [`map plot`](CALM/manuals/map_plot.md) - plot mean curvature or
    thickness, averaged over the trajectory.
  - [`map dynamic_plot`](CALM/manuals/map_dynamic_plot.md) - render a
    rolling-window-averaged curvature/thickness video.

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

## License

GPLv3, see [LICENSE](LICENSE).
