# Tests

```console
pip3 install .[dev]
pytest
```

## Coverage

- `test_fourier_core.py` - `Fourier_Series_Function` (representation,
  derivatives) and `get_fourier_modes`.
- `test_fourier_fit.py` - `fit_coefficients`: coefficient recovery,
  regularization, and its underdetermined/low-redundancy/oversampling
  diagnostics.
- `test_fourier_build.py` - the `--Remove-TMD` hole-detection pipeline
  (`_tmd_threshold`, `_hole_mask_for_layer`, `_close_enclosed_gaps`,
  `_one_frame`) and the dynamic-leaflet-tracking path.
- `test_fourier_sft.py` - `SFT.write`/`SFT.from_directory` (the `--sft`
  load path).
- `test_leaflet.py` - the leaflet-detection/tracking algorithm
  (`core/leaflet.py`).
- `test_packing.py` - `median_multiple_threshold`.
- `test_curvature.py` - `shape_operator_curvatures`.
- `test_analyze_rotation.py` - the rotation path in `analyze/analyze.py`.
- `test_argument_parser.py` - `write_replay_file` and replay-checksum
  verification.
- `test_calibrate.py` - the `calibrate` CLI scaffold.
- `test_analyze_lipids.py` - per-species lipid-composition assignment
  (`_residue_centers`, `_assign_nearest_leaflet`, `_lipid_kernel_fractions`,
  `_true_surface_area`, `_one_lipid_frame`) and the trajectory-averaged
  `area_per_lipid.csv` (`_write_area_per_lipid_csv`).
- `test_map_plot.py` - `map/plot.py`'s loading, hole-masking, nematic
  direction averaging, sign alignment, and rendering.
- `test_dynamic_plot.py` - `map/dynamic_plot.py`'s rolling-window video.
- `test_radial_plot.py` - `map/radial_plot.py`'s radial binning and
  upper/lower-only rendering.
- `test_lipids_plot.py` - `map/lipids_plot.py`'s reserved entry point.
- `test_vmd_xtc.py` - `get_vmd_visualisation`'s NaN-grid-point handling,
  `vmd_xtc`'s rotation-TCL auto-detection, and `_trajectory_hole_union`'s
  per-trajectory hole combining.
- `test_vmd_vectors.py` - `vmd_vectors`'s static/dynamic principal-direction
  TCL scripts: arrow endpoints, `--which`/`--layer` filtering, `--Remove-TMD`
  hole exclusion, and `--scale`.
- `test_write_ndx.py` - `utilize/write_ndx.py`'s CLI/file-I/O wrapper.

## Conventions

- Synthetic `MDAnalysis.Universe` objects (built in-memory via
  `Universe.empty`), not real trajectory files - self-contained, no
  external data required.
- `tmp_path` for anything written to disk.
- Numerical thresholds (e.g. `--Remove-TMD`'s far-fallback multiplier) are
  validated empirically here, not assumed - see the sweeps in
  `test_fourier_build.py` for the pattern.
