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
- `test_map_plot.py` - `map/plot.py`'s loading, hole-masking, and
  rendering.
- `test_dynamic_plot.py` - `map/dynamic_plot.py`'s rolling-window video.
- `test_get_vmd_visualization.py` - `get_vmd_visualisation`'s NaN-grid-point
  handling.
- `test_write_ndx.py` - `utilize/write_ndx.py`'s CLI/file-I/O wrapper.

## Conventions

- Synthetic `MDAnalysis.Universe` objects (built in-memory via
  `Universe.empty`), not real trajectory files - self-contained, no
  external data required.
- `tmp_path` for anything written to disk.
- Numerical thresholds (e.g. `--Remove-TMD`'s far-fallback multiplier) are
  validated empirically here, not assumed - see the sweeps in
  `test_fourier_build.py` for the pattern.
