from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from .plot import _frame_filtered_glob, _frame_number, _load_and_mask, draw

_MODE_PATTERNS = {
    "mean": "*_mean_curvature.npy",
    "gaussian": "*_gaussian_curvature.npy",
    "principal": "*_principal_curvatures.npy",
    "thickness": "*_thickness.npy",
}


def _available_frame_numbers(Dir: str, mode: str) -> list[int]:
    """Sorted, de-duplicated frame numbers with saved output for `mode`."""
    if not Dir.endswith("/"):
        Dir += "/"
    files = _frame_filtered_glob(Dir + _MODE_PATTERNS[mode], None)
    return sorted({_frame_number(f) for f in files})


def _windows(frame_numbers: list[int], window: int) -> list[list[int]]:
    """One `window`-sized window (list of frame numbers) per entry in `frame_numbers`, built by position.

    Position-based (rather than raw frame-number distance) so a
    strided/gapped sequence averages the N nearest computed frames. Edge
    entries share the same boundary window.
    """
    n = len(frame_numbers)
    window = max(1, min(int(window), n))
    half = window // 2
    result = []
    for i in range(n):
        lo = i - half
        hi = lo + window
        if lo < 0:
            lo, hi = 0, window
        elif hi > n:
            lo, hi = n - window, n
        result.append(frame_numbers[lo:hi])
    return result


def _windowed_minmax(Dir: str, mode: str, windows: list[list[int]]) -> tuple[float, float]:
    """Color-scale range spanning every rolling window's averaged data,
    excluding grid points NaN in the full-trajectory average.

    A grid point NaN'd by a hole mask in even one frame across the whole
    trajectory poisons the full-trajectory average at that point (plain
    np.mean, not nanmean - see _load_and_mask). A shorter window may not
    include that particular frame, so the point isn't NaN there and can
    still report a value - but one that's on the same unreliable point, so
    it's excluded from the scale here too.
    """
    if not Dir.endswith("/"):
        Dir += "/"
    try:
        sft = SFT.from_directory(Dir)
    except FileNotFoundError:
        sft = None

    pattern = _MODE_PATTERNS[mode]
    layer_sources = None if mode == "thickness" else (
        ["upper", "upper", "lower", "lower", "union", "union"] if mode == "principal" else ["upper", "lower", "union"]
    )

    full_avg = _load_and_mask(_frame_filtered_glob(Dir + pattern, None), pattern, Dir, sft, layer_sources)
    exclude = np.isnan(full_avg)

    Minimum, Maximum = np.inf, -np.inf
    for win in windows:
        arr = _load_and_mask(_frame_filtered_glob(Dir + pattern, win), pattern, Dir, sft, layer_sources)
        valid = arr[~exclude & ~np.isnan(arr)]
        if valid.size:
            Minimum = min(Minimum, float(valid.min()))
            Maximum = max(Maximum, float(valid.max()))

    if Minimum == Maximum:
        Maximum = Minimum + 1e-6
    return Minimum, Maximum


def draw_dynamic(
    Dir: str,
    mode: str = "mean",
    out_gif: str = "dynamic.gif",
    window: int = 5,
    spf: float = 0.2,
    minmax: list[float] | None = None,
    show_vectors: bool = True,
) -> None:
    """Render a GIF: one frame per trajectory frame, each a rolling-window average of nearby frames.

    Reuses `map/plot.py`'s `draw()` for all loading, rotation-awareness,
    hole-masking, and rendering - only the frame subset given to each call
    differs. The color scale is fixed once for the whole video (see
    `minmax`), not recomputed per frame, so it doesn't flicker. In `--mode
    mean` with thickness data present, the thickness subpanel gets its own
    fixed scale the same way.
    """
    if mode not in _MODE_PATTERNS:
        raise ValueError("mode must be 'mean', 'gaussian', 'principal', or 'thickness'")

    frame_numbers = _available_frame_numbers(Dir, mode)
    if not frame_numbers:
        raise FileNotFoundError(f"No frames found for mode '{mode}' in {Dir}")
    windows = _windows(frame_numbers, window)

    fixed_minmax = list(minmax) if minmax is not None else list(_windowed_minmax(Dir, mode, windows))

    fixed_thickness_minmax = None
    if mode == "mean" and _available_frame_numbers(Dir, "thickness"):
        fixed_thickness_minmax = list(_windowed_minmax(Dir, "thickness", windows))

    tmp_dir = tempfile.mkdtemp(prefix="dynamic_plot_")
    try:
        frame_paths = []
        for i, win in tqdm(enumerate(windows),total=len(windows)):
            frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            draw(
                Dir, mode=mode, minmax=fixed_minmax, thickness_minmax=fixed_thickness_minmax,
                filename=frame_path, show_vectors=show_vectors, frame_numbers=win,
            )
            frame_paths.append(frame_path)

        duration_ms = max(20, int(round(spf * 1000)))  # >= 20ms to avoid viewer clamping to 0
        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for p in frame_paths]
        images[0].save(out_gif, save_all=True, append_images=images[1:],
                      duration=duration_ms, loop=0, optimize=False, disposal=2)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def dynamic_plot(argv: list[str]) -> None:
    """CLI entry: render a rolling-window-averaged curvature/thickness video
    from a 'CALM analyze full' output directory."""
    parser = argparse.ArgumentParser(description="Render a rolling-window-averaged curvature/thickness video")
    parser.add_argument(
        '-i', '--numpys_directory', type=str, required=True,
        help="'CALM analyze full' output directory",
    )
    parser.add_argument(
        '--mode', choices=["mean", "gaussian", "principal", "thickness"], default="mean",
        help="quantity to plot (default: mean)",
    )
    parser.add_argument(
        '-o', '--outfile', type=str, default="dynamic.gif",
        help="output GIF path (default: dynamic.gif)",
    )
    parser.add_argument('--window', type=int, default=5, help="rolling window size in frames (default: 5)")
    parser.add_argument('--spf', type=float, default=0.2, help="seconds per video frame (default: 0.2)")
    parser.add_argument('--minimum', type=float, default=None, help="fix the color scale's lower bound")
    parser.add_argument('--maximum', type=float, default=None, help="fix the color scale's upper bound")
    parser.add_argument('--vectors', action="store_true", default=False, help="overlay principal-direction vectors")
    add_manual(parser, "map_dynamic_plot")

    ns = parser.parse_args(argv)
    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None

    draw_dynamic(
        Dir=ns.numpys_directory, mode=ns.mode, out_gif=ns.outfile,
        window=ns.window, spf=ns.spf, minmax=minmax, show_vectors=ns.vectors,
    )


if __name__ == "__main__":
    pass
