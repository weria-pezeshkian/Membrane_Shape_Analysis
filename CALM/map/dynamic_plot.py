from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from .plot import _frame_filtered_glob, _frame_number, _load_and_mask, draw

_MODE_PATTERNS = {
    "mean": "*_mean_curvature.npy",
    "gaussian": "*_gaussian_curvature.npy",
    "principal": "*_principal_curvatures.npy",
    "thickness": "*_thickness.npy",
}


def _available_frame_numbers(Dir: str, mode: str) -> List[int]:
    """Sorted, de-duplicated frame numbers with saved output for `mode`."""
    if not Dir.endswith("/"):
        Dir += "/"
    files = _frame_filtered_glob(Dir + _MODE_PATTERNS[mode], None)
    return sorted({_frame_number(f) for f in files})


def _windows(frame_numbers: List[int], window: int) -> List[List[int]]:
    """One centered window (of frame numbers) per entry in `frame_numbers`.

    Windows are built by position in `frame_numbers`, not by numeric
    distance, so a strided/gapped sequence still averages the N nearest
    computed frames rather than however many happen to fall within a
    numeric range. Windows shrink at the sequence's edges. `window` is
    clamped to at least 1.
    """
    window = max(1, int(window))
    half = window // 2
    n = len(frame_numbers)
    return [frame_numbers[max(0, i - half):min(n, i + half + 1)] for i in range(n)]


def _auto_minmax(Dir: str, mode: str) -> Tuple[float, float]:
    """The same auto color-scale draw() derives when minmax is None, computed over the full (unwindowed) trajectory."""
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
    arr = _load_and_mask(_frame_filtered_glob(Dir + pattern, None), pattern, Dir, sft, layer_sources)
    valid = arr[~np.isnan(arr)]

    Minimum, Maximum = float(valid.min()), float(valid.max())
    if Minimum == Maximum:
        Maximum = Minimum + 1e-6
    return Minimum, Maximum


def draw_dynamic(
    Dir: str,
    mode: str = "mean",
    out_gif: str = "dynamic.gif",
    window: int = 5,
    spf: float = 0.2,
    minmax: Optional[List[float]] = None,
    show_vectors: bool = True,
) -> None:
    """Render a GIF: one frame per trajectory frame, each a rolling-window average of nearby frames.

    Reuses `map/plot.py`'s `draw()` for all loading, rotation-awareness,
    hole-masking, and rendering - only the frame subset given to each call
    differs. The color scale is fixed once for the whole video (see
    `minmax`), not recomputed per frame, so it doesn't flicker.
    """
    if mode not in _MODE_PATTERNS:
        raise ValueError("mode must be 'mean', 'gaussian', 'principal', or 'thickness'")

    fixed_minmax = list(minmax) if minmax is not None else list(_auto_minmax(Dir, mode))

    frame_numbers = _available_frame_numbers(Dir, mode)
    if not frame_numbers:
        raise FileNotFoundError(f"No frames found for mode '{mode}' in {Dir}")
    windows = _windows(frame_numbers, window)

    tmp_dir = tempfile.mkdtemp(prefix="dynamic_plot_")
    try:
        frame_paths = []
        for i, win in enumerate(windows):
            frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            draw(
                Dir, mode=mode, minmax=fixed_minmax, filename=frame_path,
                show_vectors=show_vectors, frame_numbers=win,
            )
            frame_paths.append(frame_path)

        duration_ms = max(20, int(round(spf * 1000)))  # >= 20ms to avoid viewer clamping to 0
        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
        images[0].save(out_gif, save_all=True, append_images=images[1:],
                      duration=duration_ms, loop=0, optimize=False, disposal=2)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def dynamic_plot(argv: List[str]) -> None:
    """CLI entry: render a rolling-window-averaged curvature/thickness video from a 'CALM analyze full' output directory."""
    parser = argparse.ArgumentParser(description="Render a rolling-window-averaged curvature/thickness video")
    parser.add_argument('-i', '--numpys_directory', type=str, required=True, help="'CALM analyze full' output directory")
    parser.add_argument('--mode', choices=["mean", "gaussian", "principal", "thickness"], default="mean", help="quantity to plot (default: mean)")
    parser.add_argument('-o', '--outfile', type=str, default="dynamic.gif", help="output GIF path (default: dynamic.gif)")
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
