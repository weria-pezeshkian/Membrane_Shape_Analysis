from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
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


def _windowed_minmax(
    Dir: str, mode: str, windows: list[list[int]], percentile: float = 0.0
) -> tuple[float, float]:
    """Color-scale range spanning every rolling window's averaged data,
    excluding grid points NaN in the full-trajectory average.

    A grid point NaN'd by a hole mask in even one frame across the whole
    trajectory poisons the full-trajectory average at that point (plain
    np.mean, not nanmean - see _load_and_mask). A shorter window may not
    include that particular frame, so the point isn't NaN there and can
    still report a value - but one that's on the same unreliable point, so
    it's excluded from the scale here too.

    `percentile` (0-100) trims that much off the two tails combined, split
    evenly between them, of every window's pooled values - e.g.
    `percentile=5` keeps the 2.5th-97.5th percentile range, so a single
    spurious point in a single window (e.g. the brentq root search - see
    TODO.md - landing on a physically implausible root) can't set the
    scale for the whole video by itself. The default, 0, is exactly the
    plain min/max, since the 0th/100th percentiles are the min/max
    themselves.
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

    pooled = []
    for win in windows:
        arr = _load_and_mask(_frame_filtered_glob(Dir + pattern, win), pattern, Dir, sft, layer_sources)
        valid = arr[~exclude & ~np.isnan(arr)]
        if valid.size:
            pooled.append(valid)
    if not pooled:
        return np.inf, -np.inf

    all_valid = np.concatenate(pooled)
    lo, hi = percentile / 2, 100 - percentile / 2
    Minimum, Maximum = float(np.percentile(all_valid, lo)), float(np.percentile(all_valid, hi))

    if Minimum == Maximum:
        Maximum = Minimum + 1e-6
    return Minimum, Maximum


def _find_ffmpeg() -> str:
    """Path to an ffmpeg executable: one already on PATH, or imageio-ffmpeg's bundled per-platform binary.

    Checking PATH first reuses an already-installed conda/apt/brew ffmpeg;
    imageio-ffmpeg's own binary makes this work on a machine with none
    installed too, with no setup step needed.
    """
    on_path = shutil.which("ffmpeg")
    if on_path is not None:
        return on_path

    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising its stderr in the exception message on a nonzero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")


def _stitch_gif_with_ffmpeg(tmp_dir: str, out_gif: str, duration_ms: int) -> None:
    """Assemble `tmp_dir`'s sequentially numbered `frame_000000.png`, `frame_000001.png`, ... into `out_gif` via ffmpeg.

    ffmpeg's own image-sequence demuxer discovers how many frames there
    are directly from the numbered files on disk. Streams them one at a
    time through ffmpeg's own pipeline, so peak memory stays flat as the
    frame count grows. The two-pass
    palettegen/paletteuse filters build a 256-color adaptive palette
    matching the quality of `_stitch_gif_in_memory`'s own Pillow-based
    quantization.
    """
    ffmpeg = _find_ffmpeg()
    fps = 1000.0 / duration_ms
    pattern = os.path.join(tmp_dir, "frame_%06d.png")
    palette_path = os.path.join(tmp_dir, "palette.png")
    out_gif = os.path.abspath(out_gif)

    _run_ffmpeg([
        ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
        "-vf", "palettegen=max_colors=256", palette_path,
    ])
    _run_ffmpeg([
        ffmpeg, "-y", "-framerate", str(fps), "-i", pattern, "-i", palette_path,
        "-lavfi", "paletteuse=dither=sierra2_4a", "-loop", "0", out_gif,
    ])


def _stitch_gif_in_memory(frame_paths: list[str], out_gif: str, duration_ms: int) -> None:
    """Assemble `frame_paths` into `out_gif` by opening every frame via Pillow at once.

    Peak memory scales with the frame count times each frame's own decoded
    size, since both this function's own `images` list and Pillow's
    internal GIF writer hold every frame simultaneously. Useful for short
    videos, or a machine with no ffmpeg available at all.
    """
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for p in frame_paths]
    images[0].save(out_gif, save_all=True, append_images=images[1:],
                  duration=duration_ms, loop=0, optimize=False, disposal=2)


def _draw_frame_isolated(
    Dir: str,
    mode: str,
    minmax: list[float],
    thickness_minmax: list[float] | None,
    filename: str,
    show_vectors: bool,
    frame_numbers: list[int],
    vector_frame: int,
    histogram: bool,
) -> None:
    """Render one frame via `draw()`, running it in its own child process.

    A process's entire memory is reclaimed by the OS the moment it exits,
    regardless of what matplotlib, its Agg backend, or any library beneath
    them are holding onto internally - the same guarantee `draw_dynamic`'s
    own loop needs across hundreds of large figures, without needing to
    find and fix each such internal retention individually. Uses the
    `spawn` context (a fresh interpreter) rather than `fork`, so a
    multi-threaded parent process's own locks can never carry into the
    child, and the same code path runs on every platform, including ones
    without `fork` at all.
    """
    process = multiprocessing.get_context("spawn").Process(
        target=draw,
        kwargs=dict(
            Dir=Dir, mode=mode, minmax=minmax, thickness_minmax=thickness_minmax,
            filename=filename, show_vectors=show_vectors, frame_numbers=frame_numbers,
            vector_frame=vector_frame, histogram=histogram,
        ),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Rendering frame '{filename}' failed in a subprocess (exit code {process.exitcode}).")


def draw_dynamic(
    Dir: str,
    mode: str = "mean",
    out_gif: str = "dynamic.gif",
    window: int = 5,
    spf: float = 0.2,
    minmax: list[float] | None = None,
    show_vectors: bool = True,
    in_memory: bool = False,
    histogram: bool = False,
    percentile: float = 0.0,
) -> None:
    """Render a GIF: one frame per trajectory frame, each a rolling-window average of nearby frames.

    Reuses `map/plot.py`'s `draw()` for all loading, rotation-awareness,
    hole-masking, and rendering - only the frame subset given to each call
    differs, via `_draw_frame_isolated` running each one in its own child
    process so peak memory stays bounded by a single frame's own render,
    however many hundreds of frames the video has. The color scale is
    fixed once for the whole video (see `minmax`, which bypasses
    `_windowed_minmax`/`percentile` entirely when given), not recomputed
    per frame, so it doesn't flicker. In `--mode mean` with thickness data
    present, the thickness subpanel gets its own fixed scale the same way.
    In `--mode principal`, the vector overlay (`vector_frame`) always
    shows that video frame's own instantaneous directions, while the
    curvature color background behind it keeps averaging over the window
    as usual.

    `in_memory` selects which of `_stitch_gif_with_ffmpeg` (the default)
    or `_stitch_gif_in_memory` assembles the per-frame PNGs into the final
    GIF - see their own docstrings for the memory tradeoff.

    `histogram` adds a live per-frame distribution strip beside each
    colorbar (see `draw`'s own docstring) - since the color scale is fixed
    for the whole video, this is the one part of each frame that keeps
    showing how the data itself is moving, frame to frame.

    `percentile` is passed straight to `_windowed_minmax` - see its own
    docstring.
    """
    if mode not in _MODE_PATTERNS:
        raise ValueError("mode must be 'mean', 'gaussian', 'principal', or 'thickness'")

    frame_numbers = _available_frame_numbers(Dir, mode)
    if not frame_numbers:
        raise FileNotFoundError(f"No frames found for mode '{mode}' in {Dir}")
    windows = _windows(frame_numbers, window)

    fixed_minmax = (
        list(minmax) if minmax is not None else list(_windowed_minmax(Dir, mode, windows, percentile))
    )

    fixed_thickness_minmax = None
    if mode == "mean" and _available_frame_numbers(Dir, "thickness"):
        fixed_thickness_minmax = list(_windowed_minmax(Dir, "thickness", windows, percentile))

    tmp_dir = tempfile.mkdtemp(prefix="dynamic_plot_")
    try:
        frame_paths = []
        for i, win in tqdm(enumerate(windows),total=len(windows)):
            frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            _draw_frame_isolated(
                Dir=Dir, mode=mode, minmax=fixed_minmax, thickness_minmax=fixed_thickness_minmax,
                filename=frame_path, show_vectors=show_vectors, frame_numbers=win,
                vector_frame=frame_numbers[i], histogram=histogram,
            )
            frame_paths.append(frame_path)

        duration_ms = max(20, int(round(spf * 1000)))  # >= 20ms to avoid viewer clamping to 0
        if in_memory:
            _stitch_gif_in_memory(frame_paths, out_gif, duration_ms)
        else:
            _stitch_gif_with_ffmpeg(tmp_dir, out_gif, duration_ms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_dynamic_plot_parser() -> argparse.ArgumentParser:
    """The 'CALM map dynamic_plot' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
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
    parser.add_argument(
        '--in-memory', action="store_true", default=False,
        help="assemble the GIF by holding every frame open via Pillow instead of streaming through ffmpeg "
             "(default: stream through ffmpeg - much lower peak memory on long videos)",
    )
    parser.add_argument(
        '--histogram', action="store_true", default=False,
        help="add a live per-frame distribution strip beside each colorbar",
    )
    parser.add_argument(
        '--percentile', type=float, default=0.0,
        help="trim this much (0-100) off the color scale's two tails combined, split evenly between them - "
             "e.g. 5 keeps the 2.5th-97.5th percentile range, guarding against a single spurious point setting "
             "the scale for the whole video (default: 0, the plain min/max)",
    )
    add_manual(parser, "map_dynamic_plot")
    return parser


def dynamic_plot(argv: list[str]) -> None:
    """CLI entry: render a rolling-window-averaged curvature/thickness video
    from a 'CALM analyze full' output directory."""
    parser = _build_dynamic_plot_parser()

    ns = parser.parse_args(argv)
    minmax = [ns.minimum, ns.maximum] if ns.minimum is not None and ns.maximum is not None else None

    draw_dynamic(
        Dir=ns.numpys_directory, mode=ns.mode, out_gif=ns.outfile,
        window=ns.window, spf=ns.spf, minmax=minmax, show_vectors=ns.vectors,
        in_memory=ns.in_memory, histogram=ns.histogram, percentile=ns.percentile,
    )


if __name__ == "__main__":
    pass
