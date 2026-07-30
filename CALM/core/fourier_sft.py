from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from .fourier_build import calc_fourier


class SFT:
    """The per-frame Fourier coefficient stack: A_mn, q_mn, box dimensions, and optional hole mask."""

    def __init__(self) -> None:
        self.A_mn: np.ndarray | None = None
        self.q_mn: np.ndarray | None = None
        self.dimensions: np.ndarray | None = None
        self.frame_indices: np.ndarray | None = None
        self.hole_mask: np.ndarray | None = None  # set only if --Remove-TMD was used to build
        self.regularized: bool | None = None  # None if unknown (older SFTs written before this was tracked)

    def read_raw(self, read_dir: str, which: str) -> tuple[np.ndarray, np.ndarray]:
        """Load per-frame `{frame}_{which}.npy` files from `read_dir/raw_sft`, stacked in frame order."""
        dir_path = Path(read_dir) / "raw_sft"
        files = sorted(dir_path.glob(f"*_{which}.npy"))

        valid_files = []
        frame_indices = []

        for f in files:
            try:
                frame_idx = int(f.stem.split("_")[0])
            except ValueError:
                continue

            valid_files.append(f)
            frame_indices.append(frame_idx)

        first = np.load(valid_files[0])
        n_frames = len(valid_files)

        Arr = np.empty((n_frames, *first.shape), dtype=first.dtype)
        Arr[0] = first

        for i, f in enumerate(valid_files[1:], start=1):
            Arr[i] = np.load(f)

        return Arr, np.array(frame_indices, dtype=int)

    def build(self, args: argparse.Namespace, universe: mda.Universe) -> None:
        """Run `calc_fourier` and load its per-frame raw output into this SFT."""
        calc_fourier(args, universe)
        self.A_mn, self.frame_indices = self.read_raw(args.out, "A_mn")
        self.q_mn, _ = self.read_raw(args.out, "q_mn")
        self.dimensions, _ = self.read_raw(args.out, "dimensions")
        if any((Path(args.out) / "raw_sft").glob("*_hole_mask.npy")):
            self.hole_mask, _ = self.read_raw(args.out, "hole_mask")
        self.regularized = bool(args.regularize)

    def write(self, out_dir: str) -> None:
        """Save the consolidated SFT (Amn.npy, qmn.npy, dimensions.npy) into out_dir.

        dimensions.npy is [frame_index, Lx, Ly, Lz] per row, as read back by
        `from_directory`. holemask.npy and regularized.npy are written only
        if known (hole_mask/regularized are not None).
        """
        assert self.A_mn is not None
        assert self.q_mn is not None
        assert self.frame_indices is not None
        assert self.dimensions is not None
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(out_path / "Amn.npy", self.A_mn)
        np.save(out_path / "qmn.npy", self.q_mn)
        combined_dimensions = np.column_stack([self.frame_indices, self.dimensions])
        np.save(out_path / "dimensions.npy", combined_dimensions)
        if self.hole_mask is not None:
            np.save(out_path / "holemask.npy", self.hole_mask)
        if self.regularized is not None:
            np.save(out_path / "regularized.npy", np.array(self.regularized))

    @classmethod
    def from_directory(cls, directory: str) -> SFT:
        """Load a previously built SFT from a directory containing Amn.npy, qmn.npy, dimensions.npy.

        Raises FileNotFoundError naming exactly which required file(s) are missing.
        """
        dir_path = Path(directory)
        required = {
            "Amn.npy": dir_path / "Amn.npy",
            "qmn.npy": dir_path / "qmn.npy",
            "dimensions.npy": dir_path / "dimensions.npy",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot load a precomputed SFT from '{dir_path}': missing "
                f"{', '.join(missing)}. All three of Amn.npy, qmn.npy and "
                "dimensions.npy must be present together (they are written "
                "as a set by 'CALM analyze sft' / 'CALM analyze full')."
            )

        sft = cls()
        sft.A_mn = np.load(required["Amn.npy"])
        sft.q_mn = np.load(required["qmn.npy"])
        combined_dimensions = np.load(required["dimensions.npy"])
        sft.frame_indices = combined_dimensions[:, 0].astype(int)
        sft.dimensions = combined_dimensions[:, 1:]

        holemask_path = dir_path / "holemask.npy"
        if holemask_path.exists():
            sft.hole_mask = np.load(holemask_path)

        regularized_path = dir_path / "regularized.npy"
        if regularized_path.exists():
            sft.regularized = bool(np.load(regularized_path))

        return sft


if __name__ == "__main__":
    pass
