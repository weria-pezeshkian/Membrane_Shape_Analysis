import numpy as np
from pathlib import Path
from .fourier_build import calc_fourier

class SFT:
    def __init__(self):
        self.A_mn=None
        self.q_mn=None
        self.dimensions=None
        self.frame_indices=None
        self.hole_mask=None  # optional: only set if --Remove-TMD was used to build

    def read_raw(self, read_dir, which):
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

    def build(self,args,universe):
        calc_fourier(args,universe)
        self.A_mn,self.frame_indices=self.read_raw(args.out,"A_mn")
        self.q_mn,_=self.read_raw(args.out,"q_mn")
        self.dimensions,_=self.read_raw(args.out,"dimensions")
        if any((Path(args.out) / "raw_sft").glob("*_hole_mask.npy")):
            self.hole_mask,_=self.read_raw(args.out,"hole_mask")

    def write(self,out_dir):
        """Save the consolidated SFT (Amn.npy, qmn.npy, dimensions.npy) into
        out_dir. dimensions.npy is [frame_index, Lx, Ly, Lz] per row - this is
        what 'CALM analyze full --sft <out_dir>' later reads back via
        from_directory(). holemask.npy is only written if --Remove-TMD was
        used to build (self.hole_mask is not None)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "Amn.npy", self.A_mn)
        np.save(out_dir / "qmn.npy", self.q_mn)
        combined_dimensions = np.column_stack([self.frame_indices, self.dimensions])
        np.save(out_dir / "dimensions.npy", combined_dimensions)
        if self.hole_mask is not None:
            np.save(out_dir / "holemask.npy", self.hole_mask)

    @classmethod
    def from_directory(cls, directory):
        """Load a previously built SFT from a directory containing Amn.npy,
        qmn.npy and dimensions.npy (as written by write()). Raises
        FileNotFoundError naming exactly which file(s) are missing."""
        directory = Path(directory)
        required = {
            "Amn.npy": directory / "Amn.npy",
            "qmn.npy": directory / "qmn.npy",
            "dimensions.npy": directory / "dimensions.npy",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cannot load a precomputed SFT from '{directory}': missing "
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

        holemask_path = directory / "holemask.npy"
        if holemask_path.exists():
            sft.hole_mask = np.load(holemask_path)

        return sft


if __name__=="__main__":
    pass
