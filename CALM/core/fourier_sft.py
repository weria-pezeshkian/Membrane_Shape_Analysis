import numpy as np
from pathlib import Path
from .fourier_build import calc_fourier

class SFT:
    def __init__(self):
        self.A_mn=None
        self.q_mn=None
        self.frame_indices=None

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

    def read(self,filename):
        file_path=Path(f"{filename}_Amn.npy")
        self.A_mn=np.load(file_path)
        file_path=Path(f"{filename}_qmn.npy")
        self.q_mn=np.load(file_path)

    def write(self,outname):
        file_path_Amn=Path(outname+"_Amn")
        file_path_qmn=Path(outname+"_qmn")
        np.save(arr=self.A_mn,file=file_path_Amn)
        np.save(arr=self.q_mn,file=file_path_qmn)

    def build(self,args,universe):
        calc_fourier(args,universe)
        self.A_mn,self.frame_indices=self.read_raw(args.out,"A_mn")
        self.q_mn,_=self.read_raw(args.out,"q_mn")

if __name__=="__main__":
    pass