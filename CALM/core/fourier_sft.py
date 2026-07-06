import numpy as np
from pathlib import Path
from .fourier_build import calc

class SFT:
    def __init__(self):
        A_mn=None
        q_mn=None

    def read_raw(self,read_dir,which):
        dir_path = Path(read_dir)
        files = sorted(dir_path.glob(f"*_{which}.npy"))

        first = np.load(files[0])
        n_frames = len(files)

        Arr = np.empty((n_frames, *first.shape), dtype=first.dtype)
        Arr[0] = first

        for i, f in enumerate(files[1:], start=1):
            Arr[i] = np.load(f)

        return Arr

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

    def build(self,out_dir, u, ndx, From=0, Until=None, Step=1,Workers=1, remove_protein=False, Nx=2,Ny=2,sqrt_n_atoms=100):
        calc(out_dir, u, ndx, From, Until, Step,Workers, remove_protein, Nx,Ny,sqrt_n_atoms)
        self.A_mn=self.read_raw(out_dir,"A_mn")
        self.q_mn=self.read_raw(out_dir,"q_mn")

if __name__=="__main__":
    pass