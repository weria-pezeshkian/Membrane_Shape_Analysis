import numpy as np


def get_fourier_modes(box_size, lambda_x=None, lambda_y=None):
    Lx, Ly = box_size[:2]

    if lambda_x is None:
        Nx = 3
    else:
        lambda_x_A=lambda_x*10
        Nx=int(Lx / lambda_x_A)
        if Nx==0:
            Nx=1
            print("WARNING: lambda_x is too large, leading to an Nx of 0, corrected to Nx = 1")

    if lambda_y is None:
        Ny = 3
    else:
        lambda_y_A=lambda_y*10
        Ny=int(Ly / lambda_y_A)
        if Ny==0:
            Ny=1
            print("WARNING: lambda_y is too large, leading to an Nx of 0, corrected to Nx = 1")
    return Nx, Ny

class Fourier_Series_Function:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.q0 = 2 * np.pi / Lx  # Fundamental frequency in x
        self.p0 = 2 * np.pi / Ly  # Fundamental frequency in y

        # Fourier coefficients initialized to zeros
        self.Anm = np.zeros((2 * Nx + 1, 2 * Ny + 1))

    def setAnm(self,Anm):
        self.Anm=Anm

    def Z(self, x, y):
        g = 0
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                g += self.Anm[idx_i, idx_j] * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return g

    def Zx(self, x, y):
        gx = np.zeros_like(x)  # Initialize a 2D array for the derivative in x
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gx += self.Anm[idx_i, idx_j] * self.q0 * i * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) -
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gx

    def Zy(self, x, y):
        gy = np.zeros_like(y)  # Initialize a 2D array for the derivative in y
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gy += self.Anm[idx_i, idx_j] * self.p0 * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) -  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gy

    def Zxx(self, x, y):
        gxx = np.zeros_like(x)  # Initialize a 2D array for the second derivative in x
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxx -= self.Anm[idx_i, idx_j] * self.q0 ** 2 * i ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gxx

    def Zyy(self, x, y):
        gyy = np.zeros_like(y)  # Initialize a 2D array for the second derivative in y
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gyy -= self.Anm[idx_i, idx_j] * self.p0 ** 2 * j ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gyy

    def Zxy(self, x, y):
        gxy = np.zeros_like(y)  # Initialize a 2D array for the mixed second derivative Zxy
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxy -= self.Anm[idx_i, idx_j] * self.q0 * self.p0 * i * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +  # Vectorized
                    np.sin(self.q0 * i * x + self.p0 * j * y)    # Vectorized
                )
        return gxy


def average_coefficients(coff1, coff2):
    """Combine two leaflets' Anm coefficients into a middle-surface Anm."""
    return 0.5 * (coff1 + coff2)


if __name__=="__main__":
    pass
