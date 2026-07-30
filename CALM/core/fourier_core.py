from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def get_fourier_modes(
    box_size: Sequence[float],
    lambda_x: float | None = None,
    lambda_y: float | None = None,
    diagnostics: list[tuple[str, str]] | None = None,
) -> tuple[int, int]:
    """Mode counts (Nx, Ny) for a box, from wavelength scales lambda_x/lambda_y (nm).

    Nx = int(Lx / (lambda_x * 10)), floored to 1; Ny likewise. Defaults to
    (3, 3) if a wavelength isn't given.

    Runs inside a per-frame worker process (via `_one_frame`). When
    `diagnostics` is given, a `(level, message)` pair is appended to it
    whenever Nx or Ny had to be floored to 1; the caller collects these
    across frames and logs them once back in the main process, keeping
    every write to the replay log single-process.
    """
    Lx, Ly = box_size[:2]

    if lambda_x is None:
        Nx = 3
    else:
        lambda_x_A = lambda_x * 10
        Nx = int(Lx / lambda_x_A)
        if Nx == 0:
            Nx = 1
            if diagnostics is not None:
                diagnostics.append((
                    "warning", "lambda_x is too large for this box (Nx would be 0) - corrected to Nx=1"
                ))

    if lambda_y is None:
        Ny = 3
    else:
        lambda_y_A = lambda_y * 10
        Ny = int(Ly / lambda_y_A)
        if Ny == 0:
            Ny = 1
            if diagnostics is not None:
                diagnostics.append((
                    "warning", "lambda_y is too large for this box (Ny would be 0) - corrected to Ny=1"
                ))
    return Nx, Ny


class Fourier_Series_Function:
    """A 2D Fourier series surface Z(x, y) = sum_ij Anm[i,j] * (cos + sin)(q0*i*x + p0*j*y)."""

    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int) -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.q0 = 2 * np.pi / Lx
        self.p0 = 2 * np.pi / Ly
        self.Anm = np.zeros((2 * Nx + 1, 2 * Ny + 1))

    def setAnm(self, Anm: np.ndarray) -> None:
        self.Anm = Anm

    def Z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # np.cos/np.sin always produce float64 terms, so the accumulator is
        # forced to dtype=float here to match, independent of x's own dtype.
        g = np.zeros_like(x, dtype=float)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                g += self.Anm[idx_i, idx_j] * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return g

    def Zx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(x)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gx += self.Anm[idx_i, idx_j] * self.q0 * i * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) -
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gx

    def Zy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gy = np.zeros_like(y)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gy += self.Anm[idx_i, idx_j] * self.p0 * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) -
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gy

    def Zxx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gxx = np.zeros_like(x)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxx -= self.Anm[idx_i, idx_j] * self.q0 ** 2 * i ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gxx

    def Zyy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gyy = np.zeros_like(y)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gyy -= self.Anm[idx_i, idx_j] * self.p0 ** 2 * j ** 2 * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gyy

    def Zxy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gxy = np.zeros_like(y)
        for i in range(-self.Nx, self.Nx + 1):
            for j in range(-self.Ny, self.Ny + 1):
                idx_i = i + self.Nx
                idx_j = j + self.Ny
                gxy -= self.Anm[idx_i, idx_j] * self.q0 * self.p0 * i * j * (
                    np.cos(self.q0 * i * x + self.p0 * j * y) +
                    np.sin(self.q0 * i * x + self.p0 * j * y)
                )
        return gxy


def average_coefficients(coff1: np.ndarray, coff2: np.ndarray) -> np.ndarray:
    """Combine two leaflets' Anm coefficients into a middle-surface Anm."""
    return 0.5 * (coff1 + coff2)


if __name__ == "__main__":
    pass
