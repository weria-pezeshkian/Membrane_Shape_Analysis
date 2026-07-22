import numpy as np


def fit_coefficients(Data_3M, Lx, Ly, Nx, Ny):
    """Solve for the least-squares Fourier coefficients (Anm) that best
    reproduce z-values at the given (x, y) points.

    Data_3M must have shape (3, M): rows are x, y, z.
    Returns an Anm array of shape (2*Nx+1, 2*Ny+1), ready for
    Fourier_Series_Function.setAnm.
    """
    if Data_3M.shape[0] != 3:
        raise ValueError("Data_3M must have shape (3, M).")

    q0 = 2 * np.pi / Lx
    p0 = 2 * np.pi / Ly

    M = Data_3M.shape[1]
    Length = (2 * Nx + 1) * (2 * Ny + 1)

    A = np.zeros((M, Length))
    b = Data_3M[2, :]

    index = 0
    for i in range(-Nx, Nx + 1):
        for j in range(-Ny, Ny + 1):
            for k in range(M):
                x, y = Data_3M[0, k], Data_3M[1, k]
                A[k, index] = (
                    np.cos(q0 * i * x + p0 * j * y) +
                    np.sin(q0 * i * x + p0 * j * y)
                )
            index += 1

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return coeffs.reshape((2 * Nx + 1, 2 * Ny + 1))


if __name__=="__main__":
    pass
