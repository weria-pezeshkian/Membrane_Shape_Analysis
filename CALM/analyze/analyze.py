import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from ..core.fourier_core import Fourier_Series_Function, get_fourier_modes
from ..core.curvature import shape_operator_curvatures
from ..core.rotation import recover_rotation_angle

def circle_cutter(arr, dimensions):
    """NaN-mask arr outside the largest circle centered on the box that fits
    within it. Spatial axes are always axes 1 and 2 (arr.shape[1:3]); axis 0
    is a layer/component axis (e.g. upper/lower/middle), and any further
    trailing axes (e.g. the 2 vector components of principal_dirs) are left
    untouched by the mask."""
    Lx, Ly = dimensions[:2]

    if arr.ndim >= 3:
        n, m = arr.shape[1:3]
    else:
        n, m = arr.shape

    x = (np.arange(n) + 0.5) * Lx / n - Lx / 2.0
    y = (np.arange(m) + 0.5) * Ly / m - Ly / 2.0

    X, Y = np.meshgrid(x, y, indexing="ij")

    radius = min(Lx, Ly) / 2.0
    mask = X**2 + Y**2 <= radius**2

    if arr.ndim >= 3:
        arr[:, ~mask] = np.nan
    else:
        arr[~mask] = np.nan

    return arr

def periodic_gradient(Z, dx, dy, periodic_x=True, periodic_y=True):
    # dZ/dx: axis 1
    if periodic_x:
        dz_dx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * dx)
    else:
        dz_dx = np.gradient(Z, dx, axis=1)

    # dZ/dy: axis 0
    if periodic_y:
        dz_dy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * dy)
    else:
        dz_dy = np.gradient(Z, dy, axis=0)

    return dz_dy, dz_dx

def f(t, interp, mx, my, mz, nx, ny, nz,Lx,Ly):
    xq = mx + t * nx
    yq = my + t * ny
    zq = mz + t * nz

    xq = np.mod(xq, Lx)
    yq = np.mod(yq, Ly)

    return zq - interp(yq, xq, grid=False)[()]

def _rotated_grid(X, Y, cx, cy, angle):
    """Coordinates to evaluate the (unrotated-as-fit) surface at, so the
    result at output grid point (X, Y) is what the surface would look like
    after rotating it by `angle` around (cx, cy)."""
    dx = X - cx
    dy = Y - cy
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    x_old = cos_a * dx - sin_a * dy + cx
    y_old = sin_a * dx + cos_a * dy + cy
    return x_old, y_old

def _rotate_direction_vectors(vecs, angle):
    """Rotate an array of 2D tangent-plane direction vectors (shape (...,2))
    by `angle`, so they're expressed in the rotated/aligned frame's basis
    instead of the original (unrotated-as-fit) frame's basis."""
    vx = vecs[..., 0]
    vy = vecs[..., 1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.stack([cos_a * vx - sin_a * vy, sin_a * vx + cos_a * vy], axis=-1)

def analysis(universe, sft, methods, args=None):
    rotated=args.rotate


    for i,frame in tqdm(enumerate(sft.frame_indices),total=len(sft.frame_indices)):
        if universe is not None:
            try:
                universe.trajectory[frame]
            except IndexError:
                universe.trajectory[i]
            dimensions = universe.dimensions[:3]
        else:
            # No trajectory available (SFT was loaded via --sft): use the box
            # size captured per-frame at build time instead.
            dimensions = sft.dimensions[i]
        x = np.linspace(0, dimensions[:3][0], args.gridsize, endpoint=False)
        y = np.linspace(0, dimensions[:3][1], args.gridsize, endpoint=False)
        X, Y = np.meshgrid(x, y)

        num_digits = len(str(max(sft.frame_indices)))
        Nx,Ny=get_fourier_modes(dimensions[:3],args.lambda_x,args.lambda_y)
        fourier0=Fourier_Series_Function(dimensions[:3][0],dimensions[:3][1],Nx,Ny)
        fourier0.setAnm(sft.A_mn[i,0])
        fourier1=Fourier_Series_Function(dimensions[:3][0],dimensions[:3][1],Nx,Ny)
        fourier1.setAnm(sft.A_mn[i,1])
        fourier2=Fourier_Series_Function(dimensions[:3][0],dimensions[:3][1],Nx,Ny)
        fourier2.setAnm(sft.A_mn[i,2])

        # Curvature/height values are rotation-invariant scalars - rather than
        # rotating Anm (would need refitting), evaluate the as-fit surface at
        # the corresponding *unrotated* coordinate for each output grid point.
        # Only direction vectors (dirs1/dirs2 below) need rotating afterward.
        if rotated:
            theta = recover_rotation_angle(sft.q_mn[i], dimensions[:3][0], dimensions[:3][1], Nx, Ny)
            X_eval, Y_eval = _rotated_grid(X, Y, dimensions[:3][0] / 2.0, dimensions[:3][1] / 2.0, theta)
        else:
            theta = 0.0
            X_eval, Y_eval = X, Y

        if "Z_fitted" in methods or "thickness" in methods:
            # Z() is already vectorized over its (x, y) arguments (it loops over
            # Fourier modes, not grid points) - call it once on the full grid
            # instead of once per point. ~100x faster, identical result.
            # Evaluated at (X_eval, Y_eval) rather than (X, Y): height is a
            # rotation-invariant scalar, so this correctly produces "the
            # rotated surface's height at output position (X, Y)" (see
            # _rotated_grid) without needing to refit Anm.
            Z_fitted_1 = fourier0.Z(X_eval, Y_eval)
            Z_fitted_2 = fourier1.Z(X_eval, Y_eval)
            Z_fitted_vmd = (Z_fitted_1 + Z_fitted_2) / 2
            Z_fitted=np.stack([Z_fitted_1, Z_fitted_2, Z_fitted_vmd], axis=0)

            if rotated:
                Z_fitted=circle_cutter(Z_fitted,dimensions)

            if "Z_fitted" in methods: 
                np.save(f"{args.out}/{frame:0{num_digits}d}_Z_fitted.npy", Z_fitted / 10)

            if "thickness" in methods:
                try:
                    # Interpolators for intersections
                    interp_upper = RectBivariateSpline(Y[:, 0], X[0, :],  Z_fitted_1)
                    interp_lower = RectBivariateSpline(Y[:, 0], X[0, :],  Z_fitted_2)

                    # ---- Thickness calculation ---- #
                    dx = dimensions[:3][0] / (X.shape[1] - 1)
                    dy = dimensions[:3][1] / (Y.shape[0] - 1)
                    dz_dy, dz_dx = periodic_gradient(Z_fitted_vmd, dx, dy, periodic_x=True, periodic_y=True)

                    Nx_arr, Ny_arr = -dz_dx, -dz_dy
                    Nz_arr = np.ones_like(Z_fitted_vmd)
                    N = np.stack((Nx_arr, Ny_arr, Nz_arr), axis=-1)
                    N /= np.linalg.norm(N, axis=-1, keepdims=True)

                    thickness_map = np.full_like(Z_fitted_vmd, np.nan, dtype=float)

                    t_max=np.nanmax(np.abs(Z_fitted_1 - Z_fitted_2)) * 2
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            x0, y0, z0 = X[i,j], Y[i,j], Z_fitted_vmd[i, j]
                            nvecx,nvecy, nvecz = N[i,j]

                            l1 = brentq(f,0.0,t_max,args=(interp_upper, x0,y0,z0, nvecx, nvecy, nvecz,dimensions[:3][0],dimensions[:3][1]))
                            l2 = brentq(f,-t_max,0.0,args=(interp_lower, x0,y0,z0, nvecx, nvecy, nvecz,dimensions[:3][0],dimensions[:3][1]))

                            #l1_map[i, j] = l1
                            #l2_map[i, j] = l2
                            thickness_map[i, j] = l1 - l2

                    if rotated:
                        thickness_map=circle_cutter(thickness_map,dimensions)

                    np.save(f"{args.out}/{frame:0{num_digits}d}_thickness.npy", thickness_map / 10)
                except ValueError:
                    print("Thickness could not be calculated. That could be an indication that the curvature is too high or that lambdas are too small.")

        if any(m in methods for m in ("mean", "gaussian", "principal", "principal_directions")):
            # H, K, k1, k2 are rotation-invariant scalars: evaluating at
            # (X_eval, Y_eval) directly gives the correct rotated-surface
            # value at output position (X, Y). dirs1/dirs2 are genuine
            # direction vectors though, so they get an extra +theta rotation
            # afterward to express them in the rotated/aligned frame's basis.
            #Upper leaflet
            H1, K1, k1_1, k2_1, dirs1_1, dirs2_1 = shape_operator_curvatures(fourier0, X_eval, Y_eval)
            #Lower leaflet
            H2, K2, k1_2, k2_2, dirs1_2, dirs2_2 = shape_operator_curvatures(fourier1, X_eval, Y_eval)
            #Middle surface
            Hmid, Kmid, k1_mid, k2_mid, dirs1_mid, dirs2_mid = shape_operator_curvatures(fourier2, X_eval, Y_eval)

            if theta != 0.0:
                dirs1_1 = _rotate_direction_vectors(dirs1_1, theta)
                dirs2_1 = _rotate_direction_vectors(dirs2_1, theta)
                dirs1_2 = _rotate_direction_vectors(dirs1_2, theta)
                dirs2_2 = _rotate_direction_vectors(dirs2_2, theta)
                dirs1_mid = _rotate_direction_vectors(dirs1_mid, theta)
                dirs2_mid = _rotate_direction_vectors(dirs2_mid, theta)

            if "mean" in methods:
                mean_curvature = np.stack([H1, H2, Hmid], axis=0) * 10
                if rotated:
                    mean_curvature = circle_cutter(mean_curvature, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_mean_curvature.npy", mean_curvature)

            if "gaussian" in methods:
                gaussian_curvature = np.stack([K1, K2, Kmid], axis=0) * 10
                if rotated:
                    gaussian_curvature = circle_cutter(gaussian_curvature, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_gaussian_curvature.npy", gaussian_curvature)

            if "principal" in methods:
                principal_curvatures = np.stack([k1_1, k2_1, k1_2, k2_2, k1_mid, k2_mid], axis=0) * 10
                if rotated:
                    principal_curvatures = circle_cutter(principal_curvatures, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_principal_curvatures.npy", principal_curvatures)

            if "principal_directions" in methods:
                principal_dirs = np.stack([dirs1_1, dirs2_1, dirs1_2, dirs2_2, dirs1_mid, dirs2_mid], axis=0)
                if rotated:
                    principal_dirs = circle_cutter(principal_dirs, dimensions)
                np.save(f"{args.out}/{frame:0{num_digits}d}_principal_dirs.npy", principal_dirs)



if __name__=="__main__":
    pass