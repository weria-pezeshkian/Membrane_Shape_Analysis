from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..core import argument_parser as arg_helper
from ..core.curvature import shape_operator_curvatures
from ..core.fourier_build import (
    Rotation_and_Center_tracker,
    _fetch_dynamic_positions,
    _fourier_by_layer,
    _init_worker,
    _read_ndx,
    _remove_tmd_hole_mask,
    _track_dynamic_leaflets,
    _worker_state,
)
from ..core.fourier_core import Fourier_Series_Function, get_fourier_modes
from ..core.headgroup import (
    _headgroup_centers,
    _named_headgroup_centers,
    _parse_lipids_argument,
    _require_bonds,
    _validate_headgroup_override,
    _validate_species_exist,
)
from ..core.manual import add_manual
from ..core.packing import median_multiple_threshold
from ..core.rotation import rotated_grid

logger = logging.getLogger(__name__)


class _LipidFrameResult(TypedDict):
    frame: int
    diagnostics: list[tuple[str, str]]


def _assign_nearest_leaflet(
    xy: np.ndarray, z: np.ndarray, fourier_upper: Fourier_Series_Function, fourier_lower: Fourier_Series_Function,
    far_multiple: float = 5.0,
) -> np.ndarray:
    """Per-point leaflet assignment: 1 (upper), -1 (lower), or 0 (neither - implausibly far from both fitted surfaces).

    A point's own distance to its nearer fitted surface is compared
    against `far_multiple` times the median such distance across every
    point passed in here (`median_multiple_threshold`, the same "how many
    multiples of the typical scale counts as anomalous" convention
    --Remove-TMD already uses for its own far-fallback rule), so a
    genuinely free-floating, flipped, or otherwise non-embedded lipid is
    caught rather than forced into whichever leaflet happens to be
    numerically closer regardless of how far away it actually sits.
    """
    if len(xy) == 0:
        return np.empty((0,), dtype=int)
    z_upper = fourier_upper.Z(xy[:, 0], xy[:, 1])
    z_lower = fourier_lower.Z(xy[:, 0], xy[:, 1])
    dist_upper = np.abs(z - z_upper)
    dist_lower = np.abs(z - z_lower)
    dist_to_nearest = np.minimum(dist_upper, dist_lower)

    assignment = np.where(dist_upper <= dist_lower, 1, -1)
    threshold = median_multiple_threshold(dist_to_nearest, k=far_multiple)
    if np.isfinite(threshold):
        assignment[dist_to_nearest > threshold] = 0
    return assignment


def _lipid_voronoi_fractions(
    species_xy: list[np.ndarray], X: np.ndarray, Y: np.ndarray, fourier: Fourier_Series_Function, Lx: float, Ly: float,
) -> np.ndarray:
    """Lipid composition per grid point for one leaflet: shape (n_species, *X.shape), 1 for the nearest species and 0 for the rest.

    Each row of `species_xy[i]` is one lipid's own (x, y). Both lipids and
    grid points are projected onto `fourier`'s own fitted height at their
    respective (x, y), so ownership is decided by chord distance along the
    leaflet's own surface rather than each lipid's real height (a
    residue's whole-body center of geometry sits at a species-dependent
    depth below the surface - a cardiolipin's four tails pull it deeper
    than a single-tailed lipid's - which would bias the competition by
    species identity rather than true packing). Every grid point's full
    weight goes to whichever lipid is nearest under that metric - a
    rasterized Voronoi tessellation, with no bandwidth parameter.
    """
    n_species = len(species_xy)
    total_lipids = sum(len(s) for s in species_xy)
    fractions = np.zeros((n_species,) + X.shape)
    if total_lipids == 0:
        return fractions

    all_xy = np.vstack(species_xy)
    labels = np.concatenate([np.full(len(s), i) for i, s in enumerate(species_xy)])

    positions = np.column_stack([all_xy[:, 0], all_xy[:, 1], fourier.Z(all_xy[:, 0], all_xy[:, 1])])
    positions[:, :2] = np.mod(positions[:, :2], [Lx, Ly])
    tree = cKDTree(positions, boxsize=[Lx, Ly, 0])
    grid_points = np.column_stack([X.ravel(), Y.ravel(), fourier.Z(X, Y).ravel()])
    _, nearest_idx = tree.query(grid_points)

    nearest_species = labels[nearest_idx]
    flat = fractions.reshape(n_species, -1)
    for species_idx in range(n_species):
        flat[species_idx] = nearest_species == species_idx

    return fractions


def _true_surface_area(fourier: Fourier_Series_Function, X: np.ndarray, Y: np.ndarray, cell_area: float) -> np.ndarray:
    """True (undulation-corrected) area per grid cell: sqrt(1 + Zx^2 + Zy^2) * cell_area."""
    fx = fourier.Zx(X, Y)
    fy = fourier.Zy(X, Y)
    return np.sqrt(1 + fx ** 2 + fy ** 2) * cell_area


def _curvature_at_points(fourier: Fourier_Series_Function, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean curvature of `fourier`'s surface at each scattered (x, y).

    `shape_operator_curvatures` computes curvature pointwise (no dependency
    between neighboring input points), so reshaping a flat array of
    scattered positions into a column lets it be reused directly for
    per-residue points, not just a meshgrid.
    """
    if len(x) == 0:
        return np.empty(0)
    H, *_ = shape_operator_curvatures(fourier, x.reshape(-1, 1), y.reshape(-1, 1))
    return H.ravel()


def _one_lipid_frame(
    frame: int,
    *,
    out_dir: str,
    species: list[str],
    dynamic_select: bool,
    dynamic_leaflets: dict[int, tuple[list[int], list[int]]] | None,
    until: int,
    Nx: float | None = 3,
    Ny: float | None = 3,
    sqrt_n_atoms: int = 100,
    regularize: bool = False,
    remove_tmd: bool | str = False,
    tmd_far_multiple: float = 5.0,
    headgroup_override: dict[str, list[str]] | None = None,
) -> _LipidFrameResult:
    """Fit one frame's leaflet surfaces and save its lipid-composition/area-per-lipid output.

    Lipid positions come from a fresh, per-species `resname` selection
    (grouped into residues via `_headgroup_centers`, or `_named_headgroup_centers`
    for any species in `headgroup_override`) - independent of whatever atoms
    `-n`/`--index` used to fit the leaflet surfaces, since that fit selection
    is one representative atom per lipid chosen for surface geometry, not
    necessarily present in every requested species.

    Saves `{frame}_lipid_fractions.npy` (species x [upper, lower] x grid),
    `{frame}_area_per_lipid.npy` (species x [upper, lower] x [flat,
    curved]), `{frame}_lipid_counts.npy` (species x [upper, lower]), and
    `{frame}_curvature_preference.npy` (species x [upper, lower]) -
    aggregating these into trajectory-averaged `area_per_lipid.csv` and
    `curvature_preference.csv` is the caller's job, once every frame is done.

    `area_per_lipid.npy` is always built from each frame's own raw,
    unrotated grid, regardless of --rotate - it's a whole-leaflet sum, and
    which physical position each grid index happens to query doesn't
    change that sum. `lipid_fractions.npy`/`hole_mask.npy` (spatial maps,
    meant to be compared frame to frame) are different: with --rotate,
    they're built a second time on a rotated query grid (see
    `rotated_grid` in `core/rotation.py` - the same "never rotate the
    underlying data, only the query point" mechanism `_one_frame` uses for
    curvature/thickness) so a real, protein-relative spatial pattern
    survives trajectory averaging instead of washing out as the protein
    itself rotates. Real lipid/protein positions are never rotated either
    way - only the second, extra query grid is.
    """
    diagnostics: list[tuple[str, str]] = []
    universe = cast(mda.Universe, _worker_state["universe"])
    layer_group = cast(mda.core.groups.AtomGroup, _worker_state["layer_group"])
    layer_group_2 = cast(mda.core.groups.AtomGroup, _worker_state["layer_group_2"])
    rotation_and_center = cast(
        "Rotation_and_Center_tracker | None", _worker_state["rotation_and_center"]
    )

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    Lx, Ly = dimensions[0], dimensions[1]
    x = np.linspace(0, Lx, sqrt_n_atoms, endpoint=False)
    y = np.linspace(0, Ly, sqrt_n_atoms, endpoint=False)
    X, Y = np.meshgrid(x, y)
    cell_area = (Lx / sqrt_n_atoms) * (Ly / sqrt_n_atoms)

    # One file per frame (not a single shared dimensions.csv, appended to by
    # every worker with no locking) - matches every other per-frame output
    # here (e.g. {frame}_lipid_fractions.npy), and each worker owns its own
    # file with zero cross-process contention.
    np.save(
        Path(out_dir) / f"{frame:0{num_digits}d}_dimensions.npy",
        np.asarray(dimensions[:3], dtype=np.float64),
    )

    if dynamic_select:
        assert dynamic_leaflets is not None
        upper_index, lower_index = dynamic_leaflets[frame]
        layer_group = universe.atoms[upper_index]
        layer_group_2 = universe.atoms[lower_index]

    theta = 0.0
    if rotation_and_center is not None:
        rotation_and_center._center()
        if rotation_and_center.rotate:
            rotation_and_center._get_vec(base=False)
            theta = rotation_and_center._get_rotation_angle()

    Nx_modes, Ny_modes = get_fourier_modes(dimensions[:3], lambda_x=Nx, lambda_y=Ny, diagnostics=diagnostics)

    fourier_upper, _ = _fourier_by_layer(
        layer_group, dimensions[:3], Nx_modes, Ny_modes, regularize=regularize, diagnostics=diagnostics
    )
    fourier_lower, _ = _fourier_by_layer(
        layer_group_2, dimensions[:3], Nx_modes, Ny_modes, regularize=regularize, diagnostics=diagnostics
    )

    area_upper = _true_surface_area(fourier_upper, X, Y, cell_area)
    area_lower = _true_surface_area(fourier_lower, X, Y, cell_area)

    # Hoisted above the per-species loop (a pure reorder - this block doesn't
    # depend on anything the loop produces) so valid_upper/valid_lower are
    # available for the curvature baseline below, computed once per frame.
    hole_mask = None
    if remove_tmd:
        hole_mask, _ = _remove_tmd_hole_mask(
            remove_tmd, rotation_and_center, universe, layer_group, layer_group_2,
            fourier_upper, fourier_lower, X, Y, Lx, Ly, Nx_modes, Ny_modes, tmd_far_multiple,
        )
        valid_upper = ~hole_mask[0]
        valid_lower = ~hole_mask[1]
    else:
        valid_upper = np.ones(X.shape, dtype=bool)
        valid_lower = np.ones(X.shape, dtype=bool)

    # The leaflet's own ambient curvature, excluding --Remove-TMD hole grid
    # points the same way area_per_lipid already excludes them from its own
    # sum - those points aren't real lipid-accessible membrane, so including
    # them would bias what "typical for this leaflet" means. Subtracted from
    # each species' own sampled curvature below, so a bulk curvature imposed
    # on the whole leaflet (e.g. by a protein) doesn't show up identically in
    # every species' preference.
    H_grid_upper, *_ = shape_operator_curvatures(fourier_upper, X, Y)
    H_grid_lower, *_ = shape_operator_curvatures(fourier_lower, X, Y)
    baseline_upper = float(H_grid_upper[valid_upper].mean())
    baseline_lower = float(H_grid_lower[valid_lower].mean())
    # [species, upper/lower], curvature relative to the leaflet's own mean
    # (Angstrom^-1), sign-adjusted so both leaflets share one convention (see
    # analyze_lipids.md) - NaN where a species has zero residues in a
    # leaflet this frame, distinct from a genuine 0.0 preference.
    curvature_preference = np.full((len(species), 2), np.nan)

    species_xy_upper: list[np.ndarray] = []
    species_xy_lower: list[np.ndarray] = []
    counts_upper = np.zeros(len(species))
    counts_lower = np.zeros(len(species))
    for i, name in enumerate(species):
        atoms = universe.select_atoms(f"resname {name}")
        if headgroup_override and name in headgroup_override:
            xy, z, hub_xy = _named_headgroup_centers(atoms, headgroup_override[name])
        else:
            xy, z, hub_xy = _headgroup_centers(atoms, fourier_upper, fourier_lower)
        assignment = _assign_nearest_leaflet(xy, z, fourier_upper, fourier_lower, tmd_far_multiple)
        is_upper = assignment == 1
        is_lower = assignment == -1
        # Every hub of an assigned residue competes in the Voronoi step (a
        # multi-hub lipid like cardiolipin contributes more than one point),
        # while counts stay per-residue so area_per_lipid divides correctly.
        upper_points = np.vstack([hub_xy[j] for j in range(len(hub_xy)) if is_upper[j]]) if is_upper.any() \
            else np.empty((0, 2))
        lower_points = np.vstack([hub_xy[j] for j in range(len(hub_xy)) if is_lower[j]]) if is_lower.any() \
            else np.empty((0, 2))
        species_xy_upper.append(upper_points)
        species_xy_lower.append(lower_points)
        counts_upper[i] = int(is_upper.sum())
        counts_lower[i] = int(is_lower.sum())
        n_unassigned = int((assignment == 0).sum())
        if n_unassigned:
            diagnostics.append((
                "warning",
                f"{n_unassigned} {name} residue(s) excluded from both leaflets - "
                "implausibly far from both fitted surfaces",
            ))

        # xy (one averaged headgroup position per residue, the same point
        # _assign_nearest_leaflet used) rather than hub_xy - curvature
        # preference is a property of the whole lipid molecule, not of each
        # structural headgroup junction separately.
        #
        # shape_operator_curvatures's raw H is negative for a patch bulging
        # away from the bilayer core toward the upper leaflet's own water
        # side (a lysolipid-like outward bulge, e.g. Z = z0 + bump(x,y), has
        # Zxx<0/Zyy<0 at its peak) - the opposite sign of the field-standard
        # convention (positive C0 = outward/conical, like a micelle). The
        # upper leaflet's own value is negated to match that convention; the
        # lower leaflet needs no negation, since its own outward direction
        # (-z) already flips the raw sign back the other way.
        if is_upper.any():
            curvature_preference[i, 0] = -(
                _curvature_at_points(fourier_upper, xy[is_upper, 0], xy[is_upper, 1]).mean() - baseline_upper
            )
        if is_lower.any():
            curvature_preference[i, 1] = (
                _curvature_at_points(fourier_lower, xy[is_lower, 0], xy[is_lower, 1]).mean() - baseline_lower
            )

    # Always built on the raw (X, Y) grid, feeding area_per_lipid only -
    # this is what keeps --rotate from touching area_per_lipid.csv at all.
    fractions_upper = _lipid_voronoi_fractions(species_xy_upper, X, Y, fourier_upper, Lx, Ly)
    fractions_lower = _lipid_voronoi_fractions(species_xy_lower, X, Y, fourier_lower, Lx, Ly)

    area_per_lipid = np.zeros((len(species), 2, 2))  # [species, leaflet(upper=0/lower=1), area(flat=0/curved=1)]
    leaflets = (
        (fractions_upper, area_upper, valid_upper, counts_upper),
        (fractions_lower, area_lower, valid_lower, counts_lower),
    )
    for leaflet_idx, (fractions, area_curved, valid_mask, counts) in enumerate(leaflets):
        for i in range(len(species)):
            if counts[i] <= 0:
                continue
            frac = fractions[i][valid_mask]
            area_per_lipid[i, leaflet_idx, 0] = float((frac * cell_area).sum() / counts[i])
            area_per_lipid[i, leaflet_idx, 1] = float((frac * area_curved[valid_mask]).sum() / counts[i])

    if theta != 0.0:
        # A second, rotation-aligned pass for the saved spatial output only -
        # lipid/protein positions stay exactly as they are; only the query
        # grid used to build these two arrays is rotated (rotated_grid maps
        # each canonical output point to the as-recorded position it
        # corresponds to once this frame is aligned to the reference
        # direction). area_per_lipid above never sees these.
        cx, cy = Lx / 2.0, Ly / 2.0
        X_out, Y_out = rotated_grid(X, Y, cx, cy, theta)
        out_fractions_upper = _lipid_voronoi_fractions(species_xy_upper, X_out, Y_out, fourier_upper, Lx, Ly)
        out_fractions_lower = _lipid_voronoi_fractions(species_xy_lower, X_out, Y_out, fourier_lower, Lx, Ly)
        out_hole_mask = None
        if remove_tmd:
            out_hole_mask, _ = _remove_tmd_hole_mask(
                remove_tmd, rotation_and_center, universe, layer_group, layer_group_2,
                fourier_upper, fourier_lower, X_out, Y_out, Lx, Ly, Nx_modes, Ny_modes, tmd_far_multiple,
            )
    else:
        out_fractions_upper, out_fractions_lower = fractions_upper, fractions_lower
        out_hole_mask = hole_mask

    counts = np.stack([counts_upper, counts_lower], axis=1)
    out_fractions = np.stack([out_fractions_upper, out_fractions_lower], axis=1)

    out = Path(out_dir)
    np.save(out / f"{frame:0{num_digits}d}_lipid_fractions.npy", out_fractions)
    np.save(out / f"{frame:0{num_digits}d}_area_per_lipid.npy", area_per_lipid)
    np.save(out / f"{frame:0{num_digits}d}_lipid_counts.npy", counts)
    np.save(out / f"{frame:0{num_digits}d}_curvature_preference.npy", curvature_preference)
    if out_hole_mask is not None:
        # Same (2, gridsize, gridsize) [upper, lower] convention as
        # calc_fourier's own {frame}_hole_mask.npy, saved flat here (not
        # under raw_sft/) since 'analyze lipids' has no SFT to consolidate
        # into - 'map lipids_plot' reads it directly per frame.
        np.save(out / f"{frame:0{num_digits}d}_hole_mask.npy", out_hole_mask)

    return {"frame": frame, "diagnostics": diagnostics}


def _write_area_per_lipid_csv(out_dir: str, species: list[str], frames: list[int], until: int) -> None:
    """Average every frame's {frame}_area_per_lipid.npy/{frame}_lipid_counts.npy into one area_per_lipid.csv.

    One row per (leaflet, species): area_per_lipid_flat, area_per_lipid_curved, mean_count.
    """
    num_digits = len(str(abs(until)))
    out = Path(out_dir)
    area_stack = [np.load(out / f"{frame:0{num_digits}d}_area_per_lipid.npy") for frame in frames]
    count_stack = [np.load(out / f"{frame:0{num_digits}d}_lipid_counts.npy") for frame in frames]

    mean_area = np.mean(np.stack(area_stack), axis=0)  # (n_species, 2, 2)
    mean_count = np.mean(np.stack(count_stack), axis=0)  # (n_species, 2)

    lines = ["leaflet,species,area_per_lipid_flat,area_per_lipid_curved,mean_count"]
    for leaflet_idx, leaflet_name in enumerate(("upper", "lower")):
        for i, name in enumerate(species):
            lines.append(
                f"{leaflet_name},{name},{mean_area[i, leaflet_idx, 0]:.6f},"
                f"{mean_area[i, leaflet_idx, 1]:.6f},{mean_count[i, leaflet_idx]:.2f}"
            )
    (out / "area_per_lipid.csv").write_text("\n".join(lines) + "\n")


def _write_curvature_preference_csv(out_dir: str, species: list[str], frames: list[int], until: int) -> None:
    """Average every frame's {frame}_curvature_preference.npy/{frame}_lipid_counts.npy into curvature_preference.csv.

    One row per (leaflet, species) for leaflet in {upper, lower, both}:
    C0_nm-1 (converted from the code's native Angstrom^-1), C0_stderr_nm-1,
    mean_count, n_frames_sampled. A species' NaN frames in a leaflet (it
    had zero residues there that frame) are skipped via nanmean/nanstd
    rather than pulled toward zero. The "both" row is a count-weighted
    average of the upper/lower C0 values (with standard error propagation),
    falling back to whichever leaflet has data when the other is entirely
    absent, and NaN when neither leaflet ever had this species.
    """
    num_digits = len(str(abs(until)))
    out = Path(out_dir)
    curvature_stack = np.stack(
        [np.load(out / f"{frame:0{num_digits}d}_curvature_preference.npy") for frame in frames]
    )  # (n_frames, n_species, 2)
    count_stack = np.stack(
        [np.load(out / f"{frame:0{num_digits}d}_lipid_counts.npy") for frame in frames]
    )  # (n_frames, n_species, 2)

    with warnings.catch_warnings():
        # A species absent from a leaflet for the whole trajectory gives an
        # all-NaN slice here - the resulting NaN is the correct answer (no
        # data), not a bug to silence differently.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c0_mean = np.nanmean(curvature_stack, axis=0) * 10.0  # (n_species, 2): Angstrom^-1 -> nm^-1
        c0_std = np.nanstd(curvature_stack, axis=0) * 10.0
    n_sampled = np.sum(~np.isnan(curvature_stack), axis=0)  # (n_species, 2)
    c0_stderr = np.where(n_sampled > 0, c0_std / np.sqrt(np.maximum(n_sampled, 1)), np.nan)
    mean_count = np.mean(count_stack, axis=0)  # (n_species, 2)

    lines = ["leaflet,species,C0_nm-1,C0_stderr_nm-1,mean_count,n_frames_sampled"]
    for i, name in enumerate(species):
        for leaflet_idx, leaflet_name in enumerate(("upper", "lower")):
            lines.append(
                f"{leaflet_name},{name},{c0_mean[i, leaflet_idx]:.6f},{c0_stderr[i, leaflet_idx]:.6f},"
                f"{mean_count[i, leaflet_idx]:.2f},{int(n_sampled[i, leaflet_idx])}"
            )

        weight_u = mean_count[i, 0] if not np.isnan(c0_mean[i, 0]) else 0.0
        weight_l = mean_count[i, 1] if not np.isnan(c0_mean[i, 1]) else 0.0
        total_weight = weight_u + weight_l
        if total_weight > 0:
            c0_both = (
                weight_u * np.nan_to_num(c0_mean[i, 0]) + weight_l * np.nan_to_num(c0_mean[i, 1])
            ) / total_weight
            var_both = (
                weight_u ** 2 * np.nan_to_num(c0_stderr[i, 0]) ** 2
                + weight_l ** 2 * np.nan_to_num(c0_stderr[i, 1]) ** 2
            ) / total_weight ** 2
            stderr_both = float(np.sqrt(var_both))
            n_both = int(n_sampled[i, 0] + n_sampled[i, 1])
        else:
            c0_both, stderr_both, n_both = float("nan"), float("nan"), 0
        lines.append(f"both,{name},{c0_both:.6f},{stderr_both:.6f},{weight_u + weight_l:.2f},{n_both}")

    (out / "curvature_preference.csv").write_text("\n".join(lines) + "\n")


def calc_lipids(args: argparse.Namespace, u: mda.Universe) -> None:
    """Fit every selected frame's leaflet surfaces and lipid composition/area-per-lipid, in parallel.

    Structurally identical to `calc_fourier`'s parallel per-frame pipeline
    (same worker init, same dynamic-leaflet tracking), calling
    `_one_lipid_frame` instead of `_one_frame` and finishing with a single
    trajectory-averaged `area_per_lipid.csv` once every frame is done.
    """
    species, headgroup_override = _parse_lipids_argument(args.lipids)
    _validate_species_exist(u, species)
    _validate_headgroup_override(u, species, headgroup_override)
    if set(species) - set(headgroup_override):
        # Bonds are only needed for species that will actually go through
        # the bond-graph classification; --headgroup-atoms covering every
        # species means the automatic method never runs at all.
        _require_bonds(u)

    Until = args.Until
    ndx = args.index

    if Until is None:
        Until = len(u.trajectory)
    else:
        Until = int(Until)
    if ndx is None:
        sys.exit("An index selection or file has to be supplied. Exiting.")
    try:
        ndx_groups = _read_ndx(ndx)
        dynamic_select = False
        dynamic_selection = None
    except FileNotFoundError:
        logger.info(
            "No ndx file found at the given --index path; treating it as a dynamic MDAnalysis selection instead."
        )
        dynamic_select = True
        dynamic_selection = ndx
        ndx_groups = None

    frames = list(range(args.From, Until, args.Step))
    Path(args.out, "lipid_species.txt").write_text("\n".join(species) + "\n")
    # 'analyze lipids' has no SFT/q_mn.npy to smuggle a rotation signal
    # through (see core/rotation.py's own note on that trick) - 'map
    # lipids_plot' needs its own explicit record of whether --rotate was
    # used, to decide whether to restrict rendering to the fixed circle
    # the same way 'map plot' does for rotated curvature/thickness output.
    np.save(Path(args.out, "rotated.npy"), np.array(bool(args.rotate)))

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(
            args.structure, args.trajectory, ndx_groups, dynamic_select,
            args.center, args.rotation_direction, args.rotate,
        ),
    ) as ex:
        dynamic_leaflets = None
        if dynamic_select:
            assert dynamic_selection is not None
            fetch = partial(_fetch_dynamic_positions, dynamic_selection=dynamic_selection)
            ordered_results = list(ex.map(fetch, frames))
            selection_global_indices = u.select_atoms(dynamic_selection).atoms.indices
            dynamic_leaflets = _track_dynamic_leaflets(
                ordered_results, selection_global_indices, args.min_balance, margin=args.margin
            )

        fn = partial(_one_lipid_frame,
                    out_dir=args.out,
                    species=species,
                    dynamic_select=dynamic_select,
                    dynamic_leaflets=dynamic_leaflets,
                    until=Until,
                    Nx=args.lambda_x,
                    Ny=args.lambda_y,
                    sqrt_n_atoms=args.gridsize,
                    regularize=args.regularize,
                    remove_tmd=args.remove_tmd,
                    headgroup_override=headgroup_override,
                    )

        futures = [ex.submit(fn, x) for x in frames]
        # logging_redirect_tqdm only patches the loggers passed to it - the
        # console handler that actually prints logger.warning/.info below
        # lives on the "CALM" logger itself (attach_replay_log_handler), not
        # root, so it must be named explicitly here or the redirect has no
        # effect and log output writes straight to stderr, corrupting the bar.
        with logging_redirect_tqdm(loggers=[logging.getLogger("CALM")]):
            for future in tqdm(futures):
                result: _LipidFrameResult = future.result()  # surfaces exceptions raised in worker processes
                frame_num = result["frame"]
                for level, message in result["diagnostics"]:
                    log_fn = logger.warning if level == "warning" else logger.info
                    log_fn(f"frame {frame_num}: {message}")

    _write_area_per_lipid_csv(args.out, species, frames, Until)
    _write_curvature_preference_csv(args.out, species, frames, Until)


def lipids(args: list[str]) -> None:
    """CLI entry: per-species lipid composition and area-per-lipid from a trajectory.

    Always re-fits the leaflet surfaces from a live trajectory (no --sft
    reuse) - lipid identity (resname) is only available from real atoms,
    which a previously-built SFT (Amn/qmn coefficients alone) doesn't
    carry. --rotate rotationally aligns the saved per-frame composition
    output (see _one_lipid_frame) across the trajectory, the same way it
    aligns 'sft'/'full's curvature/thickness output - useful for spotting
    real spatial composition patterns (e.g. lipid sorting around a
    particular face of a protein) that would otherwise wash out under
    trajectory averaging as the protein itself rotates. It does not
    affect area_per_lipid.csv, which stays computed on each frame's own
    raw, unrotated grid throughout. --center still applies (needed by
    --Remove-TMD and by --rotate itself).
    """
    parser = argparse.ArgumentParser(
        description="Per-species lipid composition and area-per-lipid from a trajectory.",
    )
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")
    required.add_argument(
        "--lipids", nargs="+", required=True, type=arg_helper.lipids_species_token,
        metavar="RESNAME[:NAME1,NAME2,...]",
        help=(
            "resnames to treat as distinct lipid species, e.g. --lipids POPC GM1 SAPE24. A "
            "species can instead give its own headgroup atom name(s) explicitly, skipping the "
            "automatic bond-graph detection for it, e.g. --lipids POPC:PO4 TCL1:PO41,PO42 SAPE24 "
            "(POPC and TCL1 use the named atoms directly; SAPE24 still auto-detects). Warns "
            "loudly when mixing the two within one run."
        ),
    )
    arg_helper.add_build_arguments(
        parser, required_group=required, optional_group=optional
    )
    add_manual(parser, "analyze_lipids")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    # Validation runs on the fully-resolved args (replay file's tokens, if
    # any, merged with the direct CLI) - required= on -f/-s/-n means a
    # replay-only invocation (no -f/-s/-n given directly) must not be
    # validated before the replay file's own values are merged in.
    ns = arg_helper.apply_replay(parser, pre_ns, remaining)
    arg_helper.validate_rotation_args(parser, ns)

    os.makedirs(ns.out, exist_ok=True)

    replay_path = ns.out_replay or arg_helper.default_replay_name(ns.out)
    arg_helper.write_replay_file(replay_path, parser, ns)
    arg_helper.attach_replay_log_handler(replay_path, logger_name="MDAnalysis")
    arg_helper.attach_replay_log_handler(
        replay_path, logger_name="CALM",
        console_level=logging.INFO if ns.loud else logging.WARNING,
    )

    if ns.clear:
        for filename in os.listdir(ns.out):
            if filename.endswith(".npy"):
                file_path = os.path.join(ns.out, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

    try:
        start = time.perf_counter()
        universe = mda.Universe(Path(ns.structure), Path(ns.trajectory))
        calc_lipids(ns, universe)
        print(f"Execution with {ns.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
