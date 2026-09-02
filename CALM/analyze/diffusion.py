from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import MDAnalysis as mda
import numpy as np
from MDAnalysis.transformations import NoJump, unwrap
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..core import argument_parser as arg_helper
from ..core.diffusion import (
    _break_into_segments,
    _fit_diffusion_coefficient,
    _multi_tau_msd,
    _project_onto_surface,
    _selection_centers,
    _SurfaceAsInterp,
)
from ..core.fourier_build import (
    _build_dynamic_leaflet_reference,
    _dynamic_leaflet_groups,
    _fourier_by_layer,
    _init_worker,
    _remove_tmd_hole_mask,
    _worker_state,
)
from ..core.fourier_core import Fourier_Series_Function, average_coefficients, get_fourier_modes
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
from ..core.rotation import lookup_mask_at_point
from .lipids import _assign_nearest_leaflet

logger = logging.getLogger(__name__)

# Angstrom^2/ps to cm^2/s: 1 A^2 = 1e-16 cm^2, 1 ps = 1e-12 s.
_A2_PS_TO_CM2_S = 1e-4

# A hard floor under --min-segment-fraction, so a very short trajectory
# doesn't accept a physically meaningless handful-of-frames segment even
# when that fraction is set loosely.
_MIN_SEGMENT_FRAMES = 20


class _TrackBlock(NamedTuple):
    label: str
    kind: str
    select_string: str
    whole: bool


class _DiffusionFrameResult(TypedDict):
    frame: int
    diagnostics: list[tuple[str, str]]


def _track_blocks(
    species: list[str], select: str | None, select_whole: bool, select_label: str
) -> list[_TrackBlock]:
    """One tracked-point block per `--lipids` species and, if given, one more for `--select`.

    A `--lipids` block's `select_string` is `resname <name>`, tracked per
    bonded fragment. A `--select` block uses that selection string
    directly, tracked per fragment, or as a single combined point when
    `select_whole` is set. `select_string` lets every downstream pass
    reselect a block's own atoms fresh from any `Universe` built on the
    same structure, in the same fragment order every time.
    """
    blocks = [_TrackBlock(label=name, kind="lipids", select_string=f"resname {name}", whole=False) for name in species]
    if select is not None:
        blocks.append(_TrackBlock(label=select_label, kind="select", select_string=select, whole=select_whole))
    return blocks


def _resolve_tracked_points(universe: mda.Universe, blocks: list[_TrackBlock]) -> np.ndarray:
    """The fixed tracked-point roster: one row per physical point every later pass will follow.

    Row order is the concatenation of each block's own bonded fragments, in
    the order `block.select_string` selects them - the same order every
    later per-frame position array uses, so a roster row's index lines up
    with that row in every per-frame array throughout the run. A
    whole-selection block (`--select-whole`) contributes exactly one row,
    with `fragindex` set to -1, marking that point as identified by its own
    block and index alone.
    """
    rows = []
    index = 0

    for block in blocks:
        if block.kind == "select" and block.whole:
            rows.append((index, block.label, -1, block.kind))
            index += 1
            continue

        atoms = universe.select_atoms(block.select_string)

        for fragindex in np.unique(atoms.fragindices):
            rows.append(
                (
                    index,
                    block.label,
                    int(fragindex),
                    block.kind,
                )
            )
            index += 1

    dtype = [
        ("index", "i8"),
        ("label", "U64"),
        ("fragindex", "i8"),
        ("kind", "U16"),
    ]

    return np.array(rows, dtype=dtype)


def _block_positions(
    universe: mda.Universe,
    block: _TrackBlock,
    fourier_upper: Fourier_Series_Function,
    fourier_lower: Fourier_Series_Function,
    headgroup_override: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """(xy, z) for one block's tracked points this frame, in the row order `_resolve_tracked_points` uses.

    A `lipids` block uses the named-atom override for its species when one
    was given, and the automatic bond-graph headgroup detection otherwise -
    the same choice `analyze lipids` makes per species, but grouped by
    bonded fragment rather than by residue (`compound="fragments"`, see
    `core/headgroup.py::_grouped_atoms`), matching every other tracked
    point here and `_resolve_tracked_points`'s own roster. A whole `select`
    block returns its entire matched atomgroup's own center of geometry as
    a single point; a per-fragment `select` block returns one point per
    fragment's own center of geometry (`_selection_centers`).
    """
    atoms = universe.select_atoms(block.select_string)
    if block.kind == "lipids":
        if block.label in headgroup_override:
            xy, z, _ = _named_headgroup_centers(atoms, headgroup_override[block.label], compound="fragments")
        else:
            xy, z, _ = _headgroup_centers(atoms, fourier_upper, fourier_lower, compound="fragments")
        return xy, z
    if block.whole:
        center = atoms.center_of_geometry()
        return center[:2].reshape(1, 2), np.array([center[2]])
    return _selection_centers(atoms)


def _assign_middle_surface(
    xy: np.ndarray, z: np.ndarray, fourier_middle: Fourier_Series_Function, far_multiple: float,
) -> np.ndarray:
    """Per-point in/out-of-membrane flag against the middle surface only: 1 (embedded) or 0 (implausibly far).

    `--force-middle`'s replacement for `_assign_nearest_leaflet`: a point
    that straddles both leaflets (e.g. a transmembrane protein's own
    center of geometry, which sits near the mid-plane by construction) has
    no real "closer" leaflet - assigning it to whichever fitted surface is
    numerically nearer would flip between upper and lower from thermal
    noise alone, and every such flip breaks the tracked point's trajectory
    into a new segment (see `_build_segments`), fragmenting what should be
    one continuous diffusive record. Every embedded point is instead always
    `1` here, with the same far-multiple exclusion `_assign_nearest_leaflet`
    uses for a point implausibly far from the membrane entirely.
    """
    if len(xy) == 0:
        return np.empty((0,), dtype=int)
    z_middle = fourier_middle.Z(xy[:, 0], xy[:, 1])
    dist_middle = np.abs(z - z_middle)
    assignment = np.ones(len(xy), dtype=int)
    threshold = median_multiple_threshold(dist_middle, k=far_multiple)
    if np.isfinite(threshold):
        assignment[dist_middle > threshold] = 0
    return assignment


def _one_diffusion_frame(
    frame: int,
    *,
    out_dir: str,
    blocks: list[_TrackBlock],
    headgroup_override: dict[str, list[str]],
    dynamic_select: bool,
    until: int,
    Nx: float | None = 3,
    Ny: float | None = 3,
    regularize: bool = False,
    remove_tmd: bool | str = False,
    tmd_far_multiple: float = 5.0,
    gridsize: int = 100,
    force_middle: bool = False,
) -> _DiffusionFrameResult:
    """Fit one frame's leaflet surfaces, assign every tracked point's leaflet and hole status, and save both.

    Every tracked point's wrapped position for this frame comes from
    `_block_positions`, in the same row order the tracked-point roster
    uses. Leaflet assignment reuses `_assign_nearest_leaflet` (1 upper, -1
    lower, 0 excluded) - unless `force_middle` is set, in which case every
    point is instead matched against the middle surface only
    (`_assign_middle_surface`: 1 embedded, 0 excluded, never -1), meant for
    tracking something that straddles both leaflets (e.g. a transmembrane
    protein) rather than living in one of them. When `remove_tmd` is
    given, a hole mask is built fresh for this frame (`_remove_tmd_hole_mask`)
    and each tracked point's own hole status is looked up from it directly
    (`lookup_mask_at_point`), using whichever leaflet's mask it was
    assigned to - with `force_middle`, that is always the upper leaflet's
    mask, since assignment is never -1 there.

    Saves `{frame}_dimensions.npy` (box dimensions), `{frame}_diffusion_meta.npy`
    (shape `(n_tracked, 2)`: `[leaflet, in_hole]`), and
    `{frame}_diffusion_surface.npy` (shape `(2, 2*Nx+1, 2*Ny+1)`: `[Anm_upper,
    Anm_lower]`) - the fitted surfaces both leaflet assignment and hole
    lookup used this frame, saved so a later pass can project onto the
    exact same surfaces without fitting them again. With `force_middle`,
    both saved slots are the same middle surface, so a later pass projects
    onto it regardless of which slot `leaflet`'s value would normally pick.
    """
    diagnostics: list[tuple[str, str]] = []
    universe = cast(mda.Universe, _worker_state["universe"])
    layer_group = cast(mda.core.groups.AtomGroup, _worker_state["layer_group"])
    layer_group_2 = cast(mda.core.groups.AtomGroup, _worker_state["layer_group_2"])

    num_digits = len(str(abs(until)))
    ts = universe.trajectory[frame]
    dimensions = ts.dimensions
    Lx, Ly = dimensions[0], dimensions[1]

    out = Path(out_dir)
    np.save(out / f"{frame:0{num_digits}d}_dimensions.npy", np.asarray(dimensions[:3], dtype=np.float64))

    if dynamic_select:
        layer_group, layer_group_2 = _dynamic_leaflet_groups(universe, dimensions)

    Nx_modes, Ny_modes = get_fourier_modes(dimensions[:3], lambda_x=Nx, lambda_y=Ny, diagnostics=diagnostics)
    fourier_upper, _ = _fourier_by_layer(
        layer_group, dimensions[:3], Nx_modes, Ny_modes, regularize=regularize, diagnostics=diagnostics
    )
    fourier_lower, _ = _fourier_by_layer(
        layer_group_2, dimensions[:3], Nx_modes, Ny_modes, regularize=regularize, diagnostics=diagnostics
    )

    xy_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    for block in blocks:
        xy, z = _block_positions(universe, block, fourier_upper, fourier_lower, headgroup_override)
        xy_parts.append(xy)
        z_parts.append(z)
    xy = np.vstack(xy_parts)
    z = np.concatenate(z_parts)

    if force_middle:
        fourier_middle = Fourier_Series_Function(Lx, Ly, Nx_modes, Ny_modes)
        fourier_middle.setAnm(average_coefficients(fourier_upper.Anm, fourier_lower.Anm))
        leaflet = _assign_middle_surface(xy, z, fourier_middle, tmd_far_multiple)
        surface_upper, surface_lower = fourier_middle.Anm, fourier_middle.Anm
    else:
        leaflet = _assign_nearest_leaflet(xy, z, fourier_upper, fourier_lower, tmd_far_multiple)
        surface_upper, surface_lower = fourier_upper.Anm, fourier_lower.Anm

    if remove_tmd:
        x = np.linspace(0, Lx, gridsize, endpoint=False)
        y = np.linspace(0, Ly, gridsize, endpoint=False)
        X, Y = np.meshgrid(x, y)
        hole_mask, _ = _remove_tmd_hole_mask(
            remove_tmd, None, universe, layer_group, layer_group_2,
            fourier_upper, fourier_lower, X, Y, Lx, Ly, Nx_modes, Ny_modes, tmd_far_multiple,
        )
        in_hole_upper = lookup_mask_at_point(hole_mask[0], xy[:, 0], xy[:, 1], Lx, Ly)
        in_hole_lower = lookup_mask_at_point(hole_mask[1], xy[:, 0], xy[:, 1], Lx, Ly)
        in_hole = np.where(leaflet == 1, in_hole_upper, in_hole_lower)
    else:
        in_hole = np.zeros(len(leaflet), dtype=bool)

    np.save(out / f"{frame:0{num_digits}d}_diffusion_meta.npy", np.column_stack([leaflet, in_hole.astype(int)]))
    np.save(
        out / f"{frame:0{num_digits}d}_diffusion_surface.npy",
        np.stack([surface_upper, surface_lower], axis=0),
    )

    return {"frame": frame, "diagnostics": diagnostics}


def _extract_whole_continuous_positions(
    structure: str, trajectory: str, blocks: list[_TrackBlock], frames: list[int], out_dir: str, until: int
) -> None:
    """Sequential pass building each tracked point's own whole, continuous xyz for every analyzed frame.

    Opens its own `Universe`, separate from the parallel surface-fitting
    pass, and chains two MDAnalysis transformations on it:
    `unwrap(tracked_ag)` keeps each tracked fragment's own atoms from
    straddling a periodic boundary within a single frame, and `NoJump()`
    keeps each atom's position continuous from one analyzed frame to the
    next. Frames are read in trajectory order, which `NoJump` requires.
    Each tracked point's position is then its block's own center (the same
    center method `_block_positions` uses for a `select` block, or a
    fragment's own center of geometry for a `lipids` block), read from this
    transformed `Universe`.

    Saves `{frame}_diffusion_positions.npy`, shape `(n_tracked, 3)`.
    """
    universe = mda.Universe(structure, trajectory)
    select_string = " or ".join(f"({block.select_string})" for block in blocks)
    tracked_ag = universe.select_atoms(select_string)
    universe.trajectory.add_transformations(unwrap(tracked_ag), NoJump())

    num_digits = len(str(abs(until)))
    out = Path(out_dir)
    for frame in frames:
        universe.trajectory[frame]
        parts: list[np.ndarray] = []
        for block in blocks:
            atoms = universe.select_atoms(block.select_string)
            if block.kind == "select" and block.whole:
                parts.append(atoms.center_of_geometry().reshape(1, 3))
            else:
                parts.append(atoms.center_of_geometry(compound="fragments"))
        np.save(out / f"{frame:0{num_digits}d}_diffusion_positions.npy", np.vstack(parts))


def _project_all_frames(
    out_dir: str, frames: list[int], until: int, n_tracked: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(leaflet, in_hole, proj_xy) for every tracked point across every analyzed frame, from the saved per-frame files.

    Rebuilds each frame's own upper/lower `Fourier_Series_Function` from
    its saved `Anm` (mode counts read back from `Anm`'s own shape) and box
    dimensions, then projects every tracked point assigned to a leaflet
    that frame (`_project_onto_surface`) onto that same leaflet's surface.
    An unassigned point (`leaflet == 0`) keeps its own raw xy for that
    frame; it plays no further part once segments are broken at every
    unassigned frame.
    """
    num_digits = len(str(abs(until)))
    out = Path(out_dir)
    n_frames = len(frames)

    leaflet = np.zeros((n_frames, n_tracked), dtype=int)
    in_hole = np.zeros((n_frames, n_tracked), dtype=bool)
    positions = np.zeros((n_frames, n_tracked, 3))
    dimensions = np.zeros((n_frames, 3))
    surfaces: list[tuple[np.ndarray, np.ndarray]] = []
    for i, frame in enumerate(frames):
        tag = f"{frame:0{num_digits}d}"
        meta = np.load(out / f"{tag}_diffusion_meta.npy")
        leaflet[i] = meta[:, 0]
        in_hole[i] = meta[:, 1].astype(bool)
        positions[i] = np.load(out / f"{tag}_diffusion_positions.npy")
        dimensions[i] = np.load(out / f"{tag}_dimensions.npy")
        surface = np.load(out / f"{tag}_diffusion_surface.npy")
        surfaces.append((surface[0], surface[1]))

    proj_xy = np.zeros((n_frames, n_tracked, 2))
    for t in range(n_frames):
        Anm_upper, Anm_lower = surfaces[t]
        Nx_t = (Anm_upper.shape[0] - 1) // 2
        Ny_t = (Anm_upper.shape[1] - 1) // 2
        Lx_t, Ly_t = float(dimensions[t, 0]), float(dimensions[t, 1])
        fourier_upper = Fourier_Series_Function(Lx_t, Ly_t, Nx_t, Ny_t)
        fourier_upper.setAnm(Anm_upper)
        fourier_lower = Fourier_Series_Function(Lx_t, Ly_t, Nx_t, Ny_t)
        fourier_lower.setAnm(Anm_lower)
        interp_upper = _SurfaceAsInterp(fourier_upper)
        interp_lower = _SurfaceAsInterp(fourier_lower)

        for i in range(n_tracked):
            if leaflet[t, i] == 0:
                proj_xy[t, i] = positions[t, i, :2]
                continue
            px, py, pz = positions[t, i]
            if leaflet[t, i] == 1:
                x_proj, y_proj, _ = _project_onto_surface(px, py, pz, fourier_upper, interp_upper)
            else:
                x_proj, y_proj, _ = _project_onto_surface(px, py, pz, fourier_lower, interp_lower)
            proj_xy[t, i] = (x_proj, y_proj)

    return leaflet, in_hole, proj_xy


def _build_segments(
    tracked_points: np.ndarray,
    leaflet: np.ndarray,
    in_hole: np.ndarray,
    proj_xy: np.ndarray,
    frames: list[int],
    min_segment_fraction: float,
    force_middle: bool = False,
) -> list[dict]:
    """Every candidate segment for every tracked point, each tagged kept/discarded and why.

    A segment comes from `_break_into_segments` on that point's own
    leaflet/hole-status columns. It is kept when it reaches both
    `min_segment_fraction` of the analyzed range and `_MIN_SEGMENT_FRAMES`;
    otherwise `discard_reason` names which bound it fell short of.
    `force_middle` labels every segment `"middle"` instead of `"upper"` -
    with `force_middle`, `leaflet` is never -1 (`_assign_middle_surface`),
    so every segment would otherwise be mislabeled `"upper"`.
    """
    n_frames = len(frames)
    segments: list[dict] = []
    for i in range(len(tracked_points)):
        label = str(tracked_points["label"][i])
        fragindex = int(tracked_points["fragindex"][i])
        for start, end in _break_into_segments(leaflet[:, i], in_hole[:, i]):
            length = end - start
            length_fraction = length / n_frames
            if length_fraction < min_segment_fraction:
                discard_reason = "shorter than --min-segment-fraction of the analyzed range"
            elif length < _MIN_SEGMENT_FRAMES:
                discard_reason = "shorter than the minimum segment length"
            else:
                discard_reason = ""
            segments.append({
                "label": label,
                "leaflet": "middle" if force_middle else ("upper" if leaflet[start, i] == 1 else "lower"),
                "fragindex": fragindex,
                "start_frame": frames[start],
                "end_frame": frames[end - 1],
                "length_frames": length,
                "length_fraction": length_fraction,
                "kept": discard_reason == "",
                "discard_reason": discard_reason,
                "xy": proj_xy[start:end, i, :],
            })
    return segments


def _pool_label(seg: dict, per_instance: bool) -> str:
    """The string a segment pools under: its block's own label, or - with `per_instance` - that label
    plus its own fragindex (`"select#3"`), so each physical tracked point gets its own pool instead of
    being combined with every other point sharing the same label."""
    if per_instance:
        return f"{seg['label']}#{seg['fragindex']}"
    return str(seg["label"])


def _fit_diffusion_pools(
    segments: list[dict],
    dt: float,
    max_tau_fraction: float,
    fit_tau_min_fraction: float,
    fit_tau_max_fraction: float,
    per_instance: bool = False,
    force_middle: bool = False,
) -> tuple[list[tuple], list[tuple]]:
    """(diffusion_rows, msd_rows): one row per `(species, leaflet)` pool with a kept segment, plus a pooled "both" row.

    A kept segment's own xy trajectory is pooled into both its own
    `(label, leaflet)` pool and that label's `"both"` pool, so a species'
    diffusion coefficient is reported per leaflet and combined across
    both. `D`/`D_stderr` are converted from Angstrom^2/ps to cm^2/s.
    `per_instance` pools by `(label, fragindex)` instead of `label` alone
    (`_pool_label`), giving every individual tracked point its own
    diffusion estimate rather than combining every point sharing a label -
    e.g. every one of ten proteins matched by the same `--select` gets its
    own row instead of all ten being combined into one. `force_middle`
    skips the `"both"` pool entirely - every segment's own leaflet is
    already `"middle"` there (see `_assign_middle_surface`), so `"both"`
    would just be a duplicate of `"middle"` by construction, not a real
    second pool.
    """
    pools: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    discarded_by_label_leaflet: dict[tuple[str, str], int] = defaultdict(int)
    discarded_by_label: dict[str, int] = defaultdict(int)
    for seg in segments:
        label = _pool_label(seg, per_instance)
        if seg["kept"]:
            pools[(label, seg["leaflet"])].append(seg["xy"])
            if not force_middle:
                pools[(label, "both")].append(seg["xy"])
        else:
            discarded_by_label_leaflet[(label, seg["leaflet"])] += 1
            discarded_by_label[label] += 1

    diffusion_rows = []
    msd_rows = []
    for (label, leaflet_key), segs_xy in pools.items():
        tau, msd, n_samples = _multi_tau_msd(segs_xy, dt, max_tau_fraction)
        D, D_stderr, r2, loglog_slope = _fit_diffusion_coefficient(tau, msd, fit_tau_min_fraction, fit_tau_max_fraction)
        n_discarded = (
            discarded_by_label[label] if leaflet_key == "both" else discarded_by_label_leaflet[(label, leaflet_key)]
        )
        tau_min = float(tau[0]) if len(tau) else float("nan")
        tau_max = float(tau[-1]) if len(tau) else float("nan")
        diffusion_rows.append((
            leaflet_key, label, D * _A2_PS_TO_CM2_S, D_stderr * _A2_PS_TO_CM2_S,
            len(segs_xy), sum(len(s) for s in segs_xy), tau_min, tau_max, r2, loglog_slope, n_discarded,
        ))
        for tau_value, msd_value, n_value in zip(tau, msd, n_samples):
            msd_rows.append((leaflet_key, label, float(tau_value), float(msd_value), int(n_value)))

    return diffusion_rows, msd_rows


def _write_diffusion_csv(out_dir: str, diffusion_rows: list[tuple], suffix: str = "") -> None:
    """Write `diffusion{suffix}.csv`: the same rows `diffusion{suffix}.npy` gets, human-readable.

    One row per `(leaflet, species)` pool with a kept segment, plus each
    species' pooled `"both"`/`"middle"` row - whatever `_fit_diffusion_pools`
    produced, in the order it produced them.
    """
    lines = [
        "leaflet,species,D_cm2_s,D_stderr_cm2_s,n_segments,n_points_pooled,"
        "tau_min_ps,tau_max_ps,fit_r2,fit_loglog_slope,n_segments_discarded_short"
    ]
    for (
        leaflet_key, label, D, D_stderr, n_segments, n_points_pooled,
        tau_min, tau_max, r2, loglog_slope, n_discarded,
    ) in diffusion_rows:
        lines.append(
            f"{leaflet_key},{label},{D:.6e},{D_stderr:.6e},{n_segments},{n_points_pooled},"
            f"{tau_min:.4f},{tau_max:.4f},{r2:.4f},{loglog_slope:.4f},{n_discarded}"
        )
    Path(out_dir, f"diffusion{suffix}.csv").write_text("\n".join(lines) + "\n")


_DIFFUSION_DTYPE = [
    ("leaflet", "U8"), ("species", "U64"), ("D_cm2_s", "f8"), ("D_stderr_cm2_s", "f8"),
    ("n_segments", "i8"), ("n_points_pooled", "i8"), ("tau_min_ps", "f8"), ("tau_max_ps", "f8"),
    ("fit_r2", "f8"), ("fit_loglog_slope", "f8"), ("n_segments_discarded_short", "i8"),
]
_MSD_DTYPE = [("leaflet", "U8"), ("species", "U64"), ("tau_ps", "f8"), ("msd_A2", "f8"), ("n_samples", "i8")]


def _write_diffusion_and_msd(
    out_dir: str, diffusion_rows: list[tuple], msd_rows: list[tuple], suffix: str = ""
) -> None:
    """Write `diffusion{suffix}.npy`, `msd_curves{suffix}.npy`, and `diffusion{suffix}.csv`.

    `suffix` distinguishes the two groupings `_finalize_diffusion` always
    computes from the same segments - `""` (pooled by species/label,
    `_fit_diffusion_pools`'s default) and `"_per_instance"` (one row per
    individual tracked point, `per_instance=True`) - so a later `CALM map
    diffusion_plot` run can pick either view without recomputing anything.
    """
    out = Path(out_dir)
    np.save(out / f"diffusion{suffix}.npy", np.array(diffusion_rows, dtype=_DIFFUSION_DTYPE))
    np.save(out / f"msd_curves{suffix}.npy", np.array(msd_rows, dtype=_MSD_DTYPE))
    _write_diffusion_csv(out_dir, diffusion_rows, suffix=suffix)


def _write_segments(out_dir: str, segments: list[dict]) -> None:
    """Write `segments.npy`: one row per candidate segment, kept or excluded, at the finest (per-point) granularity."""
    segments_dtype = [
        ("label", "U64"), ("leaflet", "U8"), ("fragindex", "i8"), ("start_frame", "i8"), ("end_frame", "i8"),
        ("length_frames", "i8"), ("length_fraction", "f8"), ("kept", "?"), ("discard_reason", "U64"),
    ]
    np.save(
        Path(out_dir, "segments.npy"),
        np.array(
            [
                (s["label"], s["leaflet"], s["fragindex"], s["start_frame"], s["end_frame"],
                 s["length_frames"], s["length_fraction"], s["kept"], s["discard_reason"])
                for s in segments
            ],
            dtype=segments_dtype,
        ),
    )


def _finalize_diffusion(
    out_dir: str,
    tracked_points: np.ndarray,
    frames: list[int],
    until: int,
    dt: float,
    min_segment_fraction: float,
    max_tau_fraction: float,
    fit_tau_min_fraction: float,
    fit_tau_max_fraction: float,
    force_middle: bool = False,
) -> None:
    """Project, segment, pool, and fit: turn every frame's saved surface/meta/position files into the output files.

    Fits and writes both groupings `_fit_diffusion_pools` can produce from
    the same segments - pooled by species/label (`diffusion.npy`) and per
    individual tracked point (`diffusion_per_instance.npy`) - so choosing
    between them is a `CALM map diffusion_plot` flag, not a re-run here.
    """
    n_tracked = len(tracked_points)
    leaflet, in_hole, proj_xy = _project_all_frames(out_dir, frames, until, n_tracked)
    segments = _build_segments(
        tracked_points, leaflet, in_hole, proj_xy, frames, min_segment_fraction, force_middle=force_middle
    )
    _write_segments(out_dir, segments)

    for per_instance, suffix in ((False, ""), (True, "_per_instance")):
        diffusion_rows, msd_rows = _fit_diffusion_pools(
            segments, dt, max_tau_fraction, fit_tau_min_fraction, fit_tau_max_fraction,
            per_instance=per_instance, force_middle=force_middle,
        )
        _write_diffusion_and_msd(out_dir, diffusion_rows, msd_rows, suffix=suffix)


def calc_diffusion(args: argparse.Namespace, u: mda.Universe) -> None:
    """Project each tracked point onto its own leaflet's surface every frame and fit a lateral diffusion coefficient.

    Runs in three passes: a parallel per-frame pass fits each frame's
    leaflet surfaces and records each tracked point's leaflet/hole status
    (`_one_diffusion_frame`), a sequential pass builds each tracked
    point's own whole, continuous raw position via MDAnalysis's own
    unwrap/NoJump transformations (`_extract_whole_continuous_positions`),
    and a final sequential pass joins both, projects onto the surface,
    breaks each point's trajectory into segments, and fits D
    (`_finalize_diffusion`).
    """
    species, headgroup_override = _parse_lipids_argument(args.lipids) if args.lipids else ([], {})
    if species:
        _validate_species_exist(u, species)
        _validate_headgroup_override(u, species, headgroup_override)
    _require_bonds(u)

    blocks = _track_blocks(species, args.select, args.select_whole, args.select_label)
    tracked_points = _resolve_tracked_points(u, blocks)
    np.save(Path(args.out, "tracked_points.npy"), tracked_points)

    Until = args.Until
    if Until is None:
        Until = len(u.trajectory)
    else:
        Until = int(Until)

    ndx_groups, dynamic_selection = arg_helper.resolve_index_source(args)
    if ndx_groups is None and dynamic_selection is None:
        sys.exit("An index selection or file has to be supplied. Exiting.")
    dynamic_select = ndx_groups is None

    frames = list(range(args.From, Until, args.Step))

    dynamic_ref = None
    if dynamic_select:
        assert dynamic_selection is not None
        dynamic_ref = _build_dynamic_leaflet_reference(
            u, frames[0], dynamic_selection, args.min_balance, args.margin
        )

    with ProcessPoolExecutor(
        max_workers=args.Workers,
        initializer=_init_worker,
        initargs=(args.structure, args.trajectory, ndx_groups, dynamic_select, dynamic_ref, None, None, False),
    ) as ex:
        fn = partial(
            _one_diffusion_frame,
            out_dir=args.out,
            blocks=blocks,
            headgroup_override=headgroup_override,
            dynamic_select=dynamic_select,
            until=Until,
            Nx=args.lambda_x,
            Ny=args.lambda_y,
            regularize=args.regularize,
            remove_tmd=args.remove_tmd,
            gridsize=args.gridsize,
            force_middle=args.force_middle,
        )

        futures = [ex.submit(fn, x) for x in frames]
        with logging_redirect_tqdm(loggers=[logging.getLogger("CALM")]):
            for future in tqdm(futures, desc="fitting surfaces"):
                result: _DiffusionFrameResult = future.result()  # surfaces exceptions raised in worker processes
                frame_num = result["frame"]
                for level, message in result["diagnostics"]:
                    log_fn = logger.warning if level == "warning" else logger.info
                    log_fn(f"frame {frame_num}: {message}")

    _extract_whole_continuous_positions(args.structure, args.trajectory, blocks, frames, args.out, Until)

    dt = float(u.trajectory.dt) * args.Step
    _finalize_diffusion(
        args.out, tracked_points, frames, Until, dt,
        args.min_segment_fraction, args.max_tau_fraction, args.fit_tau_min_fraction, args.fit_tau_max_fraction,
        force_middle=args.force_middle,
    )


def _build_diffusion_parser() -> argparse.ArgumentParser:
    """The 'CALM analyze diffusion' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(
        description="Curvature-aware lateral diffusion coefficient per lipid species and/or MDAnalysis selection.",
    )
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--lipids", nargs="+", default=None, type=arg_helper.lipids_species_token,
        metavar="RESNAME[:NAME1,NAME2,...]",
        help=(
            "resnames to track as distinct lipid species, e.g. --lipids POPC GM1 SAPE24. A "
            "species can instead give its own headgroup atom name(s) explicitly, skipping the "
            "automatic bond-graph detection for it, e.g. --lipids POPC:PO4 TCL1:PO41,PO42. At "
            "least one of --lipids/--select is required."
        ),
    )
    optional.add_argument(
        "--select", default=None, type=str, metavar="SELECTION",
        help="MDAnalysis selection to track, one point per bonded fragment unless --select-whole is given. "
             "At least one of --lipids/--select is required.",
    )
    optional.add_argument(
        "--select-whole", dest="select_whole", default=False, action="store_true",
        help="track --select's entire match as a single combined point",
    )
    optional.add_argument(
        "--select-label", dest="select_label", default="select", type=str,
        help="label for --select's output rows (default: 'select')",
    )
    optional.add_argument(
        "--force-middle", dest="force_middle", default=False, action="store_true",
        help="track against the middle surface (the upper/lower average) instead of assigning each point to "
             "whichever leaflet it's nearer - for something that straddles both leaflets, e.g. a transmembrane "
             "protein, where a real leaflet assignment would flip on thermal noise alone and fragment its "
             "trajectory into spurious segments (see --man)",
    )
    optional.add_argument(
        "--min-segment-fraction", dest="min_segment_fraction", default=0.1, type=float,
        help="segments shorter than this fraction of the analyzed range are excluded from the fit (default: 0.1)",
    )
    optional.add_argument(
        "--max-tau-fraction", dest="max_tau_fraction", default=0.25, type=float,
        help="caps tau at this fraction of each segment's own length (default: 0.25)",
    )
    optional.add_argument(
        "--fit-tau-min-fraction", dest="fit_tau_min_fraction", default=0.1, type=float,
        help="lower edge, as a fraction of the pooled curve's own max tau, of the window used for the linear D fit "
             "(default: 0.1)",
    )
    optional.add_argument(
        "--fit-tau-max-fraction", dest="fit_tau_max_fraction", default=0.5, type=float,
        help="upper edge, as a fraction of the pooled curve's own max tau, of the window used for the linear D fit "
             "(default: 0.5)",
    )
    required, optional = arg_helper.add_build_arguments(
        parser, include_rotation=False, include_center=False, required_group=required, optional_group=optional,
    )
    for action in optional._group_actions:
        if action.dest == "remove_tmd":
            action.help = (
                "flag unsupported grid points as holes, using this selection to identify protein "
                "atoms (e.g. --Remove-TMD 'name BB SC1'). A selection is always required for this "
                "command."
            )
    add_manual(parser, "analyze_diffusion")
    return parser


def diffusion(args: list[str]) -> None:
    """CLI entry: curvature-aware lateral diffusion coefficient per lipid species and/or MDAnalysis selection.

    Each tracked point's real position is projected onto its own assigned
    leaflet's fitted surface every frame before displacement is measured,
    so membrane undulation is separated from lateral motion. A tracked
    point's trajectory splits into a new segment at a leaflet flip or a
    --Remove-TMD hole-status change; segments shorter than
    --min-segment-fraction of the analyzed range, or shorter than a fixed
    minimum frame count, are excluded from the fit. D comes from a
    multi-tau, ensemble-and-time-averaged MSD curve pooled across every
    kept segment. Leaflet surfaces are always fit live from the
    trajectory. --structure must carry bond information (e.g. a GROMACS
    .tpr): both the automatic headgroup detection and the PBC-aware
    position extraction need it. --Remove-TMD must always be given its own
    selection here (e.g. --Remove-TMD 'name BB SC1'). --force-middle tracks
    against the middle surface instead, for something that straddles both
    leaflets (e.g. a transmembrane protein) rather than living in one.
    """
    parser = _build_diffusion_parser()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--replay")
    pre_ns, remaining = pre.parse_known_args(args)

    ns = arg_helper.apply_replay(parser, pre_ns, remaining)
    arg_helper.validate_index_arguments(parser, ns, required=True)

    if not ns.lipids and not ns.select:
        parser.error("At least one of --lipids or --select is required.")
    if ns.remove_tmd is True:
        parser.error(
            "--Remove-TMD requires its own selection for this command, e.g. --Remove-TMD 'name BB SC1'."
        )

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
        calc_diffusion(ns, universe)
        print(f"Execution with {ns.Workers} Workers took {round(time.perf_counter()-start,2)} seconds.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    pass
