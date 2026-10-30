from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np

from ..core.fourier_build import (
    _bootstrap_dynamic_leaflet_split,
    _DynamicLeafletReference,
    _dynamic_leaflet_split,
)
from ..core.fourier_core import get_fourier_modes

# Exact replica of core/fourier_fit.py::fit_coefficients' own tiering
# (Length>=M / M<3*Length / M<10*Length / else) - that file's thresholds are
# finished and correct; these names only ever label the same boundaries,
# never redefine them.
TIER_UNDERDETERMINED = "underdetermined"
TIER_LOW_REDUNDANCY = "low redundancy"
TIER_LOW_OVERSAMPLING = "low oversampling"
TIER_SAFE = "safe"

# Below this raw median bond length, positions are raw, unconverted nm
# mislabeled as Angstrom. Real bond lengths in this codebase's own
# systems range from atomistic covalent bonds (~0.09-0.17 nm) through
# Martini-style coarse-grained bead-bonds (~0.3-0.6 nm); this sits above
# all of those, while the shortest real bond, once correctly converted
# to Angstrom, is already well above it.
RAW_NM_THRESHOLD = 0.7

# Final sanity range, in Angstrom, for the median bond length after
# whichever correction was or wasn't applied - spans atomistic covalent
# bonds (~0.9-1.7 A) through Martini-style coarse-grained bead-bonds
# (~3-6 A) without being so loose it would accept corrupted data.
PLAUSIBLE_BOND_LOW_ANGSTROM = 0.3
PLAUSIBLE_BOND_HIGH_ANGSTROM = 10.0

# Lindahl & Edholm (2000), Biophys. J. 79:426-433: fitting a DPPC bilayer's
# own spectral decomposition, modes with wavelength long enough to involve
# more than ~50 lipids showed genuine q^-4 collective (undulatory) behavior,
# while modes involving fewer than ~25 lipids were dominated by independent
# per-lipid thermal protrusions instead (q^-2) - the same physical
# conflation core/fourier_fit.py's own 10x M/Length cutoff is guarding
# against, but expressed there in a number never validated against real
# data. "Lipids involved in a mode" there is, in this fit's own terms, the
# atom count within one finest-resolved real-space patch: a candidate
# lambda's own fit resolves features down to (Lx/Nx, Ly/Ny), so a patch of
# that size contains, on average, M/(Nx*Ny) fit atoms - see
# `_evaluate_group`. 50 sits at the genuine-undulation end of their
# 25-50 crossover band, not its midpoint, so this is the conservative
# choice of the two numbers they report.
LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE = 50.0


def oversampling_tier(M: int, Length: int) -> str:
    """Which of fit_coefficients' own four tiers (M, Length) falls into.

    `M` = fit atoms, `Length` = Fourier coefficient count
    (2*Nx+1)*(2*Ny+1). Boundaries copied verbatim from
    core/fourier_fit.py::fit_coefficients (lines ~53-88 there): Length>=M is
    underdetermined, M<3*Length is low redundancy, M<10*Length is low
    oversampling (the 10x cutoff is an "empirical judgment call, not
    analytically derived" per that file's own comment), otherwise safe.
    """
    if Length >= M:
        return TIER_UNDERDETERMINED
    if M < 3 * Length:
        return TIER_LOW_REDUNDANCY
    if M < 10 * Length:
        return TIER_LOW_OVERSAMPLING
    return TIER_SAFE


@dataclass
class CandidateResult:
    """One lambda's own worst-case oversampling outcome across every sampled frame x leaflet.

    `lambda_nm` is one representative value, not necessarily the only
    lambda that produces this (Nx, Ny) on every sampled frame -
    `sweep_lambda_candidates` groups candidates that evaluate identically
    and only keeps one representative per distinct (Nx, Ny) signature (see
    its own docstring for why, and which value within a tied group is kept).
    """

    lambda_nm: float
    Nx: int
    Ny: int
    Length: int
    worst_case_M: int
    worst_case_ratio: float
    tier: str
    worst_case_source: tuple[int, str]  # (sampled-frame position, "upper"/"lower")
    diagnostics: list[tuple[str, str]]  # from get_fourier_modes, e.g. Nx floored to 1
    lipids_per_mode: float  # worst-case M/(Nx*Ny) - see LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE


def _position_unit_scale_factor(universe: mda.Universe) -> float:
    """1.0 normally; 10.0 if this universe's own bonds reveal its positions are actually in nm.

    Checked directly against this universe's own bonds (if any exist),
    rather than by name against one specific reader class, so this also
    catches the same class of bug in a reader this wasn't written
    against. Confirmed in practice against MDAnalysis 2.10.0's brand-new
    TPRReader (added for reading a .tpr as its own trajectory): it declares
    `units={"length": "nm", ...}` and `convert_units=True`, but its own
    `_read_first_frame` never actually applies that conversion to the raw
    parsed positions it stores - every bond then comes out roughly 10x too
    short (e.g. an atomistic N-H bond at ~0.1 "Angstrom" instead of the
    correct ~1.0, or a Martini coarse-grained bead-bond at ~0.47
    "Angstrom" instead of the correct ~4.7).

    Uses `universe.bonds.bonds()` - every bond's length, computed by
    MDAnalysis itself - rather than a sample: a bond that straddles the
    periodic boundary in raw (unwrapped) coordinates reads as a long
    cross-box distance rather than a true bond length, and the full
    population's median tolerates that kind of contamination as long as
    it stays a minority, with no need to sample around it.

    Raises ValueError if the median, interpreted either way (already
    Angstrom, or raw nm needing the x10 fix), lands outside a plausible
    real bond length - silently picking a scale for data that fits
    neither story would be worse than refusing.
    """
    try:
        bonds = universe.bonds
    except Exception:
        return 1.0

    if len(bonds) == 0:
        return 1.0

    median_length_raw = float(np.median(bonds.bonds()))
    needs_correction = median_length_raw < RAW_NM_THRESHOLD
    median_length_angstrom = median_length_raw * 10.0 if needs_correction else median_length_raw

    if not (PLAUSIBLE_BOND_LOW_ANGSTROM <= median_length_angstrom <= PLAUSIBLE_BOND_HIGH_ANGSTROM):
        raise ValueError(
            f"could not determine this structure's coordinate scale: median bond length "
            f"{median_length_raw:.4g} (raw) gives an implausible {median_length_angstrom:.4g} "
            f"Angstrom bond length even {'with' if needs_correction else 'without'} a x10 "
            f"correction - expected {PLAUSIBLE_BOND_LOW_ANGSTROM}-{PLAUSIBLE_BOND_HIGH_ANGSTROM}."
        )

    return 10.0 if needs_correction else 1.0


def _frame_dimensions(
    universe: mda.Universe, frame: int, structure_path: str,
    diagnostics: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """This frame's own box dimensions, seeked and, if missing, approximated.

    Some structure formats don't reliably surface box dimensions when
    opened without a separate trajectory - seen in practice with a .tpr
    used as the sole -s/--structure source: MDAnalysis's own TPR coordinate
    reader reads real box bytes from the file (its header correctly flags
    `bBox`) but never stores the parsed result on the Timestep - a
    confirmed bug in MDAnalysis itself, not anything this tool can correct
    upstream. `np.array(None, dtype=float)` doesn't raise on this - it
    silently becomes a 0-d nan scalar - so left unchecked this would only
    surface later as a cryptic internal MDAnalysis error deep inside
    whichever geometry call first tries to use it as a real box.

    When the real dimensions are missing, this falls back to a rectangular
    box built from this frame's own atom positions (min to max per axis) -
    reasonable for a well-behaved, PBC-wrapped frame where the real
    simulation box is in fact close to rectangular. It refuses that
    fallback (raises instead) only when there IS partial unit-cell
    information available (non-zero box lengths) and it shows real,
    non-90-degree angles - clear evidence of a genuinely triclinic box a
    position-derived rectangular approximation would misrepresent. With no
    partial information at all (the all-zero case this exact MDAnalysis
    bug produces), there is no such evidence either way, so the
    approximation proceeds - recorded in `diagnostics` (same
    `(level, message)` convention `core/fourier_core.py::get_fourier_modes`
    already uses) so it's never silent.
    """
    if diagnostics is None:
        diagnostics = []

    universe.trajectory[frame]
    dimensions = np.array(universe.dimensions, dtype=np.float64)

    if dimensions.shape == (6,) and np.all(np.isfinite(dimensions)):
        return dimensions

    partial = np.asarray(getattr(universe.trajectory.ts, "_unitcell", np.zeros(6)), dtype=np.float64)
    lengths_known = partial.shape == (6,) and np.any(np.abs(partial[:3]) > 1e-9)

    if lengths_known:
        angles = partial[3:6]
        if not np.allclose(angles, 90.0, atol=1.0):
            raise ValueError(
                f"'{structure_path}' (frame {frame})'s box dimensions are missing from "
                f"`.dimensions`, and the partial unit-cell data that IS available "
                f"(angles={angles.tolist()}) shows a non-rectangular (triclinic) box - "
                "approximating this from atom positions would be wrong. Try giving "
                "-f/--trajectory instead, or a structure format that reliably carries box info."
            )

    positions = universe.atoms.positions
    if positions.shape[0] == 0:
        raise ValueError(
            f"'{structure_path}' (frame {frame}) has no usable box dimensions and no atoms "
            "to approximate one from. Try giving -f/--trajectory instead."
        )

    scale = _position_unit_scale_factor(universe)
    if scale != 1.0:
        positions = positions * scale
        diagnostics.append((
            "warning",
            f"frame {frame}: positions appear to be in nm, not Å - scaled by {scale:g}x.",
        ))

    box_lengths = positions.max(axis=0) - positions.min(axis=0)
    diagnostics.append((
        "warning",
        f"frame {frame}: box dimensions missing - estimated from atom positions "
        f"({box_lengths[0]:.1f} x {box_lengths[1]:.1f} x {box_lengths[2]:.1f} Å).",
    ))

    return np.array([box_lengths[0], box_lengths[1], box_lengths[2], 90.0, 90.0, 90.0])


def collect_frame_samples(
    structure_path: str,
    trajectory_path: str | None,
    ndx_groups: dict[str, list[int]] | None,
    dynamic_selection: str | None,
    frame_indices: Sequence[int],
    min_balance: float,
    margin: float,
    diagnostics: list[tuple[str, str]] | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """(box_sizes, leaflet_counts), one entry per frame in sorted(frame_indices).

    `diagnostics`, if given, is appended to in place by `_frame_dimensions`
    whenever a frame's real box dimensions are missing and had to be
    approximated from atom positions instead (see its own docstring).

    Built through the same leaflet-detection primitives `CALM analyze sft`
    itself uses for a dynamic selection (core/fourier_build.py's
    `_bootstrap_dynamic_leaflet_split` / `_dynamic_leaflet_split`, the same
    two steps `_build_dynamic_leaflet_reference` composes - called directly
    here rather than through that function, see the dynamic-selection
    branch below for why) - called directly in-process, no
    ProcessPoolExecutor, since this only counts atoms, it never fits a
    surface, so the real build's per-frame worker-pool parallelism buys
    nothing here. Deliberately NOT going through `_init_worker`/
    `_worker_state` (the real build's own ProcessPoolExecutor initializer):
    those always open the structure and trajectory as two separate files,
    which this function cannot always do (see `-s` alone, below), and
    `_dynamic_leaflet_split` itself takes everything it needs as plain
    parameters, with no hidden global state to thread through.

    Exactly one of `ndx_groups`/`dynamic_selection` must be given, mirroring
    `core/argument_parser.py::resolve_index_source`'s own contract.
    """
    if (ndx_groups is None) == (dynamic_selection is None):
        raise ValueError("Exactly one of ndx_groups or dynamic_selection must be given.")

    dynamic_select = ndx_groups is None

    # -s alone, no real trajectory: build the Universe from the structure
    # file's own embedded coordinates (a single positional argument) - NOT
    # by also passing it explicitly as a second, "trajectory" argument.
    # That second form routes MDAnalysis through that file FORMAT's own
    # dedicated coordinate-reader lookup instead of letting the topology
    # parser expose its own embedded coordinates directly, and not every
    # format's coordinate reader supports being reused this way (e.g. a
    # .tpr opened as its own "trajectory" this way can come back with
    # invalid/empty box dimensions), even though the structure-only,
    # single-argument form works correctly for every format.
    universe = (
        mda.Universe(structure_path) if trajectory_path is None
        else mda.Universe(structure_path, trajectory_path)
    )

    if not dynamic_select:
        assert ndx_groups is not None
        layer_group = universe.atoms[[x - 1 for x in ndx_groups["Upper"]]]
        layer_group_2 = universe.atoms[[x - 1 for x in ndx_groups["Lower"]]]

    if dynamic_select:
        assert dynamic_selection is not None
        # Inlines _build_dynamic_leaflet_reference's own two-line body
        # (core/fourier_build.py) instead of calling it directly: that
        # function does its own independent universe.dimensions read, which
        # would silently miss this function's own approximated dimensions
        # for a frame whose real box is missing (see _frame_dimensions) -
        # writing the approximation back onto the Timestep so a later,
        # independent re-read would pick it up isn't reliable across every
        # reader (some reconstruct the Timestep fresh on every seek, even
        # to the same frame, wiping a transient patch like that). Calling
        # its own lower-level primitives - _bootstrap_dynamic_leaflet_split,
        # _DynamicLeafletReference - directly, with the dimensions already
        # in hand here, sidesteps the problem entirely.
        frame0 = sorted(frame_indices)[0]
        frame0_dimensions = _frame_dimensions(universe, frame0, structure_path, diagnostics)
        selection = universe.select_atoms(dynamic_selection)
        first_upper, first_lower, cutoff = _bootstrap_dynamic_leaflet_split(
            selection.positions.copy(), frame0_dimensions, min_balance, margin,
        )
        dynamic_ref = _DynamicLeafletReference(
            selection=dynamic_selection, first_upper=first_upper, first_lower=first_lower,
            cutoff=cutoff, selection_global_indices=selection.atoms.indices, margin=margin,
        )

    box_sizes: list[np.ndarray] = []
    leaflet_counts: list[tuple[int, int]] = []

    for frame in sorted(frame_indices):
        dimensions = _frame_dimensions(universe, frame, structure_path, diagnostics)

        if dynamic_select:
            upper_local, lower_local = _dynamic_leaflet_split(
                selection.positions, dimensions,
                dynamic_ref.first_upper, dynamic_ref.first_lower, dynamic_ref.cutoff, margin,
            )
            n_upper, n_lower = len(upper_local), len(lower_local)
        else:
            n_upper, n_lower = len(layer_group), len(layer_group_2)

        box_sizes.append(dimensions[:3])
        leaflet_counts.append((n_upper, n_lower))

    return box_sizes, leaflet_counts


def _mode_signature(lambda_nm: float, box_sizes: Sequence[np.ndarray]) -> tuple[tuple[int, int], ...]:
    """(Nx, Ny) from get_fourier_modes, once per sampled frame, for this lambda.

    The dedup key `sweep_lambda_candidates` groups candidates by:
    `get_fourier_modes` floors via `int(Lx / (lambda_nm*10))`, so many
    different lambda_nm values land on the identical (Nx, Ny) for every
    sampled frame - those are numerically indistinguishable to evaluate
    (same Length, same M, same ratio, same tier), so there is no reason to
    score them more than once.
    """
    return tuple(get_fourier_modes(box_size, lambda_nm, lambda_nm) for box_size in box_sizes)


def _evaluate_group(
    representative_lambda: float,
    box_sizes: Sequence[np.ndarray],
    leaflet_counts: Sequence[tuple[int, int]],
) -> CandidateResult:
    """This lambda's own worst case (smallest M/Length ratio) across every (sampled frame) x (upper, lower leaflet) pair.

    Separately tracks the worst-case `lipids_per_mode` (M/(Nx*Ny), see
    `LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE`) the same way - not necessarily at
    the same (frame, leaflet) pair as the worst M/Length ratio, since the
    two ratios depend on (Nx, Ny) differently ((2Nx+1)(2Ny+1) vs Nx*Ny).
    """
    worst: tuple[float, int, int, int, int, int, str] | None = None
    worst_lipids_per_mode: float | None = None
    diagnostics: list[tuple[str, str]] = []

    for frame_pos, (box_size, (M_upper, M_lower)) in enumerate(zip(box_sizes, leaflet_counts)):
        Nx, Ny = get_fourier_modes(
            box_size, representative_lambda, representative_lambda, diagnostics=diagnostics,
        )
        Length = (2 * Nx + 1) * (2 * Ny + 1)

        for leaflet_name, M in (("upper", M_upper), ("lower", M_lower)):
            ratio = M / Length

            if worst is None or ratio < worst[0]:
                worst = (ratio, M, Length, Nx, Ny, frame_pos, leaflet_name)

            lipids_per_mode = M / (Nx * Ny)
            if worst_lipids_per_mode is None or lipids_per_mode < worst_lipids_per_mode:
                worst_lipids_per_mode = lipids_per_mode

    assert worst is not None  # caller guarantees len(box_sizes) >= 1
    assert worst_lipids_per_mode is not None
    ratio, M, Length, Nx, Ny, frame_pos, leaflet_name = worst

    return CandidateResult(
        lambda_nm=representative_lambda, Nx=Nx, Ny=Ny, Length=Length,
        worst_case_M=M, worst_case_ratio=ratio, tier=oversampling_tier(M, Length),
        worst_case_source=(frame_pos, leaflet_name), diagnostics=diagnostics,
        lipids_per_mode=worst_lipids_per_mode,
    )


def _decimal_digit_count(value: float, max_decimals: int = 6) -> int:
    """How many decimal digits `value` actually needs, after rounding to `max_decimals` and stripping trailing zeros."""
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return len(text.split(".", 1)[1]) if "." in text else 0


def sweep_lambda_candidates(
    box_sizes: Sequence[np.ndarray],
    leaflet_counts: Sequence[tuple[int, int]],
    candidates: Sequence[float],
) -> list[CandidateResult]:
    """One CandidateResult per DISTINCT (Nx, Ny) signature among `candidates` - not one per raw candidate.

    `candidates` is swept isotropically (each one used as both lambda_x and
    lambda_y). Grouped by `_mode_signature` first; every member of a tied
    group evaluates identically on the frames actually sampled, so which
    one is kept as that group's single representative value is otherwise
    immaterial - the cleanest-looking one is: fewest decimal digits
    (`_decimal_digit_count`), ties broken by the smallest value.
    """
    groups: dict[tuple[tuple[int, int], ...], list[float]] = {}

    for lam in candidates:
        signature = _mode_signature(lam, box_sizes)
        groups.setdefault(signature, []).append(lam)

    return [
        _evaluate_group(
            min(lambdas, key=lambda lam: (_decimal_digit_count(lam), lam)),
            box_sizes, leaflet_counts,
        )
        for lambdas in groups.values()
    ]


def recommend_lambda(results: Sequence[CandidateResult]) -> CandidateResult:
    """The smallest lambda clearing both TIER_SAFE and LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE.

    `fit_coefficients`' own M/Length>=10x tier guards against the same
    physical risk (per-lipid thermal protrusions misread as membrane
    curvature) that Lindahl & Edholm (2000) measured directly, in lipid
    count rather than a coefficient-count ratio - requiring both lets a
    candidate that only clears the (approximate, unvalidated) M/Length
    proxy but not the (directly measured) lipid-count threshold get passed
    over for a slightly coarser one that clears both. Both improve
    monotonically as lambda grows for a fixed box, so the first (smallest)
    lambda clearing both is also the finest resolution that does.

    Always returns something usable even when every candidate is
    imperfect: if none clear both, falls back to the single candidate with
    the best worst_case_ratio - the returned CandidateResult.tier then
    honestly reflects that (e.g. TIER_LOW_OVERSAMPLING), so callers can
    tell a confident pick from a best-of-a-bad-set one. The only case this
    raises is an empty `results` (a configuration error - no candidates
    were ever generated - not "none were good enough").
    """
    if not results:
        raise ValueError("No candidate lambda values to evaluate.")

    for result in sorted(results, key=lambda r: r.lambda_nm):
        if result.tier == TIER_SAFE and result.lipids_per_mode >= LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE:
            return result

    return max(results, key=lambda r: r.worst_case_ratio)


def calibrate_lambda(
    structure_path: str,
    trajectory_path: str | None = None,
    *,
    ndx_groups: dict[str, list[int]] | None = None,
    dynamic_selection: str | None = None,
    from_frame: int = 0,
    until_frame: int | None = None,
    step: int = 1,
    n_frames: int = 5,
    seed: int = 0,
    candidates: Sequence[float] | None = None,
    lambda_min: float = 1.0,
    lambda_max: float | None = None,
    n_lambda: int = 20,
    min_balance: float = 0.6,
    margin: float = 2.0,
) -> tuple[CandidateResult, list[CandidateResult]]:
    """Recommend a lambda_x/lambda_y for `structure_path` (optionally sampling `trajectory_path`).

    Picks frame_indices (just [0] if no trajectory; otherwise
    `np.random.default_rng(seed)`-sampled, without replacement, from the
    `[from_frame, until_frame, step)` pool, `min(n_frames, pool size)` of
    them), calls `collect_frame_samples`, builds the candidate lambda grid
    (explicit `candidates` if given, else
    `np.geomspace(lambda_min, lambda_max, n_lambda)`), calls
    `sweep_lambda_candidates` then `recommend_lambda`.

    Returns `(winner, all_results)` - `winner.lambda_nm` is already the
    single representative value for its tied group
    (`sweep_lambda_candidates`' own dedup, see its docstring). This is THE
    function a pipeline calls directly (no CLI/argparse/printing involved)
    to get a recommended lambda_x/lambda_y programmatically:
    `winner, _ = calibrate_lambda(...)`.

    `lambda_max=None` (the default) means: after `collect_frame_samples`,
    use the largest Lx/Ly observed across every sampled frame, converted
    Angstrom->nm (divide by 10) - i.e. sweep up to the box's own size. Past
    that point `get_fourier_modes` already floors Nx/Ny to 1 on every
    sampled frame, so a fixed larger default would just add candidates
    that collapse into the same single "coarsest possible" dedup group;
    deriving it from the real box removes a magic number and adapts to
    whatever system is actually being calibrated. Still overridable via
    `lambda_max` directly if a caller wants to test past that point anyway.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}.")

    if trajectory_path is None:
        frame_indices: list[int] = [0]
    else:
        universe = mda.Universe(structure_path, trajectory_path)
        n_total = len(universe.trajectory)
        pool = list(range(from_frame, until_frame if until_frame is not None else n_total, step))

        if not pool:
            raise ValueError(
                f"No frames in range [from_frame={from_frame}, until_frame={until_frame}, "
                f"step={step}) for a trajectory with {n_total} frames."
            )

        rng = np.random.default_rng(seed)
        size = min(n_frames, len(pool))
        frame_indices = sorted(int(i) for i in rng.choice(pool, size=size, replace=False))

    frame_diagnostics: list[tuple[str, str]] = []
    box_sizes, leaflet_counts = collect_frame_samples(
        structure_path, trajectory_path, ndx_groups, dynamic_selection,
        frame_indices, min_balance, margin, diagnostics=frame_diagnostics,
    )

    if candidates is not None:
        candidate_list: list[float] = list(candidates)
    else:
        effective_lambda_max = lambda_max
        if effective_lambda_max is None:
            max_box_angstrom = max(
                max(box[0] for box in box_sizes),
                max(box[1] for box in box_sizes),
            )
            effective_lambda_max = float(max_box_angstrom) / 10.0
        candidate_list = [float(v) for v in np.geomspace(lambda_min, effective_lambda_max, n_lambda)]

    all_results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidate_list)
    # frame_diagnostics (e.g. a box approximated from atom positions)
    # apply equally to every candidate, not to one lambda specifically -
    # folded into each result's own diagnostics so the existing report
    # (which already prints every CandidateResult.diagnostics) surfaces
    # them without a separate reporting channel.
    for result in all_results:
        result.diagnostics = frame_diagnostics + result.diagnostics
    winner = recommend_lambda(all_results)

    return winner, all_results


def resolve_build_lambda(
    lambda_x: float | None,
    lambda_y: float | None,
    structure_path: str,
    trajectory_path: str | None,
    *,
    ndx_groups: dict[str, list[int]] | None,
    dynamic_selection: str | None,
    from_frame: int,
    until_frame: int,
    step: int,
) -> tuple[float | None, float | None, CandidateResult | None]:
    """`(lambda_x, lambda_y)` unchanged if either was already given; otherwise both replaced by `calibrate_lambda`'s own recommendation.

    Shared by every real build entry point (`core/fourier_build.py::calc_fourier`,
    `analyze/lipids.py::calc_lipids`, `analyze/diffusion.py::calc_diffusion`) so
    "no --lambda_x/--lambda_y given" means "the finest resolution CALM
    calibrate lambda itself would call safe for this exact structure", not
    `core/fourier_core.py::get_fourier_modes`' own fixed Nx=Ny=3 fallback,
    which has no relationship to the box being fit.

    Only triggers when BOTH are None - a caller giving exactly one of them
    keeps today's existing per-axis behavior (the other still falls through
    to `get_fourier_modes`' own fixed default), since `calibrate_lambda`
    only ever sweeps lambda_x/lambda_y together (isotropically), with no way
    to resolve one axis independently of the other.

    The third return value is the full `CandidateResult` behind an
    auto-selected lambda, for a caller to log - `None` if nothing was
    auto-selected (both were already given).
    """
    if lambda_x is not None or lambda_y is not None:
        return lambda_x, lambda_y, None

    winner, _ = calibrate_lambda(
        structure_path, trajectory_path,
        ndx_groups=ndx_groups, dynamic_selection=dynamic_selection,
        from_frame=from_frame, until_frame=until_frame, step=step,
    )
    return winner.lambda_nm, winner.lambda_nm, winner
