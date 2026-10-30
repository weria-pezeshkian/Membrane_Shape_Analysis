"""Tests for calibrate/lambda_calibration.py: the lambda_x/lambda_y advisor's pure logic
(no CLI/argparse) - oversampling_tier's exact replica of core/fourier_fit.py's own
thresholds, sweep_lambda_candidates' dedup, recommend_lambda's selection/fallback rule,
collect_frame_samples' leaflet-construction, and calibrate_lambda's end-to-end composition.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest

from CALM.calibrate.lambda_calibration import (
    TIER_LOW_OVERSAMPLING,
    TIER_LOW_REDUNDANCY,
    TIER_SAFE,
    TIER_UNDERDETERMINED,
    _decimal_digit_count,
    _frame_dimensions,
    _position_unit_scale_factor,
    calibrate_lambda,
    collect_frame_samples,
    oversampling_tier,
    recommend_lambda,
    sweep_lambda_candidates,
)


# =============================================================================
# oversampling_tier
# =============================================================================

def test_oversampling_tier_length_equals_m_is_underdetermined() -> None:
    assert oversampling_tier(M=10, Length=10) == TIER_UNDERDETERMINED


def test_oversampling_tier_just_below_3x_is_low_redundancy() -> None:
    assert oversampling_tier(M=29, Length=10) == TIER_LOW_REDUNDANCY  # 29 < 3*10


def test_oversampling_tier_exactly_3x_is_low_oversampling() -> None:
    assert oversampling_tier(M=30, Length=10) == TIER_LOW_OVERSAMPLING  # 30 < 30 is False


def test_oversampling_tier_just_below_10x_is_low_oversampling() -> None:
    assert oversampling_tier(M=99, Length=10) == TIER_LOW_OVERSAMPLING  # 99 < 100


def test_oversampling_tier_exactly_10x_is_safe() -> None:
    assert oversampling_tier(M=100, Length=10) == TIER_SAFE  # 100 < 100 is False


# =============================================================================
# sweep_lambda_candidates
# =============================================================================

def test_sweep_takes_the_worst_case_not_the_average_across_frames_and_leaflets() -> None:
    # Two frames, same box: frame 0 has a huge upper leaflet (safe) and a
    # tiny lower leaflet (underdetermined); the candidate's own tier must
    # be driven by the worst (lower) leaflet, not an average of the two.
    box_sizes = [np.array([100.0, 100.0, 50.0])]
    leaflet_counts = [(100000, 1)]

    [result] = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[10.0])
    assert result.tier == TIER_UNDERDETERMINED
    assert result.worst_case_M == 1
    assert result.worst_case_source == (0, "lower")


def test_sweep_collects_get_fourier_modes_diagnostics() -> None:
    # A lambda far larger than the box floors Nx/Ny to 1, which
    # get_fourier_modes reports as a diagnostic.
    box_sizes = [np.array([10.0, 10.0, 50.0])]
    leaflet_counts = [(1000, 1000)]

    [result] = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[50.0])
    assert result.Nx == 1 and result.Ny == 1
    assert any("corrected to Nx=1" in msg for _, msg in result.diagnostics)


def test_sweep_dedups_candidates_sharing_the_same_mode_signature() -> None:
    # 1.99 and 2.0 both floor to Nx=Ny=50 on this box (int(1000/19.9) ==
    # int(1000/20.0) == 50); 2.5 floors to 40, genuinely distinct.
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(5000, 5000)]

    results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[1.99, 2.0, 2.5])
    assert len(results) == 2

    tied = next(r for r in results if r.Nx == 50)
    assert tied.lambda_nm == 2.0, "representative must be the cleanest-looking tied value (fewest decimal digits)"


def test_sweep_tiebreak_prefers_fewer_decimals_even_over_a_larger_value() -> None:
    # 4.8 (1 decimal) and 4.95001 (5 decimals) both floor to Nx=Ny=20 on
    # this box (int(100/4.8) == int(100/4.95001) == 20) - the LARGER value
    # has MORE decimals, so this specifically discriminates "fewest
    # decimals wins" from the old "largest wins" rule (which would have
    # picked 4.95001 here).
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(5000, 5000)]

    [result] = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[4.8, 4.95001])
    assert result.Nx == 20
    assert result.lambda_nm == 4.8


def test_decimal_digit_count() -> None:
    assert _decimal_digit_count(2.0) == 0
    assert _decimal_digit_count(1.99) == 2
    assert _decimal_digit_count(4.95001) == 5
    assert _decimal_digit_count(0.000001) == 6  # capped at max_decimals


def test_sweep_single_candidate_group_has_no_dedup_effect() -> None:
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(5000, 5000)]

    results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[2.0])
    assert len(results) == 1
    assert results[0].lambda_nm == 2.0


# =============================================================================
# recommend_lambda
# =============================================================================

def test_recommend_picks_the_smallest_lambda_that_clears_the_safe_tier() -> None:
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(5000, 5000)]
    # 10.0 -> Nx=10, Length=441, ratio=11.3x (safe); 20.0 -> Nx=5, Length=121,
    # ratio=41.3x (also safe, but coarser) - the finer, still-safe one must win.
    results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[2.0, 5.0, 10.0, 20.0])

    winner = recommend_lambda(results)
    assert winner.tier == TIER_SAFE
    assert winner.lambda_nm == 10.0


def test_recommend_skips_a_safe_tier_candidate_that_misses_the_lipids_per_mode_threshold() -> None:
    # lambda=5.0 -> Nx=Ny=20, Length=1681, ratio=17650/1681=10.50x (clears
    # the safe tier) but lipids_per_mode=17650/(20*20)=44.1 (misses the
    # LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE=50 threshold) - recommend_lambda
    # must pass it over for the coarser lambda=10.0 (Nx=Ny=10, Length=441,
    # ratio=40.02x, lipids_per_mode=17650/100=176.5), which clears both.
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(17650, 17650)]
    results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[5.0, 10.0])

    finer = next(r for r in results if r.lambda_nm == 5.0)
    assert finer.tier == TIER_SAFE
    assert finer.lipids_per_mode < 50.0, "test fixture must actually exercise the discriminating case"

    winner = recommend_lambda(results)
    assert winner.lambda_nm == 10.0


def test_recommend_falls_back_to_best_worst_case_when_nothing_is_safe() -> None:
    box_sizes = [np.array([1000.0, 1000.0, 100.0])]
    leaflet_counts = [(5000, 5000)]
    # All three candidates land well below the safe tier.
    results = sweep_lambda_candidates(box_sizes, leaflet_counts, candidates=[1.99, 2.0, 2.5])

    winner = recommend_lambda(results)
    assert winner.tier != TIER_SAFE
    assert winner.worst_case_ratio == max(r.worst_case_ratio for r in results)


def test_recommend_raises_only_on_empty_results() -> None:
    with pytest.raises(ValueError):
        recommend_lambda([])


# =============================================================================
# collect_frame_samples
# =============================================================================

def _write_two_leaflet_structure(
    path: Path, Lx: float = 1000.0, Ly: float = 1000.0, Lz: float = 100.0, spacing: float = 50.0,
) -> int:
    """Writes a .gro with two flat leaflets of 'P' beads; returns atoms-per-leaflet."""
    xs, ys = np.meshgrid(np.arange(0, Lx, spacing), np.arange(0, Ly, spacing))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    n = xy.shape[0]
    positions = np.vstack([
        np.column_stack([xy, np.full(n, 70.0)]),
        np.column_stack([xy, np.full(n, 30.0)]),
    ])
    u = mda.Universe.empty(n_atoms=positions.shape[0], trajectory=True)
    u.add_TopologyAttr("name", ["P"] * positions.shape[0])
    u.atoms.positions = positions
    u.dimensions = [Lx, Ly, Lz, 90.0, 90.0, 90.0]
    u.atoms.write(str(path))
    return n


def test_collect_frame_samples_static_index_gives_fixed_m_per_frame(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro)
    ndx_groups = {"Upper": list(range(1, n + 1)), "Lower": list(range(n + 1, 2 * n + 1))}

    box_sizes, leaflet_counts = collect_frame_samples(
        structure_path=str(gro), trajectory_path=None,
        ndx_groups=ndx_groups, dynamic_selection=None,
        frame_indices=[0], min_balance=0.6, margin=2.0,
    )
    assert len(box_sizes) == 1
    assert np.allclose(box_sizes[0], [1000.0, 1000.0, 100.0])
    assert leaflet_counts == [(n, n)]


def test_collect_frame_samples_dynamic_selection_rederives_m_per_frame(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro, Lx=300.0, Ly=300.0, spacing=15.0)

    box_sizes, leaflet_counts = collect_frame_samples(
        structure_path=str(gro), trajectory_path=None,
        ndx_groups=None, dynamic_selection="name P",
        frame_indices=[0], min_balance=0.6, margin=2.0,
    )
    assert len(box_sizes) == 1
    assert leaflet_counts == [(n, n)]


def test_collect_frame_samples_rejects_both_or_neither_index_source(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    _write_two_leaflet_structure(gro)

    with pytest.raises(ValueError):
        collect_frame_samples(str(gro), None, {"Upper": [1], "Lower": [2]}, "name P", [0], 0.6, 2.0)

    with pytest.raises(ValueError):
        collect_frame_samples(str(gro), None, None, None, [0], 0.6, 2.0)


def test_collect_frame_samples_opens_structure_only_as_a_single_argument(tmp_path: Path) -> None:
    # Regression test: a structure-only mda.Universe(structure_path,
    # structure_path) call (the same path passed twice) routes MDAnalysis
    # through that FILE FORMAT's own dedicated coordinate-reader lookup for
    # the second argument instead of the topology parser's own embedded
    # coordinates - not every format's coordinate reader supports being
    # reused that way (seen in practice with a .tpr: "Invalid box
    # information"), even though every format works correctly when opened
    # with a single argument. collect_frame_samples must never pass
    # structure_path twice when trajectory_path is None.
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro)
    ndx_groups = {"Upper": list(range(1, n + 1)), "Lower": list(range(n + 1, 2 * n + 1))}

    with patch("CALM.calibrate.lambda_calibration.mda.Universe", wraps=mda.Universe) as mock_universe:
        collect_frame_samples(
            structure_path=str(gro), trajectory_path=None,
            ndx_groups=ndx_groups, dynamic_selection=None,
            frame_indices=[0], min_balance=0.6, margin=2.0,
        )

    mock_universe.assert_called_once_with(str(gro))


def test_collect_frame_samples_dynamic_selection_structure_only_single_argument(tmp_path: Path) -> None:
    # Same regression, exercised through the dynamic -n path specifically
    # (the exact reported failure: `CALM calibrate lambda -s prod.tpr -n
    # "name P"`, no -f/--trajectory) - _build_dynamic_leaflet_reference
    # must also receive a structure-only, single-argument Universe.
    gro = tmp_path / "structure.gro"
    _write_two_leaflet_structure(gro, Lx=300.0, Ly=300.0, spacing=15.0)

    with patch("CALM.calibrate.lambda_calibration.mda.Universe", wraps=mda.Universe) as mock_universe:
        collect_frame_samples(
            structure_path=str(gro), trajectory_path=None,
            ndx_groups=None, dynamic_selection="name P",
            frame_indices=[0], min_balance=0.6, margin=2.0,
        )

    mock_universe.assert_called_once_with(str(gro))


def _write_structure_with_no_box(path: Path, n_atoms: int = 4) -> None:
    """A structure whose Universe comes back with dimensions=None - reproduces, without
    depending on an external .tpr, the real failure: some structure formats don't reliably
    surface box dimensions when opened as the sole -s/--structure source, even when the file
    itself has real box data (seen in practice with a .tpr). Positions are spread out (not all
    at the origin) so the position-derived fallback box has a real, non-degenerate extent."""
    u = mda.Universe.empty(n_atoms=n_atoms, trajectory=True)
    u.add_TopologyAttr("name", ["P"] * n_atoms)
    u.atoms.positions = np.array([[i * 20.0, i * 15.0, i * 10.0] for i in range(n_atoms)])
    u.atoms.write(str(path))  # .gro always carries a box; .pdb with none left unset does not


def test_collect_frame_samples_approximates_missing_dimensions_static(tmp_path: Path) -> None:
    pdb = tmp_path / "structure.pdb"
    _write_structure_with_no_box(pdb)

    diagnostics: list[tuple[str, str]] = []
    box_sizes, leaflet_counts = collect_frame_samples(
        structure_path=str(pdb), trajectory_path=None,
        ndx_groups={"Upper": [1, 2], "Lower": [3, 4]}, dynamic_selection=None,
        frame_indices=[0], min_balance=0.6, margin=2.0, diagnostics=diagnostics,
    )

    assert len(box_sizes) == 1
    assert np.all(box_sizes[0] > 0), "fallback box must have a real, non-degenerate extent"
    assert any("box dimensions missing" in msg for _, msg in diagnostics)


def test_collect_frame_samples_approximates_missing_dimensions_dynamic(tmp_path: Path) -> None:
    pdb = tmp_path / "structure.pdb"
    _write_structure_with_no_box(pdb)

    diagnostics: list[tuple[str, str]] = []
    box_sizes, leaflet_counts = collect_frame_samples(
        structure_path=str(pdb), trajectory_path=None,
        ndx_groups=None, dynamic_selection="name P",
        frame_indices=[0], min_balance=0.6, margin=2.0, diagnostics=diagnostics,
    )

    assert len(box_sizes) == 1
    assert np.all(box_sizes[0] > 0)
    assert any("box dimensions missing" in msg for _, msg in diagnostics)


def test_collect_frame_samples_refuses_to_approximate_a_known_triclinic_box() -> None:
    # A universe whose Timestep DOES carry partial unit-cell info (unlike
    # the all-zero "nothing at all" case above) showing real, non-90-degree
    # angles - approximating a rectangular box from positions here would
    # misrepresent genuinely triclinic data, so this must raise instead.
    # Timestep.dimensions is an immutable-type C-extension property that
    # can't be monkeypatched, so this exercises _frame_dimensions' own
    # triclinic-detection branch directly against a minimal fake carrying
    # only what it actually reads (.trajectory[frame], .dimensions,
    # .trajectory.ts._unitcell) rather than a real MDAnalysis Universe.
    class _FakeTS:
        _unitcell = np.array([100.0, 100.0, 100.0, 70.0, 80.0, 75.0])  # genuinely triclinic

    class _FakeTrajectory:
        ts = _FakeTS()

        def __getitem__(self, frame: int) -> "_FakeTS":
            return self.ts

    class _FakeUniverse:
        dimensions = None
        trajectory = _FakeTrajectory()

    with pytest.raises(ValueError, match="triclinic"):
        _frame_dimensions(_FakeUniverse(), 0, "fake_structure.gro")


def test_collect_frame_samples_corrects_positions_mistakenly_left_in_nm(tmp_path: Path) -> None:
    # Regression test for a confirmed MDAnalysis 2.10.0 bug: its brand-new
    # TPRReader declares units={"length": "nm", ...} and convert_units=True
    # but never actually applies that conversion - real covalent bonds then
    # come out ~10x too short (e.g. ~0.1 "Angstrom" for an N-H bond that's
    # really ~1.0). Reproduced here without a real .tpr: a structure whose
    # own bonds are deliberately nm-scale, with no box, so the fallback
    # must detect and correct for this before approximating a box.
    pdb = tmp_path / "structure.pdb"
    # 3 separate, short (~0.12 "Angstrom" = real nm-scale bond) pairs,
    # spread apart from each other so the overall atom cloud still spans a
    # plausible box extent - every SAMPLED bond must be nm-scale for the
    # median-based detection to fire (a single short bond among otherwise
    # unbonded, widely-spaced atoms would not be representative).
    u = mda.Universe.empty(n_atoms=6, trajectory=True)
    u.add_TopologyAttr("name", ["P"] * 6)
    u.add_bonds([(0, 1), (2, 3), (4, 5)])
    u.atoms.positions = np.array([
        [0.0, 0.0, 0.0], [0.12, 0.0, 0.0],
        [5.0, 5.0, 0.0], [5.12, 5.0, 0.0],
        [15.0, 15.0, 13.0], [15.12, 15.0, 13.0],
    ])
    u.atoms.write(str(pdb))

    diagnostics: list[tuple[str, str]] = []
    box_sizes, _ = collect_frame_samples(
        structure_path=str(pdb), trajectory_path=None,
        ndx_groups={"Upper": [1, 3, 5], "Lower": [2, 4, 6]}, dynamic_selection=None,
        frame_indices=[0], min_balance=0.6, margin=2.0, diagnostics=diagnostics,
    )

    assert np.allclose(box_sizes[0], [151.2, 150.0, 130.0]), "box must be scaled 10x, not left nm-sized"
    assert any("appear to be in nm" in msg for _, msg in diagnostics)


def test_collect_frame_samples_corrects_nm_positions_despite_a_minority_of_long_bonds(
    tmp_path: Path,
) -> None:
    # The nm-detection uses the median over every one of this universe's
    # bonds - robust to a minority of long outliers (e.g. a bond that
    # straddles the periodic boundary in raw, unwrapped coordinates reads
    # as a long cross-box distance, not a true bond length), as long as
    # they stay a minority. Reproduced directly: 15 separate, genuinely
    # nm-scale (~0.12) bonded pairs (the majority) alongside 10 decoy
    # bonds at a uniform 5.0 (A) - squarely outside the nm range on its
    # own (the minority).
    pdb = tmp_path / "structure.pdb"
    n_pairs = 15
    n_decoys = 10
    n_chain = n_decoys + 1
    u = mda.Universe.empty(n_atoms=n_chain + 2 * n_pairs, trajectory=True)
    u.add_TopologyAttr("name", ["P"] * (n_chain + 2 * n_pairs))
    pair_bonds = [(n_chain + 2 * i, n_chain + 2 * i + 1) for i in range(n_pairs)]
    u.add_bonds([(i, i + 1) for i in range(n_chain - 1)] + pair_bonds)
    chain_positions = np.array([[i * 5.0, 100.0, 100.0] for i in range(n_chain)])
    # Each pair spread well apart from the others and from the chain.
    pair_positions = np.array([
        pos
        for i in range(n_pairs)
        for pos in ([i * 20.0, 500.0, 0.0], [i * 20.0 + 0.12, 500.0, 0.0])
    ])
    u.atoms.positions = np.vstack([chain_positions, pair_positions])
    u.atoms.write(str(pdb))

    diagnostics: list[tuple[str, str]] = []
    collect_frame_samples(
        structure_path=str(pdb), trajectory_path=None,
        ndx_groups={"Upper": [1, 3], "Lower": [2, 4]}, dynamic_selection=None,
        frame_indices=[0], min_balance=0.6, margin=2.0, diagnostics=diagnostics,
    )

    assert any("appear to be in nm" in msg for _, msg in diagnostics)


def test_position_unit_scale_factor_raises_when_implausible_under_either_hypothesis() -> None:
    # A median bond length of 50.0 is implausible as a real bond whether
    # read as already-Angstrom (50.0 A, far past any real bond) or as
    # raw nm needing the x10 fix (it isn't below RAW_NM_THRESHOLD, so
    # that correction is never even considered) - corrupted/nonsensical
    # data like this must raise rather than silently pass through
    # uncorrected.
    u = mda.Universe.empty(n_atoms=4, trajectory=True)
    u.add_TopologyAttr("name", ["P"] * 4)
    u.add_bonds([(0, 1), (2, 3)])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [50.0, 50.0, 0.0]])

    with pytest.raises(ValueError, match="implausible"):
        _position_unit_scale_factor(u)


def test_collect_frame_samples_does_not_rescale_normal_angstrom_positions(tmp_path: Path) -> None:
    # The same missing-box fallback, but with ordinary, correctly-scaled
    # (Angstrom) bond lengths - must NOT trigger the nm-correction. Two
    # extra, unbonded marker atoms pin the box's own corners, independent
    # of the bonded chain's own (small, realistic) positions.
    pdb = tmp_path / "structure.pdb"
    u = mda.Universe.empty(n_atoms=6, trajectory=True)
    u.add_TopologyAttr("name", ["P"] * 6)
    u.add_bonds([(0, 1), (1, 2), (2, 3)])
    u.atoms.positions = np.array([
        [10.0, 10.0, 10.0], [11.2, 10.0, 10.0], [11.2, 11.3, 10.0], [11.2, 11.3, 11.4],
        [0.0, 0.0, 0.0], [150.0, 150.0, 130.0],
    ])
    u.atoms.write(str(pdb))

    diagnostics: list[tuple[str, str]] = []
    box_sizes, _ = collect_frame_samples(
        structure_path=str(pdb), trajectory_path=None,
        ndx_groups={"Upper": [1, 2], "Lower": [3, 4]}, dynamic_selection=None,
        frame_indices=[0], min_balance=0.6, margin=2.0, diagnostics=diagnostics,
    )

    assert np.allclose(box_sizes[0], [150.0, 150.0, 130.0])
    assert not any("appear to be in nm" in msg for _, msg in diagnostics)


# =============================================================================
# calibrate_lambda
# =============================================================================

def test_calibrate_lambda_returns_winner_and_all_results(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro)
    ndx_groups = {"Upper": list(range(1, n + 1)), "Lower": list(range(n + 1, 2 * n + 1))}

    winner, all_results = calibrate_lambda(str(gro), None, ndx_groups=ndx_groups)
    assert winner in all_results
    assert winner.lambda_nm > 0


def test_calibrate_lambda_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    gro = tmp_path / "structure.gro"
    n = _write_two_leaflet_structure(gro, spacing=43.0)
    ndx_groups = {"Upper": list(range(1, n + 1)), "Lower": list(range(n + 1, 2 * n + 1))}

    winner1, _ = calibrate_lambda(str(gro), None, ndx_groups=ndx_groups, seed=42)
    winner2, _ = calibrate_lambda(str(gro), None, ndx_groups=ndx_groups, seed=42)
    assert winner1.lambda_nm == winner2.lambda_nm
    assert winner1.Nx == winner2.Nx


def test_calibrate_lambda_default_lambda_max_scales_with_box_size(tmp_path: Path) -> None:
    small_gro = tmp_path / "small.gro"
    n_small = _write_two_leaflet_structure(small_gro, Lx=200.0, Ly=200.0, spacing=20.0)
    ndx_small = {"Upper": list(range(1, n_small + 1)), "Lower": list(range(n_small + 1, 2 * n_small + 1))}

    large_gro = tmp_path / "large.gro"
    n_large = _write_two_leaflet_structure(large_gro, Lx=2000.0, Ly=2000.0, spacing=20.0)
    ndx_large = {"Upper": list(range(1, n_large + 1)), "Lower": list(range(n_large + 1, 2 * n_large + 1))}

    _, results_small = calibrate_lambda(str(small_gro), None, ndx_groups=ndx_small)
    _, results_large = calibrate_lambda(str(large_gro), None, ndx_groups=ndx_large)

    assert max(r.lambda_nm for r in results_large) > max(r.lambda_nm for r in results_small)
