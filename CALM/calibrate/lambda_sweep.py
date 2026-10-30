from __future__ import annotations

import argparse

from ..core import argument_parser as arg_helper
from ..core.manual import add_manual
from .lambda_calibration import LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE, TIER_SAFE, CandidateResult, calibrate_lambda

# Defaults repeated here (not just in calibrate_lambda's own signature) so
# "--lambda-candidates was given alongside a non-default grid flag" can be
# detected after parsing, without an argparse.SUPPRESS/sentinel trick.
_DEFAULT_LAMBDA_MIN = 1.0
_DEFAULT_N_LAMBDA = 20


def _build_lambda_sweep_parser() -> argparse.ArgumentParser:
    """The 'CALM calibrate lambda' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator).

    Deliberately not built on core/argument_parser.py's add_build_arguments:
    that pulls in --rotate/--Workers/--clear/replay-log machinery this tool
    doesn't need - it never persists an SFT or fits a real surface, only
    counts atoms (see calibrate/lambda_calibration.py).
    """
    parser = argparse.ArgumentParser(
        description="Sweep candidate lambda_x/lambda_y values and recommend the finest resolution "
                     "that would not trigger 'CALM analyze sft/full/lipids/diffusion's own "
                     "underdetermined/low-redundancy/low-oversampling fit warnings.",
        prog="CALM calibrate lambda",
    )
    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")

    required.add_argument('-s', '--structure', type=str, required=True, help="structure file (.tpr/.gro)")
    required.add_argument(
        '-n', '--index', type=str, default=None,
        help="MDAnalysis dynamic selection string for per-frame leaflet detection, e.g. 'name PO4' "
             "(ignored, with a warning, if --index-file is also given)",
    )
    required.add_argument(
        '--index-file', type=str, default=None,
        help="GROMACS-style index (.ndx) file with Upper/Lower groups (e.g. as written by "
             "'CALM link write_ndx'), for static leaflet membership instead of per-frame detection. "
             "Takes precedence over -n/--index if both are given. One of -n/--index-file is required.",
    )

    optional.add_argument(
        '-f', '--trajectory', type=str, default=None,
        help="trajectory file (.xtc). Without this, only -s/--structure's own single coordinate set "
             "is evaluated - no frame sampling.",
    )
    optional.add_argument('-F', '--From', dest='from_frame', default=0, type=int, help="first eligible frame, inclusive (default: 0)")
    optional.add_argument('-U', '--Until', dest='until_frame', default=None, type=arg_helper.none_or_int, help="last eligible frame, exclusive (default: end)")
    optional.add_argument('-S', '--Step', dest='step', default=1, type=int, help="eligible-frame stride (default: 1)")
    optional.add_argument(
        '--n-frames', default=5, type=int,
        help="how many frames to randomly sample from the eligible range (default: 5). Ignored, with "
             "a note, if -f/--trajectory isn't given.",
    )
    optional.add_argument('--seed', default=0, type=int, help="random seed for frame sampling (default: 0)")

    optional.add_argument('--lambda-min', default=_DEFAULT_LAMBDA_MIN, type=float, help="smallest candidate lambda, nm (default: 1.0)")
    optional.add_argument(
        '--lambda-max', default=None, type=float,
        help="largest candidate lambda, nm (default: derived from the sampled frames' own largest "
             "box dimension - sweeping past that point only adds candidates that collapse into the "
             "same coarsest-possible configuration)",
    )
    optional.add_argument('--n-lambda', default=_DEFAULT_N_LAMBDA, type=int, help="candidate grid size (default: 20)")
    optional.add_argument(
        '--lambda-candidates', nargs='+', type=float, default=None,
        help="explicit candidate lambda values (nm), overriding --lambda-min/--lambda-max/--n-lambda",
    )

    optional.add_argument('--min-balance', default=0.6, type=float, help="dynamic leaflet-split balance threshold (default: 0.6, only used with -n)")
    optional.add_argument('--margin', default=2.0, type=float, help="dynamic leaflet-split margin (default: 2.0, only used with -n)")

    add_manual(parser, "calibrate_lambda")
    return parser


def _format_report(
    structure: str, frame_indices_note: str, all_results: list[CandidateResult], winner: CandidateResult,
) -> str:
    lines = [f"Lambda calibration for {structure}", frame_indices_note, ""]

    lines.append(f"  {'lambda (nm)':>12}  {'Nx':>4}  {'Ny':>4}  {'worst-case M/Length':>20}  {'lipids/mode':>11}  tier")
    lines.append("  " + "-" * 83)
    for result in sorted(all_results, key=lambda r: r.lambda_nm):
        lines.append(
            f"  {result.lambda_nm:>12.3f}  {result.Nx:>4}  {result.Ny:>4}  "
            f"{result.worst_case_ratio:>19.2f}x  {result.lipids_per_mode:>11.1f}  {result.tier}"
        )
    lines.append("")

    diagnostics = {msg for result in all_results for _, msg in result.diagnostics}
    for msg in sorted(diagnostics):
        lines.append(f"  note: {msg}")
    if diagnostics:
        lines.append("")

    clears_both = winner.tier == TIER_SAFE and winner.lipids_per_mode >= LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE
    if clears_both:
        lines.append(
            f"Recommended: lambda_x = lambda_y = {winner.lambda_nm:.3f} nm "
            f"(Nx={winner.Nx}, Ny={winner.Ny}, worst-case oversampling {winner.worst_case_ratio:.2f}x, "
            f"{winner.lipids_per_mode:.1f} lipids/mode, tier: {winner.tier})"
        )
    else:
        lines.append(
            f"Best available (no candidate clears both the safe tier and the "
            f"{LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE:g}-lipids-per-mode threshold): "
            f"lambda_x = lambda_y = {winner.lambda_nm:.3f} nm "
            f"(Nx={winner.Nx}, Ny={winner.Ny}, worst-case oversampling {winner.worst_case_ratio:.2f}x, "
            f"{winner.lipids_per_mode:.1f} lipids/mode, tier: {winner.tier})"
        )
    lines.append(
        "(lipids/mode: fit atoms per finest-resolved patch, M/(Nx*Ny) - the recommendation "
        f"requires >= {LINDAHL_EDHOLM_MIN_LIPIDS_PER_MODE:g}, the genuine-collective-undulation "
        "threshold from Lindahl & Edholm, Biophys. J. 79:426-433, 2000)"
    )

    return "\n".join(lines)


def lambda_sweep(args: list[str]) -> None:
    """CLI entry: recommend lambda_x/lambda_y for a structure (optionally sampling a trajectory)."""
    parser = _build_lambda_sweep_parser()
    ns = parser.parse_args(args)

    arg_helper.validate_index_arguments(parser, ns, required=True)
    ndx_groups, dynamic_selection = arg_helper.resolve_index_source(ns)

    if ns.trajectory is None and (ns.n_frames != 5 or ns.seed != 0):
        print("note: --n-frames/--seed are ignored without -f/--trajectory (only one frame to sample).")

    if ns.lambda_candidates is not None and (
        ns.lambda_min != _DEFAULT_LAMBDA_MIN or ns.lambda_max is not None or ns.n_lambda != _DEFAULT_N_LAMBDA
    ):
        print("note: --lambda-candidates was given; --lambda-min/--lambda-max/--n-lambda are ignored.")

    winner, all_results = calibrate_lambda(
        structure_path=ns.structure,
        trajectory_path=ns.trajectory,
        ndx_groups=ndx_groups,
        dynamic_selection=dynamic_selection,
        from_frame=ns.from_frame,
        until_frame=ns.until_frame,
        step=ns.step,
        n_frames=ns.n_frames,
        seed=ns.seed,
        candidates=ns.lambda_candidates,
        lambda_min=ns.lambda_min,
        lambda_max=ns.lambda_max,
        n_lambda=ns.n_lambda,
        min_balance=ns.min_balance,
        margin=ns.margin,
    )

    frame_indices_note = (
        "Evaluated -s/--structure's own single coordinate set (no -f/--trajectory given)."
        if ns.trajectory is None
        # Up to n_frames - calibrate_lambda clips to the eligible frame pool's own size if smaller.
        else f"Sampled up to {ns.n_frames} frame(s) from {ns.trajectory} (seed={ns.seed})."
    )

    print(_format_report(ns.structure, frame_indices_note, all_results, winner))


if __name__ == "__main__":
    pass
