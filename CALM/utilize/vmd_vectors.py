from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

from ..core.fourier_sft import SFT
from ..core.manual import add_manual
from ..core.rotation import recover_all_rotation_angles, rotation_was_used
from ..map.plot import (
    _align_signs_to_lower_z,
    _average_principal_directions,
    _frame_number,
    _hole_masks_for_frame,
    _load_and_mask,
)


def _layer_dir_indices(layer: str) -> tuple[int, int]:
    """(k1, k2) slice-index pair into the 6-slice principal_dirs/principal_curvatures stack for one layer."""
    return {"upper": (0, 1), "lower": (2, 3), "middle": (4, 5)}[layer]


def _z_layer_index(layer: str) -> int:
    """Index into Z_fitted's 3-layer stack (upper, lower, middle) for one layer."""
    return {"upper": 0, "lower": 1, "middle": 2}[layer]


# Neighbors whose stored k1 and k2 continue each other with at least this pairing confidence
# (see `_pairing_confidence`; about 32 degrees between their axes) share a region.
_REGION_CONFIDENCE = 0.3


def _grid_edges(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat index pairs (p, q) of every 4-connected, periodic pair of distinct valid grid points."""
    ny, nx = valid.shape
    index = np.arange(ny * nx).reshape(ny, nx)
    p = np.concatenate([index.ravel(), index.ravel()])
    q = np.concatenate([np.roll(index, -1, axis=1).ravel(), np.roll(index, -1, axis=0).ravel()])
    flat_valid = valid.ravel()
    keep = flat_valid[p] & flat_valid[q] & (p != q)
    return p[keep], q[keep]


def _pairing_confidence(d1: np.ndarray, d2: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """For each grid edge (p, q) of flat (N, 3) k1 and k2 direction fields: how much better q's
    k1 and k2 continue p's k1 and k2 than p's k2 and k1. 1 when q's axes are p's axes, -1 when
    they are p's axes swapped, 0 when they sit 45 degrees from p's."""
    same = np.abs(np.sum(d1[p] * d1[q], axis=-1)) + np.abs(np.sum(d2[p] * d2[q], axis=-1))
    crossed = np.abs(np.sum(d1[p] * d2[q], axis=-1)) + np.abs(np.sum(d2[p] * d1[q], axis=-1))
    return (same - crossed) / 2.0


def family_swap_mask(dirs1: np.ndarray, dirs2: np.ndarray) -> np.ndarray:
    """(Ny, Nx) bool mask of the grid points whose stored k1 and k2 directions are drawn as
    each other's family, so that each drawn family continues smoothly across the grid.

    Stored k1 is the larger signed principal curvature at each point. On an undulating
    membrane it therefore runs across the undulation in troughs and along it on crests, and
    turns 90 degrees at every inflection line. Neighbors whose stored directions clearly
    continue each other (pairing confidence above `_REGION_CONFIDENCE`) form regions that
    keep their stored labels. Each pair of touching regions sums the pairing confidence over
    their shared border: a negative sum means one region continues the other with k1 and k2
    swapped. The regions are linked by a maximum spanning tree over those sums' magnitudes,
    so the clearest borders decide, and walked from the largest region of each linked group,
    which keeps its stored labels. Points NaN in either field are never swapped.
    """
    ny, nx = dirs1.shape[:2]
    n_points = ny * nx
    valid = ~(np.isnan(dirs1).any(axis=-1) | np.isnan(dirs2).any(axis=-1))
    flat1 = dirs1.reshape(-1, 3)
    flat2 = dirs2.reshape(-1, 3)
    p, q = _grid_edges(valid)
    no_swap = np.zeros((ny, nx), dtype=bool)
    if p.size == 0:
        return no_swap

    confidence = _pairing_confidence(flat1, flat2, p, q)
    joined = confidence > _REGION_CONFIDENCE
    joins = coo_matrix((np.ones(int(joined.sum())), (p[joined], q[joined])), shape=(n_points, n_points))
    n_regions, region = connected_components(joins, directed=False)
    border = region[p] != region[q]
    if not border.any():
        return no_swap

    low = np.minimum(region[p[border]], region[q[border]])
    high = np.maximum(region[p[border]], region[q[border]])
    # Duplicate (low, high) entries are summed into one border total per region pair.
    border_sum = coo_matrix((confidence[border], (low, high)), shape=(n_regions, n_regions)).tocsr()
    border_sum.sum_duplicates()
    totals = border_sum.tocoo()
    # A cost of 1/|total| makes the minimum spanning tree the maximum one over |total|.
    cost = coo_matrix(
        (1.0 / (np.abs(totals.data) + 1e-12), (totals.row, totals.col)), shape=(n_regions, n_regions)
    ).tocsr()
    tree = minimum_spanning_tree(cost)
    symmetric_sum = (border_sum + border_sum.T).tocsr()

    n_groups, group = connected_components(tree, directed=False)
    region_size = np.bincount(region[valid.ravel()], minlength=n_regions)
    group_size = np.bincount(group, minlength=n_groups)
    region_swap = np.zeros(n_regions, dtype=bool)
    for label in np.flatnonzero(group_size > 1):
        members = np.flatnonzero(group == label)
        root = int(members[np.argmax(region_size[members])])
        order, parent = breadth_first_order(tree, root, directed=False, return_predecessors=True)
        children = order[1:]
        crossed = np.asarray(symmetric_sum[parent[children], children]).ravel() < 0
        for child, child_crossed in zip(children, crossed, strict=True):
            region_swap[child] = region_swap[parent[child]] ^ child_crossed

    return (region_swap[region] & valid.ravel()).reshape(ny, nx)


def _continues_crossed(
    red: np.ndarray, blue: np.ndarray, previous_red: np.ndarray, previous_blue: np.ndarray
) -> bool:
    """Whether (red, blue) continue (previous_red, previous_blue) better with the two swapped,
    summed over the points valid in all four (Ny, Nx, 3) fields."""
    fields = (red, blue, previous_red, previous_blue)
    valid = ~np.any([np.isnan(f).any(axis=-1) for f in fields], axis=0)
    r, b, pr, pb = (f[valid] for f in fields)
    same = np.abs(np.sum(r * pr, axis=-1)).sum() + np.abs(np.sum(b * pb, axis=-1)).sum()
    crossed = np.abs(np.sum(r * pb, axis=-1)).sum() + np.abs(np.sum(b * pr, axis=-1)).sum()
    return bool(crossed > same)


def _layer_families(
    dirs: np.ndarray,
    k_values: np.ndarray,
    layer: str,
    previous: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(red dirs, red k, blue dirs, blue k) for one layer: the stored k1 and k2 directions and
    curvatures, swapped where `family_swap_mask` says so. With `previous`, the previous
    frame's (red dirs, blue dirs) of this layer, the two families are swapped as a whole when
    that continues the previous frame's families better."""
    k1_slice, k2_slice = _layer_dir_indices(layer)
    swap = family_swap_mask(dirs[k1_slice], dirs[k2_slice])
    red = np.where(swap[..., None], dirs[k2_slice], dirs[k1_slice])
    blue = np.where(swap[..., None], dirs[k1_slice], dirs[k2_slice])
    if previous is not None and _continues_crossed(red, blue, *previous):
        swap = ~swap
        red, blue = blue, red
    red_k = np.where(swap, k_values[k2_slice], k_values[k1_slice])
    blue_k = np.where(swap, k_values[k1_slice], k_values[k2_slice])
    return red, red_k, blue, blue_k


def _frame_vector_list(
    dirs_slice: np.ndarray,
    k_values: np.ndarray,
    z_layer: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    step: int,
    k_index: int,
) -> list[tuple[float, float, float, float, float, float, int, float]]:
    """(x, y, z, dx, dy, dz, k_index, k_value) tuples for one unit-direction field, subsampled by `step`.

    Positions are real coordinates: x, y directly from the box (Angstrom,
    matching sft.dimensions), z from Z_fitted*10 (matching
    get_vmd_visualisation's own nm -> Angstrom convention). (dx, dy, dz) is
    a dimensionless unit tangent vector, unaffected by that unit choice.
    Points where `dirs_slice` is NaN (hole-masked or outside the --rotate
    circle) are skipped. Masking `dirs_slice` decides which points get
    drawn; `k_values` is read only for those same points.
    """
    vectors = []
    ny, nx = X.shape
    for i in range(0, ny, step):
        for j in range(0, nx, step):
            d = dirs_slice[i, j]
            if np.isnan(d).any():
                continue
            vectors.append((
                float(X[i, j]), float(Y[i, j]), float(z_layer[i, j]),
                float(d[0]), float(d[1]), float(d[2]),
                k_index, float(k_values[i, j]),
            ))
    return vectors


def _arrow_endpoint(
    vector: tuple[float, float, float, float, float, float, int, float], dynamic_length: bool, scale: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """(start, end, color) for one (x, y, z, dx, dy, dz, k_index, k_value) vector tuple.

    Arrow length is a fixed 15 Angstrom by default, or 10 * |k_value|
    Angstrom with `dynamic_length` (k_value is in nm^-1, matching
    principal_curvatures.npy), then multiplied by `scale`. Color encodes
    k_index: red for k1, blue for k2.
    """
    x, y, z, dx, dy, dz, k_index, k_value = vector
    length = (10.0 * abs(k_value) if dynamic_length else 15.0) * scale
    end = (x + dx * length, y + dy * length, z + dz * length)
    color = "red" if k_index == 1 else "blue"
    return (x, y, z), end, color


_CALM_DRAW_ARROW_PROC = [
    "proc calm_draw_arrow {start end color} {",
    "    draw color $color",
    "    set diff [vecsub $end $start]",
    "    set len [veclength $diff]",
    "    if {$len < 0.001} { return {} }",
    "    set unit [vecscale [expr {1.0/$len}] $diff]",
    "    set shaft_end [vecadd $start [vecscale [expr {$len*0.7}] $unit]]",
    "    set id1 [draw cylinder $start $shaft_end radius 0.3 resolution 12 filled yes]",
    "    set id2 [draw cone $shaft_end $end radius 0.7 resolution 12]",
    "    return [list $id1 $id2]",
    "}",
]


def _static_vector_tcl_lines(vectors: list[tuple], dynamic_length: bool, scale: float) -> list[str]:
    """One `calm_draw_arrow` call per vector, drawn once with no frame tracking."""
    lines = []
    for vector in vectors:
        (x, y, z), (x2, y2, z2), color = _arrow_endpoint(vector, dynamic_length, scale)
        lines.append(
            f"calm_draw_arrow {{{x:.4f} {y:.4f} {z:.4f}}} {{{x2:.4f} {y2:.4f} {z2:.4f}}} {color}"
        )
    return lines


def _frame_endpoint_tcl_list(vectors: list[tuple], dynamic_length: bool, scale: float) -> str:
    """One frame's vectors as a TCL sublist of `{x1 y1 z1 x2 y2 z2 color}` entries."""
    entries = []
    for vector in vectors:
        (x, y, z), (x2, y2, z2), color = _arrow_endpoint(vector, dynamic_length, scale)
        entries.append(f"{{{x:.4f} {y:.4f} {z:.4f} {x2:.4f} {y2:.4f} {z2:.4f} {color}}}")
    return "    { " + " ".join(entries) + " }"


def _selected_vectors(
    families: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    step: int,
    which: str,
) -> list[tuple]:
    """Every requested (layer, family) combination's vector list, concatenated: red (k1)
    and blue (k2) from each layer's `_layer_families` result, each sign-aligned."""
    vectors: list[tuple] = []
    for layer, (red, red_k, blue, blue_k) in families.items():
        z_layer = z[_z_layer_index(layer)]
        if which in ("k1", "both"):
            vectors += _frame_vector_list(_align_signs_to_lower_z(red), red_k, z_layer, X, Y, step, 1)
        if which in ("k2", "both"):
            vectors += _frame_vector_list(_align_signs_to_lower_z(blue), blue_k, z_layer, X, Y, step, 2)
    return vectors


def build_static_vectors_tcl(
    sft: SFT,
    curvature_dir: str,
    out_path: str,
    which: str = "both",
    layers: list[str] | None = None,
    step: int = 5,
    dynamic_length: bool = False,
    scale: float = 10.0,
) -> None:
    """Write a VMD TCL script drawing one static set of principal-direction arrows.

    Directions come from the trajectory-mean (nematic-averaged)
    `principal_dirs.npy`; positions from the plain mean of `Z_fitted.npy` -
    the same average `average_structure.gro` itself is built from. Source
    this script against `average_structure.gro`. `scale` multiplies the base
    arrow length (see `_arrow_endpoint`); the default of 10 matches CALM's
    other nm-to-Angstrom conversions, since VMD's own coordinate system is
    Angstrom.
    """
    layers = layers if layers is not None else ["upper", "lower", "middle"]
    if not curvature_dir.endswith("/"):
        curvature_dir += "/"

    dir_files = sorted(glob.glob(curvature_dir + "*_principal_dirs.npy"))
    if not dir_files:
        raise FileNotFoundError(
            f"No *_principal_dirs.npy files found in {curvature_dir} - "
            "build with 'CALM analyze full --method principal_directions' first."
        )
    z_files = sorted(glob.glob(curvature_dir + "*_Z_fitted.npy"))
    if not z_files:
        raise FileNotFoundError(f"No *_Z_fitted.npy files found in {curvature_dir}")

    layer_sources = ["upper", "upper", "lower", "lower", "union", "union"]
    dirs_mean = _average_principal_directions(dir_files, "*_principal_dirs.npy", curvature_dir, sft, layer_sources)

    if dynamic_length:
        curv_files = sorted(glob.glob(curvature_dir + "*_principal_curvatures.npy"))
        if not curv_files:
            raise FileNotFoundError(
                f"--dynamic-length needs *_principal_curvatures.npy in {curvature_dir} - "
                "build with 'CALM analyze full --method principal' too."
            )
        k_mean = _load_and_mask(curv_files, "*_principal_curvatures.npy", curvature_dir, sft, layer_sources)
    else:
        k_mean = np.zeros(dirs_mean.shape[:-1])

    assert sft.dimensions is not None
    box_size = sft.dimensions[0]
    gridsize = dirs_mean.shape[1]
    x = np.linspace(0, box_size[0], gridsize, endpoint=False)
    y = np.linspace(0, box_size[1], gridsize, endpoint=False)
    X, Y = np.meshgrid(x, y)

    avg_z = np.zeros_like(np.load(z_files[0]))
    for f in z_files:
        avg_z += np.load(f)
    avg_z = avg_z / len(z_files) * 10

    families = {layer: _layer_families(dirs_mean, k_mean, layer) for layer in layers}
    vectors = _selected_vectors(families, avg_z, X, Y, step, which)

    lines = [
        "# Auto-generated by CALM link vmd_vectors",
        "# Static principal-direction arrows (trajectory-averaged).",
        "# Source against average_structure.gro (from CALM link vmd_xtc).",
        "",
        *_CALM_DRAW_ARROW_PROC,
        "",
        *_static_vector_tcl_lines(vectors, dynamic_length, scale),
        "",
        f'puts "Drew {len(vectors)} principal-direction arrows."',
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_dynamic_vectors_tcl(
    sft: SFT,
    curvature_dir: str,
    out_path: str,
    which: str = "both",
    layers: list[str] | None = None,
    step: int = 5,
    dynamic_length: bool = False,
    scale: float = 10.0,
) -> None:
    """Write a VMD TCL script that redraws principal-direction arrows for the current frame.

    Each fit-frame's own directions and heights are used. Source this script
    against `trajectory.xtc`; it registers a `vmd_frame` trace so the arrows
    redraw automatically whenever the displayed frame changes, and draws
    frame 0 immediately. `scale` multiplies the base arrow length (see
    `_arrow_endpoint`); the default of 10 matches CALM's other nm-to-Angstrom
    conversions, since VMD's own coordinate system is Angstrom.
    """
    layers = layers if layers is not None else ["upper", "lower", "middle"]
    if not curvature_dir.endswith("/"):
        curvature_dir += "/"

    dir_files = sorted(glob.glob(curvature_dir + "*_principal_dirs.npy"))
    if not dir_files:
        raise FileNotFoundError(
            f"No *_principal_dirs.npy files found in {curvature_dir} - "
            "build with 'CALM analyze full --method principal_directions' first."
        )
    z_files = sorted(glob.glob(curvature_dir + "*_Z_fitted.npy"))
    if not z_files:
        raise FileNotFoundError(f"No *_Z_fitted.npy files found in {curvature_dir}")

    if dynamic_length and not glob.glob(curvature_dir + "*_principal_curvatures.npy"):
        raise FileNotFoundError(
            f"--dynamic-length needs *_principal_curvatures.npy in {curvature_dir} - "
            "build with 'CALM analyze full --method principal' too."
        )

    layer_sources = ["upper", "upper", "lower", "lower", "union", "union"]
    assert sft.dimensions is not None
    box_size = sft.dimensions[0]

    sft_with_holes = sft if sft.hole_mask is not None else None
    thetas = None
    if sft_with_holes is not None and rotation_was_used(sft_with_holes):
        thetas = recover_all_rotation_angles(sft_with_holes)

    frame_lines = []
    previous: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dir_file, z_file in zip(dir_files, z_files, strict=True):
        dirs = np.load(dir_file)
        z = np.load(z_file) * 10

        if dynamic_length:
            curv_file = dir_file.replace("_principal_dirs.npy", "_principal_curvatures.npy")
            k_values = np.load(curv_file)
        else:
            k_values = np.zeros(dirs.shape[:-1])

        if sft_with_holes is not None:
            assert sft_with_holes.frame_indices is not None
            matches = np.nonzero(sft_with_holes.frame_indices == _frame_number(dir_file))[0]
            if matches.size:
                idx = matches[0]
                theta = thetas[idx] if thetas is not None else 0.0
                upper_hole, lower_hole = _hole_masks_for_frame(sft_with_holes, idx, theta)
                sources = {"upper": upper_hole, "lower": lower_hole, "union": upper_hole | lower_hole}
                dirs = dirs.copy()
                for layer_idx, source in enumerate(layer_sources):
                    dirs[layer_idx][sources[source]] = np.nan

        gridsize = dirs.shape[1]
        x = np.linspace(0, box_size[0], gridsize, endpoint=False)
        y = np.linspace(0, box_size[1], gridsize, endpoint=False)
        X, Y = np.meshgrid(x, y)

        families = {layer: _layer_families(dirs, k_values, layer, previous.get(layer)) for layer in layers}
        previous = {layer: (red, blue) for layer, (red, _, blue, _) in families.items()}
        vectors = _selected_vectors(families, z, X, Y, step, which)
        frame_lines.append(_frame_endpoint_tcl_list(vectors, dynamic_length, scale))

    lines = [
        "# Auto-generated by CALM link vmd_vectors",
        "# Dynamic principal-direction arrows: redraws every frame change.",
        "# Source against trajectory.xtc (from CALM link vmd_xtc).",
        "",
        *_CALM_DRAW_ARROW_PROC,
        "",
        "set calm_frame_vectors {",
        *frame_lines,
        "}",
        "",
        "set calm_molid [molinfo top]",
        "set calm_draw_ids {}",
        "",
        "proc calm_redraw_vectors {args} {",
        "    global calm_frame_vectors calm_draw_ids calm_molid",
        "    foreach id $calm_draw_ids { draw delete $id }",
        "    set calm_draw_ids {}",
        "    set f [molinfo $calm_molid get frame]",
        "    if {$f >= [llength $calm_frame_vectors]} { return }",
        "    foreach entry [lindex $calm_frame_vectors $f] {",
        "        lassign $entry x1 y1 z1 x2 y2 z2 color",
        "        lappend calm_draw_ids {*}[calm_draw_arrow [list $x1 $y1 $z1] [list $x2 $y2 $z2] $color]",
        "    }",
        "}",
        "",
        "set nf [molinfo $calm_molid get numframes]",
        'if {$nf != [llength $calm_frame_vectors]} {',
        '    puts "WARNING: molecule has $nf frames but this script has [llength $calm_frame_vectors] - '
        'check you sourced this against the matching trajectory.xtc."',
        '}',
        "trace add variable vmd_frame($calm_molid) write calm_redraw_vectors",
        "calm_redraw_vectors",
        'puts "Sourced principal-direction vectors for $nf frames: they redraw automatically as you change frames."',
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_vmd_vectors_parser() -> argparse.ArgumentParser:
    """The 'CALM link vmd_vectors' parser alone, with no side effects - shared by the CLI entry point
    below and by anything else that needs this command's own flags (e.g. the GUI's form generator)."""
    parser = argparse.ArgumentParser(
        description="Write VMD TCL scripts drawing principal-direction arrows in real space"
    )
    parser.add_argument("-i", "--input", required=True, help="'CALM analyze full' output directory")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument(
        "--which", choices=["k1", "k2", "both"], default="both",
        help="which direction family to draw: k1 (red), k2 (blue), or both (default: both)",
    )
    parser.add_argument(
        "--layer", dest="layers", nargs="+", choices=["upper", "lower", "middle"],
        default=["upper", "lower", "middle"], help="which layer(s) to draw (default: all three)",
    )
    parser.add_argument("--step", type=int, default=5, help="grid subsampling stride (default: 5)")
    parser.add_argument(
        "--dynamic-length", dest="dynamic_length", action="store_true", default=False,
        help="scale arrow length by the curvature along each arrow's direction instead of a fixed length",
    )
    parser.add_argument(
        "--scale", type=float, default=10.0,
        help="multiplier on arrow length (default: 10, converting CALM's nm-based "
             "lengths to VMD's Angstrom coordinate system)",
    )
    parser.add_argument(
        "--draw-all-frames", dest="all_frames", action="store_true", default=False,
        help="write an animated tcl to draw principal vectors for all frames (default: false)"
    )
    add_manual(parser, "link_vmd_vectors")
    return parser


def vmd_vectors(args: list[str]) -> None:
    """CLI entry: write VMD TCL scripts drawing principal-direction arrows in real space.

    Writes `principal_vectors_static.tcl` (source against
    `average_structure.gro`) and `principal_vectors_dynamic.tcl` (source
    against `trajectory.xtc`, redraws automatically as the displayed frame
    changes) - both from `CALM link vmd_xtc`'s output directory.
    """
    parser = _build_vmd_vectors_parser()
    ns = parser.parse_args(args)

    os.makedirs(ns.output, exist_ok=True)

    sft = SFT.from_directory(ns.input)

    static_path = os.path.join(ns.output, "principal_vectors_static.tcl")
    build_static_vectors_tcl(
        sft, ns.input, static_path, ns.which, ns.layers, ns.step, ns.dynamic_length, ns.scale
    )
    print(f"Wrote {static_path} (source against average_structure.gro)")

    if ns.all_frames:
        dynamic_path = os.path.join(ns.output, "principal_vectors_dynamic.tcl")
        build_dynamic_vectors_tcl(
            sft, ns.input, dynamic_path, ns.which, ns.layers, ns.step, ns.dynamic_length, ns.scale
        )
        print(f"Wrote {dynamic_path} (source against trajectory.xtc)")


if __name__ == "__main__":
    pass
