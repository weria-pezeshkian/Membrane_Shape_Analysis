import argparse, logging, os, glob, shutil, tempfile, subprocess, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from PIL import Image
from tqdm import tqdm
from ..core.calc_vectors import _rel_indices_within_protein
from ..core.calc_vectors import get_rotation_and_protein


logger = logging.getLogger(__name__)

# ====================== utilty functions ======================

def _require(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")

def _inscribed_disk_mask(shape, inset_px=0.5):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    cx, cy = (W-1)/2.0, (H-1)/2.0
    r = min(cx, cy) - inset_px
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    return dist <= r

def _theta_from_OP_box(O_box, P_box, Lx, Ly):
    """
    Changes the angle (degrees) from the vector O→P to the reference (x=Lx/2, y=Ly).
    - O_box, P_box are in box (nm) coordinates.
    - The reference direction points from O toward the top-center of the box.
    """
    O = np.asarray(O_box, float); P = np.asarray(P_box, float)
    side = np.array([Lx/2.0, Ly], float)
    ba = P - O
    bc = side - O
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0: return 0.0
    cos_t = np.clip(np.dot(ba, bc)/denom, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_t))
    cross = ba[0]*bc[1] - ba[1]*bc[0]
    return +theta if cross > 0 else -theta

def _rotate_image(arr2d, theta_deg):
    try:
        from scipy.ndimage import rotate
        return rotate(arr2d, theta_deg, reshape=False, order=0,
                      mode="nearest", prefilter=False)
    except Exception:
        return arr2d

def _rotate_points_about_center(points_xy, theta_deg, center_xy):
    pts = np.asarray(points_xy, float)
    c = np.asarray(center_xy, float)
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    return (pts - c) @ R.T + c

# ====================== loading ======================

def _load_box(dirpath):
    path = os.path.join(dirpath, "boxsize.npy")
    _require(path, "boxsize.npy")
    box = np.load(path).astype(float)
    return float(box[0]), float(box[1])

def _sorted_frame_files(dirpath, pattern):
    """
    Glob and sort by the numeric token located at [-2] when splitting
    the basename on underscores. This assumes filenames like:
        prefix_<frame>_<Layer>.npy
    If your naming differs, update this sorter (regex might be safer).
    """
    files = sorted(glob.glob(os.path.join(dirpath, pattern)),
                   key=lambda p: int(os.path.basename(p).split("_")[-2]))
    return files

def _load_field_stack(dirpath, layer, key):
    pat = f"Z_fitted_*_{layer}.npy" if key=="Zfit" else f"curvature_frame_*_{layer}.npy"
    files = _sorted_frame_files(dirpath, pat)
    if not files:
        raise FileNotFoundError(f"No frames matched: {pat}")
    sample = np.load(files[0])
    H, W = sample.shape
    stack = np.empty((len(files), H, W), dtype=sample.dtype)
    for i, fp in enumerate(files):
        stack[i] = np.load(fp)
    return stack

def _load_vectors_and_protein(dirpath):
    o_path = os.path.join(dirpath, "rotation_vectors_o.npy")
    p_path = os.path.join(dirpath, "rotation_vectors_p.npy")
    prot_path = os.path.join(dirpath, "protein_atom_positions_rotation.npy")
    _require(o_path, "rotation_vectors_o.npy")
    _require(p_path, "rotation_vectors_p.npy")
    _require(prot_path, "protein_atom_positions_rotation.npy")

    o_all = np.asarray(np.load(o_path))[:, :2].astype(float)   # (F,2)
    p_all = np.asarray(np.load(p_path))[:, :2].astype(float)   # (F,2)

    prot_raw = np.asarray(np.load(prot_path))
    if prot_raw.ndim == 3:
        prot_all_box = prot_raw[:, :, :2].astype(float)
        logging.info(f"Protein per-frame overlay detected: {prot_all_box.shape}")
    elif prot_raw.ndim == 2:
        prot_all_box = prot_raw[None, :, :2].astype(float)
        logging.info(f"Protein single-frame overlay detected: {prot_all_box.shape}")
    elif prot_raw.ndim == 1:
        prot_all_box = prot_raw[None, None, :2].astype(float)
        logging.info(f"Protein single-atom overlay detected: {prot_all_box.shape}")
    else:
        raise ValueError(f"Unexpected protein array shape: {prot_raw.shape}")

    return o_all, p_all, prot_all_box

# ====================== per-frame transforms ======================

def _nm_to_px(pt_box, W, H, Lx, Ly):
    x_px = pt_box[0] / Lx * (W-1)
    y_px = pt_box[1] / Ly * (H-1)
    return np.array([x_px, y_px], float)

def _px_shift_to_box(sx, sy, W, H, Lx, Ly):
    return sx * (Lx/(W-1)), sy * (Ly/(H-1))

def _recenter_roll(arr2d, O_px):
    """
    - This is a periodic (wraparound) shift in both axes.
    - The returned (sx, sy) are in pixels; later converted back to nm so we can
      adjust P and protein coordinates consistently in box units.
    """
    H, W = arr2d.shape
    cx, cy = (W-1)/2.0, (H-1)/2.0
    sx = int(round(cx - O_px[0]))
    sy = int(round(cy - O_px[1]))
    rec = np.roll(np.roll(arr2d, sx, axis=1), sy, axis=0)
    return rec, (sx, sy)

def _calc_frame_recenter_and_theta(arr_raw, O_box, P_box, Lx, Ly):
    """
      1) convert O from nm→px and recenter the array via circular roll,
      2) convert the applied pixel shift back to nm (dx_box, dy_box),
      3) compute θ using the recentered O' = (Lx/2, Ly/2) and shifted P' = P + (dx,dy).
    """

    H, W = arr_raw.shape
    O_px = _nm_to_px(O_box, W, H, Lx, Ly)
    rec, (sx, sy) = _recenter_roll(arr_raw, O_px)
    dx_box, dy_box = _px_shift_to_box(sx, sy, W, H, Lx, Ly)
    O_prime_box = np.array([Lx/2.0, Ly/2.0], float)
    P_prime_box = np.array([P_box[0] + dx_box, P_box[1] + dy_box], float)
    theta = _theta_from_OP_box(O_prime_box, P_prime_box, Lx, Ly)
    return rec, theta, dx_box, dy_box

def _protein_for_frame(prot_all_box, i):
    F = prot_all_box.shape[0]
    i0 = i if F > 1 else 0
    return prot_all_box[i0]  # (N,2)

# ====================== figure helpers ======================

def _set_axes_style(ax, Lx, Ly):
    ax.set_aspect(Lx/Ly)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_xticks([0, Lx]); ax.set_yticks([0, Ly])
    ax.set_xticklabels(['0', 'L$_x$']); ax.set_yticklabels(['0', 'L$_y$'])
    ax.set_facecolor((0.1, 0.1, 0.1))
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

def _colorbar_with_ticks(fig, cax, vmin, vmax, cmap):
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"Curvature ($\mathrm{nm^{-1}}$)", color='white')
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cb.ax.tick_params(labelsize=10, colors='white')
    for tick in cb.ax.get_yticklabels(): tick.set_color('white')
    cb.outline.set_edgecolor('white')
    return cb

def _fig_dual(Lx, Ly, vmin, vmax, cmap):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.15)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    _set_axes_style(axL, Lx, Ly)
    _set_axes_style(axR, Lx, Ly)
    fig.patch.set_facecolor('black')
    _colorbar_with_ticks(fig, cax, vmin, vmax, cmap)
    return fig, axL, axR

def _fig_single(Lx, Ly, vmin, vmax, cmap):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.055], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    _set_axes_style(ax, Lx, Ly)
    fig.patch.set_facecolor('black')
    _colorbar_with_ticks(fig, cax, vmin, vmax, cmap)
    return fig, ax

# --------- contour helpers ---------

def _draw_contour(ax, arr_box, Lx, Ly, vmin, vmax, cmap, mask_bool, levels):
    """
    Draw a contourf where x ∈ [0, Lx], y ∈ [0, Ly] (box units, nm).
    If 'mask_bool' is provided, values outside the mask are set to NaN.
    """
    a = np.array(arr_box, float)
    if mask_bool is not None:
        a = np.where(mask_bool, a, np.nan)
    cs = ax.contourf(
        np.linspace(0, Lx, a.shape[1]),
        np.linspace(0, Ly, a.shape[0]),
        a, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
    )
    return cs

def _clear_contour(cs):
    if cs is None: return
    for coll in list(getattr(cs, "collections", [])):
        try: coll.remove()
        except Exception: pass

def _update_contour(ax, old_cs, arr_box, Lx, Ly, vmin, vmax, cmap, mask_bool, levels):
    """
      1) Remove all previous contour collections from 'old_cs' (if any).
      2) Draw a new filled contour ('arr_box') using the same style and limits.
    """
    _clear_contour(old_cs)
    return _draw_contour(ax, arr_box, Lx, Ly, vmin, vmax, cmap, mask_bool, levels)

# --------- protein overlay ---------

def _scatter_protein(ax, pts, origin_idx, p2_idx):
    arts = []
    if pts.size == 0:
        return arts
    n = pts.shape[0]
    mask_other = np.ones(n, dtype=bool)
    if 0 <= origin_idx < n: mask_other[origin_idx] = False
    if 0 <= p2_idx < n: mask_other[p2_idx] = False
    if mask_other.any():
        arts.append(ax.scatter(pts[mask_other,0], pts[mask_other,1],
                               s=20, c='white', edgecolors="black",
                               linewidths=0.5, alpha=0.9, zorder=10))
    if 0 <= origin_idx < n:
        p = pts[origin_idx]
        arts.append(ax.scatter([p[0]],[p[1]], s=22, c='red',
                               edgecolors="black", linewidths=0.7,
                               alpha=0.95, zorder=12))
    if 0 <= p2_idx < n:
        p = pts[p2_idx]
        arts.append(ax.scatter([p[0]],[p[1]], s=22, c='lime',
                               edgecolors="black", linewidths=0.7,
                               alpha=0.95, zorder=12))
    return arts

# ====================== COM helpers (optional selection files) ======================

def _try_load_selection_indices(dirpath):
    fO = os.path.join(dirpath, "rotation_origin_indices.npy")
    fP = os.path.join(dirpath, "rotation_p2_indices.npy")
    O_idx = np.load(fO) if os.path.exists(fO) else None
    P_idx = np.load(fP) if os.path.exists(fP) else None
    # allow either 1D or 2D; keep as list of ints
    if O_idx is not None:
        O_idx = np.asarray(O_idx).astype(int).ravel()
    if P_idx is not None:
        P_idx = np.asarray(P_idx).astype(int).ravel()
    return O_idx, P_idx

def _scatter_com_markers(ax, pts, O_idx, P_idx, O_color="red", P_color="lime"):
    arts = []
    if pts.size == 0:
        return arts
    n = pts.shape[0]
    def _safe_com(indices):
        valid = [i for i in indices if 0 <= i < n]
        if len(valid) == 0:
            return None
        sub = pts[valid]
        return np.array([sub[:,0].mean(), sub[:,1].mean()], float)

    if O_idx is not None:
        comO = _safe_com(O_idx)
        if comO is not None:
            arts.append(ax.scatter([comO[0]], [comO[1]], s=28, c=O_color,
                                   edgecolors="black", linewidths=0.8, zorder=15))
    if P_idx is not None:
        comP = _safe_com(P_idx)
        if comP is not None:
            arts.append(ax.scatter([comP[0]], [comP[1]], s=28, c=P_color,
                                   edgecolors="black", linewidths=0.8, zorder=15))
    return arts

# ====================== main draw ======================

def draw(Dir, out_png="", video_layer=None, dual=False, spf=3.0, bins=None):
    """
    Modes:
      - Static (no --video): produce a 2×2 panel of mean fields:
          [Z_fitted approx, Upper curvature, Lower curvature, Both curvature].
      - Video (--video {Upper|Lower|Both}): produce a binned GIF for that layer.
        * 'bins' splits the timeline; each bin is averaged to a single frame.
        * 'dual' shows left=recentered, right=rotated; otherwise only rotated.
        * 'spf' sets seconds per frame (default 3.0 s).

    Color scaling:
      - vmin/vmax are computed from ALL rotated frames across Upper/Lower/Both
        so that multi-layer outputs share a consistent scale.

    Overlays:
      - Protein atoms are recentred and then rotated by +θ.
      - Optional COM markers from 'rotation_origin_indices.npy' and
        'rotation_p2_indices.npy' are plotted as red/green dots.
    """
    
    logging.info("=== curv: plot_curvature ===")

    # Box + vectors + protein
    Lx, Ly = _load_box(Dir)
    logging.info(f"Box (Lx, Ly) = ({Lx}, {Ly})")
    o_all, p_all, prot_all_box = _load_vectors_and_protein(Dir)
    F = len(o_all)
    logging.info(f"Frames loaded: {F}")

    # Optional selection indices → COM markers
    O_idx, P_idx = _try_load_selection_indices(Dir)
    if O_idx is not None or P_idx is not None:
        logging.info("Found selection index files; red/green markers will be COMs of those selections.")

    # Load raw stacks - THIS LOGIC WILL NEED TO BE CHANGES WHEN ZFIT WORKS
    layers = {"Zfit": "Both", "Upper": "Upper", "Lower": "Lower", "Both": "Both"}
    stacks_raw = {k: _load_field_stack(Dir, v, k) for k, v in layers.items()}

    # Color scale from rotated stacks
    logging.info("Computing color scale from rotated stacks …")
    all_rot_vals = []
    for key in ("Upper", "Lower", "Both"):
        raw = stacks_raw[key]
        vals = []
        for i in tqdm(range(F), desc=f"Scan {key}", unit="frame"):
            rec_i, theta_i, *_ = _calc_frame_recenter_and_theta(raw[i], o_all[i], p_all[i], Lx, Ly)
            vals.append(_rotate_image(rec_i, -theta_i))  # membrane uses -θ
        all_rot_vals.append(np.stack(vals, 0))
    all_vals = np.concatenate([x.ravel() for x in all_rot_vals])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    logging.info(f"Auto color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # Global mask
    mask = _inscribed_disk_mask(stacks_raw["Upper"][0].shape, inset_px=0.5)
    logging.info("Using global circular mask (inscribed disk).")

    # -------- VIDEO --------
    if video_layer:
        layer_name = str(video_layer)
        if layer_name not in ("Upper", "Lower", "Both"):
            raise ValueError(f"--video must be one of Upper/Lower/Both, got {layer_name}")

        # bins for mean-per-bin frames
        if bins is None or bins <= 1 or bins > F:
            bins = F
        logging.info(f"Binning frames: total={F}, bins={bins} (avg ~{F/bins:.1f} frames/bin)")
        idx_bins = np.array_split(np.arange(F), bins)

        # timing
        # --- Timing setup ---
        # Use seconds per frame (spf). Default = 3.0 s if not provided.
        try:
            seconds_per_frame = float(spf)
        except (TypeError, ValueError):
            logging.warning(f"Invalid --spf={spf!r}; falling back to 3.0 s/frame.")
            seconds_per_frame = 3.0

        # Ensure positive, finite value
        if not np.isfinite(seconds_per_frame) or seconds_per_frame <= 0:
            logging.warning(f"Invalid --spf={spf!r}; falling back to 3.0 s/frame.")
            seconds_per_frame = 3.0

        logging.info(f"Video timing: {seconds_per_frame:.3f} s/frame (~{1.0/seconds_per_frame:.3f} fps)")

        tmp_dir = tempfile.mkdtemp(prefix=f"curv_frames_{layer_name}_")
        logging.info(f"Rendering PNG frames to: {tmp_dir} (bins={bins})")

        cmap = "plasma"
        levels = np.linspace(vmin, vmax, 20)

        # figure & first draw
        if dual:
            fig, axL, axR = _fig_dual(Lx, Ly, vmin, vmax, cmap)
            csL = _draw_contour(axL, stacks_raw[layer_name][0], Lx, Ly, vmin, vmax, cmap, mask, levels)
            csR = _draw_contour(axR, stacks_raw[layer_name][0], Lx, Ly, vmin, vmax, cmap, mask, levels)
        else:
            fig, axR = _fig_single(Lx, Ly, vmin, vmax, cmap)
            csR = _draw_contour(axR, stacks_raw[layer_name][0], Lx, Ly, vmin, vmax, cmap, mask, levels)

        for bi, ids in enumerate(tqdm(idx_bins, desc="Rendering PNG frames (binned)", unit="bin")):
            rec_list, rot_list = [], []
            thetas, shifts = [], []

            for i in ids:
                rec_i, theta_i, dx_box_i, dy_box_i = _calc_frame_recenter_and_theta(
                    stacks_raw[layer_name][i], o_all[i], p_all[i], Lx, Ly
                )
                rec_list.append(rec_i)                          # left
                rot_list.append(_rotate_image(rec_i, -theta_i)) # right: membrane -θ
                thetas.append(theta_i)                          # +θ for protein overlay
                shifts.append((dx_box_i, dy_box_i))

            AiL = np.nanmean(np.stack(rec_list, 0), axis=0) if dual else None
            AiR = np.nanmean(np.stack(rot_list, 0), axis=0)

            # representative frame for overlay
            i_rep = ids[len(ids)//2]
            theta_rep = thetas[len(thetas)//2]
            dx_box, dy_box = shifts[len(shifts)//2]
            prot_rep = _protein_for_frame(prot_all_box, i_rep)  # (N,2)

            # overlays → recenter (+ rotate on right)
            center = np.array([Lx/2.0, Ly/2.0], float)
            prot_recent = prot_rep + np.array([dx_box, dy_box], float)
            prot_rot = _rotate_points_about_center(prot_recent, theta_rep, center)

            # update contour fields (clear then redraw)
            if dual:
                csL = _update_contour(axL, csL, AiL, Lx, Ly, vmin, vmax, cmap, mask, levels)
                csR = _update_contour(axR, csR, AiR, Lx, Ly, vmin, vmax, cmap, mask, levels)
            else:
                csR = _update_contour(axR, csR, AiR, Lx, Ly, vmin, vmax, cmap, mask, levels)

            # protein dots (white) + COM markers (red/green) if available
            artists = []
            if dual:
                # left: not rotated
                artists += _scatter_protein(axL, prot_recent, -1, -1)  # no colored atoms
                artists += _scatter_com_markers(axL, prot_recent, O_idx, P_idx)
                # right: rotated
                artists += _scatter_protein(axR, prot_rot, -1, -1)
                artists += _scatter_com_markers(axR, prot_rot, O_idx, P_idx)
            else:
                artists += _scatter_protein(axR, prot_rot, -1, -1)
                artists += _scatter_com_markers(axR, prot_rot, O_idx, P_idx)

            fig.suptitle(f"{layer_name} — Bin {bi+1}/{bins}",
                         fontsize=14, color='white')
            out_png_frame = os.path.join(tmp_dir, f"frame_{bi:05d}.png")
            fig.savefig(out_png_frame, dpi=170, facecolor='black', bbox_inches="tight")

            # clear per-frame markers (contours already refreshed)
            for a in artists:
                try: a.remove()
                except Exception: pass

        # ---- encode output ----
        paths = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png")))
        if not paths:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("No PNG frames found to encode.")

        if True:
            out_name = f"video_{layer_name}{'_dual' if dual else ''}.gif"
            # seconds → ms; ≥ 20 ms to avoid viewer clamping to 0
            duration_ms = max(20, int(round(seconds_per_frame * 1000)))
            logging.info(f"Encoding GIF with Pillow (duration={seconds_per_frame:.3f}s/frame) …")
            frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in paths]
            frames[0].save(out_name, save_all=True, append_images=frames[1:],
                           duration=duration_ms, loop=0, optimize=False, disposal=2)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info(f"Saved: {out_name}")
        return

    # -------- static 2×2 means --------
    logging.info("Producing 2×2 mean images (no video).")

    def _mean_rotated(raw_stack):
        vals = []
        for i in range(F):
            rec_i, theta_i, *_ = _calc_frame_recenter_and_theta(raw_stack[i], o_all[i], p_all[i], Lx, Ly)
            vals.append(_rotate_image(rec_i, -theta_i))
        return np.nanmean(np.stack(vals,0), axis=0)

    mean_Z = _mean_rotated(stacks_raw["Zfit"])
    mean_U = _mean_rotated(stacks_raw["Upper"])
    mean_L = _mean_rotated(stacks_raw["Lower"])
    mean_B = _mean_rotated(stacks_raw["Both"])

    vmin2 = min(map(np.nanmin, (mean_U, mean_L, mean_B)))
    vmax2 = max(map(np.nanmax, (mean_U, mean_L, mean_B)))
    vmin_plot, vmax_plot = float(vmin2), float(vmax2)

    fig, axes = plt.subplots(2,2, figsize=(16,16))
    fig.patch.set_facecolor('black')
    axes = axes.ravel()
    arrays = [mean_Z, mean_U, mean_L, mean_B]
    titles = ["Fourier Approximation", "Curvature Upper", "Curvature Lower", "Curvature Both"]


    cmap = "plasma"
    levels = np.linspace(vmin_plot, vmax_plot, 20)
    mask2 = _inscribed_disk_mask(mean_U.shape, 0.5)

    # pick a representative frame for overlays
    i_rep = F // 2
    rec_i, theta_i, dx_box, dy_box = _calc_frame_recenter_and_theta(
        stacks_raw["Upper"][i_rep], o_all[i_rep], p_all[i_rep], Lx, Ly
    )
    prot_rep = _protein_for_frame(prot_all_box, i_rep)[:, :2]
    center = np.array([Lx/2.0, Ly/2.0], float)
    prot_recent = prot_rep + np.array([dx_box, dy_box], float)
    prot_rot = _rotate_points_about_center(prot_recent, theta_i, center)

    for ax, title, arr in zip(axes, titles, arrays):
        _set_axes_style(ax, Lx, Ly)
        a = np.where(mask2, arr, np.nan)
        cs = _draw_contour(ax, a, Lx, Ly, vmin_plot, vmax_plot, cmap, None, levels)
        ax.set_title(title, color='white')

        # overlay protein (white atoms + COM markers if available)
        _ = _scatter_protein(ax, prot_rot, -1, -1)
        _ = _scatter_com_markers(ax, prot_rot, O_idx, P_idx)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    _colorbar_with_ticks(fig, cax, vmin_plot, vmax_plot, cmap)
    plt.tight_layout(rect=[0.05,0.02,0.9,0.98])

    if out_png:
        fig.savefig(out_png, dpi=220, facecolor='black', bbox_inches="tight")
        logging.info(f"Saved: {out_png}")
    else:
        plt.show()
    plt.close(fig)


# ====================== bins image (multi-panel PNG) ======================

def _grid_shape(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def _add_colorbar(fig, vmin, vmax, cmap):
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"Curvature ($\mathrm{nm^{-1}}$)", color='white')
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    cb.ax.tick_params(labelsize=10, colors='white')
    for tick in cb.ax.get_yticklabels(): tick.set_color('white')
    cb.outline.set_edgecolor('white')

def draw_bins_image(Dir, layer, bins, outfile, cmap="plasma"):
    """
    Render a single PNG containing 'bins' temporal bins for a chosen layer.

    For each bin:
      - Recenter + rotate(-θ) each frame, average within the bin,
      - Plot the mean field with consistent color scale across bins,
      - Overlay protein atoms (recentred, then rotated by +θ) and optional COMs.

    Output:
      - Writes 'outfile' (PNG).
    """
    
    logging.info("=== curv: bin image ===")
    Lx, Ly = _load_box(Dir)
    o_all, p_all, prot_all_box = _load_vectors_and_protein(Dir)
    F = len(o_all)

    # selection indices for COM overlay (optional)
    O_idx, P_idx = _try_load_selection_indices(Dir)

    key = "Zfit" if layer == "Zfit" else layer
    stack = _load_field_stack(Dir, layer if layer!="Zfit" else "Both", key)
    if stack.shape[0] != F:
        n = min(stack.shape[0], F)
        stack = stack[:n]; o_all = o_all[:n]; p_all = p_all[:n]
        if prot_all_box.shape[0] > 1: prot_all_box = prot_all_box[:n]
        F = n
        logging.info(f"Truncated to {F} frames to match arrays.")

    logging.info(f"Recenter/rotate frames for '{layer}' …")
    rots, thetas, shifts = [], [], []
    for i in tqdm(range(F), desc=f"Rotate {layer}", unit="frame"):
        rec_i, theta_i, dx_box, dy_box = _calc_frame_recenter_and_theta(
            stack[i], o_all[i], p_all[i], Lx, Ly
        )
        rots.append(_rotate_image(rec_i, -theta_i))
        thetas.append(theta_i)
        shifts.append((dx_box, dy_box))
    rots = np.stack(rots, 0)
    thetas = np.asarray(thetas)
    shifts = np.asarray(shifts)

    field = rots
    vmin = float(np.nanmin(field)); vmax = float(np.nanmax(field))
    logging.info(f"Color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")
    mask = _inscribed_disk_mask(field[0].shape, inset_px=0.5)
    levels = np.linspace(vmin, vmax, 20)

    bins = max(1, int(bins))
    idx_bins = np.array_split(np.arange(F), bins)
    rows, cols = _grid_shape(bins)
    fig_w = 4.8*cols + 2.0
    fig_h = 4.8*rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor('black')

    for bi, ids in enumerate(idx_bins):
        r, c = divmod(bi, cols)
        ax = axes[r, c]
        _set_axes_style(ax, Lx, Ly)

        A = np.nanmean(field[ids], axis=0)
        a = np.where(mask, A, np.nan)
        _draw_contour(ax, a, Lx, Ly, vmin, vmax, cmap, None, levels)

        # overlay at representative frame
        i_rep = ids[len(ids)//2]
        theta_rep = thetas[i_rep]
        dx_box, dy_box = shifts[i_rep]
        prot_rep = _protein_for_frame(prot_all_box, i_rep)[:, :2]
        center = np.array([Lx/2.0, Ly/2.0], float)
        prot_recent = prot_rep + np.array([dx_box, dy_box], float)
        prot_rot = _rotate_points_about_center(prot_recent, theta_rep, center)

        # protein dots (white)
        _ = _scatter_protein(ax, prot_rot, -1, -1)
        # COM markers if available
        _ = _scatter_com_markers(ax, prot_rot, O_idx, P_idx)

        start, end = int(ids[0])+1, int(ids[-1])+1
        ax.set_title(f"Frames {start}–{end}", color='white', fontsize=11)

    # turn off unused axes
    for k in range(bins, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')

    _add_colorbar(fig, vmin, vmax, cmap)
    plt.tight_layout(rect=[0.03, 0.02, 0.9, 0.98])

    if not outfile:
        outfile = f"bins_{layer}_{bins}.png"
    fig.savefig(outfile, dpi=220, facecolor='black', bbox_inches="tight")
    logging.info(f"Saved: {outfile}")
    plt.close(fig)

# ====================== CLI ======================

def Map(argv):
    p = argparse.ArgumentParser(
        description="Curvature plots/videos. Dual mode: left recentred, right rotated (membrane -θ, protein +θ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-d","--numpys_directory", required=True)
    p.add_argument("-o","--outfile", default="")
    p.add_argument("-v","--video", choices=["Upper","Lower","Both"])
    p.add_argument("--dual", action="store_true", help="Dual view: left recentred, right rotated.")
    p.add_argument("--spf", type=float, default=3.0, help="Seconds per frame (default: 3.0 s per frame).")
    p.add_argument("--bins-video", type=int, default=None, help="Temporal bins (mean per bin) for video.")
    p.add_argument("--bins-image", type=int, default=None,
                   help="If set, render a single PNG with this many temporal bins for the chosen layer, then exit.")
    p.add_argument("--bins-layer", choices=["Upper","Lower","Both","Zfit"], default="Upper",
                   help="Layer to use when --bins-image is requested.")

    ## calc_vectors arguements

    p.add_argument('-f','--trajectory', type=str, required=True,
                        help="Path to trajectory file (e.g., .xtc, .dcd)")
    p.add_argument('-s','--structure', type=str, required=True,
                        help="Path to structure/topology file (e.g., .pdb, .psf, .gro)")
    p.add_argument('-F','--From', default=0, type=int,
                        help="First frame index (inclusive)")
    p.add_argument('-U','--Until', default=None, type=int,
                        help="Stop before this frame index (exclusive); None = end")
    p.add_argument('-S','--Step', default=1, type=int,
                        help="Stride between frames")
    p.add_argument("--np-dir", default="", type=str,
                        help="Directory containing curvature numpys (default: current dir)")
    p.add_argument('-p1','--selection1', type=str, required=True,
                        help="Atom selection for reference point 1 (O)")
    p.add_argument('-p2','--selection2', type=str, required=True,
                        help="Atom selection for reference point 2 (P)")

    
    ns = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)


    try:
        u = mda.Universe(ns.structure, ns.trajectory)
        get_rotation_and_protein(
            out_dir=ns.np_dir or "./",
            u=u, From=ns.From, Until=ns.Until, Step=ns.Step,
            sele1=ns.selection1, sele2=ns.selection2,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


    try:
        if ns.bins_image is not None:
            logging.info(f"Producing bins image: layer={ns.bins_layer}, bins={ns.bins_image}")
            draw_bins_image(
                ns.numpys_directory,
                layer=ns.bins_layer,
                bins=ns.bins_image,
                outfile=ns.outfile or f"bins_{ns.bins_layer}_{ns.bins_image}.png", cmap="plasma"
            )
            return

        draw(ns.numpys_directory, out_png=ns.outfile, video_layer=ns.video, dual=ns.dual, spf=ns.spf, bins=ns.bins_video)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

