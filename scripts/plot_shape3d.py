"""Figures of the planned cook on a reconstructed cut (figures/shadow3d/<name>/).

Re-simulates the plan on the cut reconstructed from photos (tools/cutvision's shape.json, a VoxelPiece through
scripts/shadow3d.jl; the schedule as scripts/run_shadow3d.py) and draws:

- shape3d_plan.png: the pit, the deepest point and the TRUE cold spot (grid minimum) with the wrap, the pull and the
  fuel addition; the bark over the exposed faces (mean and range) beside the smoke deposition; the mass.
- shape3d_sections.png: temperature and moisture on the two vertical sections through the deepest point (along x and
  along y) at 2 h, at the wrap and at serve, interpolated across the cells within the reconstruction's outline, the
  height drawn twice its scale.
- shape3d_cut.png: the cut at serve in 3-D on the simulation's cells: whole, its surface by the model's L* (bark and
  smoke); and cut through the deepest point, the section by temperature, the fat paler and the bone white.
- shape3d_cook.gif: the cook in 3-D: the cut halved through its deepest point, its smooth surface and the section by
  temperature on one scale through the whole cook, the camera turning.
- shape3d_slice.gif: the slice down the middle (the lengthwise section through the deepest point): temperature and
  moisture through the cook within the reconstruction's outline, the height drawn twice its scale.
- shape3d_smooth.png: the same on the smooth surface the exposed areas come from (lit, coloured by the exposed
  faces' states blended over a cell), the section's temperature interpolated across the cells within the reconstruction's own
  outline.

Usage (from app/):
    uv run python ../scripts/plot_shape3d.py --shape <name>          # data/cut/<name>/shape.json
"""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from PIL import Image
from scipy.spatial import cKDTree
from shadow3d_common import HR, REPO, load_shape, simulate

BARK = mcolors.LinearSegmentedColormap.from_list("bark", ["#1f120b", "#5a3420", "#a0603a", "#d9a46c"])
EXAG = 2.0                                                            # the sections' height drawn twice its scale
DIRS6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def cells(sim):
    """Each node's grid index (i, j, k), 0-based, and the cell size [m]."""
    H = sim["coarsen"] * sim["shape"]["h"]
    ijk = np.floor(np.asarray(sim["coords"]) / H).astype(int)
    return ijk, H


def grid_of(sim, values):
    """A per-node vector on the cut's grid (NaN outside the meat)."""
    ijk, _ = cells(sim)
    G = np.full(ijk.max(axis=0) + 1, np.nan)
    G[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values
    return G


def smooth_section(sim, values, axis, at):
    """A per-node field on the vertical section `axis` = `at` [m] (axis 1: the x–z plane at y = at; 0: the y–z plane
    at x = at), interpolated across the cells (trilinear, the cells' values extended to the outline) at half the
    reconstruction's resolution, NaN outside its smooth outline. Returns the image (horizontal × height) and its
    horizontal and vertical extents [m]."""
    ijk, H = cells(sim)
    h = sim["shape"]["h"]; lab = sim["shape"]["labels"]
    G = grid_of(sim, values)
    near = ndimage.distance_transform_edt(np.isnan(G), return_distances=False, return_indices=True)
    Gf = G[tuple(near)]
    if "_occ" not in sim:
        sim["_occ"] = ndimage.gaussian_filter((lab != 0).astype(float), 1.0)
    occ = sim["_occ"]
    step = h / 2
    other = 0 if axis == 1 else 1
    us = np.arange(0.0, lab.shape[other] * h, step) + step / 2; zs = np.arange(0.0, lab.shape[2] * h, step) + step / 2
    U, Z = np.meshgrid(us, zs, indexing="ij")
    pos = [None, None, Z]; pos[other] = U; pos[axis] = np.full_like(U, at)
    fine = np.stack([p / h - 0.5 for p in pos]); coarse = np.stack([p / H - 0.5 for p in pos])
    inside = ndimage.map_coordinates(occ, fine.reshape(3, -1), order=1, mode="nearest").reshape(U.shape) >= 0.5
    img = ndimage.map_coordinates(Gf, coarse.reshape(3, -1), order=1, mode="nearest").reshape(U.shape)
    img[~inside] = np.nan
    return img, lab.shape[other] * h, lab.shape[2] * h


def snapshots(sim):
    t = sim["t"] / HR
    out = [(int(np.searchsorted(t, 2.0)), "2 h")]
    tw = sim["decisions"]["t_wrap"] / HR
    if tw < 1e4:
        k = int(np.searchsorted(t, tw)); out.append((k, f"wrap ({t[k]:.1f} h)"))
    out.append((len(t) - 1, f"serve ({t[-1]:.0f} h)"))
    return out


def title_of(sim):
    m = sim["shape"]["meta"]
    return (f"{m['meat'].replace('_', ' ')} reconstructed from photos: {np.sum(sim['vol']) * 1e3:.2f} L, "
            f"{sim['area'] * 1e4:.0f} cm² exposed, cap {m['cap_mm']:.1f} mm, {sim['grid']}")


def plot_plan(sim, out):
    t = sim["t"] / HR
    dec = sim["decisions"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]})
    ax = axes[0]
    ax.plot(t, sim["Tpit"] - 273.15, color="tab:red", lw=2, label="pit")
    ax.plot(t, sim["Tcore"] - 273.15, color="tab:blue", lw=2, label="the cut's deepest point")
    ax.plot(t, sim["Tcold"] - 273.15, color="tab:cyan", lw=2, ls="--", label="TRUE cold spot (grid min)")
    ax.set_ylabel("temperature [°C]")
    for lab, tt, ls in (("wrap", dec["t_wrap"], ":"), ("pull", dec["t_pull"], "--"), ("fuel add", dec["t_add"], "-.")):
        if tt < 1e8 and tt / HR <= t[-1]:
            ax.axvline(tt / HR, color="gray", lw=1, ls=ls)
            ax.annotate(lab, (tt / HR, ax.get_ylim()[0] + 5), xytext=(-4, 0), textcoords="offset points", color="gray",
                        fontsize=9, ha="right", rotation=90)
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"3-D shadow of {sim['plan']}: {title_of(sim)}\ncold spot at serve {sim['Tcold'][-1] - 273.15:.1f} °C, "
                 f"deepest point {sim['Tcore'][-1] - 273.15:.1f} °C", fontsize=10)
    ax = axes[1]
    B = np.asarray(sim["bark_dist"])
    ax.fill_between(t, B.min(axis=0), B.max(axis=0), color="tab:brown", alpha=0.25, lw=0, label="bark, range over the faces")
    ax.plot(t, sim["bark_b"], color="tab:brown", lw=2, label="bark (area mean)")
    ax.plot(t, sim["bark_s"], color="dimgray", lw=2, label="smoke deposition (area mean)")
    ax.set_ylabel("surface state [-]"); ax.set_ylim(-0.02, 1.02); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    ax.annotate(f"serve: bark {sim['bark_b'][-1]:.2f} ± {sim['bark_std'][-1]:.2f} · ring {1000 * sim['ring'][-1]:.1f} mm · "
                f"water retained {sim['w_retained']:.2f} · fat rendered {sim['fat_rendered']:.2f}", (0.99, 0.05),
                xycoords="axes fraction", ha="right", fontsize=9, color="dimgray")
    ax = axes[2]
    ax.plot(t, sim["mass"], color="tab:olive", lw=2); ax.set_ylabel("mass [kg]"); ax.set_xlabel("time [h]"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "shape3d_plan.png", dpi=140); plt.close(fig)
    print("wrote shape3d_plan.png")


def plot_sections(sim, out):
    ijk, H = cells(sim)
    ci, cj, _ = ijk[sim["core"]]
    xc, yc = (ci + 0.5) * H, (cj + 0.5) * H
    snaps = snapshots(sim)
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15), ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig, axes = plt.subplots(2 * len(rows), len(snaps), figsize=(5.0 * len(snaps) + 1.2, 9.0), squeeze=False, constrained_layout=True)
    for r, (name, label, cmap, f) in enumerate(rows):
        secs = [[smooth_section(sim, f(np.asarray(sim[name])[:, k]), 1, yc), smooth_section(sim, f(np.asarray(sim[name])[:, k]), 0, xc)]
                for k, _ in snaps]
        vlo = min(np.nanmin(sc[0]) for pair in secs for sc in pair); vhi = max(np.nanmax(sc[0]) for pair in secs for sc in pair)
        for c, ((_, title), pair) in enumerate(zip(snaps, secs, strict=True)):
            for q, ((sec, wh, ht), axis_label) in enumerate(zip(pair, ("x", "y"), strict=True)):
                ax = axes[2 * r + q, c]
                im = ax.imshow(sec.T, origin="lower", cmap=cmap, vmin=vlo, vmax=vhi, interpolation="bilinear",
                               extent=(0, wh * 100, 0, EXAG * ht * 100))
                ax.set_yticks(EXAG * np.arange(0, ht * 100 + 0.1, 2.0), [f"{v:.0f}" for v in np.arange(0, ht * 100 + 0.1, 2.0)])
                ax.set_aspect("equal")
                if r == 0 and q == 0:
                    ax.set_title(title, fontsize=10)
                ax.set_xlabel(f"{axis_label} [cm] (section through the deepest point)", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"height [cm] (×{EXAG:.0f})", fontsize=8)
                ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[2 * r:2 * r + 2, :], shrink=0.85, label=label)
    fig.suptitle(f"Sections through the deepest point: {title_of(sim)}", fontsize=10)
    fig.savefig(out / "shape3d_sections.png", dpi=140); plt.close(fig)
    print("wrote shape3d_sections.png")


def face_quad(c, d, H):
    """The square of side H on the face of the cell centred at c toward the axis direction d."""
    a = int(np.argmax(np.abs(d))); s = np.sign(d[a])
    u, v = [x for x in range(3) if x != a]
    base = np.array(c, dtype=float); base[a] += s * H / 2
    out = []
    for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = base.copy(); p[u] += du * H / 2; p[v] += dv * H / 2; out.append(p)
    return out


def plot_cut(sim, out):
    ijk, H = cells(sim)
    xyz = np.asarray(sim["coords"]) * 100; Hc = H * 100
    lm = sim["lstar_map"]
    Bk = np.asarray(sim["bark_dist"])[:, -1]; Sk = np.asarray(sim["smoke_dist"])[:, -1]
    Ls = lm["L0_bark"] - lm["dL_B"] * Bk - lm["dL_S"] * Sk
    lo, hi = lm["L0_bark"] - lm["dL_B"] - lm["dL_S"], lm["L0_bark"]
    fc_l = BARK(np.clip((Ls - lo) / (hi - lo), 0, 1))
    nodes = np.asarray(sim["boundary_nodes"]); dirs = sim["faces"]
    T = np.asarray(sim["T_field"])[:, -1] - 273.15
    # every open face of the cut's cells, coloured by its own exposed face's L* (a face the smooth surface gave no
    # area to takes its node's other faces' mean, or the cut's)
    occ = np.zeros(ijk.max(axis=0) + 1, dtype=bool); occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    face_l = {(int(n), tuple(int(v) for v in d)): fc_l[b] for b, (n, d) in enumerate(zip(nodes, dirs, strict=True))}
    node_l = {}
    for (n, _), c in face_l.items():
        node_l.setdefault(n, []).append(c)
    mean_l = np.mean(fc_l, axis=0)
    open_faces = []
    for n, (i, j, k) in enumerate(ijk):
        for d in DIRS6:
            q = (i + d[0], j + d[1], k + d[2])
            inside = all(0 <= q[a] < occ.shape[a] for a in range(3)) and occ[q]
            if inside or (d == (0, 0, -1) and k == 0):
                continue                                              # a neighbour, or the board's side: not seen
            c = face_l.get((n, d), np.mean(node_l[n], axis=0) if n in node_l else mean_l)
            open_faces.append((n, np.array(d), c))
    tn = mcolors.Normalize(np.min(T), np.max(T))
    cj = ijk[sim["core"], 1]
    fig = plt.figure(figsize=(14, 6.2))
    span = xyz.max(axis=0) - xyz.min(axis=0) + Hc
    for col, cut in enumerate((False, True)):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        polys, fcol = [], []
        for n, d, c in open_faces:
            if cut and ijk[n, 1] < cj:
                continue
            polys.append(face_quad(xyz[n], d, Hc)); fcol.append(c)
        if cut:                                                       # the section's faces, by temperature
            for n in np.flatnonzero(ijk[:, 1] == cj):
                polys.append(face_quad(xyz[n], np.array([0, -1, 0]), Hc))
                m = sim["mat"][n]
                fcol.append((0.97, 0.97, 0.94, 1.0) if sim["bone"][n] > 0.5 else plt.cm.inferno(tn(T[n])) if m == 1
                            else tuple(0.6 * np.array(plt.cm.inferno(tn(T[n]))[:3]) + 0.4) + (1.0,))
        pc = Poly3DCollection(polys, facecolors=fcol, edgecolors=fcol, linewidths=0.4)   # edged in their own colour: no seams
        ax.add_collection3d(pc)
        mn = xyz.min(axis=0) - Hc; mx = xyz.max(axis=0) + Hc
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(0, mx[2])
        ax.set_box_aspect(tuple(span)); ax.view_init(elev=28, azim=-60); ax.set_axis_off()
        ax.set_title("the surface at serve by the model's L*" if not cut else
                     "cut through the deepest point: temperature at serve (fat paler, bone white)", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=tn, cmap="inferno")
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="temperature at serve [°C]")
    fig.suptitle(title_of(sim) + " (the colours illustrate the modelled states, they are not a photograph)", fontsize=10)
    fig.savefig(out / "shape3d_cut.png", dpi=140); plt.close(fig)
    print("wrote shape3d_cut.png")


def plot_smooth(sim, out):
    """The cut at serve on the smooth surface: whole, lit and coloured by its exposed faces' L* blended over a cell; and cut through the
    deepest point, the section's temperature interpolated across the cells (trilinear, the cells' values extended to
    the outline) within the reconstruction's outline at its own 2 mm resolution."""
    ijk, H = cells(sim)
    h = sim["shape"]["h"]; lab = sim["shape"]["labels"]
    V = np.asarray(sim["tris"]).reshape(-1, 3, 3) * 100.0             # cm
    lm = sim["lstar_map"]
    Ls = lm["L0_bark"] - lm["dL_B"] * np.asarray(sim["bark_dist"])[:, -1] - lm["dL_S"] * np.asarray(sim["smoke_dist"])[:, -1]
    lo, hi = lm["L0_bark"] - lm["dL_B"] - lm["dL_S"], lm["L0_bark"]
    # each triangle's L*: a Gaussian blend (σ 1.5 cells) of the nearby exposed faces'. On a dome the cells' terraces
    # leave a cell with two exposed faces at each terrace edge, which browns a little faster: a staircase artefact of the
    # cells, not the meat, that one face's value would print as rings
    xyz = np.asarray(sim["coords"]) * 100; Hc = H * 100
    fcen = xyz[np.asarray(sim["boundary_nodes"])] + 0.5 * Hc * np.asarray(sim["faces"])
    d, nb = cKDTree(fcen).query(V.mean(axis=1), k=24)
    w = np.exp(-(d / (1.5 * Hc)) ** 2 / 2) + 1e-12
    Ltri = (w * Ls[nb]).sum(axis=1) / w.sum(axis=1)
    tri_rgb = BARK(np.clip((Ltri - lo) / (hi - lo), 0, 1))[:, :3]
    # Lambert shading on each triangle's normal (either orientation)
    # normals from a smoother field than the surface's own (σ 2.5 cells), so the reconstruction's 2 mm terraces
    # don't print as rings in the shading
    soft = ndimage.gaussian_filter((lab != 0).astype(float), 2.5)
    cen = V.mean(axis=1) / 100.0 / h - 0.5
    nrm = -np.stack([ndimage.map_coordinates(np.gradient(soft, axis=a), cen.T, order=1, mode="nearest") for a in range(3)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.5, 1.0]); light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.abs(nrm @ light)
    tri_rgb = np.clip(tri_rgb * shade[:, None], 0, 1)
    # the section through the deepest point (y = its cell's centre)
    yc = (ijk[sim["core"], 1] + 0.5) * H
    T = np.asarray(sim["T_field"])[:, -1] - 273.15
    G = grid_of(sim, T)
    near = ndimage.distance_transform_edt(np.isnan(G), return_distances=False, return_indices=True)
    Gf = G[tuple(near)]                                               # the cells' values extended outward
    occ = ndimage.gaussian_filter((lab != 0).astype(float), 1.0)
    step = h / 2
    xs = np.arange(0.0, lab.shape[0] * h, step) + step / 2; zs = np.arange(0.0, lab.shape[2] * h, step) + step / 2
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    fine = np.stack([X / h - 0.5, np.full_like(X, yc / h - 0.5), Z / h - 0.5])
    inside = ndimage.map_coordinates(occ, fine.reshape(3, -1), order=1, mode="nearest").reshape(X.shape) >= 0.5
    coarse = np.stack([X / H - 0.5, np.full_like(X, yc / H - 0.5), Z / H - 0.5])
    Ts = ndimage.map_coordinates(Gf, coarse.reshape(3, -1), order=1, mode="nearest").reshape(X.shape)
    # the fat and the bone on the section by the same smooth level set as the outline (not cell by cell)
    def smooth_region(code):
        f = ndimage.gaussian_filter((lab == code).astype(float), 1.0)
        return ndimage.map_coordinates(f, fine.reshape(3, -1), order=1, mode="nearest").reshape(X.shape) >= 0.5
    tn = mcolors.Normalize(np.nanmin(T), np.nanmax(T))
    sec = plt.cm.inferno(tn(Ts))
    fat = smooth_region(2); bone = smooth_region(3)
    sec[fat, :3] = 0.6 * sec[fat, :3] + 0.4                           # fat paler
    sec[bone, :3] = (0.97, 0.97, 0.94)                                # bone
    sec[~inside, 3] = 0.0
    fig = plt.figure(figsize=(14, 6.2))
    mn = V.reshape(-1, 3).min(axis=0); mx = V.reshape(-1, 3).max(axis=0)
    for col, cut in enumerate((False, True)):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d", computed_zorder=False)
        keep = np.ones(len(V), bool) if not cut else V[:, :, 1].mean(axis=1) >= 100 * yc
        pc = Poly3DCollection(V[keep], facecolors=tri_rgb[keep], edgecolors=tri_rgb[keep], linewidths=0.2, zorder=1)
        ax.add_collection3d(pc)
        if cut:
            ax.plot_surface(100 * X, np.full_like(X, 100 * yc), 100 * Z, facecolors=sec, rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False, zorder=2)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(0, mx[2])
        ax.set_box_aspect(tuple(mx - np.array([mn[0], mn[1], 0]))); ax.view_init(elev=28, azim=-60); ax.set_axis_off()
        ax.set_title("the surface at serve by the model's L*" if not cut else
                     "cut through the deepest point: temperature at serve (fat paler, bone white)", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=tn, cmap="inferno")
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="temperature at serve [°C]")
    fig.suptitle(title_of(sim) + " · drawn on the smooth surface (the colours illustrate the modelled states)", fontsize=10)
    fig.savefig(out / "shape3d_smooth.png", dpi=140); plt.close(fig)
    print("wrote shape3d_smooth.png")


def frame_image(fig, dpi):
    """The figure as an RGB image at a fixed size (the whole canvas, so every frame of a GIF matches)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, path, fps, hold=8):
    """The frames as a looping GIF, the last held `hold` frames."""
    frames = frames + [frames[-1]] * hold
    pal = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)


def frame_times(sim, frames):
    t = sim["t"]
    return np.unique(np.round(np.linspace(0, len(t) - 1, frames)).astype(int))


def stamp(sim, k):
    """A frame's caption: the time, what the plan is doing, the pit and the cut."""
    t = sim["t"][k] / HR
    dec = sim["decisions"]
    phase = "resting" if t >= dec["t_pull"] / HR else "wrapped" if t >= dec["t_wrap"] / HR else "smoking"
    return (f"t = {t:4.1f} h · {phase} · pit {sim['Tpit'][k] - 273.15:5.1f} °C · deepest point "
            f"{sim['Tcore'][k] - 273.15:5.1f} °C · cold spot {sim['Tcold'][k] - 273.15:5.1f} °C")


def plot_cook_gif(sim, out, frames=48, fps=8):
    """The cook in 3-D: the half beyond the section through the deepest point, its smooth surface coloured by the
    exposed faces' temperature (blended over 1.5 cells) and the section by the cells' temperature interpolated
    within the outline, on one scale for the whole cook; the camera turns through 40 degrees over it."""
    ijk, H = cells(sim)
    h = sim["shape"]["h"]
    lab = sim["shape"]["labels"]
    V = np.asarray(sim["tris"]).reshape(-1, 3, 3) * 100.0
    yc = (ijk[sim["core"], 1] + 0.5) * H
    V = V[V[:, :, 1].mean(axis=1) >= 100 * yc]
    xyz = np.asarray(sim["coords"]) * 100
    Hc = H * 100
    bn = np.asarray(sim["boundary_nodes"])
    fcen = xyz[bn] + 0.5 * Hc * np.asarray(sim["faces"])
    d, nb = cKDTree(fcen).query(V.mean(axis=1), k=24)
    w = np.exp(-(d / (1.5 * Hc)) ** 2 / 2) + 1e-12
    w /= w.sum(axis=1, keepdims=True)
    soft = ndimage.gaussian_filter((lab != 0).astype(float), 2.5)
    cen = V.mean(axis=1) / 100.0 / h - 0.5
    nrm = -np.stack([ndimage.map_coordinates(np.gradient(soft, axis=a), cen.T, order=1, mode="nearest")
                     for a in range(3)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.5, 1.0])
    light /= np.linalg.norm(light)
    shade = (0.6 + 0.4 * np.abs(nrm @ light))[:, None]
    Tall = np.asarray(sim["T_field"]) - 273.15
    tn = mcolors.Normalize(np.min(Tall), np.max(Tall))
    # the section on a grid at the reconstruction's own resolution (lighter than the stills' half)
    xs = np.arange(0.0, lab.shape[0] * h, h) + h / 2
    zs = np.arange(0.0, lab.shape[2] * h, h) + h / 2
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    mn = V.reshape(-1, 3).min(axis=0)
    mx = V.reshape(-1, 3).max(axis=0)
    imgs = []
    ks = frame_times(sim, frames)
    for f, k in enumerate(ks):
        Tk = Tall[:, k]
        tri_rgb = np.clip(plt.cm.inferno(tn((w * Tk[bn][nb]).sum(axis=1)))[:, :3] * shade, 0, 1)
        sec, _, _ = smooth_section(sim, Tk, 1, yc)
        sec = sec[::2, ::2][:X.shape[0], :X.shape[1]]
        rgba = plt.cm.inferno(tn(np.nan_to_num(sec, nan=0.0)))
        rgba[np.isnan(sec), 3] = 0.0
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = fig.add_axes((-0.04, -0.04, 0.92, 0.94), projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(V, facecolors=tri_rgb, edgecolors=tri_rgb, linewidths=0.2, zorder=1))
        ax.plot_surface(100 * X, np.full_like(X, 100 * yc), 100 * Z, facecolors=rgba, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False, zorder=2)
        ax.set_xlim(mn[0], mx[0])
        ax.set_ylim(mn[1], mx[1])
        ax.set_zlim(0, mx[2])
        ax.set_box_aspect(tuple(mx - np.array([mn[0], mn[1], 0])), zoom=1.1)
        ax.view_init(elev=26, azim=-75 + 40 * f / max(len(ks) - 1, 1))
        ax.set_axis_off()
        cax = fig.add_axes((0.89, 0.2, 0.02, 0.6))
        fig.colorbar(plt.cm.ScalarMappable(norm=tn, cmap="inferno"), cax=cax, label="temperature [°C]")
        fig.text(0.5, 0.95, stamp(sim, k), ha="center", fontsize=10, family="monospace")
        fig.text(0.5, 0.91, title_of(sim) + " · cut through the deepest point", ha="center", fontsize=8, color="dimgray")
        imgs.append(frame_image(fig, 100))
        plt.close(fig)
    save_gif(imgs, out / "shape3d_cook.gif", fps)
    print(f"wrote shape3d_cook.gif ({len(imgs)} frames)")


def plot_slice_gif(sim, out, frames=60, fps=10):
    """The slice down the middle: the lengthwise section through the deepest point, temperature and moisture through
    the cook, each on one scale for the whole cook, the height drawn twice its scale; below, the cook's temperatures
    with the frame's time marked."""
    ijk, H = cells(sim)
    yc = (ijk[sim["core"], 1] + 0.5) * H
    Tall = np.asarray(sim["T_field"]) - 273.15
    Wall = np.asarray(sim["W_field"])
    tn = mcolors.Normalize(np.min(Tall), np.max(Tall))
    wn = mcolors.Normalize(0.0, np.max(Wall))
    t = sim["t"] / HR
    dec = sim["decisions"]
    imgs = []
    for k in frame_times(sim, frames):
        fig = plt.figure(figsize=(11, 6.4))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55], width_ratios=[1, 0.025], hspace=0.45, wspace=0.03,
                              left=0.07, right=0.93, top=0.9, bottom=0.08)
        rows = ((Tall[:, k], tn, "inferno", "temperature [°C]"), (Wall[:, k], wn, "viridis", "moisture [kg/kg db]"))
        for r, (vals, norm, cmap, label) in enumerate(rows):
            sec, wh, ht = smooth_section(sim, vals, 1, yc)
            ax = fig.add_subplot(gs[r, 0])
            im = ax.imshow(sec.T, origin="lower", cmap=cmap, norm=norm, interpolation="bilinear",
                           extent=(0, wh * 100, 0, EXAG * ht * 100))
            ax.set_aspect("equal")
            ax.set_ylabel(f"height [cm] (×{EXAG:.0f})", fontsize=8)
            ax.tick_params(labelsize=7)
            ticks = np.arange(0, ht * 100 + 0.1, 2.0)
            ax.set_yticks(EXAG * ticks, [f"{v:.0f}" for v in ticks])
            ax.set_xlabel("length [cm] (the slice down the middle, through the deepest point)", fontsize=8)
            fig.colorbar(im, cax=fig.add_subplot(gs[r, 1]), label=label)
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(t, sim["Tpit"] - 273.15, color="tab:red", lw=1.5, label="pit")
        ax.plot(t, sim["Tcore"] - 273.15, color="tab:blue", lw=1.5, label="deepest point")
        ax.plot(t, sim["Tcold"] - 273.15, color="tab:cyan", lw=1.5, ls="--", label="cold spot")
        for tt, ls in ((dec["t_wrap"], ":"), (dec["t_pull"], "--")):
            if tt < 1e8:
                ax.axvline(tt / HR, color="gray", lw=1, ls=ls)
        ax.axvline(t[k], color="black", lw=1.5)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("[°C]", fontsize=8)
        ax.set_xlabel("time [h] (dotted: wrap; dashed: pull)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.grid(alpha=0.3)
        fig.text(0.5, 0.95, stamp(sim, k), ha="center", fontsize=10, family="monospace")
        imgs.append(frame_image(fig, 90))
        plt.close(fig)
    save_gif(imgs, out / "shape3d_slice.gif", fps)
    print(f"wrote shape3d_slice.gif ({len(imgs)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--shape", required=True, help="data/cut/<name>/shape.json or its name")
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--no-gif", action="store_true", help="skip the two animations")
    args = ap.parse_args()
    shape = load_shape(args.shape)
    out = REPO / "figures" / "shadow3d" / shape["name"]
    out.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook on", shape["path"], "...")
    sim = simulate(args.plan, shape=shape, coarsen=args.coarsen)
    plot_plan(sim, out)
    plot_sections(sim, out)
    plot_cut(sim, out)
    plot_smooth(sim, out)
    if not args.no_gif:
        plot_slice_gif(sim, out)
        plot_cook_gif(sim, out)


if __name__ == "__main__":
    main()
