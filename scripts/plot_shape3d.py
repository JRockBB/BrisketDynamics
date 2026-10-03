"""Figures of the planned cook on a reconstructed cut (figures/shadow3d/<name>/).

Re-simulates the plan on the cut reconstructed from photos (tools/cutvision's shape.json, a VoxelPiece through
scripts/shadow3d.jl; the schedule as scripts/run_shadow3d.py) and draws:

- shape3d_plan.png: the pit, the deepest point and the TRUE cold spot (grid minimum) with the wrap, the pull and the
  fuel addition; the bark over the exposed faces (mean and range) beside the smoke deposition; the mass.
- shape3d_sections.png: temperature and moisture on the two vertical sections through the deepest point (along x and
  along y) at 2 h, at the wrap and at serve, the height drawn twice its scale.
- shape3d_cut.png: the cut at serve in 3-D: whole, its surface by the model's L* (bark and smoke); and cut through the
  deepest point, the section by temperature, the fat and bone outlined.

Usage (from app/):
    uv run python ../scripts/plot_shape3d.py --shape <name>          # data/cut/<name>/shape.json
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    snaps = snapshots(sim)
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15), ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig, axes = plt.subplots(2 * len(rows), len(snaps), figsize=(5.0 * len(snaps) + 1.2, 9.0), squeeze=False, constrained_layout=True)
    for r, (name, label, cmap, f) in enumerate(rows):
        grids = [grid_of(sim, f(np.asarray(sim[name])[:, k])) for k, _ in snaps]
        vlo = min(np.nanmin(g) for g in grids); vhi = max(np.nanmax(g) for g in grids)
        for c, ((_, title), G) in enumerate(zip(snaps, grids, strict=True)):
            for q, (sec, axis_label) in enumerate(((G[:, cj, :], "x"), (G[ci, :, :], "y"))):
                ax = axes[2 * r + q, c]
                n_h, n_v = sec.shape
                im = ax.imshow(sec.T, origin="lower", cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest",
                               extent=(0, n_h * H * 100, 0, EXAG * n_v * H * 100))
                ax.set_yticks(EXAG * np.arange(0, n_v * H * 100 + 0.1, 2.0), [f"{v:.0f}" for v in np.arange(0, n_v * H * 100 + 0.1, 2.0)])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--shape", required=True, help="data/cut/<name>/shape.json or its name")
    ap.add_argument("--coarsen", type=int, default=4)
    args = ap.parse_args()
    shape = load_shape(args.shape)
    out = REPO / "figures" / "shadow3d" / shape["name"]
    out.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook on", shape["path"], "...")
    sim = simulate(args.plan, shape=shape, coarsen=args.coarsen)
    plot_plan(sim, out)
    plot_sections(sim, out)
    plot_cut(sim, out)


if __name__ == "__main__":
    main()
