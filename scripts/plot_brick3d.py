"""3-D figures of the planned cook on a brick of the cut (figures/brick3d/).

Re-simulates the plan on the 3-D brick (scripts/shadow3d.jl, one eighth by symmetry; the geometry and the
schedule as scripts/run_shadow3d.py) and draws what the third dimension adds:

- brick3d_plan.png: the pit, the centre and the TRUE cold spot (grid minimum) with the wrap and the pull; the bark
  over the six faces (mean and the face-to-face range) beside the smoke deposition; the thickness and the mass.
- brick3d_fields.png: temperature and moisture on the mid-thickness plane (the whole footprint, mirrored) at
  2 h, at the wrap and at serve: the corners and edges lead and dry first.
- brick3d_surface.png: the top face at serve: Maillard bark, smoke deposition and smoke-ring depth.

Usage (from app/):
    uv run python ../scripts/plot_brick3d.py
    uv run python ../scripts/plot_brick3d.py --plan schedule_robust.json --nx 6 --ny 8 --nz 12
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brick3d_common import HR, REPO, simulate

OUT = REPO / "figures" / "brick3d"


def grid_of(sim, field, k):
    """A node field at save index k as (nx, ny, nz) (x through the thickness from the centre plane, y across the
    half-width, z along the half-length; the mesh's x-fastest node order)."""
    nx, ny, nz = sim["dims"]
    return np.asarray(field)[:, k].reshape((nx, ny, nz), order="F")


def mirror2(a):
    """Reflect a quarter-plane (y, z) array across both symmetry planes: the whole footprint."""
    a = np.vstack([a[::-1, :], a[1:, :]])
    return np.hstack([a[:, ::-1], a[:, 1:]])


def top_face(sim, dist, k):
    """A surface state on the top face (the first ny·nz exposed entries, y fastest) as (ny, nz)."""
    _, ny, nz = sim["dims"]
    return np.asarray(dist)[: ny * nz, k].reshape((ny, nz), order="F")


def footprint_nodes(sim):
    """The mirrored node coordinates of the footprint [cm]: along the length (z) and across the width (y), so
    every value is drawn at its node (the faces at the cut's edges, not half a cell inside them)."""
    nx, ny, nz = sim["dims"]
    c = np.asarray(sim["coords"])
    y = 100.0 * c[[nx * j for j in range(ny)], 1]
    z = 100.0 * c[[nx * ny * k for k in range(nz)], 2]
    return np.concatenate([-z[::-1], z[1:]]), np.concatenate([-y[::-1], y[1:]])


def draw_map(ax, sim, v, **kw):
    zz, yy = footprint_nodes(sim)
    im = ax.pcolormesh(zz, yy, v, shading="gouraud", **kw)
    ax.set_aspect("equal")
    ax.set_xlim(zz[0], zz[-1])
    ax.set_ylim(yy[0], yy[-1])
    return im


def node_place(sim, n):
    """Where node n sits in words: centre, mid-face, edge or corner of the cut."""
    nx, ny, nz = sim["dims"]
    i, j, kk = n % nx, (n // nx) % ny, n // (nx * ny)
    on = [i == nx - 1, j == ny - 1, kk == nz - 1]
    return {0: "inside", 1: "on a face", 2: "on an edge", 3: "at a corner"}[sum(on)]


def plot_plan(sim):
    t = sim["t"] / HR
    dec = sim["decisions"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(t, sim["Tpit"] - 273.15, color="tab:red", lw=2, label="pit")
    ax.plot(t, sim["Tcore"] - 273.15, color="tab:blue", lw=2, label="centre of the cut")
    ax.plot(t, sim["Tcold"] - 273.15, color="tab:cyan", lw=2, ls="--", label="TRUE cold spot (grid min)")
    ax.set_ylabel("temperature [°C]")
    for key, ls in (("t_wrap", ":"), ("t_pull", "--")):
        if dec[key] < 1e8:
            ax.axvline(dec[key] / HR, color="gray", lw=1, ls=ls)
            ax.annotate(key[2:], (dec[key] / HR, ax.get_ylim()[0] + 5), xytext=(-4, 0),
                        textcoords="offset points", color="gray", fontsize=9, ha="right", rotation=90)
    if dec["t_add"] < 1e8 and dec["t_add"] / HR <= t[-1]:
        ax.axvline(dec["t_add"] / HR, color="gray", lw=1, ls="-.")
        ax.annotate("fuel add", (dec["t_add"] / HR, ax.get_ylim()[0] + 5), xytext=(-4, 0),
                    textcoords="offset points", color="gray", fontsize=9, ha="right", rotation=90)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    serve_n = int(np.argmin(sim["T_field"][:, -1]))
    ax.set_title(f"3-D shadow of the planned cook ({sim['plan']}, a "
                 f"{100 * sim['geometry']['thickness']:.1f} × {100 * sim['geometry']['width']:.1f} × "
                 f"{100 * sim['geometry']['length']:.1f} cm flat, rested to serve)\n"
                 f"cold spot at serve {sim['Tcold'][-1] - 273.15:.1f} °C, {node_place(sim, serve_n)}; "
                 f"centre {sim['Tcore'][-1] - 273.15:.1f} °C", fontsize=11)

    ax = axes[1]
    B = np.asarray(sim["bark_dist"])
    ax.fill_between(t, B.min(axis=0), B.max(axis=0), color="tab:brown", alpha=0.25, lw=0,
                    label="bark, face-to-face range")
    ax.plot(t, sim["bark_b"], color="tab:brown", lw=2, label="bark (area mean)")
    ax.plot(t, sim["bark_s"], color="dimgray", lw=2, label="smoke deposition (area mean)")
    ax.set_ylabel("surface state [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.annotate(f"serve: bark {sim['bark_b'][-1]:.2f} ± {sim['bark_std'][-1]:.2f} · ring "
                f"{1000 * sim['ring'][-1]:.1f} mm · water retained {sim['w_retained']:.2f}",
                (0.99, 0.05), xycoords="axes fraction", ha="right", fontsize=9, color="dimgray")

    ax = axes[2]
    ax.plot(t, 200.0 * np.asarray(sim["l_thick"]), color="tab:green", lw=2)
    ax.set_ylabel("thickness [cm]")
    ax.grid(alpha=0.3)
    ax = axes[3]
    ax.plot(t, sim["mass"], color="tab:olive", lw=2)
    ax.set_ylabel("mass [kg]")
    ax.set_xlabel("time [h]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "brick3d_plan.png", dpi=140)
    plt.close(fig)
    print("wrote brick3d_plan.png")


def plot_fields(sim):
    t = sim["t"] / HR
    k_wrap = int(np.searchsorted(t, sim["decisions"]["t_wrap"] / HR)) if sim["decisions"]["t_wrap"] < 1e8 else None
    snaps = [(int(np.searchsorted(t, 2.0)), "2 h")]
    if k_wrap is not None and k_wrap < len(t):
        snaps.append((k_wrap, f"wrap ({t[k_wrap]:.1f} h)"))
    snaps.append((len(t) - 1, f"serve ({t[-1]:.0f} h)"))
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15),
            ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig, axes = plt.subplots(len(rows), len(snaps), figsize=(4.6 * len(snaps) + 1.4, 2.6 * len(rows) + 0.9),
                             squeeze=False, constrained_layout=True)
    for r, (name, label, cmap, f) in enumerate(rows):
        vals = [f(mirror2(grid_of(sim, sim[name], k)[0])) for k, _ in snaps]
        vlo, vhi = min(v.min() for v in vals), max(v.max() for v in vals)
        for c, ((k, title), v) in enumerate(zip(snaps, vals, strict=True)):
            ax = axes[r, c]
            im = draw_map(ax, sim, v, cmap=cmap, vmin=vlo, vmax=vhi)
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel("length [cm]")
            if c == 0:
                ax.set_ylabel("width [cm]")
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, label=label)
    fig.suptitle("The cut in 3-D: the mid-thickness plane over the whole footprint", fontsize=11)
    fig.savefig(OUT / "brick3d_fields.png", dpi=140)
    plt.close(fig)
    print("wrote brick3d_fields.png")


def plot_surface(sim):
    panels = (("bark_dist", "Maillard bark [-]", "YlOrBr", 1.0),
              ("smoke_dist", "smoke deposition [-]", "Greys", 1.0),
              ("ring_dist", "smoke-ring depth [mm]", "RdPu", 1000.0))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.4), constrained_layout=True)
    for ax, (name, label, cmap, scale) in zip(axes, panels, strict=True):
        v = scale * mirror2(top_face(sim, sim[name], -1))
        im = draw_map(ax, sim, v, cmap=cmap)
        ax.set_xlabel("length [cm]")
        ax.set_ylabel("width [cm]")
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
    fig.suptitle("The top face at serve: the edges brown, smoke and ring differently from the middle", fontsize=11)
    fig.savefig(OUT / "brick3d_surface.png", dpi=140)
    plt.close(fig)
    print("wrote brick3d_surface.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=12)
    ap.add_argument("--t-salt-h", type=float, default=0.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook in 3-D ...")
    sim = simulate(args.plan, args.nx, args.ny, args.nz, t_salt_s=args.t_salt_h * HR)
    plot_plan(sim)
    plot_fields(sim)
    plot_surface(sim)


if __name__ == "__main__":
    main()
