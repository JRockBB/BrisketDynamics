"""3-D figures of the planned cook on a trimmed brisket flat (figures/shadow3d/).

Re-simulates the plan on the flat (scripts/shadow3d.jl: tapering in thickness and width from the thick end, a fat
cap on top, half of it solved across its width; the schedule as scripts/run_shadow3d.py) and draws what the third
dimension adds:

- shadow3d_plan.png: the pit, the centre and the TRUE cold spot (grid minimum) with the wrap, the pull and the fuel
  addition; the bark over the faces (mean and range) beside the smoke deposition; the core's depth below the
  nearest surface and the mass.
- shadow3d_fields.png: temperature and moisture on the lengthwise section through the centre plane at 2 h, at the
  wrap and at serve: the thin end leads, the thick end lags, the fat cap insulates the top.
- shadow3d_surface.png: the top face at serve over the whole footprint: Maillard bark, smoke deposition and
  smoke-ring depth.

Usage (from app/):
    uv run python ../scripts/plot_shadow3d.py
    uv run python ../scripts/plot_shadow3d.py --plan schedule_robust.json --nl 12 --nw 6 --nt 8
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shadow3d_common import HR, REPO, coords_grid, face_state, field_at, mirror_width, simulate

OUT = REPO / "figures" / "shadow3d"
SECTION_EXAG = 2.0     # the sections' height drawn twice its scale


def node_place(sim, n):
    """Where node n sits in words: inside, on a face, an edge or a corner, and toward which end."""
    nl, nw, nt = sim["dims"]
    i, j, k = n % nl, (n // nl) % nw, n // (nl * nw)
    on = [i in (0, nl - 1), j == nw - 1, k in (0, nt - 1)]
    where = {0: "inside", 1: "on a face", 2: "on an edge", 3: "at a corner"}[sum(on)]
    end = "the thick end" if i < nl / 3 else ("the thin end" if i > 2 * nl / 3 else "mid-length")
    return f"{where} toward {end}"


def plot_plan(sim):
    t = sim["t"] / HR
    dec = sim["decisions"]
    g = sim["geometry"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(t, sim["Tpit"] - 273.15, color="tab:red", lw=2, label="pit")
    ax.plot(t, sim["Tcore"] - 273.15, color="tab:blue", lw=2, label="the cut's deepest point")
    ax.plot(t, sim["Tcold"] - 273.15, color="tab:cyan", lw=2, ls="--", label="TRUE cold spot (grid min)")
    ax.set_ylabel("temperature [°C]")
    marks = [("wrap", dec["t_wrap"], ":"), ("pull", dec["t_pull"], "--"), ("fuel add", dec["t_add"], "-.")]
    for lab, tt, ls in marks:
        if tt < 1e8 and tt / HR <= t[-1]:
            ax.axvline(tt / HR, color="gray", lw=1, ls=ls)
            ax.annotate(lab, (tt / HR, ax.get_ylim()[0] + 5), xytext=(-4, 0), textcoords="offset points",
                        color="gray", fontsize=9, ha="right", rotation=90)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    serve_n = int(np.argmin(np.asarray(sim["T_field"])[:, -1]))
    ax.set_title(f"3-D shadow of the planned cook ({sim['plan']}) on a trimmed flat: {100 * g['length']:.0f} cm long, "
                 f"{100 * g['thickness_thick']:.1f} → {100 * g['thickness_thin']:.1f} cm thick, "
                 f"{1000 * g['cap']:.0f} mm fat cap\n"
                 f"cold spot at serve {sim['Tcold'][-1] - 273.15:.1f} °C, {node_place(sim, serve_n)}; "
                 f"deepest point {sim['Tcore'][-1] - 273.15:.1f} °C", fontsize=11)

    ax = axes[1]
    B = np.asarray(sim["bark_dist"])
    ax.fill_between(t, B.min(axis=0), B.max(axis=0), color="tab:brown", alpha=0.25, lw=0, label="bark, range over the faces")
    ax.plot(t, sim["bark_b"], color="tab:brown", lw=2, label="bark (area mean)")
    ax.plot(t, sim["bark_s"], color="dimgray", lw=2, label="smoke deposition (area mean)")
    ax.set_ylabel("surface state [-]")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.annotate(f"serve: bark {sim['bark_b'][-1]:.2f} ± {sim['bark_std'][-1]:.2f} · ring "
                f"{1000 * sim['ring'][-1]:.1f} mm · water retained {sim['w_retained']:.2f} · fat rendered "
                f"{sim['fat_rendered']:.2f}", (0.99, 0.05), xycoords="axes fraction", ha="right", fontsize=9,
                color="dimgray")

    ax = axes[2]
    ax.plot(t, 100.0 * np.asarray(sim["l_thick"]), color="tab:green", lw=2)
    ax.set_ylabel("core depth [cm]")
    ax.grid(alpha=0.3)
    ax = axes[3]
    ax.plot(t, sim["mass"], color="tab:olive", lw=2)
    ax.set_ylabel("mass [kg]")
    ax.set_xlabel("time [h]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "shadow3d_plan.png", dpi=140)
    plt.close(fig)
    print("wrote shadow3d_plan.png")


def snapshots(sim):
    t = sim["t"] / HR
    out = [(int(np.searchsorted(t, 2.0)), "2 h")]
    tw = sim["decisions"]["t_wrap"] / HR
    if tw < 1e4:
        k = int(np.searchsorted(t, tw)); out.append((k, f"wrap ({t[k]:.1f} h)"))
    out.append((len(t) - 1, f"serve ({t[-1]:.0f} h)"))
    return out


def plot_fields(sim):
    S, Y, Z = coords_grid(sim)
    snaps = snapshots(sim)
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15),
            ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig, axes = plt.subplots(len(rows), len(snaps), figsize=(5.2 * len(snaps) + 1.4, 2.3 * len(rows) + 1.0),
                             squeeze=False, constrained_layout=True)
    Ss, Zs = S[:, 0, :], SECTION_EXAG * Z[:, 0, :]                  # the centre plane: length × height
    for r, (name, label, cmap, f) in enumerate(rows):
        vals = [f(field_at(sim, name, k)[:, 0, :]) for k, _ in snaps]
        vlo, vhi = min(v.min() for v in vals), max(v.max() for v in vals)
        for c, ((k, title), v) in enumerate(zip(snaps, vals, strict=True)):
            ax = axes[r, c]
            im = ax.pcolormesh(Ss, Zs, v, shading="gouraud", cmap=cmap, vmin=vlo, vmax=vhi)
            ax.set_aspect("equal")
            ax.set_xlim(Ss.min() - 0.5, Ss.max() + 0.5); ax.set_ylim(-0.5, Zs.max() + 0.5)
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel("length from the thick end [cm]")
            if c == 0:
                ax.set_ylabel(f"height [cm] (×{SECTION_EXAG:.0f})")
            ax.set_yticks(SECTION_EXAG * np.array([0.0, 2.0, 4.0]), ["0", "2", "4"])
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, label=label)
    fig.suptitle("The flat cut lengthwise through its centre: the thin end leads, the thick end lags", fontsize=11)
    fig.savefig(OUT / "shadow3d_fields.png", dpi=140)
    plt.close(fig)
    print("wrote shadow3d_fields.png")


def plot_surface(sim):
    S, Y, Z = coords_grid(sim)
    St = mirror_width(S[:, :, -1]); Yt = mirror_width(Y[:, :, -1], negate=True)
    panels = (("bark_dist", "Maillard bark [-]", "YlOrBr", 1.0),
              ("smoke_dist", "smoke deposition [-]", "Greys", 1.0),
              ("ring_dist", "smoke-ring depth [mm]", "RdPu", 1000.0))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0), constrained_layout=True)
    for ax, (name, label, cmap, scale) in zip(axes, panels, strict=True):
        v = scale * mirror_width(face_state(sim, name, -1, "top"))
        im = ax.pcolormesh(St, Yt, v, shading="gouraud", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlim(St.min() - 0.5, St.max() + 0.5); ax.set_ylim(Yt.min() - 0.5, Yt.max() + 0.5)
        ax.set_xlabel("length from the thick end [cm]")
        ax.set_ylabel("width [cm]")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.suptitle("The top face (the fat cap) at serve over the whole footprint", fontsize=11)
    fig.savefig(OUT / "shadow3d_surface.png", dpi=140)
    plt.close(fig)
    print("wrote shadow3d_surface.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nl", type=int, default=12)
    ap.add_argument("--nw", type=int, default=6)
    ap.add_argument("--nt", type=int, default=8)
    ap.add_argument("--t-salt-h", type=float, default=0.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook in 3-D ...")
    sim = simulate(args.plan, args.nl, args.nw, args.nt, t_salt_s=args.t_salt_h * HR)
    plot_plan(sim)
    plot_fields(sim)
    plot_surface(sim)


if __name__ == "__main__":
    main()
