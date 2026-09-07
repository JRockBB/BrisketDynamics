"""2-D figures and cook animations (figures/slab2d/).

Re-simulates the planned cook on the 2-D axisymmetric solver (full
fields, via juliacall like render_animation.py) and renders the
grid-only story: the plan traces with the TRUE cold spot, temperature /
moisture / doneness field snapshots on the shrinking cut, the brine
layer and the bark distribution, plus two animations — the cook itself
(T | W | C with the cold-spot marker and the domain contracting as the
meat shrinks) and the salt-and-drying front.

Usage (from app/, so juliacall resolves):
    uv run python ../scripts/plot_slab2d.py
    uv run python ../scripts/plot_slab2d.py --plan schedule_robust.json --fps 10
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

from brisket_app.calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES
from brisket_app.coupled import HR
from brisket_app.shadow2d import THETA2_IDX, equivalent_radius

OUT = REPO / "figures" / "slab2d"


def simulate(args):
    """One 2-D coupled cook of the plan, full fields, via juliacall."""
    import juliacall

    mp = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([mp[n] for n, _ in THETA_SITES], dtype=float)
    phi = np.array(
        [mp[n] for n, _ in PHI_SITES]
        + [mp.get(n, d.mean) for n, d in PHI_SITES_IGNITION],
        dtype=float,
    )
    plan = json.loads((REPO / "data" / "control" / args.plan).read_text())
    u = np.asarray(plan.get("u_robust", plan.get("u_opt")), dtype=float)
    t_end = float(plan.get("cook_h", 11.0)) * HR
    tw = plan.get("t_wrap_robust_h", plan.get("t_wrap_plan_h"))
    tsp = plan.get("t_spritz_robust_h", plan.get("t_spritz_plan_h"))
    dts = plan.get("dt_spritz_robust_min", plan.get("dt_spritz_plan_min"))
    tp = plan.get("t_pull_robust_h", plan.get("t_pull_plan_h"))

    jl = juliacall.newmodule("Slab2dPlot")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jv = lambda x: juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )
    tsave = np.arange(0.0, t_end + 1, 600.0)
    utk = np.linspace(0.0, t_end, len(u))
    kw = dict(
        R=(args.radius if args.radius is not None else equivalent_radius()),
        Hz=2.0 * theta[3],
        nr=args.nr,
        nz=args.nz,
        h=20.0,
        t_wrap=(tw * HR if tw is not None else 1.0e9),
        t_pull=(tp * HR if tp is not None else 1.0e9),
        t_salt=args.t_salt_h * HR,
        t_spritz=(tsp * HR if tsp is not None else 1.0e9),
        dt_spritz=(dts * 60.0 if dts is not None else 3600.0),
    )
    rest = plan.get("rest") or {}
    if rest:  # the plan's rest environment (cambro / board), as the monolith took it
        kw["cfg"] = jl.MeatConfig(
            T_rest=float(rest.get("rest_t", 293.15)),
            h_rest=float(rest.get("rest_h", 1.5)),
            ε_rest=float(rest.get("rest_eps", 0.10)),
            η_rest=float(rest.get("rest_eta", 0.9)),
        )
    o = jl.coupled_solve_2d(jv(theta[list(THETA2_IDX)]), jv(phi), jv(u),
                            jv(utk), jv(tsave), **kw)
    get = lambda name: np.asarray(jl.getproperty(o, jl.Symbol(name)))
    sim = {
        n: get(n)
        for n in (
            "Tpit", "Tcore", "Tcold", "T_field", "W_field", "C_field",
            "bark_rim_t", "bark_ends_t", "R_eff_t", "Hz_eff_t", "salt_field",
            "m_fuel",
        )
    }
    sim["w_retained"] = float(get("w_retained"))
    sim["bark_mean"] = float(get("bark_mean"))
    sim["bark_std"] = float(get("bark_std"))
    sim["t"] = tsave
    sim["t_wrap_h"] = tw
    sim["t_pull_h"] = tp
    sim["radius_m"] = kw["R"]
    sim["plan"] = args.plan
    return sim


def mirror(f2d):
    """Reflect an (nr, nz) field across the axis for a full-width view."""
    return np.vstack([f2d[::-1, :], f2d])


def extent_at(sim, k):
    """Physical extent [cm] of the mirrored field at save index k."""
    r = 100.0 * sim["R_eff_t"][k]
    z = 100.0 * sim["Hz_eff_t"][k]
    return [-r, r, 0.0, z]


def fig_plan(sim):
    """Traces + bark band + shrink — the 2-D read of the planned cook."""
    th = sim["t"] / HR
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.4]})
    ax = axes[0]
    ax.plot(th, sim["Tpit"] - 273.15, color="tab:red", lw=2, label="pit")
    ax.plot(th, sim["Tcore"] - 273.15, color="tab:blue", lw=2,
            label="core (axis)")
    ax.plot(th, sim["Tcold"] - 273.15, color="tab:cyan", lw=2, ls="--",
            label="TRUE cold spot (grid min)")
    if sim["t_wrap_h"]:
        ax.axvline(sim["t_wrap_h"], color="gray", lw=1, ls=":")
        ax.annotate("wrap", (sim["t_wrap_h"], ax.get_ylim()[0] + 5),
                    color="gray", fontsize=9, ha="right", rotation=90)
    if sim["t_pull_h"]:
        ax.axvline(sim["t_pull_h"], color="gray", lw=1, ls="--")
        ax.annotate("pull", (sim["t_pull_h"], ax.get_ylim()[0] + 5),
                    color="gray", fontsize=9, ha="right", rotation=90)
    ax.set_ylabel("temperature [\N{DEGREE SIGN}C]")
    ax.legend(loc="upper left", fontsize=9)  # the pull marker lives lower right
    ax.set_title(f"2-D shadow of the planned cook ({sim['plan']}, "
                 "overnight dry brine, rested to serve)")

    ax = axes[1]
    rim = sim["bark_rim_t"]  # (nz, nt)
    ends = sim["bark_ends_t"]  # (nr, 2, nt)
    bark_all = np.vstack([rim, ends[:, 0, :], ends[:, 1, :]])
    bmean = bark_all.mean(axis=0)
    blo, bhi = bark_all.min(axis=0), bark_all.max(axis=0)
    ax.fill_between(th, blo, bhi, color="tab:brown", alpha=0.25,
                    label="face min-max")
    ax.plot(th, bmean, color="tab:brown", lw=2, label="bark (face mean)")
    ax.set_ylabel("Maillard bark [-]")
    ax.legend(loc="upper left", fontsize=9)
    ax.annotate(
        f"serve: mean {sim['bark_mean']:.2f} \N{PLUS-MINUS SIGN} "
        f"{sim['bark_std']:.2f} \N{MIDDLE DOT} water retained "
        f"{sim['w_retained']:.2f}",
        (0.98, 0.08), xycoords="axes fraction", ha="right", fontsize=9,
        color="dimgray")

    ax = axes[2]
    ax.plot(th, 100 * sim["R_eff_t"], color="tab:olive", lw=2, label="radius")
    ax.plot(th, 100 * sim["Hz_eff_t"], color="tab:green", lw=2,
            label="thickness")
    ax.set_ylabel("size [cm]")
    ax.set_xlabel("time [h]")
    ax.legend(loc="upper right", fontsize=9)
    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "slab2d_plan.png", dpi=140)
    plt.close(fig)


def fig_fields(sim):
    """T and W snapshots on the shrinking cut at three moments."""
    th = sim["t"] / HR
    picks = [np.argmin(np.abs(th - x)) for x in
             (2.0, sim["t_wrap_h"] or th[-1] / 2, th[-1])]
    labels = ["2 h", f"wrap ({th[picks[1]]:.1f} h)", f"serve ({th[-1]:.0f} h)"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6), sharey=True,
                         layout="constrained")
    Tmin, Tmax = 5.0, np.max(sim["T_field"]) - 273.15
    for c, (k, lab) in enumerate(zip(picks, labels, strict=True)):
        ext = extent_at(sim, k)
        imT = axes[0, c].imshow(mirror(sim["T_field"][:, :, k]).T - 273.15,
                                origin="lower", extent=ext, aspect="equal",
                                cmap="inferno", vmin=Tmin, vmax=Tmax)
        imW = axes[1, c].imshow(mirror(sim["W_field"][:, :, k]).T,
                                origin="lower", extent=ext, aspect="equal",
                                cmap="viridis", vmin=0.0,
                                vmax=np.max(sim["W_field"]))
        axes[0, c].set_title(lab, fontsize=10)
        # the cut outline (it shrinks between panels)
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xlim(-100 * sim["R_eff_t"][0], 100 * sim["R_eff_t"][0])
            ax.set_ylim(0, 100 * sim["Hz_eff_t"][0])
            ax.set_xlabel("r [cm]")
    axes[0, 0].set_ylabel("z [cm]  (temperature)")
    axes[1, 0].set_ylabel("z [cm]  (moisture)")
    fig.colorbar(imT, ax=axes[0, :], shrink=0.85, label="T [\N{DEGREE SIGN}C]")
    fig.colorbar(imW, ax=axes[1, :], shrink=0.85, label="W [kg/kg db]")
    fig.suptitle("The cut in 2-D: fields on the shrinking domain")
    fig.savefig(OUT / "slab2d_fields.png", dpi=140)
    plt.close(fig)


def fig_salt_bark(sim):
    """The brine layer (t = 0 vs serve) and the bark distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    ext0 = extent_at(sim, 0)
    vmax = max(float(np.max(sim["salt_field"][:, :, 0])), 1e-6)
    im = axes[0].imshow(mirror(sim["salt_field"][:, :, 0]).T, origin="lower",
                        extent=ext0, aspect="equal", cmap="cividis",
                        vmin=0, vmax=vmax)
    axes[0].set_title("dry brine at light-the-fire", fontsize=10)
    im2 = axes[1].imshow(mirror(sim["salt_field"][:, :, -1]).T, origin="lower",
                         extent=extent_at(sim, len(sim["t"]) - 1),
                         aspect="equal", cmap="cividis", vmin=0, vmax=vmax)
    axes[1].set_title("salt at serve (diffused inward)", fontsize=10)
    for a in axes[:2]:
        a.set_xlim(-100 * sim["R_eff_t"][0] - 0.5, 100 * sim["R_eff_t"][0] + 0.5)
        a.set_ylim(-0.2, 100 * sim["Hz_eff_t"][0] + 0.3)
        a.set_xlabel("r [cm]")
        a.set_ylabel("z [cm]")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="salt [kg/kg db]")
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label="salt [kg/kg db]")

    ax = axes[2]
    nz = sim["bark_rim_t"].shape[0]
    nr = sim["bark_ends_t"].shape[0]
    zfrac = np.linspace(0, 1, nz)
    rfrac = np.linspace(0, 1, nr)
    ax.plot(zfrac, sim["bark_rim_t"][:, -1], color="tab:brown", lw=2,
            label="rim (vs height)")
    ax.plot(rfrac, sim["bark_ends_t"][:, 0, -1], color="tab:orange", lw=2,
            label="bottom face (vs radius)")
    ax.plot(rfrac, sim["bark_ends_t"][:, 1, -1], color="tab:red", lw=2,
            ls="--", label="top face (vs radius)")
    ax.set_xlabel("position along face [-]")
    ax.set_ylabel("Maillard bark [-]")
    ax.set_title("bark distribution at serve", fontsize=10)
    ax.set_ylim(min(0.44, float(np.min(sim["bark_ends_t"][:, :, -1])) - 0.03),
                1.04)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / "slab2d_salt_bark.png", dpi=140)
    plt.close(fig)


def anim_cook(sim, fps):
    """The flagship: T | W | C evolving on the shrinking cut."""
    th = sim["t"] / HR
    nt = len(th)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), layout="constrained")
    ext0 = extent_at(sim, 0)
    Tmax = np.max(sim["T_field"]) - 273.15
    Wmax = np.max(sim["W_field"])
    ims, cms, labels = [], ["inferno", "viridis", "magma"], [
        "temperature [\N{DEGREE SIGN}C]", "moisture [kg/kg db]",
        "doneness (collagen) [-]"]
    vlims = [(5.0, Tmax), (0.0, Wmax), (0.0, 1.0)]
    fields = [sim["T_field"] - 273.15, sim["W_field"], sim["C_field"]]
    for ax, cm, lab, (vlo, vhi), f in zip(axes, cms, labels, vlims, fields,
                                          strict=True):
        im = ax.imshow(mirror(f[:, :, 0]).T, origin="lower", extent=ext0,
                       aspect="equal", cmap=cm, vmin=vlo, vmax=vhi)
        ims.append(im)
        ax.set_xlim(-100 * sim["R_eff_t"][0], 100 * sim["R_eff_t"][0])
        ax.set_ylim(0, 100 * sim["Hz_eff_t"][0])
        ax.set_xlabel("r [cm]")
        fig.colorbar(im, ax=ax, shrink=0.75, label=lab)
    axes[0].set_ylabel("z [cm]")
    (cold_dot,) = axes[0].plot([], [], "o", mfc="none", mec="cyan", mew=2,
                               ms=10)
    title = fig.suptitle("", fontsize=12)

    def draw(k):
        for im, f in zip(ims, fields, strict=True):
            im.set_data(mirror(f[:, :, k]).T)
            im.set_extent(extent_at(sim, k))
        Tf = sim["T_field"][:, :, k]
        i, j = np.unravel_index(np.argmin(Tf), Tf.shape)
        nr, nz = Tf.shape
        rk, zk = 100 * sim["R_eff_t"][k], 100 * sim["Hz_eff_t"][k]
        cold_dot.set_data([rk * i / (nr - 1)], [zk * j / (nz - 1)])
        state = ""
        if sim["t_wrap_h"] and th[k] >= sim["t_wrap_h"]:
            state = " \N{MIDDLE DOT} wrapped"
        if sim["t_pull_h"] and th[k] >= sim["t_pull_h"]:
            state += " \N{MIDDLE DOT} resting"
        title.set_text(
            f"the cook in 2-D \N{MIDDLE DOT} t = {th[k]:4.1f} h{state} "
            f"\N{MIDDLE DOT} cold spot {np.min(Tf) - 273.15:5.1f} "
            f"\N{DEGREE SIGN}C \N{MIDDLE DOT} size "
            f"{200 * sim['R_eff_t'][k]:.1f} x "
            f"{100 * sim['Hz_eff_t'][k]:.1f} cm")
        return [*ims, cold_dot, title]

    ani = FuncAnimation(fig, draw, frames=nt, blit=False)
    ani.save(OUT / "cook2d.gif", writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def anim_salt(sim, fps):
    """The brine and the drying front: W field with salt contours."""
    th = sim["t"] / HR
    nt = len(th)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    ext0 = extent_at(sim, 0)
    Wmax = np.max(sim["W_field"])
    im = ax.imshow(mirror(sim["W_field"][:, :, 0]).T, origin="lower",
                   extent=ext0, aspect="equal", cmap="viridis", vmin=0.0,
                   vmax=Wmax)
    ax.set_xlim(-100 * sim["R_eff_t"][0], 100 * sim["R_eff_t"][0])
    ax.set_ylim(0, 100 * sim["Hz_eff_t"][0])
    ax.set_xlabel("r [cm]")
    ax.set_ylabel("z [cm]")
    fig.colorbar(im, ax=ax, shrink=0.85, label="moisture [kg/kg db]")
    smax = max(float(np.max(sim["salt_field"])), 1e-9)
    levels = [0.25 * smax, 0.5 * smax, 0.75 * smax]
    title = ax.set_title("")
    state = {"cs": None}

    def draw(k):
        im.set_data(mirror(sim["W_field"][:, :, k]).T)
        im.set_extent(extent_at(sim, k))
        if state["cs"] is not None:
            state["cs"].remove()
        ext = extent_at(sim, k)
        nrm, nzm = mirror(sim["salt_field"][:, :, k]).shape
        xs = np.linspace(ext[0], ext[1], nrm)
        zs = np.linspace(ext[2], ext[3], nzm)
        state["cs"] = ax.contour(xs, zs, mirror(sim["salt_field"][:, :, k]).T,
                                 levels=levels, colors="white",
                                 linewidths=1.0, alpha=0.8)
        title.set_text(f"brine layer + drying fronts \N{MIDDLE DOT} "
                       f"t = {th[k]:4.1f} h")
        return [im, title]

    ani = FuncAnimation(fig, draw, frames=nt, blit=False)
    ani.save(OUT / "salt2d.gif", writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main():
    """Simulate the planned cook in 2-D and render figures + animations."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--radius", type=float, default=None,
                    help="cylinder radius [m]; default: the 1-D exposed-area equivalent")
    ap.add_argument("--nr", type=int, default=24)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--t-salt-h", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook in 2-D ...")
    sim = simulate(args)
    print(f"  w_retained {sim['w_retained']:.3f}  bark {sim['bark_mean']:.2f}"
          f" +/- {sim['bark_std']:.2f}")
    fig_plan(sim)
    print("wrote slab2d_plan.png")
    fig_fields(sim)
    print("wrote slab2d_fields.png")
    fig_salt_bark(sim)
    print("wrote slab2d_salt_bark.png")
    anim_cook(sim, args.fps)
    print("wrote cook2d.gif")
    anim_salt(sim, args.fps)
    print("wrote salt2d.gif")


if __name__ == "__main__":
    main()
