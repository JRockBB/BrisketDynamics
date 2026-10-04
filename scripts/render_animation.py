"""The animated cook: full T(x,t), W(x,t), F(x,t) fields of the real replay.

The calibrated model (MAP-real parameters) replays the real cook under the
measured fan schedule.

Run from app/:  uv run python -u ../scripts/render_animation.py [--stride N]

Writes figures/model/cook_animation.gif (and .mp4 if ffmpeg is available).
Visualization only — calls the Julia solvers directly (full fields are not
Tesseract outputs; no gradients are involved).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

REPO = Path(__file__).resolve().parents[1]


def main():
    """Render the animated-cook GIF from the calibrated real-cook replay.

    Re-solves the cook with ``monolith_solve`` directly in Julia (the full
    meat fields are not Tesseract outputs), then animates three panels: the
    temperature profile through the slab, the normalized water and
    un-rendered-fat profiles, and the pit/core/surface time series with the
    observed core data, a moving time cursor, and a status line (doneness,
    water retained, fat rendered, stall flag). Writes
    figures/model/cook_animation.gif at ``--fps`` frames/s, one frame per
    ``--stride`` time-grid points.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2, help="frame every Nth grid point")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    from brisket_app import cooklog
    from brisket_app.calibrate import THETA_SITES, from_lighting, lighting_grids, phi_full

    log = cooklog.load(dt_min=10.0)
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = [m[n] for n, _ in THETA_SITES]
    phi = list(phi_full(m))

    # --- Julia: the monolith solve, full meat fields ---
    import juliacall

    jl = juliacall.newmodule("Anim")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jv = lambda x: juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )

    t = log["t"]
    cfg_meat = jl.MeatConfig(T0=log["meta"]["tcore0_K"])
    if from_lighting(log):
        # the cook from lighting (its documented preheat, the fitted preheat damper u_pre); the animation shows the
        # cook from meat-on
        grids, n = lighting_grids(log)
        tl = np.asarray(grids.tknots)
        u = np.concatenate([np.full(n, m["u_pre"]), log["u"]])
        cfg_pit = jl.PitConfig(Tpit0=300.0, T_amb=300.0)
        sol = jl.monolith_solve(
            jv(theta), jv(phi), jv(u), jv(tl), jv(tl), cfg=cfg_meat, pit=cfg_pit, t_load=float(tl[n])
        )
    else:
        n = 0
        cfg_pit = jl.PitConfig(Tpit0=log["meta"]["tpit0_K"], T_amb=300.0)
        sol = jl.monolith_solve(
            jv(theta), jv(phi), jv(log["u"]), jv(t), jv(t), cfg=cfg_meat, pit=cfg_pit
        )

    tpit = np.asarray(sol.Tpit).copy()[n:] - 273.15
    T = np.asarray(sol.T_field).copy()[:, n:] - 273.15  # nx × nt
    W = np.asarray(sol.W_field).copy()[:, n:]  # dry-basis
    C = np.asarray(sol.C_field).copy()[:, n:]
    F = np.asarray(sol.F_field).copy()[:, n:]  # un-rendered fat, dry-basis
    j_evap = np.asarray(sol.j_evap).copy()[n:] * 1e6  # mg/m2/s
    W0 = float(cfg_meat.W0)
    F0 = float(cfg_meat.F0)
    nx, nt = T.shape
    L_cm = theta[3] * 100.0
    x = np.linspace(0.0, L_cm, nx)  # center -> surface [cm]
    th = t / 3600.0

    # --- figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.5], hspace=0.35, wspace=0.28)
    axT = fig.add_subplot(gs[0, 0])
    axW = fig.add_subplot(gs[1, 0])
    axS = fig.add_subplot(gs[:, 1])

    (lnT,) = axT.plot([], [], color="tab:red", lw=2)
    axT.set(
        xlim=(0, L_cm),
        ylim=(0, 115),
        xlabel="depth from center [cm]",
        ylabel="T [°C]",
        title="temperature through the slab",
    )
    axT.axhline(100, color="gray", lw=0.6, ls=":")

    # normalized profiles share one axis; the 1.05-1.3 band is reserved
    # headroom for labels so curves (<=1.02) can never collide with them
    (lnW,) = axW.plot([], [], color="tab:blue", lw=2)
    (lnF,) = axW.plot([], [], color="darkgoldenrod", lw=2)
    axW.set(
        xlim=(0, L_cm),
        ylim=(0, 1.3),
        xlabel="depth from center [cm]",
        ylabel="fraction of initial",
        title="water & fat through the slab",
    )
    axW.axhline(1.0, color="gray", lw=0.6, ls=":")
    axW.text(
        0.02,
        0.94,
        "water W/W₀",
        fontsize=8,
        color="tab:blue",
        va="center",
        transform=axW.transAxes,
    )
    axW.text(
        0.40,
        0.94,
        "un-rendered fat F/F₀",
        fontsize=8,
        color="darkgoldenrod",
        va="center",
        transform=axW.transAxes,
    )
    # Oswin isotherm (k_ox = 0.10, n_ox = 1.5): a_w = 0.9 at W = 0.43
    w90 = 0.10 / (1 / 0.9 - 1) ** (1 / 1.5) / W0
    axW.axhline(w90, color="tab:blue", lw=0.6, ls=":")
    axW.text(
        0.02,
        0.85,
        f"dotted: surface water activity 0.9 (W/W₀ = {w90:.2f})",
        fontsize=7,
        color="tab:blue",
        va="center",
        transform=axW.transAxes,
    )

    axS.plot(th, tpit, color="tab:orange", lw=1, alpha=0.8, label="pit")
    axS.plot(th, T[0], color="tab:red", lw=1.4, label="core")
    axS.plot(th, T[-1], color="tab:pink", lw=1, alpha=0.9, label="surface")
    obs = log["obs_core"] - 273.15
    axS.plot(
        th[log["mask_core"]],
        obs[log["mask_core"]],
        ".",
        ms=2,
        color="k",
        alpha=0.35,
        label="observed core",
    )
    cursor = axS.axvline(0, color="k", lw=1, ymax=0.76)  # stop below text band
    # ylim headroom: data tops out ~112 °C; the 115-145 band is reserved
    # for the 3-line status text so it never sits on the pit trace
    axS.set(
        xlabel="time from meat-on [h]",
        ylabel="T [°C]",
        ylim=(15, 145),
        title='the calibrated cook (Mache/TVWBB, 18.5" WSM)',
    )
    axS.legend(fontsize=8, loc="lower right", framealpha=1).set_zorder(6)
    txt = axS.text(
        0.02,
        0.97,
        "",
        transform=axS.transAxes,
        fontsize=9,
        va="top",
        family="monospace",
    )

    frames = range(0, nt, args.stride)

    def update(i):
        lnT.set_data(x, T[:, i])
        lnW.set_data(x, W[:, i] / W0)
        lnF.set_data(x, F[:, i] / F0)
        cursor.set_xdata([th[i], th[i]])
        stall = (
            "STALL: evap cooling ≈ heat input"
            if (
                55 <= T[0, i] <= 75
                and i > nt // 8
                and T[0, i] - T[0, max(0, i - 6)] < 1.5
            )
            else ""
        )
        w_ret = W[:, i].mean() / W0
        f_ren = 1.0 - F[:, i].mean() / F0
        txt.set_text(
            f"t = {th[i]:5.1f} h   core {T[0, i]:5.1f} °C   "
            f"evap {j_evap[i]:6.0f} mg/m²/s\n"
            f"doneness {C[0, i]:.2f}   water retained {w_ret:4.0%}   "
            f"fat rendered {f_ren:4.0%}\n"
            f"{stall}"
        )
        return lnT, lnW, lnF, cursor, txt

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = REPO / "figures" / "model" / "cook_animation.gif"
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"wrote {out} ({len(list(frames))} frames)")


if __name__ == "__main__":
    main()
