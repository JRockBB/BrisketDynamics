"""The animated cook: T(x,t) and W(x,t) fields of the calibrated model
replaying the real cook (MAP-real parameters, measured fan schedule).

Run from app/:  uv run python -u ../scripts/render_animation.py [--stride N]

Writes figures/cook_animation.gif (and .mp4 if ffmpeg is available).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2, help="frame every Nth grid point")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    from brisket_app import cooklog
    from brisket_app.calibrate import THETA_SITES, PHI_SITES

    log = cooklog.load(dt_min=10.0)
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = [m[n] for n, _ in THETA_SITES]
    phi = [m[n] for n, _ in PHI_SITES]

    # --- Julia: 2 Picard iterations of the coupled cook, full meat fields ---
    import juliacall

    jl = juliacall.newmodule("Anim")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jv = lambda x: juliacall.convert(jl.Vector, np.ascontiguousarray(
        np.asarray(x, dtype=np.float64)))

    t = log["t"]
    cfg_pit = jl.PitConfig(Tpit0=log["meta"]["tpit0_K"], T_amb=300.0)
    cfg_meat = jl.MeatConfig(T0=log["meta"]["tcore0_K"])
    q = np.zeros(len(t))
    for _ in range(2):
        pit = jl.pit_solve(jv(phi), jv(log["u"]), jv(t), jv(q), jv(t), jv(t),
                           cfg=cfg_pit)
        meat = jl.meat_solve(jv(theta), pit.Tpit, jv(t), jv(t), cfg=cfg_meat)
        q = np.asarray(meat.q_meat).copy()

    tpit = np.asarray(pit.Tpit).copy() - 273.15
    T = np.asarray(meat.T_field).copy() - 273.15   # nx × nt
    W = np.asarray(meat.W_field).copy()            # dry-basis
    C = np.asarray(meat.C_field).copy()
    j_evap = np.asarray(meat.j_evap).copy() * 1e6  # mg/m2/s
    nx, nt = T.shape
    L_cm = theta[3] * 100.0
    x = np.linspace(0.0, L_cm, nx)                 # center -> surface [cm]
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

    lnT, = axT.plot([], [], color="tab:red", lw=2)
    axT.set(xlim=(0, L_cm), ylim=(0, 115), xlabel="depth from center [cm]",
            ylabel="T [°C]", title="temperature through the slab")
    axT.axhline(100, color="gray", lw=0.6, ls=":")

    lnW, = axW.plot([], [], color="tab:blue", lw=2)
    axW.set(xlim=(0, L_cm), ylim=(0, 2.5), xlabel="depth from center [cm]",
            ylabel="W [kg/kg dry]", title="moisture through the slab")
    wcrit = 0.40
    axW.axhline(wcrit, color="tab:blue", lw=0.6, ls=":")
    axW.annotate("evaporation throttles below this", (0.02, wcrit + 0.06),
                 fontsize=7, color="tab:blue")

    axS.plot(th, tpit, color="tab:orange", lw=1, alpha=0.8, label="pit")
    axS.plot(th, T[0], color="tab:red", lw=1.4, label="core")
    axS.plot(th, T[-1], color="tab:pink", lw=1, alpha=0.9, label="surface")
    obs = log["obs_core"] - 273.15
    axS.plot(th[log["mask_core"]], obs[log["mask_core"]], ".", ms=2,
             color="k", alpha=0.35, label="observed core")
    cursor = axS.axvline(0, color="k", lw=1)
    axS.set(xlabel="time [h]", ylabel="T [°C]",
            title="the calibrated cook (Mache/TVWBB, 18.5\" WSM)")
    axS.legend(fontsize=8, loc="lower right")
    txt = axS.text(0.02, 0.97, "", transform=axS.transAxes, fontsize=9,
                   va="top", family="monospace")

    frames = range(0, nt, args.stride)

    def update(i):
        lnT.set_data(x, T[:, i])
        lnW.set_data(x, W[:, i])
        cursor.set_xdata([th[i], th[i]])
        stall = "STALL: evaporative cooling ~ heat input" if \
            (55 <= T[0, i] <= 75 and i > nt // 8 and T[0, i] - T[0, max(0, i - 6)] < 1.5) else ""
        txt.set_text(f"t = {th[i]:5.1f} h   core {T[0, i]:5.1f} °C   "
                     f"evap {j_evap[i]:6.0f} mg/m²/s   doneness {C[0, i]:.2f}\n{stall}")
        return lnT, lnW, cursor, txt

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = REPO / "figures" / "cook_animation.gif"
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"wrote {out} ({len(list(frames))} frames)")


if __name__ == "__main__":
    main()
