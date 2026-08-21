"""Roadmap stretch item 3: differentiable wrap-time (Texas crutch).

Joint optimization of the damper schedule AND the wrap moment at the
MAP-real smoker: wrapping suppresses evaporation (kills the stall, keeps
moisture) but freezes smoke deposition and slows browning — bark quality
here is the MECHANISTIC bark model's predicted final surface L*
(docs/bark_plan.md), so "when to wrap" trades juiciness and speed against
actual predicted color.

Run from app/:  uv run python -u ../scripts/run_wraptime.py
Writes figures/wraptime.png + data/control/wraptime_result.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig, optimize_schedule
from brisket_app.coupled import HR, CookGrids, coupled_cook

REPO = Path(__file__).resolve().parents[1]
T_F = 10 * HR
GRIDS = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=14)
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}
ITERS = 60


def simulate(theta, phi, u, meat_tx, pit_tx, t_wrap=1.0e9):
    """Run one coupled cook on the module grids and return NumPy outputs.

    Parameters
    ----------
    theta : array_like, shape (7,)
        Meat parameters (here the MAP-real fit).
    phi : array_like, shape (7,)
        Pit parameters (here the MAP-real fit).
    u : array_like
        Damper schedule on the GRIDS control knots, values in [0, 1].
    meat_tx, pit_tx
        Tesseract clients for the meat and pit solvers.
    t_wrap : float, optional
        Texas-crutch wrap time [s]; the default 1e9 means never wrap.

    Returns
    -------
    dict of str to numpy.ndarray
        The ``coupled_cook`` outputs converted to NumPy arrays.
    """
    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.asarray(u),
        meat_tx,
        pit_tx,
        GRIDS,
        K=2,
        ics=ICS,
        t_wrap=t_wrap,
    )
    return {k: np.asarray(v) for k, v in sim.items()}


def metrics(sim, tk):
    """Summarize cook quality from one simulation.

    Parameters
    ----------
    sim : dict of str to numpy.ndarray
        ``simulate`` output.
    tk : numpy.ndarray
        Uniform time knots [s] matching the simulation outputs.

    Returns
    -------
    dict
        Scalar metrics: final core doneness (``doneness``), retained
        moisture fraction (``w_retained``), final bark color and states
        (``l_star_end``, ``bark_b``, ``bark_s``), the longest stretch of
        core temperature climbing slower than 1.5 °C/h inside the 60-82 °C
        window (``stall_h``, hours), and the final core temperature
        (``core_end_C``, °C).
    """
    TcoreC = sim["tcore"] - 273.15
    dt_h = (tk[1] - tk[0]) / HR
    rate = np.diff(TcoreC) / dt_h
    in_stall = (rate < 1.5) & (TcoreC[:-1] >= 60) & (TcoreC[:-1] <= 82)
    best = cur = 0
    for b in in_stall:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return {
        "doneness": float(sim["c_field_final"][0]),
        "w_retained": float(sim["w_retained"]),
        "l_star_end": float(sim["l_star"][-1]),
        "bark_b": float(sim["bark_b"][-1]),
        "bark_s": float(sim["bark_s"][-1]),
        "stall_h": best * dt_h,
        "core_end_C": float(TcoreC[-1]),
    }


def main():
    """Optimize the wrap time jointly with the damper schedule and plot it.

    Finds the no-wrap optimal schedule first, then jointly optimizes
    (schedule, t_wrap) from a 3-point multi-start over the initial wrap
    logit, and finally sweeps a 1-D objective slice over wrap time at the
    optimal schedule. Writes data/control/wraptime_result.json and
    figures/wraptime.png: no-wrap versus wrapped core/surface trajectories
    with the chosen wrap moment, and the objective-versus-wrap-time slice
    with the optimizer's choice marked.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    meat_tx, pit_tx = meat_client(), pit_client()
    tk = np.asarray(GRIDS.tknots)

    print("1) no-wrap optimum (juiciness- and bark-aware)")
    cc0 = ControlConfig(n_iter=ITERS)
    u0, _ = optimize_schedule(
        theta, phi, meat_tx, pit_tx, grids=GRIDS, cc=cc0, u_init=0.5, ics=ICS
    )
    sim0 = simulate(theta, phi, u0, meat_tx, pit_tx)
    m0 = metrics(sim0, tk)
    print(f"   no-wrap: {m0}")

    print("2) joint (schedule, t_wrap) optimization, 3-point multi-start")
    cc = ControlConfig(opt_wrap=True, n_iter=ITERS)
    best = None
    for zw0 in (-2.0, 0.0, 2.0):
        u, h = optimize_schedule(
            theta,
            phi,
            meat_tx,
            pit_tx,
            grids=GRIDS,
            cc=cc,
            u_init=u0,
            ics=ICS,
            z_w_init=zw0,
        )
        print(
            f"   start z_w={zw0:+.0f}: J={h['losses'][-1]:.4f}  "
            f"t_wrap={h['t_wrap'] / HR:.2f} h"
        )
        if best is None or h["losses"][-1] < best[1]["losses"][-1]:
            best = (u, h)
    u_w, h_w = best
    t_wrap = h_w["t_wrap"]
    sim_w = simulate(theta, phi, u_w, meat_tx, pit_tx, t_wrap=t_wrap)
    m_w = metrics(sim_w, tk)
    print(f"   optimal wrap at {t_wrap / HR:.2f} h: {m_w}")

    print("3) 1-D objective slice over t_wrap at the optimal schedule")
    from brisket_app.control import objective

    slice_tw = np.linspace(0.1 * T_F, T_F, 15)
    zs = np.log(np.clip(u_w, 1e-3, 1 - 1e-3) / (1 - np.clip(u_w, 1e-3, 1 - 1e-3)))
    js = []
    for twv in slice_tw:
        zw = np.log(twv / (T_F - twv)) if twv < T_F else 8.0
        j = float(
            objective(
                jnp.asarray(zs),
                jnp.asarray(zw),
                jnp.asarray(theta),
                jnp.asarray(phi),
                meat_tx,
                pit_tx,
                GRIDS,
                cc,
                {"abstol": 1e-6, "reltol": 1e-6},
                ics=ICS,
            )
        )
        js.append(j)
        print(f"   t_wrap={twv / HR:5.2f} h  J={j:.4f}")

    (REPO / "data" / "control").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "control" / "wraptime_result.json").write_text(
        json.dumps(
            {
                "t_wrap_h": t_wrap / HR,
                "no_wrap": m0,
                "wrapped": m_w,
                "slice_t_wrap_h": (slice_tw / HR).tolist(),
                "slice_J": js,
                "u_no_wrap": u0.tolist(),
                "u_wrap": u_w.tolist(),
            },
            indent=2,
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, MUTED = "#256abf", "#eb6834", "#52514e"
    th = tk / HR
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(th, sim0["tcore"] - 273.15, "--", color=MUTED, label="no wrap: core")
    ax.plot(th, sim_w["tcore"] - 273.15, "-", color=BLUE, lw=2, label="wrapped: core")
    ax.plot(
        th,
        sim_w["tsurf"] - 273.15,
        "-",
        color=BLUE,
        lw=1,
        alpha=0.5,
        label="wrapped: surface",
    )
    ax.axvline(t_wrap / HR, color=ORANGE, lw=1.5, ls=":")
    ax.annotate(
        f"wrap at {t_wrap / HR:.1f} h",
        (t_wrap / HR + 0.15, 30),
        color=ORANGE,
        fontsize=10,
    )
    ax.set(
        xlabel="time [h]",
        ylabel="temperature [°C]",
        title=f"the model decides when to wrap "
        f"(doneness {m_w['doneness']:.3f}, juiciness "
        f"{m_w['w_retained']:.2f} vs {m0['w_retained']:.2f}, "
        f"bark L* {m_w['l_star_end']:.0f} vs {m0['l_star_end']:.0f})",
    )
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    ax.plot(slice_tw / HR, js, "o-", color=BLUE, lw=2)
    ax.axvline(t_wrap / HR, color=ORANGE, lw=1.5, ls=":")
    ax.annotate(
        f"optimizer: {t_wrap / HR:.1f} h",
        (
            t_wrap / HR + 0.15,
            float(np.min(js)) + 0.75 * (float(np.max(js)) - float(np.min(js))),
        ),
        color=ORANGE,
        fontsize=10,
    )
    ax.set(
        xlabel="wrap time [h]",
        ylabel="objective J",
        title="objective vs wrap time (fixed optimal schedule)",
    )

    fig.suptitle(
        "Differentiable Texas crutch: wrap time optimized jointly "
        "with the damper schedule",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "wraptime.png", dpi=140, bbox_inches="tight")
    print("wrote figures/wraptime.png")


if __name__ == "__main__":
    main()
