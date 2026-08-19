"""The full quality composition: DURATION x WRAP TIME x SCHEDULE.

Outer scan over planned durations; inside each, the damper schedule AND
the Texas-crutch wrap time are optimized jointly against the full quality
objective — doneness band (undershoot catastrophic, gentle overdone
penalty above 0.95), juiciness, mechanistic bark color L*, bark-burn
surface constraint, fuel. The perfect brisket picks its own duration AND
its own wrap moment.

Run from app/:  uv run python -u ../scripts/run_fullplan.py
Writes figures/fullplan.png + data/control/fullplan_result.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp

from brisket_app.clients import meat_client, pit_client
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.control import ControlConfig
from brisket_app.coupled import HR, coupled_cook
from brisket_app.timeplan import optimal_duration, plan_duration

REPO = Path(__file__).resolve().parents[1]
# override the scan grid from the command line: run_fullplan.py 14 15 16 17 18
DURATIONS_H = ([float(a) for a in sys.argv[1:]]
               if len(sys.argv) > 1 else
               [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
RESULT_NAME = ("fullplan_result_ext.json" if len(sys.argv) > 1
               else "fullplan_result.json")
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}
CC = ControlConfig(asymmetric_done=True, w_juicy=5.0, opt_wrap=True,
                   l_star_target=30.0, w_bark=0.5, n_iter=60)


def main():
    # startup self-check: prove the RUNNING process imports the band we
    # think it does (stale-import debugging cost us a scan once)
    import jax
    _k = 4.0 / (1.0 - CC.c_overdone)
    _ov = float((jax.nn.softplus(_k * (0.9995 - CC.c_overdone)) / _k)
                / (1.0 - CC.c_overdone))
    print(f"band self-check: penalty at c=0.9995 -> "
          f"{CC.w_overdone * _ov**2:.1f} (expect ~24)", flush=True)
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    meat_tx, pit_tx = meat_client(), pit_client()

    seed = None
    if RESULT_NAME.endswith("_ext.json"):
        # extension run: warm-start from the base scan's best (a CONVERGED
        # optimum). A cold 60-iter start at 14 h parked in doneness
        # saturation (J = 27.9 vs the warm chain's -0.163) and would have
        # poisoned every duration above it.
        base = json.loads(
            (REPO / "data" / "control" / "fullplan_result.json").read_text())
        seed = {"u": base["best_u"], "t_f_h": base["best_t_f_h"],
                "t_wrap_frac": base["best_t_wrap_h"] / base["best_t_f_h"]}
        print(f"extension seeded from base optimum at {base['best_t_f_h']} h")
    print("full composition scan: duration x wrap x schedule")
    records = plan_duration(theta, phi, meat_tx, pit_tx, DURATIONS_H,
                            cc=CC, ics=ICS, seed=seed)
    best = optimal_duration(records)
    print(f"THE COOK: {best['t_f_h']:.0f} h total, wrap at "
          f"{best['t_wrap_h']:.2f} h — doneness {best['doneness']:.3f}, "
          f"juiciness {best['w_retained']:.3f}, bark L* {best['l_star_end']:.1f}")

    (REPO / "data" / "control").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "control" / RESULT_NAME).write_text(json.dumps({
        "durations_h": [r["t_f_h"] for r in records],
        "j_quality": [r["j_quality"] for r in records],
        "t_wrap_h": [r["t_wrap_h"] for r in records],
        "doneness": [r["doneness"] for r in records],
        "w_retained": [r["w_retained"] for r in records],
        "l_star_end": [r["l_star_end"] for r in records],
        "best_t_f_h": best["t_f_h"],
        "best_t_wrap_h": best["t_wrap_h"],
        "best_u": best["u_opt"].tolist(),
    }, indent=2))

    sim = coupled_cook(jnp.asarray(theta), jnp.asarray(phi),
                       jnp.asarray(best["u_opt"]), meat_tx, pit_tx,
                       best["grids"], K=2, ics=ICS,
                       t_wrap=best["t_wrap_h"] * HR)
    sim = {k: np.asarray(v) for k, v in sim.items()}
    th = np.asarray(best["grids"].tknots) / HR

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED = "#256abf", "#eb6834", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    t_fs = np.array([r["t_f_h"] for r in records])
    js = np.array([r["j_quality"] for r in records])
    ax.plot(t_fs, js, "o-", color=BLUE, lw=2)
    for r in records:
        ax.annotate(f"wrap\n{r['t_wrap_h']:.1f}h", (r["t_f_h"], r["j_quality"]),
                    textcoords="offset points", xytext=(0, -22),
                    color=MUTED, fontsize=7, ha="center")
    ax.axvline(best["t_f_h"], color=ORANGE, lw=1.5, ls=":")
    ax.annotate(f"the cook: {best['t_f_h']:.0f} h",
                (best["t_f_h"] + 0.12, float(js.max()) * 0.85),
                color=ORANGE, fontsize=10)
    ax.set(xlabel="planned duration t_f [h]", ylabel="quality objective J",
           ylim=(float(js.min()) - abs(float(js.min())) * 0.35 - 0.2,
                 float(js.max()) + 0.3),
           title="duration and wrap time chosen together\n"
                 "(gray: each duration's own optimal wrap moment)")

    ax = axes[1]
    ax.plot(th, sim["tcore"] - 273.15, "-", color=BLUE, lw=2, label="core")
    ax.plot(th, sim["tsurf"] - 273.15, "-", color=BLUE, lw=1, alpha=0.5,
            label="surface")
    ax2c = MUTED
    ax.axvline(best["t_wrap_h"], color=ORANGE, lw=1.5, ls=":")
    ax.annotate(f"wrap at {best['t_wrap_h']:.1f} h",
                (best["t_wrap_h"] - 0.15, 25), color=ORANGE, fontsize=10,
                ha="right")
    ax.set(xlabel="time [h]", ylabel="temperature [°C]",
           title=f"the chosen cook: {best['t_f_h']:.0f} h · "
                 f"doneness {best['doneness']:.3f} · juiciness "
                 f"{best['w_retained']:.2f} · bark L* {best['l_star_end']:.0f}")
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("The perfect brisket picks its own duration and wrap moment "
                 "(MAP smoker; bark constants uncalibrated)", y=1.0)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "fullplan.png", dpi=140,
                bbox_inches="tight")
    print("wrote figures/fullplan.png")


if __name__ == "__main__":
    main()
