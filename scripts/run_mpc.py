"""Receding-horizon MPC defending THE assessed optimal cook.

The plan under defense is the fullplan optimum (17 h, wrap at ~3.5 h —
data/control/fullplan_result_ext.json), not a freshly derived schedule.
The calibrated smoker (MAP-real parameters) is the plant. At t = 2 h —
BEFORE the planned wrap — a cold front drops ambient by 15 K, which the
controller's model does not know. Compare:
  reference: the assessed plan, no disturbance
  open-loop: the assessed plan executed blindly through the disturbance
             (wrap fires at its planned moment regardless)
  MPC:       hourly re-plans schedule AND wrap moment from measured state
             (full-field handoff; wrap irreversible once executed)

Run from app/:  uv run python -u ../scripts/run_mpc.py
Writes figures/mpc_demo.png + data/control/mpc_result.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import mpc
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig
from brisket_app.coupled import HR, CookGrids

REPO = Path(__file__).resolve().parents[1]
T_DIST = 2 * HR  # before the planned ~3.5 h wrap: exercises wrap re-timing
T_AMB_BASE, T_AMB_COLD = 300.0, 285.0


def main():
    """Run the MPC-defends-the-plan demo and render the comparison figure.

    Loads the ASSESSED optimal cook (fullplan: 17 h, wrap ~3.5 h) as the
    plan under defense, then runs three cooks against the plant: a
    reference with no cold front, the plan executed open-loop through the
    t = 2 h ambient drop, and MPC re-planning schedule + wrap hourly from
    measured state. Prints the final doneness of each, writes
    data/control/mpc_result.json, and renders figures/mpc_demo.png.
    """
    wrap_mode = True  # the assessed plan wraps; the demo is wrap-aware

    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(meat_client(), pit_client())

    cold_front = lambda t: T_AMB_COLD if t >= T_DIST else T_AMB_BASE
    no_dist = lambda t: T_AMB_BASE
    cc = ControlConfig(opt_wrap=True)

    # THE plan under defense: the CONVERGED optimum when present (see
    # run_fullplan_converge.py), else the scan incumbent
    conv = REPO / "data" / "control" / "fullplan17_converged.json"
    if conv.exists():
        b = json.loads(conv.read_text())
        T_F = b["t_f_h"] * HR
        t_wrap_star = b["t_wrap_h"] * HR
        u0_plan = np.asarray(b["u_opt"])
        print("using CONVERGED incumbent (fullplan17_converged.json)")
    else:
        best = json.loads(
            (REPO / "data" / "control" / "fullplan_result_ext.json").read_text()
        )
        T_F = best["best_t_f_h"] * HR
        t_wrap_star = best["best_t_wrap_h"] * HR
        u0_plan = np.asarray(best["best_u"])
    grids0 = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=len(u0_plan))
    print(
        f"plan under defense: {T_F / HR:.0f} h cook, "
        f"wrap at {t_wrap_star / HR:.2f} h, "
        f"u in [{u0_plan.min():.2f}, {u0_plan.max():.2f}]"
    )

    print("reference (no disturbance):")
    ref = mpc.run_cook(
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
    )
    print("open-loop through the cold front:")
    ol = mpc.run_cook(
        theta,
        phi,
        cold_front,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
    )
    print("MPC, NO disturbance (discriminator: plan-refinement vs weather):")
    mp0 = mpc.run_cook(
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="mpc",
        model_t_amb=T_AMB_BASE,
        cc=cc,
        t_wrap_init_abs=t_wrap_star,
        u_incumbent=u0_plan,
        u_incumbent_grids=grids0,
    )
    print("MPC through the cold front:")
    mp = mpc.run_cook(
        theta,
        phi,
        cold_front,
        t_f=T_F,
        mode="mpc",
        model_t_amb=T_AMB_BASE,
        cc=cc,
        t_wrap_init_abs=t_wrap_star,
        u_incumbent=u0_plan,
        u_incumbent_grids=grids0,
    )
    if wrap_mode and mp.get("wrapped"):
        print(
            f"  crutch executed at {mp['t_wrap_abs'] / HR:.2f} h "
            f"(replanned {len(mp['wrap_history'])}x)"
        )

    for name, r in [
        ("reference", ref),
        ("open-loop", ol),
        ("MPC-nominal", mp0),
        ("MPC", mp),
    ]:
        jj = mpc.executed_objective(r, cc, T_F)
        r["J_executed"] = jj
        print(
            f"  {name}: J={jj['J']:.3f}  doneness {r['doneness']:.3f} (band "
            f"{cc.c_target}-{cc.c_overdone})  juiciness {r['juiciness']:.3f}  "
            f"L* {r['l_star_end']:.1f}  smoke {r['smoke_end']:.2f}"
        )
        print(f"    components: {jj}")

    out = REPO / "data" / "control"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mpc_result.json").write_text(
        json.dumps(
            {
                **{
                    f"{k}_{tag}": r[k]
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                    for k in ("doneness", "juiciness", "l_star_end", "smoke_end")
                },
                **{
                    f"J_{tag}": r["J_executed"]
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                },
                "t_disturbance_h": T_DIST / HR,
                "dT_amb": T_AMB_BASE - T_AMB_COLD,
                **(
                    {
                        "wrapped": bool(mp.get("wrapped")),
                        "t_wrap_abs_h": (
                            mp["t_wrap_abs"] / HR
                            if mp.get("t_wrap_abs") is not None
                            else None
                        ),
                        "wrap_history_h": [
                            [t / HR, w / HR] for t, w in mp.get("wrap_history", [])
                        ],
                    }
                    if wrap_mode
                    else {}
                ),
            },
            indent=2,
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for r, name, ls in [
        (ref, "reference (no cold front)", ":"),
        (ol, "open-loop", "--"),
        (mp0, "MPC (no front)", "-."),
        (mp, "MPC", "-"),
    ]:
        axes[0].plot(r["t"] / HR, r["tcore"] - 273.15, ls, label=name)
        axes[1].plot(r["t"] / HR, r["tpit"] - 273.15, ls, label=name)
        axes[2].step(r["t"] / HR, r["u"], ls, where="post", label=name)
    for ax in axes[:2]:
        ax.axvline(T_DIST / HR, color="steelblue", lw=1, ls="-.")
    if mp0.get("t_wrap_abs") is not None and mp0.get("wrapped"):
        axes[0].axvline(mp0["t_wrap_abs"] / HR, color="#52514e", lw=1.2, ls=":")
    if wrap_mode and mp.get("t_wrap_abs") is not None:
        axes[0].axvline(mp["t_wrap_abs"] / HR, color="#eb6834", lw=1.5, ls=":")
        axes[0].annotate(
            f"crutch executed\nat {mp['t_wrap_abs'] / HR:.1f} h",
            (mp["t_wrap_abs"] / HR + 0.2, 55),
            fontsize=8,
            color="#eb6834",
        )
    axes[0].annotate(
        "cold front\n(−15 K ambient)",
        (T_DIST / HR + 0.2, 30),
        fontsize=8,
        color="steelblue",
    )
    axes[0].set_title(
        f"done/juicy/L*:  ref {ref['doneness']:.2f}/{ref['juiciness']:.2f}/"
        f"{ref['l_star_end']:.0f} · OL {ol['doneness']:.2f}/"
        f"{ol['juiciness']:.2f}/{ol['l_star_end']:.0f} · MPC "
        f"{mp['doneness']:.2f}/{mp['juiciness']:.2f}/{mp['l_star_end']:.0f} "
        f"· J: {ref['J_executed']['J']:.2f}/{ol['J_executed']['J']:.2f}/"
        f"{mp['J_executed']['J']:.2f}",
        fontsize=10,
    )
    axes[0].set(xlabel="time [h]", ylabel="core [°C]")
    axes[1].set(xlabel="time [h]", ylabel="pit [°C]", title="pit temperature")
    axes[2].set(
        xlabel="time [h]",
        ylabel="damper fraction",
        ylim=(0, 1),
        title="executed schedules (MPC re-plans hourly)",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(
        "Receding-horizon MPC absorbs an unmodeled disturbance"
        + (" — and re-times the crutch" if wrap_mode else "")
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "mpc_demo.png", dpi=140)
    print("wrote figures/mpc_demo.png")


if __name__ == "__main__":
    main()
