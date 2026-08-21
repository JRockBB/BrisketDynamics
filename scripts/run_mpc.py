"""Roadmap item 5: receding-horizon MPC demo.

The calibrated smoker (MAP-real parameters) is the plant. At t = 6 h a cold
front drops ambient by 15 K — the controller's model does not know. Compare:
  reference: initial plan, no disturbance
  open-loop: initial plan executed blindly through the disturbance
  MPC:       hourly re-plans from measured state (full-field handoff)

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
from brisket_app.control import ControlConfig, optimize_schedule
from brisket_app.coupled import HR, CookGrids

REPO = Path(__file__).resolve().parents[1]
T_F = 12 * HR
T_DIST = 6 * HR
T_AMB_BASE, T_AMB_COLD = 300.0, 285.0


def main():
    """Run the receding-horizon MPC demo and render the comparison figure.

    Plans a 12-h damper schedule at the MAP-real parameters (no disturbance
    knowledge), then runs three cooks against the plant: a reference with
    no cold front, the initial plan executed open-loop through the t = 6 h
    ambient drop, and MPC re-planning hourly from measured state. Prints
    the final doneness of each, writes data/control/mpc_result.json, and
    renders figures/mpc_demo.png: core temperature, pit temperature, and
    the executed damper schedules for the three cases.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(meat_client(), pit_client())

    cold_front = lambda t: T_AMB_COLD if t >= T_DIST else T_AMB_BASE
    no_dist = lambda t: T_AMB_BASE
    cc = ControlConfig()

    print("initial 12-h plan (controller model, no disturbance knowledge)")
    grids0 = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=16)
    u0_plan, _ = optimize_schedule(
        theta,
        phi,
        *mpc._CLIENTS,
        grids=grids0,
        cc=ControlConfig(n_iter=60),
        u_init=0.5,
        ics={"tpit0": 293.15, "t_amb": T_AMB_BASE, "t0_meat": 281.15},
    )
    print(f"  plan u in [{u0_plan.min():.2f}, {u0_plan.max():.2f}]")

    print("reference (no disturbance):")
    ref = mpc.run_cook(
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
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
    )
    print("MPC through the cold front:")
    mp = mpc.run_cook(
        theta, phi, cold_front, t_f=T_F, mode="mpc", model_t_amb=T_AMB_BASE
    )

    print(
        f"doneness: reference {ref['doneness']:.3f}  "
        f"open-loop {ol['doneness']:.3f}  MPC {mp['doneness']:.3f} "
        f"(target {cc.c_target})"
    )

    out = REPO / "data" / "control"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mpc_result.json").write_text(
        json.dumps(
            {
                "doneness_reference": ref["doneness"],
                "doneness_openloop": ol["doneness"],
                "doneness_mpc": mp["doneness"],
                "t_disturbance_h": T_DIST / HR,
                "dT_amb": T_AMB_BASE - T_AMB_COLD,
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
        (mp, "MPC", "-"),
    ]:
        axes[0].plot(r["t"] / HR, r["tcore"] - 273.15, ls, label=name)
        axes[1].plot(r["t"] / HR, r["tpit"] - 273.15, ls, label=name)
        axes[2].step(r["t"] / HR, r["u"], ls, where="post", label=name)
    for ax in axes[:2]:
        ax.axvline(T_DIST / HR, color="steelblue", lw=1, ls="-.")
    axes[0].annotate(
        "cold front\n(−15 K ambient)",
        (T_DIST / HR + 0.2, 30),
        fontsize=8,
        color="steelblue",
    )
    axes[0].set(
        xlabel="time [h]",
        ylabel="core [°C]",
        title=f"doneness: ref {ref['doneness']:.2f} · "
        f"open-loop {ol['doneness']:.2f} · MPC {mp['doneness']:.2f} "
        f"(target {cc.c_target})",
    )
    axes[1].set(xlabel="time [h]", ylabel="pit [°C]", title="pit temperature")
    axes[2].set(
        xlabel="time [h]",
        ylabel="damper fraction",
        ylim=(0, 1),
        title="executed schedules (MPC re-plans hourly)",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Receding-horizon MPC absorbs an unmodeled disturbance")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "mpc_demo.png", dpi=140)
    print("wrote figures/mpc_demo.png")


if __name__ == "__main__":
    main()
