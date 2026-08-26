"""Phase C: optimize the damper schedule at MAP parameters.

Run from app/:  uv run python ../scripts/run_control.py [--iters N]

Produces figures/control_before_after.png (objective curve + before/after
trajectories + recovered schedule) and data/control/schedule_opt.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig, optimize_schedule, safety_report
from brisket_app.coupled import HR, CookGrids, check_picard, coupled_cook

REPO = Path(__file__).resolve().parents[1]


def main():
    """Optimize the damper schedule at the MAP parameters and plot the result.

    Loads the MAP fit (data/calibration/map_result.json), runs a baseline
    flat-damper cook, then optimizes the schedule with Optax (Adam) against
    the doneness/bark-limit objective. Writes
    data/control/schedule_opt.json and control_losses.npy, plus
    figures/control_before_after.png: the objective curve, before/after
    core and surface trajectories with the bark limit, and the recovered
    damper schedule.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--u-init", type=float, default=0.5)
    args = ap.parse_args()

    map_file = REPO / "data" / "calibration" / "map_result.json"
    m = json.loads(map_file.read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    print(f"MAP params from {map_file.name}: theta={theta}, phi={phi}")

    meat_tx, pit_tx = meat_client(), pit_client()
    grids = CookGrids(dt_exchange=600.0, n_u=16)
    cc = ControlConfig(n_iter=args.iters)

    baseline = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.full(grids.n_u, args.u_init),
        meat_tx,
        pit_tx,
        grids,
        K=2,
    )
    c0 = float(baseline["c_field_final"][0])
    print(
        f"baseline (u={args.u_init} flat): core doneness {c0:.3f} "
        f"(target {cc.c_target})"
    )

    u_opt, hist = optimize_schedule(
        theta,
        phi,
        meat_tx,
        pit_tx,
        grids=grids,
        cc=cc,
        u_init=args.u_init,
        callback=lambda i, l, u: print(
            f"  iter {i:4d}  J={l:10.4f}  u=[{u.min():.2f}..{u.max():.2f}]"
        ),
    )
    after = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.asarray(u_opt),
        meat_tx,
        pit_tx,
        grids,
        K=2,
    )
    c1 = float(after["c_field_final"][0])
    print(
        f"optimized: core doneness {c1:.3f}, J {hist['losses'][0]:.2f} -> "
        f"{hist.get('best_loss', hist['losses'][-1]):.2f} in {hist['wall_s']:.0f} s"
    )
    check_picard(after)
    sr = safety_report(after["tsurf"], grids.tknots, cc)
    print(
        f"safety (surface, intact-muscle assumption): "
        f"F={sr['F_s']:.1f}s (ok={sr['lethality_ok']})  "
        f"dwell={sr['danger_h']:.2f}h (ok={sr['dwell_ok']})"
    )

    out_dir = REPO / "data" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schedule_opt.json").write_text(
        json.dumps(
            {
                "u_opt": u_opt.tolist(),
                "u_tknots_h": (grids.u_tknots / HR).tolist(),
                "doneness_before": c0,
                "doneness_after": c1,
                "J_first": float(hist["losses"][0]),
                "J_last": float(hist["losses"][-1]),
                "iters": args.iters,
                "map_source": "map_result.json",
            },
            indent=2,
        )
    )
    np.save(out_dir / "control_losses.npy", hist["losses"])

    # --- figure: objective + before/after core & surface + schedule ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = np.asarray(grids.tknots) / HR
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].semilogy(hist["losses"])
    axes[0].set(
        xlabel="iteration",
        ylabel="objective J",
        title="Optax (Adam) on damper schedule",
    )
    axes[1].plot(
        th,
        np.asarray(baseline["tcore"]) - 273.15,
        "--",
        label=f"core, u={args.u_init} flat",
    )
    axes[1].plot(th, np.asarray(after["tcore"]) - 273.15, label="core, optimized")
    axes[1].plot(
        th,
        np.asarray(baseline["tsurf"]) - 273.15,
        "--",
        alpha=0.5,
        label="surface, flat",
    )
    axes[1].plot(
        th, np.asarray(after["tsurf"]) - 273.15, alpha=0.7, label="surface, optimized"
    )
    axes[1].axhline(
        cc.t_surf_max - 273.15, color="r", lw=0.8, ls=":", label="bark limit"
    )
    axes[1].set(
        xlabel="time [h]",
        ylabel="temperature [C]",
        title=f"doneness {c0:.2f} -> {c1:.2f} (target {cc.c_target})",
    )
    axes[1].legend(fontsize=8)
    axes[2].step(grids.u_tknots / HR, u_opt, where="mid", label="optimized")
    axes[2].axhline(args.u_init, ls="--", color="gray", label="baseline")
    axes[2].set(
        xlabel="time [h]",
        ylabel="damper fraction",
        ylim=(0, 1),
        title="recovered schedule",
    )
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig_path = REPO / "figures" / "control_before_after.png"
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
