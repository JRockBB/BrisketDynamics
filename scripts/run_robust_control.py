"""Phase C stretch: damper-schedule optimization in expectation over the
NUTS posterior — control under calibrated uncertainty.

Run from app/:  uv run python -u ../scripts/run_robust_control.py [--iters N]

Optimizes E[J] over N_OPT decorrelated posterior draws (warm-started from the
MAP-optimal schedule), then evaluates BOTH schedules on N_EVAL other draws.
Figure: doneness distributions under each schedule + schedule overlay.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp

from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig, optimize_schedule_expected
from brisket_app.coupled import HR, CookGrids, coupled_cook
from brisket_app.calibrate import THETA_SITES, PHI_SITES

REPO = Path(__file__).resolve().parents[1]
N_OPT = 8    # posterior draws in the optimization objective
N_EVAL = 30  # held-out draws for evaluating both schedules


def draws(samples, idx):
    th = np.stack([np.stack([samples[n][i] for n, _ in THETA_SITES]) for i in idx])
    ph = np.stack([np.stack([samples[n][i] for n, _ in PHI_SITES]) for i in idx])
    return th, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    samples = dict(np.load(REPO / "data" / "calibration" / "nuts_samples.npz"))
    n_total = len(samples[THETA_SITES[0][0]])
    opt_idx = np.linspace(0, n_total - 1, N_OPT, dtype=int)          # decorrelated
    eval_idx = np.linspace(0, n_total - 1, N_EVAL + 2, dtype=int)[1:-1]
    th_opt, ph_opt = draws(samples, opt_idx)
    th_ev, ph_ev = draws(samples, eval_idx)

    u_map = np.asarray(json.loads(
        (REPO / "data" / "control" / "schedule_opt.json").read_text())["u_opt"])

    meat_tx, pit_tx = meat_client(), pit_client()
    grids = CookGrids(dt_exchange=600.0, n_u=len(u_map))
    cc = ControlConfig(n_iter=args.iters)

    print(f"optimizing E[J] over {N_OPT} posterior draws, warm start from MAP schedule")
    u_rob, hist = optimize_schedule_expected(
        th_opt, ph_opt, meat_tx, pit_tx, grids=grids, cc=cc, u_init=u_map,
        callback=lambda i, l, u: print(f"  iter {i:4d}  E[J]={l:9.4f}", flush=True))
    print(f"E[J] {hist['losses'][0]:.3f} -> {hist['losses'][-1]:.3f} "
          f"in {hist['wall_s']:.0f} s")

    def doneness(u, th, ph):
        out = coupled_cook(jnp.asarray(th), jnp.asarray(ph), jnp.asarray(u),
                           meat_tx, pit_tx, grids, K=2)
        return float(out["c_field_final"][0])

    print(f"evaluating both schedules on {N_EVAL} held-out draws")
    d_map = np.array([doneness(u_map, th_ev[i], ph_ev[i]) for i in range(N_EVAL)])
    d_rob = np.array([doneness(u_rob, th_ev[i], ph_ev[i]) for i in range(N_EVAL)])
    for name, d in [("MAP-optimal", d_map), ("robust", d_rob)]:
        print(f"  {name}: doneness mean={d.mean():.3f} sd={d.std():.3f} "
              f"min={d.min():.3f} max={d.max():.3f}")

    out_dir = REPO / "data" / "control"
    (out_dir / "schedule_robust.json").write_text(json.dumps({
        "u_robust": u_rob.tolist(),
        "doneness_map_eval": d_map.tolist(),
        "doneness_robust_eval": d_rob.tolist(),
        "n_opt": N_OPT, "n_eval": N_EVAL, "iters": args.iters,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(min(d_map.min(), d_rob.min()) - 0.02,
                       max(d_map.max(), d_rob.max()) + 0.02, 16)
    axes[0].hist(d_map, bins=bins, alpha=0.6, label=f"MAP-optimal (sd={d_map.std():.3f})")
    axes[0].hist(d_rob, bins=bins, alpha=0.6, label=f"robust (sd={d_rob.std():.3f})")
    axes[0].axvline(0.90, color="r", ls=":", lw=1, label="target")
    axes[0].set(xlabel="core doneness at 12 h", ylabel="posterior draws",
                title=f"doneness under parameter uncertainty ({N_EVAL} draws)")
    axes[0].legend(fontsize=8)
    axes[1].step(grids.u_tknots / HR, u_map, where="mid", label="MAP-optimal")
    axes[1].step(grids.u_tknots / HR, u_rob, where="mid", label="robust (E over posterior)")
    axes[1].set(xlabel="time [h]", ylabel="damper fraction", ylim=(0, 1),
                title="schedules")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "control_robust.png", dpi=140)
    print(f"wrote figures/control_robust.png")


if __name__ == "__main__":
    main()
