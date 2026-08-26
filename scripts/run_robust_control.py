"""Phase C stretch: damper-schedule optimization over the NUTS posterior.

Optimizing in expectation over posterior draws — control under calibrated
uncertainty.

Run from app/:  uv run python -u ../scripts/run_robust_control.py [--iters N]

Optimizes E[J] over N_OPT decorrelated posterior draws (warm-started from the
MAP-optimal schedule), then evaluates BOTH schedules on N_EVAL other draws.
Figure: doneness distributions under each schedule + schedule overlay.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import (
    PHI_IGNITION_INERT,
    PHI_SITES,
    PHI_SITES_IGNITION,
    THETA_SITES,
)
from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig, optimize_schedule_expected
from brisket_app.coupled import HR, CookGrids, coupled_cook

REPO = Path(__file__).resolve().parents[1]
N_OPT = 8  # posterior draws in the optimization objective
N_EVAL = 30  # held-out draws for evaluating both schedules


def draws(samples, idx):
    """Assemble theta/phi parameter matrices for selected posterior draws.

    Parameters
    ----------
    samples : dict of str to numpy.ndarray
        Stacked NUTS samples keyed by site name.
    idx : array_like of int
        Indices of the draws to extract.

    Returns
    -------
    th : numpy.ndarray, shape (len(idx), 7)
        Meat parameter rows in THETA_SITES order.
    ph : numpy.ndarray, shape (len(idx), 7)
        Pit parameter rows: PHI_SITES followed by the ignition sites, with
        inert defaults substituted for ignition sites absent from the
        posterior (pre-ignition-state sample files).
    """
    th = np.stack([np.stack([samples[n][i] for n, _ in THETA_SITES]) for i in idx])

    def phi_row(i):
        base = [samples[n][i] for n, _ in PHI_SITES]
        ign = [
            samples[n][i] if n in samples else d
            for (n, _), d in zip(PHI_SITES_IGNITION, PHI_IGNITION_INERT, strict=True)
        ]
        return np.array(base + ign)

    ph = np.stack([phi_row(i) for i in idx])
    return th, ph


def main():
    """Optimize the robust schedule and compare it to the MAP-optimal one.

    Optimizes E[J] over N_OPT decorrelated posterior draws (warm-started
    from the MAP-optimal schedule), evaluates both schedules on N_EVAL
    held-out draws, prints the doneness statistics, and writes
    data/control/schedule_robust.json plus figures/control_robust.png:
    doneness histograms under parameter uncertainty for each schedule, and
    the two schedules overlaid.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--samples",
        default="nuts_samples_real.npz",
        help="posterior sample file(s) under data/calibration, "
        "comma-separated (chains are concatenated)",
    )
    ap.add_argument(
        "--plan",
        default="fullplan_result_ext.json",
        help="assessed cook json under data/control (u, t_f, t_wrap)",
    )
    ap.add_argument(
        "--figure",
        default="control_robust.png",
        help="output figure name under figures/",
    )
    ap.add_argument(
        "--out",
        default="schedule_robust.json",
        help="output json name under data/control/",
    )
    args = ap.parse_args()

    parts = [
        dict(np.load(REPO / "data" / "calibration" / f.strip()))
        for f in args.samples.split(",")
    ]
    samples = {
        k: np.concatenate([np.asarray(p[k]).reshape(-1) for p in parts])
        for k in parts[0]
    }
    n_total = len(samples[THETA_SITES[0][0]])
    opt_idx = np.linspace(0, n_total - 1, N_OPT, dtype=int)  # decorrelated
    eval_idx = np.linspace(0, n_total - 1, N_EVAL + 2, dtype=int)[1:-1]
    th_opt, ph_opt = draws(samples, opt_idx)
    th_ev, ph_ev = draws(samples, eval_idx)

    # the plan under uncertainty-hedging: the ASSESSED fullplan optimum
    # (17 h, wrap ~3.5 h), not the legacy 12 h MAP schedule
    best = json.loads((REPO / "data" / "control" / args.plan).read_text())
    print(f"plan: {args.plan}  samples: {args.samples}", flush=True)
    u_map = np.asarray(best["best_u"])
    t_wrap_map = best["best_t_wrap_h"] * HR
    t_f = best["best_t_f_h"] * HR

    meat_tx, pit_tx = meat_client(), pit_client()
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=len(u_map))
    # opt_wrap: the posterior-EXPECTED wrap moment is a decision variable
    # too — the interesting question is whether parameter uncertainty
    # hedges the crutch earlier or later than the MAP's choice
    cc = ControlConfig(n_iter=args.iters, opt_wrap=True)

    print(f"optimizing E[J] over {N_OPT} posterior draws, warm start from MAP schedule")
    frac = np.clip(t_wrap_map / t_f, 1e-3, 1 - 1e-3)
    u_rob, hist = optimize_schedule_expected(
        th_opt,
        ph_opt,
        meat_tx,
        pit_tx,
        grids=grids,
        cc=cc,
        u_init=u_map,
        z_w_init=float(np.log(frac / (1 - frac))),
        callback=lambda i, l, u: print(f"  iter {i:4d}  E[J]={l:9.4f}", flush=True),
    )
    t_wrap_rob = hist["t_wrap"]
    print(
        f"wrap: plan {t_wrap_map / HR:.2f} h -> "
        f"posterior-expected {t_wrap_rob / HR:.2f} h"
    )
    print(
        f"E[J] {hist['losses'][0]:.3f} -> {hist['losses'][-1]:.3f} "
        f"in {hist['wall_s']:.0f} s"
    )

    def metrics(u, tw, th, ph):
        out = coupled_cook(
            jnp.asarray(th),
            jnp.asarray(ph),
            jnp.asarray(u),
            meat_tx,
            pit_tx,
            grids,
            K=2,
            t_wrap=tw,
        )
        return float(out["c_field_final"][0]), float(out["juiciness"])

    print(f"evaluating both plans on {N_EVAL} held-out draws")
    m_map = [metrics(u_map, t_wrap_map, th_ev[i], ph_ev[i]) for i in range(N_EVAL)]
    m_rob = [metrics(u_rob, t_wrap_rob, th_ev[i], ph_ev[i]) for i in range(N_EVAL)]
    d_map = np.array([x[0] for x in m_map])
    d_rob = np.array([x[0] for x in m_rob])
    j_map = np.array([x[1] for x in m_map])
    j_rob = np.array([x[1] for x in m_rob])
    for name, d, j in [("plan-optimal", d_map, j_map), ("robust", d_rob, j_rob)]:
        print(
            f"  {name}: doneness mean={d.mean():.3f} sd={d.std():.3f} "
            f"min={d.min():.3f} max={d.max():.3f}  "
            f"juiciness mean={j.mean():.3f} sd={j.std():.3f}"
        )

    out_dir = REPO / "data" / "control"
    (out_dir / args.out).write_text(
        json.dumps(
            {
                "u_robust": u_rob.tolist(),
                "t_wrap_plan_h": t_wrap_map / HR,
                "t_wrap_robust_h": t_wrap_rob / HR,
                "doneness_map_eval": d_map.tolist(),
                "doneness_robust_eval": d_rob.tolist(),
                "juiciness_map_eval": j_map.tolist(),
                "juiciness_robust_eval": j_rob.tolist(),
                "n_opt": N_OPT,
                "n_eval": N_EVAL,
                "iters": args.iters,
                "samples": args.samples,
                "plan": args.plan,
            },
            indent=2,
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(
        min(d_map.min(), d_rob.min()) - 0.02, max(d_map.max(), d_rob.max()) + 0.02, 16
    )
    axes[0].hist(
        d_map, bins=bins, alpha=0.6, label=f"plan-optimal (sd={d_map.std():.3f})"
    )
    axes[0].hist(d_rob, bins=bins, alpha=0.6, label=f"robust (sd={d_rob.std():.3f})")
    axes[0].axvline(0.90, color="r", ls=":", lw=1, label="target")
    axes[0].set(
        xlabel=f"core doneness at {t_f / HR:.0f} h",
        ylabel="posterior draws",
        title=f"doneness under parameter uncertainty ({N_EVAL} draws)",
    )
    axes[0].legend(fontsize=8)
    axes[1].step(grids.u_tknots / HR, u_map, where="mid", label="plan-optimal")
    axes[1].step(
        grids.u_tknots / HR, u_rob, where="mid", label="robust (E over posterior)"
    )
    axes[1].set(
        xlabel="time [h]", ylabel="damper fraction", ylim=(0, 1), title="schedules"
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / args.figure, dpi=140)
    print(f"wrote figures/{args.figure}")


if __name__ == "__main__":
    main()
