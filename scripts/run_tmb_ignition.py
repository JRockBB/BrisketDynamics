"""TMB nested Laplace on (a0, k_spread): local Fisher Newton + log det.

At each ignition grid cell the rest starts at the Laplace conditional,
then one Gauss-Newton step uses the *local* J^T Sigma^{-1} J + prior
(not the MAP Hessian -- that is what collapsed --refine-rest). The
marginal is -U(theta, u-hat) - 1/2 log det H_rest, the TMB / INLA
formula. Writes its own npz so the profile nested grid is not overwritten.

Run from app/:  uv run python -u ../scripts/run_tmb_ignition.py
Writes data/calibration/tmb_ignition_grid_real.npz,
tmb_ignition_samples_real.npz, tmb_ignition_real.json,
figures/calibration/tmb_ignition_marginal.png.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import calibrate, cooklog
from brisket_app.approx import (
    IGNITION_NAMES,
    ignition_moments_from_grid,
    load_laplace_hess,
    nested_tmb_grid,
    sample_nested_laplace,
    sd_z_physics,
)
from brisket_app.monolith import (
    ALL_NAMES_SORTED,
    NAMES,
    Whitened,
    effective_sigma,
    gn_rest_newton,
)

REPO = Path(__file__).resolve().parents[1]


def main():
    """TMB nested ignition grid on the real cook."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=11)
    ap.add_argument(
        "--n-sd",
        type=float,
        nargs="+",
        default=[4.0, 7.0],
        help="Laplace-sd half-widths for (a0, k_spread).",
    )
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-json", default="map_real.json")
    ap.add_argument(
        "--n-steps",
        type=int,
        default=8,
        help="cap on GN Newton steps on rest (converges early at --tol)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.02,
        help="inner-Newton |dz|_inf convergence tolerance (0 = fixed n-steps; "
        "the TMB marginal formula assumes the inner optimum was reached)",
    )
    ap.add_argument("--clip", type=float, default=0.5)
    args = ap.parse_args()
    n_sd = args.n_sd[0] if len(args.n_sd) == 1 else args.n_sd
    cal = REPO / "data" / "calibration"
    z0, _h, cov = load_laplace_hess(cal / "laplace_hess_real.npz")
    init = {
        k: float(v)
        for k, v in json.loads((cal / args.init_json).read_text()).items()
        if not k.startswith("_")
    }
    from brisket_app.clients import monolith_client

    log = cooklog.load(dt_min=10.0)
    wh = Whitened.build(log, *effective_sigma(init))
    model = calibrate.bound_model_real(log, monolith_client())
    _zt, names, _unr, inv, pot = calibrate.unconstrained_potential(
        model, init, seed=args.seed
    )
    assert names == ALL_NAMES_SORTED
    print(
        f"TMB nested ignition: {args.n_grid}x{args.n_grid} n_sd={n_sd} "
        f"n_steps<={args.n_steps} tol={args.tol}",
        flush=True,
    )

    n_calls = 0
    logdets = []
    steps = []

    def cell_fn(z):
        nonlocal n_calls
        z_hat, logdet, step, _hrr = gn_rest_newton(
            z, log, wh, clip=args.clip, n_steps=args.n_steps, tol=args.tol
        )
        n_calls += 1
        logdets.append(logdet)
        steps.append(step)
        u = float(pot(z_hat))
        if n_calls == 1 or n_calls % 5 == 0:
            print(
                f"  cell {n_calls}  U={u:.1f}  logdet={logdet:.2f}  "
                f"|dz|={step:.3f}",
                flush=True,
            )
        return u, logdet

    t0 = time.perf_counter()
    nested = nested_tmb_grid(
        z0, cov, cell_fn, n_grid=args.n_grid, n_sd=n_sd, names=names
    )
    wall = time.perf_counter() - t0
    print(f"grid {args.n_grid}x{args.n_grid} in {wall:.0f} s", flush=True)

    rng = np.random.default_rng(args.seed)
    z_draws = sample_nested_laplace(nested, args.draws, rng)
    stacked = calibrate.constrain_draws(z_draws, names, inv)
    prefix = "tmb_ignition"
    np.savez(cal / f"{prefix}_samples_real.npz", **stacked)
    logdet_grid = np.asarray(logdets, dtype=float).reshape(args.n_grid, args.n_grid)
    step_grid = np.asarray(steps, dtype=float).reshape(args.n_grid, args.n_grid)
    np.savez(
        cal / f"{prefix}_grid_real.npz",
        z0=nested.z0,
        z1=nested.z1,
        log_marg=nested.log_marg,
        logdet=logdet_grid,
        step=step_grid,
        names=np.array(nested.names),
    )

    ign_mean, ign_cov = ignition_moments_from_grid(
        nested.z0, nested.z1, nested.log_marg
    )
    sd_g = np.sqrt(np.diag(ign_cov))
    corr = float(ign_cov[0, 1] / (sd_g[0] * sd_g[1]))
    print(
        f"TMB grid a0={ign_mean[0]:+.3f} k_spread={ign_mean[1]:+.3f}  "
        f"sd {sd_g[0]:.3f}/{sd_g[1]:.3f}  corr={corr:+.3f}",
        flush=True,
    )
    print(
        f"  logdet range {logdet_grid.min():.2f} .. {logdet_grid.max():.2f}  "
        f"mean |dz|={step_grid.mean():.3f}",
        flush=True,
    )

    lap_sd = {
        n: float(np.sqrt(cov[names.index(n), names.index(n)])) for n in IGNITION_NAMES
    }
    tmb_sd = {n: float(sd_g[i]) for i, n in enumerate(IGNITION_NAMES)}
    nuts_sd = None
    nuts_path = cal / "nuts_mono_samples_real.npz"
    if nuts_path.exists():
        nuts_sd = sd_z_physics(dict(np.load(nuts_path)))
    cols = {"Laplace-GN": lap_sd, "TMB-ign": tmb_sd}
    if nuts_sd:
        cols["NUTS-mono"] = {n: nuts_sd[n] for n in IGNITION_NAMES}
    print(f"{'site':10s}" + "".join(f"{c:>12s}" for c in cols))
    for n in IGNITION_NAMES:
        print(f"{n:10s}" + "".join(f"{cols[c][n]:12.3f}" for c in cols))

    report = {
        "method": "tmb_nested",
        "n_grid": args.n_grid,
        "n_sd": list(np.atleast_1d(n_sd).astype(float)),
        "n_steps": args.n_steps,
        "tol": args.tol,
        "wall_s": wall,
        "grid_mean": ign_mean.tolist(),
        "grid_cov": ign_cov.tolist(),
        "grid_corr": corr,
        "sd_z_tmb": tmb_sd,
        "sd_z_laplace": lap_sd,
        "sd_z_nuts": {n: nuts_sd[n] for n in IGNITION_NAMES} if nuts_sd else None,
        "logdet_min": float(logdet_grid.min()),
        "logdet_max": float(logdet_grid.max()),
        "step_mean": float(step_grid.mean()),
        "step_max": float(step_grid.max()),
    }
    (cal / f"{prefix}_real.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {prefix}_samples_real.npz")
    _figure(
        nested,
        nuts_path if nuts_path.exists() else None,
        REPO,
        fig_name=f"{prefix}_marginal.png",
    )


def _figure(nested, nuts_path, repo, fig_name="tmb_ignition_marginal.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    log_m = nested.log_marg - np.max(nested.log_marg)
    cs = ax.contour(
        nested.z1, nested.z0, log_m, levels=np.linspace(-8, 0, 9), cmap="Blues"
    )
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    ign = nested.ign
    mu = nested.mean[ign]
    s = nested.cov[np.ix_(ign, ign)]
    theta = np.linspace(0, 2 * np.pi, 80)
    circ = np.column_stack([np.cos(theta), np.sin(theta)])
    ell = mu + circ @ np.linalg.cholesky(0.5 * (s + s.T) + 1e-12 * np.eye(2)).T
    ax.plot(ell[:, 1], ell[:, 0], color="tab:orange", lw=1.2, label="Laplace 1s")
    gmu, gs = ignition_moments_from_grid(nested.z0, nested.z1, nested.log_marg)
    gell = gmu + circ @ np.linalg.cholesky(0.5 * (gs + gs.T) + 1e-12 * np.eye(2)).T
    ax.plot(gell[:, 1], gell[:, 0], color="tab:green", lw=1.4, label="TMB 1s")
    if nuts_path is not None:
        nuts = dict(np.load(nuts_path))
        from brisket_app.monolith import to_z

        z_n = np.stack(
            [
                to_z([float(nuts[n][i]) for n in NAMES])
                for i in range(min(400, len(nuts[NAMES[0]])))
            ]
        )
        ax.plot(
            z_n[:, NAMES.index("k_spread")],
            z_n[:, NAMES.index("a0")],
            ".",
            ms=2,
            alpha=0.25,
            color="tab:blue",
            label="NUTS",
        )
    ax.set(
        xlabel="k_spread (z)",
        ylabel="a0 (z)",
        title="TMB nested Laplace ignition marginal",
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(repo / "figures" / "calibration" / fig_name, dpi=140)
    plt.close(fig)
    print(f"wrote figures/calibration/{fig_name}")


if __name__ == "__main__":
    main()
