"""PSIS-reweight Laplace draws with the true ODE potential.

Draws from the Gauss-Newton Laplace, scores ``-U_true(z) - log q_Laplace(z)``,
Pareto-smooths, resamples, and writes constrained draws like the other
posteriors. Pareto-k ≲ 0.7 means the reweighted sample is usable; a large k
concentrates on ignition.

Run from app/:  uv run python -u ../scripts/run_psis_laplace.py
Writes data/calibration/psis_laplace_samples_real.npz and psis_laplace_real.json.
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

from brisket_app import approx, calibrate, cooklog
from brisket_app.approx import (
    IGNITION_NAMES,
    khat_threshold,
    load_laplace_hess,
    moment_match_psis,
    mvn_logpdf,
    psis_smooth,
)
from brisket_app.clients import monolith_client
from brisket_app.monolith import ALL_NAMES_SORTED, NAMES

REPO = Path(__file__).resolve().parents[1]


def _eval_pot(pot, Z, every: int = 25):
    """Sequential potential evaluations (Tesseract is not vmappable)."""
    u = np.empty(len(Z))
    t0 = time.perf_counter()
    for i, z in enumerate(Z):
        u[i] = float(pot(z))
        if (i + 1) % every == 0 or i + 1 == len(Z):
            print(
                f"  potential {i + 1}/{len(Z)}  "
                f"({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    return u


def main():
    """PSIS-reweight the Laplace proposal against the true (or linear) potential."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--linear",
        action="store_true",
        help="score the Gauss-Newton tangent model instead of the ODE (smoke test)",
    )
    ap.add_argument("--init-json", default="map_real.json")
    ap.add_argument(
        "--proposal",
        choices=("laplace", "tmb"),
        default="laplace",
        help="laplace: GN Gaussian; tmb: nested TMB grid x Laplace rest",
    )
    ap.add_argument(
        "--moment-match",
        action="store_true",
        help="when khat is above its threshold, adapt the proposal draws by "
        "importance-weighted moment matching (Paananen et al. 2021; each "
        "accepted/tried transform costs one ODE solve per draw)",
    )
    ap.add_argument(
        "--mm-evals",
        type=int,
        default=4,
        help="cap on moment-matching transform attempts",
    )
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    z0, _H, cov = load_laplace_hess(cal / "laplace_hess_real.npz")
    assert z0.size == len(ALL_NAMES_SORTED)

    init = {
        k: float(v)
        for k, v in json.loads((cal / args.init_json).read_text()).items()
        if not k.startswith("_")
    }
    log = cooklog.load(dt_min=10.0)
    rng = np.random.default_rng(args.seed)
    from brisket_app.approx import (
        NestedLaplace,
        ignition_indices,
        nested_log_q,
        sample_mvn,
        sample_nested_laplace,
    )

    nested = None
    if args.proposal == "tmb":
        gpath = cal / "tmb_ignition_grid_real.npz"
        if not gpath.exists():
            raise SystemExit("tmb proposal needs data/calibration/tmb_ignition_grid_real.npz")
        g = np.load(gpath)
        nested = NestedLaplace(
            names=IGNITION_NAMES,
            z0=np.asarray(g["z0"], dtype=float),
            z1=np.asarray(g["z1"], dtype=float),
            log_marg=np.asarray(g["log_marg"], dtype=float),
            mean=z0,
            cov=cov,
            ign=ignition_indices(),
        )
        Z = sample_nested_laplace(nested, args.draws, rng)
        print(f"TMB-nested proposal n={args.draws}", flush=True)
    else:
        Z = sample_mvn(z0, cov, args.draws, rng)

    if args.linear:
        from brisket_app.calibrate import phi_full
        from brisket_app.monolith import N_THETA, model_mean

        x0 = np.concatenate([[init[k] for k in NAMES[:N_THETA]], phi_full(init)])
        mu0, J = model_mean(x0, log)
        model = calibrate.bound_model_real_linear(log, mu0, J, x0)
        print(f"linear surrogate J {J.shape}", flush=True)
    else:
        model = calibrate.bound_model_real(log, monolith_client())
        print("true ODE potential (one solve per draw)", flush=True)

    _z_trace, names, _unravel, inv, pot = calibrate.unconstrained_potential(
        model, init, seed=args.seed
    )
    assert names == ALL_NAMES_SORTED
    U = _eval_pot(pot, Z)
    if args.proposal == "tmb":
        log_q = nested_log_q(Z, nested)
        tag = "tmb"
    else:
        log_q = mvn_logpdf(Z, z0, cov)
        tag = "laplace"
    lw = -U - log_q
    psis = psis_smooth(lw)
    thr = khat_threshold(psis.n)
    print(
        f"PSIS khat={psis.khat:.3f} (threshold {thr:.2f} at n={psis.n})  "
        f"ESS={psis.ess:.1f}/{psis.n}  "
        f"({'usable' if psis.khat < thr else 'proposal tail too thin'})"
    )
    mm = None
    Z_final, U_final, lw_s = Z, U, psis.log_weights
    khat, ess = psis.khat, psis.ess
    if args.moment_match and (not np.isfinite(psis.khat) or psis.khat >= thr):
        print("moment-matching the proposal draws (implicitly adaptive IS)")
        mm = moment_match_psis(
            Z, U, log_q, pot, max_evals=args.mm_evals, verbose=print
        )
        Z_final, U_final, lw_s = mm.Z, mm.u, mm.log_weights
        khat, ess = mm.khat, mm.ess
        print(
            f"moment-match: transforms={list(mm.transforms)}  "
            f"khat={mm.khat:.3f}  ESS={mm.ess:.1f}/{psis.n}  "
            f"({mm.n_evals} x {psis.n} extra solves)"
        )
    idx = approx.resample_weights(lw_s, n=args.draws, rng=rng)
    stacked = calibrate.constrain_draws(Z_final[idx], names, inv)
    out = cal / f"psis_{tag}_samples_real.npz"
    np.savez(out, **stacked)
    # raw proposal draws + weights so khat can be re-scored without ODE runs
    np.savez(
        cal / f"psis_{tag}_raw_real.npz",
        Z=Z,
        u=U,
        log_q=log_q,
        Z_final=Z_final,
        u_final=U_final,
        log_weights=lw_s,
        khat=khat,
    )
    sd = approx.sd_z_physics(stacked)
    nuts_path = cal / "nuts_mono_samples_real.npz"
    report = {
        "proposal": args.proposal,
        "khat": khat,
        "ess": ess,
        "khat_proposal": psis.khat,
        "ess_proposal": psis.ess,
        "khat_threshold": thr,
        "moment_match": (
            {"transforms": list(mm.transforms), "n_evals": mm.n_evals}
            if mm is not None
            else None
        ),
        "n": psis.n,
        "linear": args.linear,
        "sd_z": sd,
        "ignition_sd_z": {n: sd[n] for n in IGNITION_NAMES if n in sd},
    }
    if nuts_path.exists():
        nuts_sd = approx.sd_z_physics(dict(np.load(nuts_path)))
        report["nuts_sd_z_ignition"] = {n: nuts_sd[n] for n in IGNITION_NAMES}
        print("sd(z) ignition:  PSIS vs NUTS vs proposal")
        prop_sd = approx.sd_z_from_draws(Z, names)
        nuts = dict(np.load(nuts_path))
        print("constrained p5/p95")
        for n in IGNITION_NAMES:
            print(
                f"  {n:10s}  PSIS {sd[n]:.3f}  NUTS {nuts_sd[n]:.3f}  "
                f"proposal {prop_sd[n]:.3f}"
            )
            pq = np.quantile(stacked[n], [0.05, 0.95])
            nq = np.quantile(nuts[n], [0.05, 0.95])
            print(
                f"             p5/p95 PSIS [{pq[0]:.3g}, {pq[1]:.3g}]  "
                f"NUTS [{nq[0]:.3g}, {nq[1]:.3g}]"
            )
    (cal / f"psis_{tag}_real.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out.name} and psis_{tag}_real.json")


if __name__ == "__main__":
    main()
