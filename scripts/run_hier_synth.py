"""Two-stage hierarchical pooling on synthetic two-cook data.

Same pit, different meat. Per-cook Gauss-Newton Laplace (minutes), then a
Gaussian pool: pit shared, meat partial, ignition iid. No joint NUTS.

Run from app/:  uv run python -u ../scripts/run_hier_synth.py
Writes data/calibration/hier_synth.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import synth
from brisket_app.approx import MEAT_PARTIAL, PIT_SHARED, hierarchical_pool
from brisket_app.calibrate import PHI_SITES, THETA_SITES
from brisket_app.coupled import HR
from brisket_app.monolith import (
    NAMES,
    SITES,
    Whitened,
    from_z,
    levenberg_marquardt,
    to_z,
)

REPO = Path(__file__).resolve().parents[1]


def _prior_median_z():
    x = []
    for _, d in SITES:
        if hasattr(d, "concentration1"):
            a, b = float(d.concentration1), float(d.concentration0)
            x.append(a / (a + b))
        else:
            x.append(float(np.exp(d.loc)))
    return to_z(np.array(x))


def _gn_laplace(log, sigma_pit, sigma_core, z0, iters: int):
    """LM + Fisher covariance in unconstrained physics space (rho=0, iid)."""
    noise = {
        "sigma_pit": sigma_pit,
        "sigma_core": sigma_core,
        "rho_pit": 0.0,
        "rho_core": 0.0,
    }
    # effective_sigma adds the digitization floor; synthetic is iid at the
    # observation sigma, so pass the extras as (sigma - floor) if positive.
    from brisket_app.calibrate import SIGMA_FLOOR_K
    from brisket_app.monolith import effective_sigma

    noise["sigma_pit"] = max(sigma_pit - SIGMA_FLOOR_K, 0.05)
    noise["sigma_core"] = max(sigma_core - SIGMA_FLOOR_K, 0.05)
    wh = Whitened.build(log, *effective_sigma(noise))
    z, F, H, _hist = levenberg_marquardt(z0, wh, log, iters=iters)
    H = 0.5 * (H + H.T)
    lam = np.linalg.eigvalsh(H)
    if lam.min() <= 0:
        H = H + (1e-4 * max(lam.max(), 1.0) - lam.min()) * np.eye(z.size)
    return z, np.linalg.inv(H), F


def _embed_physics(z_phys, cov_phys):
    """Pad physics-only Laplace into ALL_NAMES_SORTED with tight noise."""
    from brisket_app.monolith import ALL_NAMES_SORTED

    d = len(ALL_NAMES_SORTED)
    mean = np.zeros(d)
    cov = np.eye(d) * 0.05
    for i, n in enumerate(NAMES):
        j = ALL_NAMES_SORTED.index(n)
        mean[j] = z_phys[i]
    for i, ni in enumerate(NAMES):
        ji = ALL_NAMES_SORTED.index(ni)
        for k, nk in enumerate(NAMES):
            jk = ALL_NAMES_SORTED.index(nk)
            cov[ji, jk] = cov_phys[i, k]
    return mean, cov


def main():
    """Generate two cooks, Laplace-fit each, pool, print recovery."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument(
        "--tau2",
        default="0.04",
        help="between-cook meat variance in z, or 'auto' (method-of-moments)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from brisket_app.clients import monolith_client
    from brisket_app.coupled import CookGrids

    tx = monolith_client()
    grids = CookGrids(
        t_end=args.hours * HR, dt_exchange=600.0, n_u=len(synth.U_COOK)
    )
    theta_a = synth.THETA_TRUE.copy()
    theta_b = synth.THETA_TRUE.copy()
    # different meat: L and D_eff; same pit
    theta_b[0] *= 1.35  # D_eff
    theta_b[3] *= 1.25  # L
    phi = synth.PHI_TRUE.copy()
    print("generating two synthetic cooks (same pit, different meat)", flush=True)
    dA = synth.generate(tx, grids=grids, seed=args.seed, theta=theta_a, phi=phi)
    dB = synth.generate(tx, grids=grids, seed=args.seed + 1, theta=theta_b, phi=phi)
    logA, logB = synth.as_cooklog(dA), synth.as_cooklog(dB)

    z0 = _prior_median_z()
    print("Gauss-Newton Laplace cook A", flush=True)
    zA, cA, fA = _gn_laplace(
        logA, synth.OBS_SIGMA_PIT, synth.OBS_SIGMA_CORE, z0, args.iters
    )
    print("Gauss-Newton Laplace cook B", flush=True)
    zB, cB, fB = _gn_laplace(
        logB, synth.OBS_SIGMA_PIT, synth.OBS_SIGMA_CORE, z0, args.iters
    )
    from brisket_app.monolith import ALL_NAMES_SORTED

    mA, kA = _embed_physics(zA, cA)
    mB, kB = _embed_physics(zB, cB)
    pool = hierarchical_pool(
        [mA, mB],
        [kA, kB],
        names=list(ALL_NAMES_SORTED),
        tau2="auto" if args.tau2 == "auto" else float(args.tau2),
    )

    xA, xB = from_z(zA), from_z(zB)
    true_pit = {n: float(phi[i]) for i, n in enumerate([n for n, _ in PHI_SITES])}
    # ignition is inert on synthetic
    true_meat_a = {n: float(theta_a[i]) for i, n in enumerate([n for n, _ in THETA_SITES])}
    true_meat_b = {n: float(theta_b[i]) for i, n in enumerate([n for n, _ in THETA_SITES])}

    pit_names = list(PIT_SHARED)
    pooled_pit_z = pool.shared_mean
    pooled_pit_x = {}
    for i, n in enumerate(pit_names):
        # invert the single-site transform
        j = NAMES.index(n)
        z1 = np.zeros(len(NAMES))
        z1[j] = pooled_pit_z[i]
        # can't invert one coordinate in isolation when using from_z on a vector
        # with zeros — log sites would be exp(0)=1. Invert directly:
        from brisket_app.monolith import _is_beta

        _, distn = SITES[j]
        v = pooled_pit_z[i]
        pooled_pit_x[n] = (
            1 / (1 + np.exp(-v)) if _is_beta(distn) else float(np.exp(v))
        )

    def err(est, truth):
        return abs(est - truth) / abs(truth)

    print(f"{'pit':10s} {'true':>12s} {'A':>12s} {'B':>12s} {'pooled':>12s}")
    pit_rows = {}
    for n in pit_names:
        i = NAMES.index(n)
        pit_rows[n] = {
            "true": true_pit[n],
            "A": float(xA[i]),
            "B": float(xB[i]),
            "pooled": pooled_pit_x[n],
            "err_A": err(xA[i], true_pit[n]),
            "err_B": err(xB[i], true_pit[n]),
            "err_pooled": err(pooled_pit_x[n], true_pit[n]),
        }
        print(
            f"{n:10s} {true_pit[n]:12.4g} {xA[i]:12.4g} {xB[i]:12.4g} "
            f"{pooled_pit_x[n]:12.4g}"
        )
    mean_err_A = float(np.mean([pit_rows[n]["err_A"] for n in pit_names]))
    mean_err_B = float(np.mean([pit_rows[n]["err_B"] for n in pit_names]))
    mean_err_P = float(np.mean([pit_rows[n]["err_pooled"] for n in pit_names]))
    pit_beats_worse = mean_err_P < max(mean_err_A, mean_err_B)

    meat_names = ("D_eff", "L")
    print(f"{'meat':10s} {'true A':>12s} {'true B':>12s} {'A':>12s} {'B':>12s}")
    meat_rows = {}
    for n in meat_names:
        i = NAMES.index(n)
        meat_rows[n] = {
            "true_A": float(true_meat_a[n]),
            "true_B": float(true_meat_b[n]),
            "A": float(xA[i]),
            "B": float(xB[i]),
        }
        print(
            f"{n:10s} {true_meat_a[n]:12.4g} {true_meat_b[n]:12.4g} "
            f"{xA[i]:12.4g} {xB[i]:12.4g}"
        )
    # cook-level partial-pool means (z); plan: meat not collapsed to one value
    iD, iL = MEAT_PARTIAL.index("D_eff"), MEAT_PARTIAL.index("L")
    zD = [float(pool.cook_partial_mean[i][iD]) for i in (0, 1)]
    zL = [float(pool.cook_partial_mean[i][iL]) for i in (0, 1)]
    meat_not_collapsed = abs(zD[0] - zD[1]) > 0.02 or abs(zL[0] - zL[1]) > 0.02
    print(
        f"mean relative pit error: A {mean_err_A:.3f}  B {mean_err_B:.3f}  "
        f"pooled {mean_err_P:.3f}"
    )
    print(f"pooled pit beats worse cook: {pit_beats_worse}")
    print(
        f"meat not collapsed (partial-pool z D_eff/L): "
        f"{zD[0]:.3f}/{zL[0]:.3f} vs {zD[1]:.3f}/{zL[1]:.3f}  "
        f"ok={meat_not_collapsed}"
    )

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report = {
        "hours": args.hours,
        "F": [float(fA), float(fB)],
        "pit": pit_rows,
        "meat": meat_rows,
        "mean_rel_err": {"A": mean_err_A, "B": mean_err_B, "pooled": mean_err_P},
        "pit_beats_worse_cook": pit_beats_worse,
        "meat_not_collapsed": meat_not_collapsed,
        "cook_partial_z": {"D_eff": zD, "L": zL},
        "tau2": args.tau2,
        "theta_A": theta_a.tolist(),
        "theta_B": theta_b.tolist(),
        "phi": phi.tolist(),
    }
    out = REPO / "data" / "calibration" / "hier_synth.json"
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
