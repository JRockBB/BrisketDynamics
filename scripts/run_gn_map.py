"""Gauss–Newton MAP + Fisher covariance on the real cook from the monolith.

Levenberg–Marquardt over the 14 physics latents in the unconstrained space,
noise parameters (σ, ρ) fixed at the adjoint MAP's values, from a cold start
at the prior medians; then the Fisher information (Jᵀ Σ⁻¹ J + prior
curvature) at the optimum, compared with the finite-difference Laplace
Hessian of the composed pipeline (data/calibration/laplace_hess_real.npz)
on the same 14 coordinates.

Run from app/:  uv run python -u ../scripts/run_gn_map.py [--warm]
Writes data/calibration/gn_map_real.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import cooklog
from brisket_app.calibrate import phi_full
from brisket_app.monolith import (
    N_PHYS,
    NAMES,
    SITES,
    Whitened,
    effective_sigma,
    from_z,
    levenberg_marquardt,
    objective,
    to_z,
)

REPO = Path(__file__).resolve().parents[1]


def prior_median_z():
    """Cold start: prior medians (LogNormal) / means (Beta), in z."""
    x = []
    for _, d in SITES:
        if hasattr(d, "concentration1"):
            a, b = float(d.concentration1), float(d.concentration0)
            x.append(a / (a + b))
        else:
            x.append(float(np.exp(d.loc)))
    return to_z(np.array(x))


def main():
    """Run LM from the cold start, then compare Fisher and FD-Laplace widths."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map",
        default="map_real.json",
        help="adjoint MAP json (noise params + reference)",
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="start from the adjoint MAP instead of the priors",
    )
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    m = json.loads((cal / args.map).read_text())
    x_map = np.concatenate([[m[n] for n, _ in SITES[:7]], phi_full(m)])
    log = cooklog.load(dt_min=10.0)
    wh = Whitened.build(log, *effective_sigma(m))
    print(
        f"cook log: {len(log['t'])} points, {wh.L.shape[0]} whitened residuals",
        flush=True,
    )

    t0 = time.perf_counter()
    z_ref = to_z(x_map)
    F_ref, g_ref, _H_ref, w_ref = objective(z_ref, wh, log)
    print(
        f"adjoint MAP as reference: F={F_ref:.3f}  0.5|w|^2={0.5 * w_ref @ w_ref:.3f}  "
        f"|g|={np.linalg.norm(g_ref):.3g}  (first Jacobian {time.perf_counter() - t0:.0f} s incl. compile)",
        flush=True,
    )

    z0 = z_ref if args.warm else prior_median_z()
    print(
        "LM from",
        "the adjoint MAP" if args.warm else "the prior medians (cold start)",
        flush=True,
    )
    t0 = time.perf_counter()
    z, F, H, hist = levenberg_marquardt(z0, wh, log, iters=args.iters)
    wall = time.perf_counter() - t0
    x = from_z(z)
    print(
        f"LM done: {len(hist) - 1} iterations, {wall:.0f} s, F={F:.3f} (reference {F_ref:.3f})",
        flush=True,
    )

    cov = np.linalg.inv(H)
    sd = np.sqrt(np.diag(cov))
    print(
        f"{'site':10s} {'GN-MAP':>11s} {'adjoint MAP':>12s} {'ratio':>7s} {'sd(z) Fisher':>13s}",
        flush=True,
    )
    for i, n in enumerate(NAMES):
        print(
            f"{n:10s} {x[i]:11.4g} {x_map[i]:12.4g} {x[i] / x_map[i]:7.3f} {sd[i]:13.3f}"
        )

    # the FD Laplace Hessian of the composed pipeline, same coordinates
    out = {
        "x": dict(zip(NAMES, map(float, x), strict=True)),
        "z": dict(zip(NAMES, map(float, z), strict=True)),
        "F": F,
        "F_reference": F_ref,
        "iterations": len(hist) - 1,
        "wall_s": wall,
        "history": hist,
        "sd_z_fisher": dict(zip(NAMES, map(float, sd), strict=True)),
        "cold_start": not args.warm,
        "noise_from": args.map,
    }
    hess = cal / "laplace_hess_real.npz"
    if hess.exists():
        c = np.load(hess)
        Hl = 0.5 * (c["H"] + c["H"].T)
        # laplace_hess_real.npz is in sorted-site order; pick the 14 physics rows
        names_sorted = sorted(
            NAMES + ["sigma_pit", "sigma_core", "rho_pit", "rho_core"]
        )
        idx = [names_sorted.index(n) for n in NAMES]
        Hl_phys = Hl[np.ix_(idx, idx)]  # noise params fixed: the conditional block
        lam = np.linalg.eigvalsh(Hl_phys)
        Hl_reg = (
            Hl_phys
            if lam.min() > 0
            else Hl_phys + (1e-6 * lam.max() - lam.min()) * np.eye(N_PHYS)
        )
        sd_l = np.sqrt(np.diag(np.linalg.inv(Hl_reg)))
        print(
            f"\n{'site':10s} {'sd(z) Fisher':>13s} {'sd(z) FD-Laplace':>17s} {'ratio':>7s}"
        )
        for i, n in enumerate(NAMES):
            print(f"{n:10s} {sd[i]:13.3f} {sd_l[i]:17.3f} {sd[i] / sd_l[i]:7.2f}")
        out["sd_z_fd_laplace"] = dict(zip(NAMES, map(float, sd_l), strict=True))
    (cal / "gn_map_real.json").write_text(json.dumps(out, indent=2))
    print("wrote data/calibration/gn_map_real.json")


if __name__ == "__main__":
    main()
