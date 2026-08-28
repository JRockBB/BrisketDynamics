"""Laplace posterior at the real-cook MAP.

A Gaussian in numpyro's unconstrained space, centred one Newton step from
the MAP (the constrained mode; the transform Jacobian shifts it slightly)
with covariance = inverse of a central-difference Hessian of the log
density (36 gradient evaluations for 18 parameters). Eigenvalues below
1e-4 of the largest are floored, so ridge directions inherit the prior's
width rather than blowing up. Draws are transformed back to the
constrained space and written like the ADVI output, so every downstream
consumer (MPC on the mean, robust control on the draws) reads them
unchanged.

Run from app/:  uv run python -u ../scripts/run_laplace.py
Writes data/calibration/laplace_samples_real.npz, laplace_mean_real.json,
figures/corner_laplace_real.png, figures/posterior_predictive_laplace_real.png.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
import jax.numpy as jnp
from brisket_app import calibrate, cooklog
from brisket_app.clients import monolith_client
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot
from jax.flatten_util import ravel_pytree
from numpyro.distributions import biject_to
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[1]
EIG_FLOOR_REL = 1e-4
NEWTON_CLIP = 0.3  # max |dz| per coordinate in unconstrained space


def main():
    """Fit the Laplace posterior, draw from it, render corner + predictive."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fd-step",
        type=float,
        default=1e-3,
        help="central-difference step in unconstrained space",
    )
    ap.add_argument("--init-json", default="map_real.json")
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"

    tx = monolith_client()
    log = cooklog.load(dt_min=10.0)
    model = calibrate.bound_model_real(log, tx)
    init = {
        k: float(v)
        for k, v in json.loads((cal / args.init_json).read_text()).items()
        if not k.startswith("_")
    }
    mi = initialize_model(
        jax.random.PRNGKey(args.seed), model, init_strategy=init_to_value(values=init)
    )
    z0_tree = mi.param_info.z
    z0, unravel = ravel_pytree(z0_tree)
    z0 = np.asarray(z0, dtype=float)
    names = sorted(z0_tree.keys())
    n = z0.size
    inv_transforms = {
        k: biject_to(site["fn"].support)
        for k, site in mi.model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }

    def pot(z):
        return mi.potential_fn(unravel(jnp.asarray(z)))

    grad = jax.grad(lambda z: mi.potential_fn(unravel(z)))
    t0 = time.perf_counter()
    p0 = float(pot(z0))
    g0 = np.asarray(grad(jnp.asarray(z0)), dtype=float)
    print(
        f"potential at {args.init_json}: {p0:.2f}  |grad| {np.linalg.norm(g0):.3g}  "
        f"({time.perf_counter() - t0:.0f} s)",
        flush=True,
    )

    # central-difference Hessian, one column per gradient pair; cached so
    # an interrupted run resumes at the next column
    cache = cal / "laplace_hess_real.npz"
    h = args.fd_step
    H, done = np.zeros((n, n)), 0
    if cache.exists():
        c = np.load(cache)
        if (
            c["z0"].shape == z0.shape
            and np.allclose(c["z0"], z0)
            and float(c["h"]) == h
        ):
            H, done = c["H"], int(c["done"])
            print(f"resuming hessian at column {done}/{n}", flush=True)
    for j in range(done, n):
        e = np.zeros(n)
        e[j] = h
        gp = np.asarray(grad(jnp.asarray(z0 + e)), dtype=float)
        gm = np.asarray(grad(jnp.asarray(z0 - e)), dtype=float)
        H[:, j] = (gp - gm) / (2 * h)
        np.savez(cache, z0=z0, h=h, H=H, done=j + 1)
        print(
            f"  hessian column {j + 1}/{n} ({names[j]})  "
            f"{(time.perf_counter() - t0) / 60:.1f} min",
            flush=True,
        )
    H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(H)
    floor = EIG_FLOOR_REL * lam.max()
    n_floor = int(np.sum(lam < floor))
    lam_reg = np.maximum(np.abs(lam), floor)
    print(
        f"hessian eigenvalues: min {lam.min():.3g}  max {lam.max():.3g}  "
        f"({n_floor} floored at {floor:.3g})",
        flush=True,
    )
    cov = (V / lam_reg) @ V.T

    dz = np.clip(-cov @ g0, -NEWTON_CLIP, NEWTON_CLIP)
    z_c = z0 + dz
    pc = float(pot(z_c))
    if pc > p0:
        print(f"newton step rejected ({pc:.2f} > {p0:.2f}); centred at the MAP")
        z_c = z0
    else:
        print(
            f"newton step: potential {p0:.2f} -> {pc:.2f}  max |dz| {np.abs(dz).max():.3f}"
        )

    rng = np.random.default_rng(args.seed)
    L = V * np.sqrt(1.0 / lam_reg)  # cov = L @ L.T
    Z = z_c[None, :] + rng.standard_normal((args.draws, n)) @ L.T
    stacked = {}
    for j, k in enumerate(names):
        stacked[k] = np.asarray(inv_transforms[k](jnp.asarray(Z[:, j]))).reshape(
            args.draws
        )
    out = cal / "laplace_samples_real.npz"
    np.savez(out, **stacked)
    mean = {k: float(np.mean(v)) for k, v in stacked.items()}
    mean["_n_draws"] = args.draws
    mean["_source"] = f"Laplace at {args.init_json} (FD Hessian, unconstrained space)"
    (cal / "laplace_mean_real.json").write_text(json.dumps(mean, indent=2))
    sd = {k: float(np.std(v)) for k, v in stacked.items()}
    print(f"wrote {out.name} ({args.draws} draws), laplace_mean_real.json")
    print("posterior sd / mean:", {k: round(sd[k] / abs(mean[k]), 3) for k in names})

    grids = CookGrids(
        t_end=float(log["t"][-1]),
        dt_exchange=float(log["t"][1] - log["t"][0]),
        n_u=len(log["t"]),
    )
    pp_data = {
        "u_knots": log["u"],
        "obs_pit": log["obs_pit"],
        "obs_core": log["obs_core"],
    }
    ics = {
        "tpit0": log["meta"]["tpit0_K"],
        "t_amb": 300.0,
        "t0_meat": log["meta"]["tcore0_K"],
    }
    corner_names = (
        [k for k, _ in calibrate.THETA_SITES]
        + [k for k, _ in calibrate.PHI_SITES]
        + [k for k, _ in calibrate.PHI_SITES_IGNITION]
        + [k for k, _ in calibrate.RHO_SITES]
    )
    corner_plot(
        stacked, REPO / "figures" / "corner_laplace_real.png", names=corner_names
    )
    posterior_predictive_plot(
        stacked,
        pp_data,
        tx,
        grids,
        REPO / "figures" / "posterior_predictive_laplace_real.png",
        ics=ics,
        masks={"pit": log["mask_pit"], "core": log["mask_core"]},
    )
    print(
        "wrote figures/corner_laplace_real.png, figures/posterior_predictive_laplace_real.png"
    )


if __name__ == "__main__":
    main()
