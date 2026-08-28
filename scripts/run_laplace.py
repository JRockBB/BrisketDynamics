"""Laplace posterior at the real-cook MAP on the Gauss-Newton Hessian.

The Hessian of the potential in numpyro's unconstrained space, by
`jax.hessian` of a surrogate model whose μ is the Tesseract's tangent model
μ₀ + J (x − x₀) at the MAP: the 14 × 14 physics block is the Gauss-Newton
curvature Jᵀ Σ⁻¹ J + prior (the solve's second derivatives dropped), the
noise block and the physics × noise cross block are exact, and numpyro
applies the constrained→unconstrained chain rule. No finite differences.
Draws are transformed back to the constrained space and written like the
ADVI output, so every downstream consumer reads them unchanged.

Run from app/:  uv run python -u ../scripts/run_laplace.py
Writes data/calibration/laplace_samples_real.npz, laplace_mean_real.json,
laplace_hess_real.npz (H, z0; sorted-site order),
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
from brisket_app.calibrate import phi_full
from brisket_app.clients import monolith_client
from brisket_app.coupled import CookGrids
from brisket_app.monolith import ALL_NAMES_SORTED, NAMES, model_mean
from brisket_app.plots import corner_plot, posterior_predictive_plot
from jax.flatten_util import ravel_pytree
from numpyro.distributions import biject_to
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[1]
NEWTON_CLIP = 0.3  # max |dz| per coordinate in unconstrained space


def _hessian_of(model, init, seed):
    """(z0, names, unravel, inverse transforms, H, grad) of a model's potential at `init`."""
    mi = initialize_model(
        jax.random.PRNGKey(seed), model, init_strategy=init_to_value(values=init)
    )
    z0, unravel = ravel_pytree(mi.param_info.z)
    names = sorted(mi.param_info.z.keys())
    assert names == ALL_NAMES_SORTED
    inv_transforms = {
        k: biject_to(site["fn"].support)
        for k, site in mi.model_trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }

    def pot(z):
        return mi.potential_fn(unravel(z))

    H = np.asarray(jax.hessian(pot)(z0), dtype=float)
    g = np.asarray(jax.grad(pot)(z0), dtype=float)
    return np.asarray(z0, dtype=float), names, unravel, inv_transforms, pot, H, g


def main():
    """Fit the Laplace posterior, draw from it, render corner + predictive."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-json", default="map_real.json")
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"

    tx = monolith_client()
    log = cooklog.load(dt_min=10.0)
    init = {
        k: float(v)
        for k, v in json.loads((cal / args.init_json).read_text()).items()
        if not k.startswith("_")
    }
    x0 = np.concatenate([[init[k] for k in NAMES[:7]], phi_full(init)])

    t0 = time.perf_counter()
    mu0, J = model_mean(x0, log)
    print(
        f"Tesseract: μ and J {J.shape} at {args.init_json} "
        f"({time.perf_counter() - t0:.0f} s incl. compile)",
        flush=True,
    )
    surrogate = calibrate.bound_model_real_linear(log, mu0, J, x0)
    z0, names, _unravel, inv_transforms, pot, H, g0 = _hessian_of(
        surrogate, init, args.seed
    )
    p0 = float(pot(jnp.asarray(z0)))
    n = z0.size
    print(f"potential at {args.init_json}: {p0:.2f}  |grad| {np.linalg.norm(g0):.3g}")
    H = 0.5 * (H + H.T)
    lam = np.linalg.eigvalsh(H)
    print(f"Gauss-Newton Hessian eigenvalues: min {lam.min():.3g}  max {lam.max():.3g}")
    if lam.min() <= 0:
        raise RuntimeError("Gauss-Newton Hessian is not positive definite")
    np.savez(cal / "laplace_hess_real.npz", z0=z0, H=H, done=n)
    cov = np.linalg.inv(H)
    sd_z = np.sqrt(np.diag(cov))
    print("sd(z):", {k: round(float(sd_z[i]), 3) for i, k in enumerate(names)})
    source = "Gauss-Newton Hessian (jax; noise and cross blocks exact)"

    dz = np.clip(-cov @ g0, -NEWTON_CLIP, NEWTON_CLIP)
    z_c = z0 + dz
    pc = float(pot(jnp.asarray(z_c)))
    if pc > p0:
        print(f"newton step rejected ({pc:.2f} > {p0:.2f}); centred at the MAP")
        z_c = z0
    else:
        print(
            f"newton step: potential {p0:.2f} -> {pc:.2f}  max |dz| {np.abs(dz).max():.3f}"
        )

    rng = np.random.default_rng(args.seed)
    L = np.linalg.cholesky(cov)
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
    mean["_source"] = f"Laplace at {args.init_json}: {source}"
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
