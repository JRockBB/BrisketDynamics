"""Laplace posterior at the real-cook MAP on the Gauss-Newton Hessian.

The Hessian of the potential in numpyro's unconstrained space, by
`jax.hessian` of a surrogate model whose μ is the Tesseract's tangent model
μ₀ + J (x − x₀) at the MAP: the 15 × 15 physics block is the Gauss-Newton
curvature Jᵀ Σ⁻¹ J + prior (the solve's second derivatives dropped), the
noise block and the physics × noise cross block are exact, and numpyro
applies the constrained→unconstrained chain rule. No finite differences.
Draws are transformed back to the constrained space and written like the
ADVI output, so every downstream consumer reads them unchanged. The latents
are the log's (`monolith.names_for`): the real cook is modelled from lighting,
k_spread and the preheat damper u_pre sampled, a0 fixed at the chimney
start's A_LIGHT and written into the samples as a constant.

Run from app/:  uv run python -u ../scripts/run_laplace.py
Writes data/calibration/laplace_samples_real.npz, laplace_mean_real.json,
laplace_hess_real.npz (H, z0; sorted-site order),
figures/calibration/corner_laplace_real.png, figures/calibration/posterior_predictive_laplace_real.png.
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
from brisket_app.coupled import A_LIGHT, HR, CookGrids
from brisket_app.monolith import NOISE_NAMES, from_lighting, model_mean, names_for
from brisket_app.plots import corner_plot, posterior_predictive_plot
from jax.flatten_util import ravel_pytree
from numpyro.distributions import biject_to
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[1]
NEWTON_CLIP = 0.3  # max |dz| per coordinate in unconstrained space


def _hessian_of(model, init, seed, all_names):
    """(z0, names, unravel, inverse transforms, H, grad) of a model's potential at `init`."""
    mi = initialize_model(
        jax.random.PRNGKey(seed), model, init_strategy=init_to_value(values=init)
    )
    z0, unravel = ravel_pytree(mi.param_info.z)
    names = sorted(mi.param_info.z.keys())
    assert names == all_names
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
    NAMES = names_for(log)
    ALL_NAMES_SORTED = sorted(NAMES + NOISE_NAMES)
    lit = from_lighting(log)
    init = {k: v for k, v in init.items() if k in ALL_NAMES_SORTED}   # a0 is fixed from lighting, not a site
    x0 = np.array([init[k] for k in NAMES])

    t0 = time.perf_counter()
    mu0, J = model_mean(x0, log)
    print(
        f"Tesseract: μ and J {J.shape} at {args.init_json} "
        f"({time.perf_counter() - t0:.0f} s incl. compile)",
        flush=True,
    )
    surrogate = calibrate.bound_model_real_linear(log, mu0, J, x0)
    z0, names, _unravel, inv_transforms, pot, H, g0 = _hessian_of(
        surrogate, init, args.seed, ALL_NAMES_SORTED
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
    if lit:
        stacked["a0"] = np.full(args.draws, A_LIGHT)  # the phi layout's lit fraction: a cook from lighting
    out = cal / "laplace_samples_real.npz"
    np.savez(out, **stacked)
    mean = {k: float(np.mean(v)) for k, v in stacked.items()}
    mean["_n_draws"] = args.draws
    mean["_source"] = f"Laplace at {args.init_json}: {source}"
    (cal / "laplace_mean_real.json").write_text(json.dumps(mean, indent=2))
    sd = {k: float(np.std(v)) for k, v in stacked.items()}
    print(f"wrote {out.name} ({args.draws} draws), laplace_mean_real.json")
    print("posterior sd / mean:", {k: round(sd[k] / abs(mean[k]), 3) for k in names})

    corner_names = NAMES + [k for k, _ in calibrate.RHO_SITES]
    corner_plot(
        stacked, REPO / "figures" / "calibration" / "corner_laplace_real.png", names=corner_names
    )
    if lit:
        _predictive_from_lighting(stacked, NAMES, log, REPO / "figures" / "calibration"
                                  / "posterior_predictive_laplace_real.png", seed=args.seed)
        print("wrote figures/calibration/corner_laplace_real.png, "
              "figures/calibration/posterior_predictive_laplace_real.png")
        return

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
    posterior_predictive_plot(
        stacked,
        pp_data,
        tx,
        grids,
        REPO / "figures" / "calibration" / "posterior_predictive_laplace_real.png",
        ics=ics,
        masks={"pit": log["mask_pit"], "core": log["mask_core"]},
    )
    print(
        "wrote figures/calibration/corner_laplace_real.png, figures/calibration/posterior_predictive_laplace_real.png"
    )


def _predictive_from_lighting(stacked, names, log, path, n_draws=30, seed=0):
    """The posterior-predictive band of the cook from lighting (the preheat included) over the log."""
    from brisket_app.monolith import preheat_trace

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_total = len(stacked[names[0]])
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)
    runs = [preheat_trace(np.array([float(stacked[k][i]) for k in names]), log) for i in idx]
    t_pre = runs[0]["t_load"]
    th = runs[0]["t"] / HR
    to = (np.asarray(log["t"]) - log["t"][0] + t_pre) / HR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, key, obs, mk in [
        (axes[0], "tpit", log["obs_pit"], log["mask_pit"]),
        (axes[1], "tcore", log["obs_core"], log["mask_core"]),
    ]:
        Y = np.stack([r[key] for r in runs]) - 273.15
        lo, med, hi = np.percentile(Y, [5, 50, 95], axis=0)
        ax.fill_between(th, lo, hi, color="tab:orange", alpha=0.3, label="posterior 90% band")
        ax.plot(th, med, color="tab:orange", lw=1.5, label="posterior median")
        ax.plot(to[mk], np.asarray(obs)[mk] - 273.15, ".", ms=2.5, color="k", alpha=0.5, label="observed")
        ax.axvline(t_pre / HR, color="0.5", lw=1.0, ls="-.")
        ax.set(xlabel="time from lighting [h]", ylabel="temperature [C]",
               title=("pit" if key == "tpit" else "core") + " (meat on at the dash-dot line)")
        ax.legend(fontsize=8)
    fig.suptitle("Real cook from lighting - Laplace posterior predictive")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
