"""Bayesian calibration of the coupled pit-brisket model (NumPyro).

Stage 1 (this file's `run_map`): MAP via SVI with a Delta guide — validates the
full gradient path through both Tesseracts cheaply and initializes NUTS.
Stage 2 (`run_nuts`): full NUTS posterior. Every leapfrog step pulls adjoint
gradients across the Julia<->JAX boundary.

Priors are LogNormal, centered on the model defaults (which double as the
physical scaling analysis; see docs/brief.md §6) with log-sds wide enough to
cover the literature ranges. u_leak (a fraction) gets a Beta prior. Every
prior's center is THETA_DEFAULT/PHI_DEFAULT — the synthetic truths in synth.py
are offset 20-40% from these, so posterior recovery is a real test.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from .coupled import CookGrids, coupled_cook

# (name, prior, index) for the two parameter vectors; order must match the
# Julia solvers' expectations.
THETA_SITES = [
    ("D_eff", dist.LogNormal(np.log(1.5e-8), 0.7)),
    ("k_m", dist.LogNormal(np.log(1.8e-3), 0.7)),
    ("h_c", dist.LogNormal(np.log(10.0), 0.4)),
    ("L", dist.LogNormal(np.log(0.03), 0.25)),
    ("k_mult", dist.LogNormal(0.0, 0.3)),
]
PHI_SITES = [
    ("C_th", dist.LogNormal(np.log(15e3), 0.5)),
    ("UA", dist.LogNormal(np.log(20.0), 0.5)),
    ("r_max", dist.LogNormal(np.log(3e-4), 0.5)),
    ("dH_eff", dist.LogNormal(np.log(1.2e7), 0.3)),
    ("u_leak", dist.Beta(2.0, 8.0)),
]
# Ignition-dynamics sites: sampled only for REAL-cook calibration. Synthetic
# data was generated fully lit (a0 = 1, spread inert), so these constants are
# appended deterministically there.
PHI_SITES_IGNITION = [
    ("k_spread", dist.LogNormal(np.log(3e-4), 0.7)),
    ("a0", dist.Beta(2.0, 4.0)),
]
PHI_IGNITION_INERT = np.array([3e-4, 1.0])  # k_spread (inert at a0=1), a0


def phi_full(params: dict) -> np.ndarray:
    """Assemble the 7-vector phi from a params dict (MAP json, posterior
    draw), filling ignition entries with inert defaults when absent —
    backward-compatible with pre-ignition-state result files."""
    base = [params[n] for n, _ in PHI_SITES]
    ign = [params.get(n, d) for (n, _), d in
           zip(PHI_SITES_IGNITION, PHI_IGNITION_INERT)]
    return np.array(base + ign)


def model(obs_pit, obs_core, u_knots, meat_tx, pit_tx, grids, tols):
    theta = jnp.stack([numpyro.sample(n, d) for n, d in THETA_SITES])
    # synthetic data was generated fully lit: ignition entries stay inert
    phi = jnp.concatenate([
        jnp.stack([numpyro.sample(n, d) for n, d in PHI_SITES]),
        jnp.asarray(PHI_IGNITION_INERT),
    ])
    sigma_pit = numpyro.sample("sigma_pit", dist.HalfNormal(3.0))
    sigma_core = numpyro.sample("sigma_core", dist.HalfNormal(2.0))

    sim = coupled_cook(theta, phi, jnp.asarray(u_knots), meat_tx, pit_tx,
                       grids, K=2, **tols)
    numpyro.sample("pit", dist.Normal(sim["tpit"], sigma_pit), obs=obs_pit)
    numpyro.sample("core", dist.Normal(sim["tcore"], sigma_core), obs=obs_core)


def _bound_model(data: dict, meat_tx, pit_tx, grids: CookGrids | None = None,
                 tols: dict | None = None):
    grids = grids or CookGrids(t_end=float(data["t"][-1]),
                               dt_exchange=float(data["t"][1] - data["t"][0]),
                               n_u=len(data["u_knots"]))
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    return partial(model,
                   jnp.asarray(data["obs_pit"]), jnp.asarray(data["obs_core"]),
                   data["u_knots"], meat_tx, pit_tx, grids, tols)


SIGMA_FLOOR_K = 1.2  # >= 2 F digitization noise floor (PROVENANCE.md)


def model_real(obs_pit, obs_core, mask_pit, mask_core, u_knots, ics,
               meat_tx, pit_tx, grids, tols):
    """Calibration model for the real cook log: masked Gaussian likelihood
    (lid events + probe-settling excluded) with the digitization noise floor,
    initial conditions fixed from the log."""
    theta = jnp.stack([numpyro.sample(n, d) for n, d in THETA_SITES])
    # real cooks start Minion-style: sample the ignition dynamics too
    phi = jnp.stack([numpyro.sample(n, d)
                     for n, d in PHI_SITES + PHI_SITES_IGNITION])
    sigma_pit = SIGMA_FLOOR_K + numpyro.sample("sigma_pit", dist.HalfNormal(3.0))
    sigma_core = SIGMA_FLOOR_K + numpyro.sample("sigma_core", dist.HalfNormal(2.0))

    sim = coupled_cook(theta, phi, jnp.asarray(u_knots), meat_tx, pit_tx,
                       grids, K=2, ics=ics, **tols)
    numpyro.sample("pit", dist.Normal(sim["tpit"][mask_pit], sigma_pit),
                   obs=obs_pit[mask_pit])
    numpyro.sample("core", dist.Normal(sim["tcore"][mask_core], sigma_core),
                   obs=obs_core[mask_core])


def bound_model_real(log: dict, meat_tx, pit_tx, tols: dict | None = None):
    """Bind model_real to a loaded cook log (see cooklog.load)."""
    grids = CookGrids(t_end=float(log["t"][-1]),
                      dt_exchange=float(log["t"][1] - log["t"][0]),
                      n_u=len(log["t"]))
    ics = {"tpit0": log["meta"]["tpit0_K"], "t_amb": 300.0,
           "t0_meat": log["meta"]["tcore0_K"]}
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    return partial(model_real,
                   jnp.asarray(log["obs_pit"]), jnp.asarray(log["obs_core"]),
                   np.asarray(log["mask_pit"]), np.asarray(log["mask_core"]),
                   log["u"], ics, meat_tx, pit_tx, grids, tols)


def time_gradient(data, meat_tx, pit_tx, **kw) -> float:
    """Seconds per log-density gradient at calibration settings — the number
    the Aug-16 NUTS go/no-go checkpoint runs on (ladder in the plan)."""
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    from numpyro.infer.util import initialize_model

    info = initialize_model(jax.random.PRNGKey(0), m)
    f = jax.value_and_grad(lambda p: info.potential_fn(p))
    f(info.param_info.z)  # warmup (JIT + Julia first-call)
    t0 = time.perf_counter()
    n = 3
    for _ in range(n):
        v, g = f(info.param_info.z)
        jax.block_until_ready(v)
    return (time.perf_counter() - t0) / n


def run_map(data, meat_tx, pit_tx, num_steps: int = 300, lr: float = 2e-2,
            seed: int = 0, **kw):
    """MAP via SVI + AutoDelta. Returns (params_dict, losses)."""
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    guide = AutoDelta(m)
    svi = SVI(m, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=True)
    params = guide.median(result.params)
    return {k: np.asarray(v) for k, v in params.items()}, np.asarray(result.losses)


def run_nuts(data, meat_tx, pit_tx, num_warmup: int = 300, num_samples: int = 300,
             max_tree_depth: int = 6, init_params: dict | None = None,
             seed: int = 0, **kw) -> MCMC:
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    kernel = NUTS(m, max_tree_depth=max_tree_depth, init_strategy=(
        numpyro.infer.init_to_value(values=init_params) if init_params
        else numpyro.infer.init_to_median()))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def recovery_table(map_params: dict, data: dict) -> str:
    """Human-readable MAP-vs-truth comparison (Milestone B1: within ~10-20%)."""
    names = [n for n, _ in THETA_SITES] + [n for n, _ in PHI_SITES]
    truths = np.concatenate([data["theta_true"], data["phi_true"]])
    lines = [f"{'param':>10} {'truth':>12} {'MAP':>12} {'err %':>8}"]
    for n, t in zip(names, truths):
        m = float(map_params[n])
        lines.append(f"{n:>10} {t:>12.4g} {m:>12.4g} {100 * (m - t) / t:>7.1f}%")
    return "\n".join(lines)
