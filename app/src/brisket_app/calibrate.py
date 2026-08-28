"""Bayesian calibration of the coupled pit-brisket model (NumPyro).

Stage 1 (`run_map`): MAP via SVI with a Delta guide — validates the
full gradient path through both Tesseracts cheaply and initializes NUTS
and VI.
Stage 2 (`run_nuts`): full NUTS posterior. Every leapfrog step pulls adjoint
gradients across the Julia<->JAX boundary.
Evening / library path (`make_vi_guide`): full-rank (or low-rank /
mean-field) ADVI on the same model. NUTS is not replaced.

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
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)

from .coupled import CookGrids, coupled_cook

# (name, prior, index) for the two parameter vectors; order must match the
# Julia solvers' expectations.
THETA_SITES = [
    ("D_eff", dist.LogNormal(np.log(7.0e-9), 0.7)),
    ("k_m", dist.LogNormal(np.log(1.8e-3), 0.7)),
    ("h_c", dist.LogNormal(np.log(10.0), 0.4)),
    ("L", dist.LogNormal(np.log(0.03), 0.25)),
    ("k_mult", dist.LogNormal(0.0, 0.3)),
    # expulsion kinetics joined theta 2026-08-21: frozen literature values
    # left the real-cook refit at loss 495 vs 245 (they shape the stall
    # entry, which the core probe sees). a_myosin prior kept sub-0.5 at 2
    # sigma so W_eq stays positive with the fixed actin amplitude.
    ("k_e", dist.LogNormal(np.log(5.6e-4), 0.5)),
    ("a_myosin", dist.LogNormal(np.log(0.20), 0.4)),
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
# Real-cook residual AR(1) on the 10-min grid. ρ is bounded at RHO_MAX
# (correlation time ≤ ~3 h): near a unit root the discrepancy process
# becomes a free offset and a smooth 15 K miss costs less than a fit.
RHO_MAX = 0.95
_RHO_PRIOR = dist.TransformedDistribution(
    dist.Beta(2.0, 2.0),
    dist.transforms.AffineTransform(
        0.0, RHO_MAX, domain=dist.constraints.unit_interval
    ),
)
RHO_SITES = [
    ("rho_pit", _RHO_PRIOR),
    ("rho_core", _RHO_PRIOR),
]
_RHO_VAR_FLOOR = 1.0e-8


def phi_full(params: dict) -> np.ndarray:
    """Assemble the 7-vector phi from a params dict.

    Accepts a MAP json or a posterior draw; ignition entries fall back to
    inert defaults when absent, keeping pre-ignition-state result files
    loadable.

    Parameters
    ----------
    params : dict
        Parameter values keyed by site name (PHI_SITES at minimum).

    Returns
    -------
    np.ndarray
        The full pit parameter vector [C_th, UA, r_max, dH_eff, u_leak,
        k_spread, a0].
    """
    base = [params[n] for n, _ in PHI_SITES]
    ign = [
        params.get(n, d)
        for (n, _), d in zip(PHI_SITES_IGNITION, PHI_IGNITION_INERT, strict=True)
    ]
    return np.array(base + ign)


def model(obs_pit, obs_core, u_knots, meat_tx, pit_tx, grids, tols):
    """NumPyro model for the synthetic cook (fully lit, unmasked Gaussian).

    Samples theta/phi/noise scales from the priors, runs the coupled
    differentiable cook, and conditions Gaussian likelihoods on both probe
    traces. The ignition entries stay inert (a0 = 1) because the synthetic
    data was generated fully lit.

    Parameters
    ----------
    obs_pit, obs_core : jnp.ndarray
        Observed pit / core temperature traces [K] on the exchange grid.
    u_knots : array-like
        Damper schedule (control input, replayed open-loop).
    meat_tx, pit_tx : Tesseract
        The two solver endpoints.
    grids : CookGrids
        Time discretization of the coupled cook.
    tols : dict
        Solver tolerances forwarded to the Tesseracts.
    """
    theta = jnp.stack([numpyro.sample(n, d) for n, d in THETA_SITES])
    # synthetic data was generated fully lit: ignition entries stay inert
    phi = jnp.concatenate(
        [
            jnp.stack([numpyro.sample(n, d) for n, d in PHI_SITES]),
            jnp.asarray(PHI_IGNITION_INERT),
        ]
    )
    sigma_pit = numpyro.sample("sigma_pit", dist.HalfNormal(3.0))
    sigma_core = numpyro.sample("sigma_core", dist.HalfNormal(2.0))

    sim = coupled_cook(
        theta, phi, jnp.asarray(u_knots), meat_tx, pit_tx, grids, K=2, **tols
    )
    numpyro.sample("pit", dist.Normal(sim["tpit"], sigma_pit), obs=obs_pit)
    numpyro.sample("core", dist.Normal(sim["tcore"], sigma_core), obs=obs_core)


def _bound_model(
    data: dict,
    meat_tx,
    pit_tx,
    grids: CookGrids | None = None,
    tols: dict | None = None,
):
    grids = grids or CookGrids(
        t_end=float(data["t"][-1]),
        dt_exchange=float(data["t"][1] - data["t"][0]),
        n_u=len(data["u_knots"]),
    )
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    return partial(
        model,
        jnp.asarray(data["obs_pit"]),
        jnp.asarray(data["obs_core"]),
        data["u_knots"],
        meat_tx,
        pit_tx,
        grids,
        tols,
    )


SIGMA_FLOOR_K = 1.2  # >= 2 F digitization noise floor (PROVENANCE.md)


def contiguous_runs(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Index tuples of contiguous True runs in a 1-D boolean mask.

    Lid events and probe-settling holes break the AR(1) chain: two
    observations separated by a masked gap are not lag-1 neighbours.
    Empty mask → empty tuple.

    Parameters
    ----------
    mask : np.ndarray
        1-D boolean; True = usable observation.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        One index tuple per contiguous True run, in order.
    """
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0 or not m.any():
        return ()
    padded = np.concatenate([[False], m, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return tuple(
        tuple(range(int(s), int(e))) for s, e in zip(starts, ends, strict=True)
    )


def ar1_log_prob(y, mu, sigma, rho):
    """Gaussian AR(1) log-density on a contiguous residual series.

    `sigma` is the *marginal* residual scale: y_0 ~ N(μ_0, σ²) and
    subsequent innovations have sd σ√(1-ρ²). ρ = 0 recovers iid N(μ, σ²).

    Parameters
    ----------
    y, mu : array-like
        Observations and model means, same length ≥ 1.
    sigma, rho : scalar
        Marginal scale (>0) and lag-1 correlation in (-1, 1).

    Returns
    -------
    jax.Array
        Scalar log-density.
    """
    y = jnp.asarray(y).reshape(-1)
    mu = jnp.asarray(mu).reshape(-1)
    e = y - mu
    lp0 = dist.Normal(0.0, sigma).log_prob(e[0])
    n = e.shape[0]
    if n == 1:
        return lp0
    innov_sd = sigma * jnp.sqrt(jnp.clip(1.0 - rho * rho, _RHO_VAR_FLOOR, 1.0))
    return lp0 + dist.Normal(rho * e[:-1], innov_sd).log_prob(e[1:]).sum()


def ar1_observe(name, y, mu, sigma, rho):
    """Condition a NumPyro model on an AR(1) Gaussian segment.

    First point carries the marginal; the rest are innovations. A
    length-1 run is just N(μ, σ²).

    Parameters
    ----------
    name : str
        Trace site prefix (must be unique per segment).
    y, mu : array-like
        Observations and model means for this contiguous run.
    sigma, rho : scalar
        Marginal residual scale and lag-1 correlation.
    """
    y = jnp.asarray(y).reshape(-1)
    mu = jnp.asarray(mu).reshape(-1)
    numpyro.sample(f"{name}_y0", dist.Normal(mu[0], sigma), obs=y[0])
    # run length is a Python constant (precomputed mask segments)
    if y.shape[0] == 1:
        return
    innov_sd = sigma * jnp.sqrt(jnp.clip(1.0 - rho * rho, _RHO_VAR_FLOOR, 1.0))
    numpyro.sample(
        f"{name}_innov",
        dist.Normal(mu[1:] + rho * (y[:-1] - mu[:-1]), innov_sd),
        obs=y[1:],
    )


def model_real(
    obs_pit,
    obs_core,
    pit_runs,
    core_runs,
    u_knots,
    ics,
    meat_tx,
    pit_tx,
    grids,
    tols,
):
    """Calibration model for the real cook log.

    Masked AR(1) Gaussian (lid events and probe-settling break the chain)
    with the digitization noise floor added to both *marginal* scales,
    initial conditions fixed from the log, and the ignition dynamics
    sampled (real cooks start Minion-style). Synthetic calibration stays
    iid: that data is generated iid.
    """
    theta = jnp.stack([numpyro.sample(n, d) for n, d in THETA_SITES])
    # real cooks start Minion-style: sample the ignition dynamics too
    phi = jnp.stack([numpyro.sample(n, d) for n, d in PHI_SITES + PHI_SITES_IGNITION])
    sigma_pit = SIGMA_FLOOR_K + numpyro.sample("sigma_pit", dist.HalfNormal(3.0))
    sigma_core = SIGMA_FLOOR_K + numpyro.sample("sigma_core", dist.HalfNormal(2.0))
    rho_pit = numpyro.sample("rho_pit", RHO_SITES[0][1])
    rho_core = numpyro.sample("rho_core", RHO_SITES[1][1])

    sim = coupled_cook(
        theta, phi, jnp.asarray(u_knots), meat_tx, pit_tx, grids, K=2, ics=ics, **tols
    )
    for k, run in enumerate(pit_runs):
        idx = jnp.array(run)
        ar1_observe(f"pit{k}", obs_pit[idx], sim["tpit"][idx], sigma_pit, rho_pit)
    for k, run in enumerate(core_runs):
        idx = jnp.array(run)
        ar1_observe(f"core{k}", obs_core[idx], sim["tcore"][idx], sigma_core, rho_core)


def bound_model_real(log: dict, meat_tx, pit_tx, tols: dict | None = None):
    """Bind model_real to a loaded cook log (see cooklog.load)."""
    grids = CookGrids(
        t_end=float(log["t"][-1]),
        dt_exchange=float(log["t"][1] - log["t"][0]),
        n_u=len(log["t"]),
    )
    ics = {
        "tpit0": log["meta"]["tpit0_K"],
        "t_amb": 300.0,
        "t0_meat": log["meta"]["tcore0_K"],
    }
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    return partial(
        model_real,
        jnp.asarray(log["obs_pit"]),
        jnp.asarray(log["obs_core"]),
        contiguous_runs(log["mask_pit"]),
        contiguous_runs(log["mask_core"]),
        log["u"],
        ics,
        meat_tx,
        pit_tx,
        grids,
        tols,
    )


def time_gradient(data, meat_tx, pit_tx, **kw) -> float:
    """Time one log-density gradient at calibration settings.

    Seconds per `jax.value_and_grad` of the potential — the number the
    NUTS go/no-go checkpoint runs on (ladder in the plan).
    """
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    from numpyro.infer.util import initialize_model

    info = initialize_model(jax.random.PRNGKey(0), m)
    f = jax.value_and_grad(lambda p: info.potential_fn(p))
    f(info.param_info.z)  # warmup (JIT + Julia first-call)
    t0 = time.perf_counter()
    n = 3
    for _ in range(n):
        v, _g = f(info.param_info.z)
        jax.block_until_ready(v)
    return (time.perf_counter() - t0) / n


def real_latent_names() -> list[str]:
    """Constrained site names sampled by `model_real` (not the observations)."""
    return (
        [n for n, _ in THETA_SITES]
        + [n for n, _ in PHI_SITES]
        + [n for n, _ in PHI_SITES_IGNITION]
        + ["sigma_pit", "sigma_core"]
        + [n for n, _ in RHO_SITES]
    )


def make_vi_guide(
    model,
    kind: str = "full",
    init_values: dict | None = None,
    rank: int = 8,
    init_scale: float = 0.1,
):
    """Build an ADVI guide for a bound NumPyro model.

    `full` is a multivariate Gaussian on the unconstrained space (can
    represent linear ridges). `lowrank` is cheaper; `meanfield` cannot
    capture the pit ridges. NUTS (`run_nuts`) remains the exact-posterior
    sampler.

    Parameters
    ----------
    model : callable
        Bound NumPyro model (no extra args).
    kind : {'full', 'lowrank', 'meanfield'}
        Variational family.
    init_values : dict or None
        Constrained site values (typically the MAP) for the guide loc.
    rank : int
        Factor rank when `kind='lowrank'`.
    init_scale : float
        Initial guide scale in the unconstrained space.

    Returns
    -------
    numpyro.infer.autoguide.AutoContinuous
        The guide.
    """
    init_loc_fn = (
        numpyro.infer.init_to_value(values=init_values)
        if init_values
        else numpyro.infer.init_to_median()
    )
    kw = {"init_loc_fn": init_loc_fn, "init_scale": init_scale}
    if kind == "full":
        return AutoMultivariateNormal(model, **kw)
    if kind == "lowrank":
        return AutoLowRankMultivariateNormal(model, rank=rank, **kw)
    if kind == "meanfield":
        return AutoNormal(model, **kw)
    raise ValueError(f"unknown VI guide kind {kind!r}; use full|lowrank|meanfield")


def draw_vi_posterior(guide, params: dict, n: int, seed: int = 1) -> dict:
    """Draw constrained samples from a fitted ADVI guide.

    Parameters
    ----------
    guide
        Fitted autoguide (same object used in SVI).
    params : dict
        `svi.get_params(state)` / `SVIRunResult.params`.
    n : int
        Number of draws.
    seed : int
        PRNG seed.

    Returns
    -------
    dict[str, np.ndarray]
        Site name → array of shape `(n,)`.
    """
    raw = guide.sample_posterior(jax.random.PRNGKey(seed), params, sample_shape=(n,))
    return {k: np.asarray(v).reshape(n) for k, v in raw.items()}


def run_map(
    data, meat_tx, pit_tx, num_steps: int = 300, lr: float = 2e-2, seed: int = 0, **kw
):
    """MAP via SVI + AutoDelta. Returns (params_dict, losses)."""
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    guide = AutoDelta(m)
    svi = SVI(m, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, progress_bar=True)
    params = guide.median(result.params)
    return {k: np.asarray(v) for k, v in params.items()}, np.asarray(result.losses)


def run_nuts(
    data,
    meat_tx,
    pit_tx,
    num_warmup: int = 300,
    num_samples: int = 300,
    max_tree_depth: int = 6,
    init_params: dict | None = None,
    seed: int = 0,
    **kw,
) -> MCMC:
    """Run full NUTS on the synthetic-cook model.

    Every leapfrog step pulls adjoint gradients across the Julia/JAX
    boundary. Initializes at `init_params` (typically the MAP) when given,
    else at the prior median.

    Parameters
    ----------
    data : dict
        Synthetic cook data (see synth.load).
    meat_tx, pit_tx : Tesseract
        Solver endpoints.
    num_warmup, num_samples : int
        NUTS phase lengths.
    max_tree_depth : int
        NUTS tree-depth cap (each doubling doubles gradient count).
    init_params : dict or None
        Site values to initialize from (e.g. the MAP).
    seed : int
        PRNG seed.
    **kw
        Forwarded to the model binding (grids, tols).

    Returns
    -------
    numpyro.infer.MCMC
        The completed sampler (samples via `.get_samples()`).
    """
    m = _bound_model(data, meat_tx, pit_tx, **kw)
    kernel = NUTS(
        m,
        max_tree_depth=max_tree_depth,
        init_strategy=(
            numpyro.infer.init_to_value(values=init_params)
            if init_params
            else numpyro.infer.init_to_median()
        ),
    )
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def recovery_table(map_params: dict, data: dict) -> str:
    """Human-readable MAP-vs-truth comparison (Milestone B1: within ~10-20%)."""
    # include ignition names so the zip stays aligned whichever era the
    # stored npz comes from (theta_true grew 5->7 in 2026-08-21; phi_true
    # carries inert ignition entries); sites absent from the MAP (ignition
    # on synthetic fits) are skipped rather than misaligned
    names = (
        [n for n, _ in THETA_SITES]
        + [n for n, _ in PHI_SITES]
        + [n for n, _ in PHI_SITES_IGNITION]
    )
    truths = np.concatenate([data["theta_true"], data["phi_true"]])
    if len(truths) != len(names):
        raise ValueError(
            f"truth vector length {len(truths)} != {len(names)} current sites; "
            "regenerate synthetic data under the current parameter schema"
        )
    lines = [f"{'param':>10} {'truth':>12} {'MAP':>12} {'err %':>8}"]
    for n, t in zip(names, truths, strict=True):
        if n not in map_params:
            continue
        m = float(map_params[n])
        lines.append(f"{n:>10} {t:>12.4g} {m:>12.4g} {100 * (m - t) / t:>7.1f}%")
    return "\n".join(lines)
