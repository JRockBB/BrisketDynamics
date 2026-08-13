"""Optimal cook control: Optax damper-schedule optimization through the
coupled differentiable pipeline (the same `coupled_cook` used by calibration).

Objective (defaults from the project plan, D4): hit target core doneness by
the deadline, penalize bark-burning surface temperatures, fuel use, and
schedule roughness. Decision variables are unconstrained reals mapped to
damper fractions u in [0, 1] by a sigmoid, so bounds hold by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .coupled import CookGrids, coupled_cook


@dataclass(frozen=True)
class ControlConfig:
    c_target: float = 0.90       # core collagen conversion at t_f
    t_surf_max: float = 368.15   # 95 C bark-burn proxy [K]
    dT_soft: float = 2.0         # softplus width for the surface penalty [K]
    w_done: float = 100.0
    w_surf: float = 1.0
    w_fuel: float = 0.1
    w_smooth: float = 0.01
    w_juicy: float = 0.0   # optional: reward retained moisture (juiciness proxy)
    lr: float = 0.05
    n_iter: int = 200


def objective(z, theta, phi, meat_tx, pit_tx, grids, cc: ControlConfig, tols):
    u = jax.nn.sigmoid(z)
    sim = coupled_cook(theta, phi, u, meat_tx, pit_tx, grids, K=2, **tols)
    c_core = sim["c_field_final"][0]                     # center node
    surf_over = jax.nn.softplus((sim["tsurf"] - cc.t_surf_max) / cc.dT_soft)
    fuel_frac = (sim["m_fuel"][0] - sim["m_fuel"][-1]) / sim["m_fuel"][0]
    return (cc.w_done * (c_core - cc.c_target) ** 2
            + cc.w_surf * jnp.mean(surf_over**2)
            + cc.w_fuel * fuel_frac
            + cc.w_smooth * jnp.sum(jnp.diff(u) ** 2)
            - cc.w_juicy * sim["w_retained"])


def optimize_schedule_expected(thetas, phis, meat_tx, pit_tx,
                               grids: CookGrids | None = None,
                               cc: ControlConfig = ControlConfig(),
                               tols: dict | None = None,
                               u_init=0.5,
                               log_every: int = 10,
                               callback=None):
    """Adam on the damper schedule against the EXPECTED objective over N
    posterior draws (thetas, phis: arrays of shape (N, 5)) — control under
    calibrated uncertainty. Sequential loop over draws inside the objective
    (vmap over Tesseract calls needs an explicit vmap_method and buys nothing
    at N=8); cost is N x the single-draw gradient. `u_init` may be a scalar
    or a schedule array (warm start)."""
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    thetas = jnp.asarray(thetas)
    phis = jnp.asarray(phis)
    n = thetas.shape[0]

    def expected_obj(z, ths, phs):
        # differentiate w.r.t. ALL differentiable inputs (see optimize_schedule)
        return sum(objective(z, ths[i], phs[i], meat_tx, pit_tx, grids, cc, tols)
                   for i in range(n)) / n

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    opt = optax.adam(cc.lr)
    state = opt.init(z)
    val_grad = jax.jit(jax.value_and_grad(expected_obj, argnums=(0, 1, 2)))

    losses = []
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        loss, (g, _, _) = val_grad(z, thetas, phis)
        updates, state = opt.update(g, state, z)
        z = optax.apply_updates(z, updates)
        losses.append(float(loss))
        if callback and (i % log_every == 0 or i == cc.n_iter - 1):
            callback(i, float(loss), np.asarray(jax.nn.sigmoid(z)))
    return np.asarray(jax.nn.sigmoid(z)), {
        "losses": np.asarray(losses),
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
    }


def optimize_schedule(theta, phi, meat_tx, pit_tx,
                      grids: CookGrids | None = None,
                      cc: ControlConfig = ControlConfig(),
                      tols: dict | None = None,
                      u_init: float = 0.5,
                      log_every: int = 20,
                      callback=None):
    """Adam on the damper schedule at fixed (theta, phi). Returns
    (u_opt, history dict). `callback(i, loss, u)` fires every `log_every`."""
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    z = jnp.full(grids.n_u, jnp.log(u_init / (1 - u_init)))  # logit(u_init)
    opt = optax.adam(cc.lr)
    state = opt.init(z)
    # Two hard-won choices here:
    # 1. Differentiate w.r.t. ALL differentiable Tesseract inputs (z via u,
    #    theta, phi) and use only the z gradient — tesseract-jax's JVP-rule
    #    input flattening misaligns when only a subset of differentiable
    #    inputs carries tangents.
    # 2. jax.jit the whole value_and_grad: the EAGER AD path re-linearizes
    #    per call and has a timing-dependent race that hands one Tesseract a
    #    degenerate input (observed: 1-element tpit_knots at the endpoint,
    #    sticky once triggered). Under jit the graph is traced once and
    #    replayed with fixed shapes — the jitted SVI/MAP path has never
    #    exhibited the race across thousands of calls.
    val_grad = jax.jit(jax.value_and_grad(
        lambda zz, th, ph: objective(zz, th, ph, meat_tx, pit_tx, grids, cc, tols),
        argnums=(0, 1, 2)))

    losses, schedules = [], []
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        # Single retry: a rare timing-dependent race in the local-client
        # dispatch can hand one call a degenerate input (shape guard in the
        # tesseract api raises a clear error). One retry has always sufficed;
        # two consecutive failures is a real bug worth surfacing.
        try:
            loss, (g, _g_theta, _g_phi) = val_grad(z, theta, phi)
        except RuntimeError as e:
            if "DEGENERATE" not in str(e) and "BoundsError" not in str(e):
                raise
            print(f"  iter {i}: degenerate-input race, retrying once ({e})",
                  flush=True)
            loss, (g, _g_theta, _g_phi) = val_grad(z, theta, phi)
        updates, state = opt.update(g, state, z)
        z = optax.apply_updates(z, updates)
        losses.append(float(loss))
        if i % log_every == 0 or i == cc.n_iter - 1:
            schedules.append(np.asarray(jax.nn.sigmoid(z)))
            if callback:
                callback(i, float(loss), np.asarray(jax.nn.sigmoid(z)))
    return np.asarray(jax.nn.sigmoid(z)), {
        "losses": np.asarray(losses),
        "schedules": np.asarray(schedules),
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
    }
