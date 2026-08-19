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
    # Texas-crutch wrap decision (differentiable t_wrap through the meat BC).
    # bark proxy = trapezoid(j_evap): integrated evaporative flux; post-wrap
    # flux is suppressed in-model, so wrapping early starves bark formation.
    # The asymmetric shortfall penalty makes "when to wrap" a real trade-off
    # (else moisture+speed always favor wrapping at t=0).
    opt_wrap: bool = False       # optimize t_wrap jointly with the schedule
    bark_target: float = 0.0     # target integral of j_evap [kg/m2]; 0 = off
    w_bark: float = 0.0
    dbark_soft: float = 0.05     # penalty width, fraction of bark_target
    # Asymmetric terminal-doneness penalty (MPC): undershooting the target at
    # the deadline is the disaster; finishing early is nearly free (the hold
    # exists for this). Symmetric quadratics 'give up' late in a losing cook:
    # thermal lag kills the doneness gradient and fuel penalties take over.
    asymmetric_done: bool = False
    lr: float = 0.05
    n_iter: int = 200


def objective(z, z_w, theta, phi, meat_tx, pit_tx, grids, cc: ControlConfig,
              tols, ics=None):
    u = jax.nn.sigmoid(z)
    # bounded wrap time: t_end * sigmoid(z_w) in (0, t_end); wrapping at
    # ~t_end is physically a no-wrap, so that corner stays reachable
    t_wrap = grids.t_end * jax.nn.sigmoid(z_w)
    sim = coupled_cook(theta, phi, u, meat_tx, pit_tx, grids, K=2, ics=ics,
                       t_wrap=t_wrap, **tols)
    c_core = sim["c_field_final"][0]                     # center node
    surf_over = jax.nn.softplus((sim["tsurf"] - cc.t_surf_max) / cc.dT_soft)
    fuel_frac = (sim["m_fuel"][0] - sim["m_fuel"][-1]) / sim["m_fuel"][0]
    if cc.asymmetric_done:
        k = 20.0  # undershoot-only, smooth: ~(target-c)^2 under, ~0 over
        done_term = (jax.nn.softplus(k * (cc.c_target - c_core)) / k) ** 2
    else:
        done_term = (c_core - cc.c_target) ** 2
    j = (cc.w_done * done_term
         + cc.w_surf * jnp.mean(surf_over**2)
         + cc.w_fuel * fuel_frac
         + cc.w_smooth * jnp.sum(jnp.diff(u) ** 2)
         - cc.w_juicy * sim["w_retained"])
    if cc.bark_target > 0.0:  # static config branch: jit-safe
        bark = jnp.trapezoid(sim["j_evap"], jnp.asarray(grids.tknots))
        shortfall = (cc.bark_target - bark) / (cc.dbark_soft * cc.bark_target)
        j = j + cc.w_bark * (cc.dbark_soft * jax.nn.softplus(shortfall)) ** 2
    return j


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

    def expected_obj(z, z_w, ths, phs):
        # differentiate w.r.t. ALL differentiable inputs (see optimize_schedule)
        return sum(objective(z, z_w, ths[i], phs[i], meat_tx, pit_tx, grids,
                             cc, tols)
                   for i in range(n)) / n

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    z_w = jnp.asarray(8.0)   # pinned no-wrap; tangent still flows (see above)
    opt = optax.adam(cc.lr)
    state = opt.init(z)
    val_grad = jax.jit(jax.value_and_grad(expected_obj, argnums=(0, 1, 2, 3)))

    losses = []
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        loss, (g, _gw, _, _) = val_grad(z, z_w, thetas, phis)
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
                      u_init=0.5,
                      log_every: int = 20,
                      callback=None,
                      ics: dict | None = None,
                      z_w_init: float = 0.0):
    """Adam on the damper schedule at fixed (theta, phi). Returns
    (u_opt, history dict). `callback(i, loss, u)` fires every `log_every`."""
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    # wrap decision variable: t_wrap = t_end*sigmoid(z_w). When opt_wrap is
    # off, z_w is pinned near-never-wrap (sigmoid(8) = 0.9997 -> t_wrap at
    # the horizon end, physically no wrap) and its updates are discarded --
    # but it ALWAYS carries a tangent, keeping one traced graph and the
    # all-inputs-tangent invariant (see below).
    z_w = jnp.asarray(z_w_init if cc.opt_wrap else 8.0)
    opt = optax.adam(cc.lr)
    state = opt.init((z, z_w))
    # Two deliberate choices here:
    # 1. Differentiate w.r.t. ALL differentiable Tesseract inputs (z via u,
    #    z_w via t_wrap, theta, phi) and discard the gradients you don't
    #    need — one traced graph regardless of which decision variables are
    #    active, and no per-caller special-casing of tangent structure.
    # 2. jax.jit the whole value_and_grad: the EAGER AD path re-linearizes
    #    per call and has a timing-dependent race that hands one Tesseract a
    #    degenerate input (observed: 1-element tpit_knots at the endpoint,
    #    sticky once triggered). Under jit the graph is traced once and
    #    replayed with fixed shapes — the jitted SVI/MAP path has never
    #    exhibited the race across thousands of calls.
    val_grad = jax.jit(jax.value_and_grad(
        lambda zz, zw, th, ph: objective(zz, zw, th, ph, meat_tx, pit_tx,
                                         grids, cc, tols, ics=ics),
        argnums=(0, 1, 2, 3)))

    losses, schedules = [], []
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        # Single retry: a rare timing-dependent race in the local-client
        # dispatch can hand one call a degenerate input (shape guard in the
        # tesseract api raises a clear error). One retry has always sufficed;
        # two consecutive failures is a real bug worth surfacing.
        try:
            loss, (g, g_w, _g_theta, _g_phi) = val_grad(z, z_w, theta, phi)
        except RuntimeError as e:
            if "DEGENERATE" not in str(e) and "BoundsError" not in str(e):
                raise
            print(f"  iter {i}: degenerate-input race, retrying once ({e})",
                  flush=True)
            loss, (g, g_w, _g_theta, _g_phi) = val_grad(z, z_w, theta, phi)
        if not cc.opt_wrap:
            g_w = jnp.zeros_like(g_w)
        updates, state = opt.update((g, g_w), state, (z, z_w))
        z, z_w = optax.apply_updates((z, z_w), updates)
        losses.append(float(loss))
        if i % log_every == 0 or i == cc.n_iter - 1:
            schedules.append(np.asarray(jax.nn.sigmoid(z)))
            if callback:
                callback(i, float(loss), np.asarray(jax.nn.sigmoid(z)))
    return np.asarray(jax.nn.sigmoid(z)), {
        "losses": np.asarray(losses),
        "schedules": np.asarray(schedules),
        "t_wrap": float(grids.t_end * jax.nn.sigmoid(z_w)),
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
    }
