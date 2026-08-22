"""Optimal cook control.

Optax damper-schedule optimization through the coupled differentiable
pipeline (the same `coupled_cook` used by calibration).

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
    """The quality objective for the coupled cook.

    The DEFAULT objective is the KCBS-weighted quality vector — appearance /
    taste / tenderness at the multipliers real judges' scores carry
    (0.56 / 2.2972 / 1.1428, normalized 0.14 / 0.57 / 0.29) — over the
    subset of each category the simulation observes, plus operational terms
    and the food-safety constraints. Every weight and band is a plain field:
    a different preference is a different config, not a different mode.

    HONEST TAXONOMY (docs/model_gaps.md is the standing audit):
      tenderness — mechanistic proxy: collagen conversion IS the physics of
        probe-tender; the C = 0.90 scale is literature kinetics,
        uncalibrated against texture data. Two-sided band: undershoot =
        tough, saturation = pot roast; overshoot is discounted by
        `over_rel` because early finishes are absorbed by the resting hold.
      taste — two correlates: smoke-phenolic mass (deposited guaiacol /
        syringol; band = depth without acrid over-deposition; smoke
        QUALITY clean-vs-dirty is not modeled) and perceptual juiciness
        (retained water + β_fat-weighted retained rendered fat from the
        fat-rendering/expulsion model — gap audit #1/#2 DONE; β_fat is a
        sensory placeholder pending the scale experiment).
        Seasoning, sweetness, beef flavor: absent.
      appearance — predicted color only (two-sided mahogany L* band;
        judges ding black bark as burnt-looking). Uniformity, sheen,
        presentation: absent.
    """

    # category weights — KCBS-cited defaults, override at will
    w_appearance: float = 0.14
    w_taste: float = 0.57
    w_tenderness: float = 0.29
    # tenderness band (collagen conversion at the center node)
    c_target: float = 0.90
    c_overdone: float = 0.95
    over_rel: float = 0.5  # overshoot severity relative to undershoot
    # appearance band (predicted surface CIELAB L*)
    l_mahogany_lo: float = 28.0  # blacker reads burnt
    l_mahogany_hi: float = 34.0  # paler reads underdeveloped
    dl_soft: float = 2.0
    # taste: smoke-deposition band + dryness
    s_smoke_lo: float = 0.55
    s_smoke_hi: float = 0.90
    ds_soft: float = 0.08
    w_dry: float = 4.0
    # operational terms
    t_surf_max: float = 368.15  # 95 C bark-burn limit [K]
    dT_soft: float = 2.0
    w_surf: float = 1.0
    w_fuel: float = 0.1
    w_smooth: float = 0.01
    # --- food safety: LETHALITY-CURVE constraint (gap audit Tier 1 #4) ---
    # Pasteurization is a time-temperature CURVE, not a threshold. The
    # z-value equivalence integrates surface lethality
    #     F = integral 10^((T_s - T_ref)/z) dt   [T_ref-equivalent seconds]
    # FSIS guidance for ~7-log Salmonella reduction in beef: ~12 min at
    # 60 C with z ~ 6 C. The constraint requires F >= leth_f_req by end of
    # cook, penalized on the log10 shortfall (curve-faithful: any
    # time-temperature path achieving the required lethality passes). A
    # danger-zone dwell cap separately guards the GROWTH phase before
    # lethal temperatures are reached. Both computed from the existing
    # surface-temperature output; applied unconditionally.
    leth_t_ref: float = 333.15  # 60 C
    leth_z: float = 6.0  # z-value [K]
    leth_f_req: float = 720.0  # 60 C-equivalent seconds (~7-log)
    w_lethality: float = 500.0
    t_danger_lo: float = 277.6  # 4.4 C
    t_danger_hi: float = 333.15  # 60 C
    danger_cap_h: float = 4.0
    w_dwell: float = 500.0
    # decision variables / optimizer
    opt_wrap: bool = False  # optimize t_wrap jointly with the schedule
    lr: float = 0.05
    n_iter: int = 200


def objective(
    z, z_w, theta, phi, meat_tx, pit_tx, grids, cc: ControlConfig, tols, ics=None
):
    """Scalar quality objective of one simulated cook (lower is better).

    KCBS-weighted appearance/taste/tenderness bands plus operational terms
    and the unconditional food-safety constraints (lethality curve +
    danger-zone dwell); see ControlConfig for every weight and band.

    Parameters
    ----------
    z : jnp.ndarray
        Unconstrained damper decision variables; u = sigmoid(z).
    z_w : jnp.ndarray
        Unconstrained wrap-time variable; t_wrap = t_end * sigmoid(z_w).
    theta, phi : jnp.ndarray
        Meat / pit parameter vectors.
    meat_tx, pit_tx : Tesseract
        Solver endpoints.
    grids : CookGrids
        Time discretization.
    cc : ControlConfig
        Weights, bands, and constraint constants.
    tols : dict
        Solver tolerances.
    ics : dict, optional
        Initial conditions (receding-horizon replans).

    Returns
    -------
    jnp.ndarray
        Scalar objective value.
    """
    u = jax.nn.sigmoid(z)
    # bounded wrap time: t_end * sigmoid(z_w) in (0, t_end); wrapping at
    # ~t_end is physically a no-wrap, so that corner stays reachable
    t_wrap = grids.t_end * jax.nn.sigmoid(z_w)
    sim = coupled_cook(
        theta, phi, u, meat_tx, pit_tx, grids, K=2, ics=ics, t_wrap=t_wrap, **tols
    )

    # tenderness: two-sided collagen band, each side normalized to O(1)
    # per its band width (undershoot 0.05; overshoot 1 - c_overdone)
    c_core = sim["c_field_final"][0]
    k = 20.0
    under = (jax.nn.softplus(k * (cc.c_target - c_core)) / k) / 0.05
    band_o = 1.0 - cc.c_overdone
    k_ov = 4.0 / band_o
    over = (jax.nn.softplus(k_ov * (c_core - cc.c_overdone)) / k_ov) / band_o
    tenderness = under**2 + cc.over_rel * over**2

    # appearance: two-sided mahogany band on predicted final L*
    Lst = sim["l_star"][-1]
    appearance = (
        jax.nn.softplus((Lst - cc.l_mahogany_hi) / cc.dl_soft) ** 2
        + jax.nn.softplus((cc.l_mahogany_lo - Lst) / cc.dl_soft) ** 2
    )

    # taste: smoke-depth band + dryness (see taxonomy)
    S_end = sim["bark_s"][-1]
    smoke_band = (
        jax.nn.softplus((cc.s_smoke_lo - S_end) / cc.ds_soft) ** 2
        + jax.nn.softplus((S_end - cc.s_smoke_hi) / cc.ds_soft) ** 2
    )
    # dryness now uses perceptual juiciness (water + rendered fat) —
    # moisture_plan.md; w_retained remains available for the scale story
    taste = smoke_band + cc.w_dry * (1.0 - sim["juiciness"]) ** 2

    # operational terms
    surf_over = jax.nn.softplus((sim["tsurf"] - cc.t_surf_max) / cc.dT_soft)
    fuel_frac = (sim["m_fuel"][0] - sim["m_fuel"][-1]) / sim["m_fuel"][0]

    # food safety (unconditional): lethality curve + growth-phase dwell
    Ts = sim["tsurf"]
    t_s = jnp.asarray(grids.tknots)
    F = jnp.trapezoid(10.0 ** ((Ts - cc.leth_t_ref) / cc.leth_z), t_s)
    leth_pen = (
        cc.w_lethality
        * jax.nn.softplus((jnp.log10(cc.leth_f_req) - jnp.log10(F + 1e-9)) / 0.2) ** 2
    )
    in_zone = jax.nn.sigmoid((Ts - cc.t_danger_lo) / 2.0) * jax.nn.sigmoid(
        (cc.t_danger_hi - Ts) / 2.0
    )
    danger_h = jnp.trapezoid(in_zone, t_s / 3600.0)
    dwell_pen = cc.w_dwell * jax.nn.softplus((danger_h - cc.danger_cap_h) / 0.5) ** 2

    return (
        cc.w_appearance * appearance
        + cc.w_taste * taste
        + cc.w_tenderness * tenderness
        + cc.w_surf * jnp.mean(surf_over**2)
        + cc.w_fuel * fuel_frac
        + cc.w_smooth * jnp.sum(jnp.diff(u) ** 2)
        + leth_pen
        + dwell_pen
    )


def optimize_schedule_expected(
    thetas,
    phis,
    meat_tx,
    pit_tx,
    grids: CookGrids | None = None,
    cc: ControlConfig | None = None,
    tols: dict | None = None,
    u_init=0.5,
    log_every: int = 10,
    callback=None,
):
    """Optimize the damper schedule against the EXPECTED objective.

    Adam over N posterior draws (thetas/phis: arrays of shape
    (N, len(THETA_SITES)) / (N, 7)) — control under calibrated uncertainty.
    Sequential loop over draws inside the objective (vmap over Tesseract
    calls needs an explicit vmap_method and buys nothing at N=8); cost is
    N x the single-draw gradient. `u_init` may be a scalar or a schedule
    array (warm start).
    """
    cc = cc or ControlConfig()
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    thetas = jnp.asarray(thetas)
    phis = jnp.asarray(phis)
    n = thetas.shape[0]

    def expected_obj(z, z_w, ths, phs):
        # differentiate w.r.t. ALL differentiable inputs (see optimize_schedule)
        return (
            sum(
                objective(z, z_w, ths[i], phs[i], meat_tx, pit_tx, grids, cc, tols)
                for i in range(n)
            )
            / n
        )

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    z_w = jnp.asarray(8.0)  # pinned no-wrap; tangent still flows (see above)
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


def optimize_schedule(
    theta,
    phi,
    meat_tx,
    pit_tx,
    grids: CookGrids | None = None,
    cc: ControlConfig | None = None,
    tols: dict | None = None,
    u_init=0.5,
    log_every: int = 20,
    callback=None,
    ics: dict | None = None,
    z_w_init: float = 0.0,
    z_w_pin: float = 8.0,
):
    """Optimize the damper schedule at fixed (theta, phi) with Adam.

    Returns (u_opt, history dict); the history carries losses, wall time,
    the grids, and the final t_wrap. `callback(i, loss, u)` fires every
    `log_every` iterations. With ``cc.opt_wrap`` off, ``z_w`` is pinned at
    ``z_w_pin`` (8 = never wrap; -8 = wrapped from the start, used by MPC
    replans after the crutch is executed).
    """
    cc = cc or ControlConfig()
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    # wrap decision variable: t_wrap = t_end*sigmoid(z_w). When opt_wrap is
    # off, z_w is PINNED at z_w_pin and its updates are discarded -- but it
    # ALWAYS carries a tangent, keeping one traced graph and the
    # all-inputs-tangent invariant (see below). z_w_pin defaults to
    # near-never-wrap (sigmoid(8) = 0.9997 -> wrap at the horizon end);
    # MPC replans of an ALREADY-WRAPPED cook pin at -8 instead
    # (wrap at ~t=0, i.e. the blend is on from the segment start).
    z_w = jnp.asarray(z_w_init if cc.opt_wrap else z_w_pin)
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
    val_grad = jax.jit(
        jax.value_and_grad(
            lambda zz, zw, th, ph: objective(
                zz, zw, th, ph, meat_tx, pit_tx, grids, cc, tols, ics=ics
            ),
            argnums=(0, 1, 2, 3),
        )
    )

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
            print(f"  iter {i}: degenerate-input race, retrying once ({e})", flush=True)
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
