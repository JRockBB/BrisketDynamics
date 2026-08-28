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

# Pull blend width [s] — mirrors MeatConfig.δ_pull (the in-pit weighting of
# the bark-burn cap must use the same blend the meat integrates)
DELTA_PULL_S = 300.0
# initial pull logit: sigmoid(2.2) ≈ 0.9 of the cook horizon
PULL_Z_INIT = 2.2


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
        tough, saturation = pot roast, weighted equally (`over_rel`); the
        rest hold is modeled, so an early finish is a decision, not a gift.
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
    over_rel: float = 1.0  # overshoot severity relative to undershoot
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
    # --- food safety: SOFT PENALTIES, not a certification ---
    # Pasteurization is a time-temperature CURVE, not a threshold. The
    # z-value equivalence integrates SURFACE lethality
    #     F = integral 10^((T_s - T_ref)/z) dt   [T_ref-equivalent seconds]
    # FSIS Appendix A requires the stated condition in all parts of the
    # product; this model assumes INTACT MUSCLE (surface pathogens) and
    # does not evaluate the cold spot. Both terms are weighted softplus
    # penalties an optimizer can trade against quality — they are not
    # hard constraints, do not reject unsafe plans, and are not a
    # food-safety certification. See safety_report() for raw residuals.
    leth_t_ref: float = 333.15  # 60 C
    leth_z: float = 6.0  # z-value [K]
    leth_f_req: float = 720.0  # 60 C-equivalent seconds (~7-log)
    w_lethality: float = 500.0
    t_danger_lo: float = 277.6  # 4.4 C
    t_danger_hi: float = 333.15  # 60 C
    danger_cap_h: float = 4.0
    w_dwell: float = 500.0
    # ambient Kalman filter (MPC state estimation; see mpc.AmbientKF).
    # R: variance of the segment-end pit innovation — from the CALIBRATED
    # pit noise (sigma_0 + sigma_pit ~ 1.42 K at the real-cook AR(1) MAP).
    # Q: ambient random-walk intensity — weather drifts a few K per hour.
    # P0: prior variance of the initial ambient belief.
    amb_r: float = 3.4  # [K^2]
    amb_q: float = 0.0025  # [K^2 / s]
    amb_p0: float = 25.0  # [K^2]
    # receding-horizon consistency: fuel is normalized by a FIXED reference
    # load and banked lethality carries across replans, so the remaining-
    # horizon objective sums to the whole-cook objective (a replan that
    # re-baselines fuel fractions taxes late burning ever harder and coasts
    # the cook as an artifact — observed before this fix)
    m0_ref: float = 7.0  # [kg] PitConfig default load
    # decision variables / optimizer
    opt_wrap: bool = False  # optimize t_wrap jointly with the schedule
    # optimize the pull time jointly: t_pull = t_end * sigmoid(z_p); the meat
    # rests from the pull to the end of the horizon (serve), so the quality
    # terms are scored at serve
    opt_pull: bool = True
    lr: float = 0.05
    n_iter: int = 200  # iteration CAP; convergence stop below usually binds
    # plateau stop: quit when the best loss has not improved by more than
    # tol_rel * |best| for `patience` consecutive iterations (0 disables)
    tol_rel: float = 1e-4
    patience: int = 20


def _plateau_stop(losses, cc: ControlConfig) -> bool:
    """Convergence check: no improvement beyond tol_rel for `patience` iters.

    Compares the best loss inside the trailing window against the best
    before it; disabled when cc.patience == 0.
    """
    if cc.patience <= 0 or len(losses) <= cc.patience:
        return False
    best_before = min(losses[: -cc.patience])
    best_recent = min(losses[-cc.patience :])
    return best_recent > best_before - cc.tol_rel * abs(best_before)


def safety_report(tsurf, t, cc: ControlConfig, leth_banked=0.0, danger_banked_h=0.0):
    """Raw surface lethality / danger-zone dwell vs the configured thresholds.

    Soft objective penalties are NOT a certification; this reports the
    residuals. Assumes intact muscle (surface pathogens); the cold-spot
    field is not evaluated.
    """
    ts = np.asarray(tsurf, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    F = leth_banked + float(
        np.trapezoid(10.0 ** ((ts - cc.leth_t_ref) / cc.leth_z), tt)
    )
    in_zone = 1.0 / (1.0 + np.exp(-(ts - cc.t_danger_lo) / 2.0))
    in_zone *= 1.0 / (1.0 + np.exp(-(cc.t_danger_hi - ts) / 2.0))
    danger_h = danger_banked_h + float(np.trapezoid(in_zone, tt / 3600.0))
    return {
        "F_s": F,
        "danger_h": danger_h,
        "lethality_ok": F >= cc.leth_f_req,
        "dwell_ok": danger_h <= cc.danger_cap_h,
        "assumption": "intact muscle; surface lethality; soft penalties",
    }


def objective(
    z,
    z_w,
    theta,
    phi,
    tx,
    grids,
    cc: ControlConfig,
    tols,
    ics=None,
    leth_banked: float = 0.0,
    danger_banked_h: float = 0.0,
    t_wrap_override: float | None = None,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    z_p=None,
):
    """Scalar quality objective of one simulated cook (lower is better).

    KCBS-weighted appearance/taste/tenderness bands plus operational terms
    and the food-safety *soft penalties* (surface lethality + danger-zone
    dwell; intact-muscle assumption — see ControlConfig / safety_report).
    With ``t_pull`` inside the horizon the quality terms are evaluated at
    SERVE (the horizon end) — after the rest's carryover, continued
    collagen conversion and juice redistribution — while the operational
    terms price the cook only (fuel at the pull, bark-burn cap in the pit).

    Parameters
    ----------
    z : jnp.ndarray
        Unconstrained damper decision variables; u = sigmoid(z).
    z_w : jnp.ndarray
        Unconstrained wrap-time variable; t_wrap = t_end * sigmoid(z_w).
    theta, phi : jnp.ndarray
        Meat / pit parameter vectors.
    tx : Tesseract
        Solver endpoints.
    grids : CookGrids
        Time discretization.
    cc : ControlConfig
        Weights, bands, and constraint constants.
    tols : dict
        Solver tolerances.
    ics : dict, optional
        Initial conditions (receding-horizon replans).
    leth_banked : float, optional
        Lethality already accumulated by the executed cook [60C-equivalent
        seconds]; replans pass it so the safety constraint sees the whole
        cook, not just the remaining horizon.
    danger_banked_h : float, optional
        Danger-zone dwell already spent by the executed cook [h]; same
        whole-cook accounting as leth_banked (the 4 h cap is cumulative,
        not per-replan).
    t_wrap_override : float, optional
        Bypass the z_w mapping with an explicit wrap time [s], possibly
        negative (wrap in the past). Used by post-wrap MPC replans: the
        sigmoid map cannot represent an already-executed wrap, and pinning
        z_w merely puts the wrap shortly after segment start (a half-open
        blend for ~δ_wrap), not saturated-on.
    t_pull : float, optional
        Fixed pull time [s] used when ``cc.opt_pull`` is off or ``z_p`` is
        None (default never).
    rest : dict, optional
        Rest-environment overrides for the meat Tesseract
        (``rest_t/rest_h/rest_eps/rest_eta``); None = faux-cambro defaults.
    z_p : jnp.ndarray, optional
        Unconstrained pull-time variable; t_pull = t_end * sigmoid(z_p) when
        ``cc.opt_pull``. The meat rests from the pull to the end of the
        horizon (serve).

    Returns
    -------
    jnp.ndarray
        Scalar objective value.
    """
    u = jax.nn.sigmoid(z)
    # With opt_wrap: t_wrap = t_end * sigmoid(z_w) in (0, t_end).
    # Without: WRAP_NEVER (1e9) plus a dummy 0*z_w so the tangent stays live.
    # t_wrap_override (possibly negative = already wrapped) bypasses the map.
    t_wrap = (
        t_wrap_override
        if t_wrap_override is not None
        else (grids.t_end * jax.nn.sigmoid(z_w) if cc.opt_wrap else 1.0e9 + 0.0 * z_w)
    )
    if cc.opt_pull and z_p is not None:
        t_pull = grids.t_end * jax.nn.sigmoid(z_p)
    elif z_p is not None:
        t_pull = t_pull + 0.0 * z_p
    sim = coupled_cook(
        theta,
        phi,
        u,
        tx,
        grids,
        ics=ics,
        t_wrap=t_wrap,
        t_pull=t_pull,
        rest=rest,
        **tols,
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

    # operational terms; fuel normalized by the FIXED reference load so
    # replans price burning identically to the whole-cook objective. The
    # fire keeps burning after the pull, so price the fuel AT the pull
    # (jnp.interp clamps: t_pull = never reads the end of the horizon
    # exactly) and cap bark-burn in the pit only ((1 - pull blend) weight)
    t_s = jnp.asarray(grids.tknots)
    surf_over = jax.nn.softplus((sim["tsurf"] - cc.t_surf_max) / cc.dT_soft)
    in_pit = 1.0 - jax.nn.sigmoid((t_s - t_pull) / DELTA_PULL_S)
    surf_term = jnp.sum(in_pit * surf_over**2) / jnp.sum(in_pit)
    m_at_pull = jnp.interp(jnp.minimum(t_pull, t_s[-1]), t_s, sim["m_fuel"])
    fuel_frac = (sim["m_fuel"][0] - m_at_pull) / cc.m0_ref

    # food safety (unconditional): lethality curve + growth-phase dwell,
    # through the rest as well (a long board rest re-enters the zone)
    Ts = sim["tsurf"]
    F = leth_banked + jnp.trapezoid(10.0 ** ((Ts - cc.leth_t_ref) / cc.leth_z), t_s)
    leth_pen = (
        cc.w_lethality
        * jax.nn.softplus((jnp.log10(cc.leth_f_req) - jnp.log10(F + 1e-9)) / 0.2) ** 2
    )
    in_zone = jax.nn.sigmoid((Ts - cc.t_danger_lo) / 2.0) * jax.nn.sigmoid(
        (cc.t_danger_hi - Ts) / 2.0
    )
    danger_h = danger_banked_h + jnp.trapezoid(in_zone, t_s / 3600.0)
    dwell_pen = cc.w_dwell * jax.nn.softplus((danger_h - cc.danger_cap_h) / 0.5) ** 2

    return (
        cc.w_appearance * appearance
        + cc.w_taste * taste
        + cc.w_tenderness * tenderness
        + cc.w_surf * surf_term
        + cc.w_fuel * fuel_frac
        + cc.w_smooth * jnp.sum(jnp.diff(u) ** 2)
        + leth_pen
        + dwell_pen
    )


def optimize_schedule_expected(
    thetas,
    phis,
    tx,
    grids: CookGrids | None = None,
    cc: ControlConfig | None = None,
    tols: dict | None = None,
    u_init=0.5,
    log_every: int = 10,
    callback=None,
    z_w_init: float | None = None,
    z_p_init: float | None = None,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    ics: dict | None = None,
):
    """Optimize the damper schedule against the EXPECTED objective.

    Adam over N posterior draws (thetas/phis: arrays of shape
    (N, len(THETA_SITES)) / (N, 7)) — control under calibrated uncertainty.
    Sequential loop over draws inside the objective (vmap over Tesseract
    calls needs an explicit vmap_method and buys nothing at N=8); cost is
    N x the single-draw gradient. `u_init` may be a scalar or a schedule
    array (warm start). The wrap and pull moments are single decisions
    shared by all draws (you wrap and pull the real brisket once).
    """
    cc = cc or ControlConfig()
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    thetas = jnp.asarray(thetas)
    phis = jnp.asarray(phis)
    n = thetas.shape[0]

    def expected_obj(z, z_w, z_p, ths, phs):
        # differentiate w.r.t. ALL differentiable inputs (see optimize_schedule)
        return (
            sum(
                objective(
                    z,
                    z_w,
                    ths[i],
                    phs[i],
                    tx,
                    grids,
                    cc,
                    tols,
                    ics=ics,
                    t_pull=t_pull,
                    rest=rest,
                    z_p=z_p,
                )
                for i in range(n)
            )
            / n
        )

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    if z_w_init is None:
        z_w_init = 0.0 if cc.opt_wrap else 8.0
    z_w = jnp.asarray(z_w_init)
    z_p = jnp.asarray(PULL_Z_INIT if z_p_init is None else z_p_init)
    opt = optax.adam(cc.lr)
    state = opt.init((z, z_w, z_p))
    val_grad = jax.jit(jax.value_and_grad(expected_obj, argnums=(0, 1, 2, 3, 4)))

    losses = []
    best_loss = np.inf
    best_z, best_z_w, best_z_p = z, z_w, z_p
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        loss, (g, g_w, g_p, _, _) = val_grad(z, z_w, z_p, thetas, phis)
        lf = float(loss)
        losses.append(lf)
        if np.isfinite(lf) and lf < best_loss:
            best_loss, best_z, best_z_w, best_z_p = lf, z, z_w, z_p
        if not cc.opt_wrap:
            g_w = jnp.zeros_like(g_w)
        if not cc.opt_pull:
            g_p = jnp.zeros_like(g_p)
        updates, state = opt.update((g, g_w, g_p), state, (z, z_w, z_p))
        z, z_w, z_p = optax.apply_updates((z, z_w, z_p), updates)
        if callback and (i % log_every == 0 or i == cc.n_iter - 1):
            callback(i, lf, np.asarray(jax.nn.sigmoid(z)))
        if _plateau_stop(losses, cc):
            break
    tw = float(grids.t_end * jax.nn.sigmoid(best_z_w)) if cc.opt_wrap else 1.0e9
    tp = float(grids.t_end * jax.nn.sigmoid(best_z_p)) if cc.opt_pull else t_pull
    return np.asarray(jax.nn.sigmoid(best_z)), {
        "losses": np.asarray(losses),
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
        "t_wrap": tw,
        "t_pull": tp,
        "best_loss": best_loss,
        "converged": _plateau_stop(losses, cc),
        "n_iter_run": len(losses),
    }


def optimize_schedule(
    theta,
    phi,
    tx,
    grids: CookGrids | None = None,
    cc: ControlConfig | None = None,
    tols: dict | None = None,
    u_init=0.5,
    log_every: int = 20,
    callback=None,
    ics: dict | None = None,
    z_w_init: float = 0.0,
    z_w_pin: float = 8.0,
    leth_banked: float = 0.0,
    danger_banked_h: float = 0.0,
    t_wrap_override: float | None = None,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    z_p_init: float | None = None,
):
    """Optimize the damper schedule at fixed (theta, phi) with Adam.

    Returns (u_opt, history dict); the history carries losses, wall time,
    the grids, and the final t_wrap / t_pull. `callback(i, loss, u)` fires
    every `log_every` iterations. With ``cc.opt_wrap`` off, wrap physics is
    WRAP_NEVER (1e9); ``z_w`` still carries a dummy tangent. Post-wrap MPC
    replans pass ``t_wrap_override`` (possibly negative) instead of a pin.
    With ``cc.opt_pull`` off the pull is the fixed ``t_pull``.
    """
    cc = cc or ControlConfig()
    grids = grids or CookGrids(dt_exchange=600.0, n_u=16)
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    u0 = jnp.broadcast_to(jnp.asarray(u_init), (grids.n_u,))
    z = jnp.log(u0 / (1 - u0))
    # wrap decision variable. When opt_wrap is off, objective uses WRAP_NEVER
    # (1e9 + 0*z_w); z_w is still traced so one graph covers both modes.
    # z_w_pin is kept for callers that still pass it; it no longer changes
    # the physics.
    z_w = jnp.asarray(z_w_init if cc.opt_wrap else z_w_pin)
    z_p = jnp.asarray(PULL_Z_INIT if z_p_init is None else z_p_init)
    opt = optax.adam(cc.lr)
    state = opt.init((z, z_w, z_p))
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
            lambda zz, zw, zp, th, ph: objective(
                zz,
                zw,
                th,
                ph,
                tx,
                grids,
                cc,
                tols,
                ics=ics,
                leth_banked=leth_banked,
                danger_banked_h=danger_banked_h,
                t_wrap_override=t_wrap_override,
                t_pull=t_pull,
                rest=rest,
                z_p=zp,
            ),
            argnums=(0, 1, 2, 3, 4),
        )
    )

    losses, schedules = [], []
    best_loss = np.inf
    best_z, best_z_w, best_z_p = z, z_w, z_p
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        # Single retry: a rare timing-dependent race in the local-client
        # dispatch can hand one call a degenerate input (shape guard in the
        # tesseract api raises a clear error). One retry has always sufficed;
        # two consecutive failures is a real bug worth surfacing.
        try:
            loss, (g, g_w, g_p, _g_theta, _g_phi) = val_grad(z, z_w, z_p, theta, phi)
        except RuntimeError as e:
            if "DEGENERATE" not in str(e) and "BoundsError" not in str(e):
                raise
            print(f"  iter {i}: degenerate-input race, retrying once ({e})", flush=True)
            loss, (g, g_w, g_p, _g_theta, _g_phi) = val_grad(z, z_w, z_p, theta, phi)
        lf = float(loss)
        losses.append(lf)
        if np.isfinite(lf) and lf < best_loss:
            best_loss, best_z, best_z_w, best_z_p = lf, z, z_w, z_p
        if not cc.opt_wrap:
            g_w = jnp.zeros_like(g_w)
        if not cc.opt_pull:
            g_p = jnp.zeros_like(g_p)
        updates, state = opt.update((g, g_w, g_p), state, (z, z_w, z_p))
        z, z_w, z_p = optax.apply_updates((z, z_w, z_p), updates)
        if i % log_every == 0 or i == cc.n_iter - 1:
            schedules.append(np.asarray(jax.nn.sigmoid(best_z)))
            if callback:
                callback(i, lf, np.asarray(jax.nn.sigmoid(best_z)))
        if _plateau_stop(losses, cc):
            break
    if t_wrap_override is not None:
        tw = float(t_wrap_override)
    elif cc.opt_wrap:
        tw = float(grids.t_end * jax.nn.sigmoid(best_z_w))
    else:
        tw = 1.0e9
    tp = float(grids.t_end * jax.nn.sigmoid(best_z_p)) if cc.opt_pull else t_pull
    return np.asarray(jax.nn.sigmoid(best_z)), {
        "losses": np.asarray(losses),
        "schedules": np.asarray(schedules),
        "t_wrap": tw,
        "t_pull": tp,
        "best_loss": best_loss,
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
        "converged": _plateau_stop(losses, cc),
        "n_iter_run": len(losses),
    }
