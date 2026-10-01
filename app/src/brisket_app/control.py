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
# fuel addition / spritz start: t = t_end * sigmoid(z), same map as the
# pull, initialised late in the horizon (a near-null decision the gradient
# can pull earlier); the spritz interval is a bounded sigmoid map
ADD_Z_INIT = 2.2
SPRITZ_Z_INIT = 2.2
SPRITZ_DT_Z_INIT = 0.0
SPRITZ_DOSE_Z_INIT = 0.0
# salting / wood-split decisions start mid-range
SALT_Z_INIT = 0.0
SPLIT_Z_INIT = 0.0
TAU_ADD_S = 300.0  # PitConfig.τ_add: the chimney dump duration


@dataclass(frozen=True)
class ControlConfig:
    """The quality objective for the coupled cook.

    The DEFAULT objective is the KCBS-weighted quality vector — appearance /
    taste / tenderness at the multipliers real judges' scores carry
    (0.56 / 2.2972 / 1.1428, normalized 0.14 / 0.57 / 0.29) — over the
    subset of each category the simulation observes, plus operational terms
    and the food-safety constraints. Every weight and band is a plain field:
    a different preference is a different config, not a different mode.

    HONEST TAXONOMY (docs/roadmap.md lists the gaps and residues):
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
    # z-value equivalence integrates lethality
    #     F = integral 10^((T - T_ref)/z) dt   [T_ref-equivalent seconds]
    # on the SURFACE (intact-muscle contamination) and on the CORE node —
    # the cold spot of the symmetric slab, the last point to heat — since
    # FSIS Appendix A requires the condition in all parts of the product.
    # The requirement is the Appendix A 6.5-log Salmonella line (60 C for
    # 12 min). Appendix A also conditions the table on the SURFACE
    # HUMIDITY during the cook (relative humidity / sealed-oven rules);
    # this model does not represent that condition and does not certify
    # it — the report says so.
    #
    # Heating come-up time (CUT): FSIS Appendix A (2021) limits the time
    # the product spends between 50 and 130 F (10-54.4 C) to 6 h against
    # S. aureus enterotoxin. It is reported on the surface and the core.
    # FSIS Scientific Gap 6 names BRISKET explicitly: a whole-muscle beef
    # cut cooked low-and-slow is EXPECTED to exceed a 6 h come-up, and the
    # CUT is not a critical operating parameter for it until more research
    # exists. So the DEFAULT (brisket) keeps the CUT a report: never a
    # penalty, never a rejection. Set cut_critical=True for products where
    # Appendix A's CUT applies (poultry, comminuted product): then it is
    # penalized and the fullplan scan rejects on it. The old consumer
    # "danger zone 40-140 F, 4 h" cap was a HOLDING rule for cooked food
    # left out, not a roasting limit, and was removed (docs/review.md).
    #
    # The terms are weighted softplus penalties an optimizer can trade
    # against quality; the fullplan scan rejects cook lengths whose raw
    # residuals fail (safety_report), the MPC only penalizes. Not a
    # certification.
    leth_t_ref: float = 333.15  # 60 C
    leth_z: float = 6.0  # z-value [K]
    leth_f_req: float = 720.0  # 60 C-equivalent seconds (FSIS Appendix A 6.5-log line)
    w_lethality: float = 500.0
    cut_t_lo: float = 283.15  # 10 C (50 F)
    cut_t_hi: float = 327.6  # 54.4 C (130 F)
    cut_cap_h: float = 6.0  # Appendix A come-up limit [h]
    w_cut: float = 500.0  # applied only when cut_critical
    cut_critical: bool = False  # brisket: FSIS Scientific Gap 6 — report, do not bind
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
    # mid-cook fuel addition (one chimney, PitConfig.m_chimney) and the
    # spritz policy (first spritz + interval): t = t_end * sigmoid(z) inside
    # the horizon, never beyond it — the same construction as wrap and pull
    opt_add: bool = True
    m_chimney: float = 1.4  # [kg] PitConfig.m_chimney; added fuel is priced at the pull
    opt_spritz: bool = True
    dt_spritz_min: float = 1200.0  # [s] spritz interval bounds
    dt_spritz_max: float = 7200.0
    m_spritz_min: float = 0.02  # [kg/m²] spritz dose bounds (a light to a heavy spray)
    m_spritz_max: float = 0.10
    # the M1 planner decisions (all OFF by default; the salt decision has
    # no saltiness counterweight in the objective yet -- enabling opt_salt
    # pins t_salt at its bound, the spritz lesson all over again)
    opt_salt: bool = False  # optimize the dry-brine time t_salt
    t_salt_max: float = 48 * 3600.0  # [s] dry-brine bound (two days)
    opt_splits: bool = False  # optimize the wood-split interval dt_add
    dt_add_min: float = 1800.0  # [s] split-interval bounds
    dt_add_max: float = 4 * 3600.0
    n_split_cap: int = 24  # pulse-train pricing terms (covers t_end/dt_add_min)
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


def _softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


SAFETY_DISCLAIMER = (
    "model output, not a USDA-certified cook: lethality is a soft curve "
    "check on a modelled surface and core, the Appendix A humidity "
    "condition is not modelled, the come-up time is reported and not "
    "binding for brisket (FSIS Scientific Gap 6); quality is not a KCBS "
    "judge score and the schedule is not a 95% on-time guarantee"
)


def lethality_cut(temps, t_s, cc: ControlConfig, xp=np):
    """Lethality F [T_ref-equivalent s] and come-up time [h] of a temperature trace.

    The come-up time is the time spent between `cut_t_lo` and `cut_t_hi`
    (FSIS Appendix A: 50-130 F), as a smooth window. `xp` is numpy or
    jax.numpy; the integrals are trapezoids on `t_s` [s].
    """
    F = xp.trapezoid(10.0 ** ((temps - cc.leth_t_ref) / cc.leth_z), t_s)
    lo = 1.0 / (1.0 + xp.exp(-(temps - cc.cut_t_lo) / 2.0))
    hi = 1.0 / (1.0 + xp.exp(-(cc.cut_t_hi - temps) / 2.0))
    return F, xp.trapezoid(lo * hi, t_s / 3600.0)


def safety_penalty(F, cut_h, cc: ControlConfig, xp=np):
    """Soft penalties: lethality shortfall (log scale) plus, when critical, the come-up excess.

    The come-up term enters only when `cc.cut_critical` (never for the
    brisket default, FSIS Scientific Gap 6).
    """
    sp = jax.nn.softplus if xp is jnp else _softplus_np
    leth = (
        cc.w_lethality * sp((xp.log10(cc.leth_f_req) - xp.log10(F + 1e-9)) / 0.2) ** 2
    )
    if not cc.cut_critical:
        return leth
    return leth + cc.w_cut * sp((cut_h - cc.cut_cap_h) / 0.5) ** 2


def safety_report(tsurf, tcore, t, cc: ControlConfig, banked: dict | None = None):
    """Raw lethality / come-up time of the surface and the core vs the thresholds.

    The core node is the cold spot of the symmetric slab. `banked` carries
    an executed cook's integrals (keys leth_f, cut_h, leth_f_core,
    cut_h_core) into the whole-cook accounting. `ok` is the lethality
    check, plus the come-up cap only when `cc.cut_critical` (never for
    the brisket default — FSIS Scientific Gap 6). Soft objective penalties
    are NOT a certification; this reports the residuals.
    """
    b = banked or {}
    tt = np.asarray(t, dtype=np.float64)
    F_s, c_s = lethality_cut(np.asarray(tsurf, dtype=np.float64), tt, cc)
    F_c, c_c = lethality_cut(np.asarray(tcore, dtype=np.float64), tt, cc)
    F_s = float(F_s) + b.get("leth_f", 0.0)
    F_c = float(F_c) + b.get("leth_f_core", 0.0)
    c_s = float(c_s) + b.get("cut_h", 0.0)
    c_c = float(c_c) + b.get("cut_h_core", 0.0)
    lethality_ok = bool(F_s >= cc.leth_f_req and F_c >= cc.leth_f_req)
    cut_ok = bool(c_s <= cc.cut_cap_h and c_c <= cc.cut_cap_h)
    return {
        "F_s": F_s,
        "F_core": F_c,
        "cut_h": c_s,
        "cut_h_core": c_c,
        "lethality_ok": lethality_ok,
        "cut_ok": cut_ok,
        "cut_critical": bool(cc.cut_critical),
        "ok": lethality_ok and (cut_ok or not cc.cut_critical),
        "assumption": (
            "1D slab: core node is the cold spot; soft penalties, not a "
            "certification; Appendix A humidity not modelled; come-up time "
            + ("binding" if cc.cut_critical else "reported, not binding (brisket, FSIS Gap 6)")
        ),
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
    cut_banked_h: float = 0.0,
    leth_banked_core: float = 0.0,
    cut_banked_core_h: float = 0.0,
    t_wrap_override: float | None = None,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    z_p=None,
    t_add: float = 1.0e9,
    z_a=None,
    t_add_override: float | None = None,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    z_s=None,
    z_d=None,
    t_spritz_override: float | None = None,
    spritz_dose: float = 0.05,
    z_m=None,
    t_salt: float = 0.0,
    z_ts=None,
    dt_add: float = 0.0,
    z_da=None,
):
    """Scalar quality objective of one simulated cook (lower is better).

    KCBS-weighted appearance/taste/tenderness bands plus operational terms
    and the food-safety *soft penalties* (lethality + come-up time on
    the surface and on the core node — see ControlConfig / safety_report).
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
    cut_banked_h : float, optional
        Come-up time already spent by the executed cook [h]; same
        whole-cook accounting as leth_banked (the 4 h cap is cumulative,
        not per-replan).
    leth_banked_core, cut_banked_core_h : float, optional
        The same two integrals on the core node.
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
    # the chimney and the spritz policy: overrides (possibly negative =
    # already executed) bypass the maps, like the wrap
    if t_add_override is not None:
        t_add = t_add_override
    elif cc.opt_add and z_a is not None:
        t_add = grids.t_end * jax.nn.sigmoid(z_a)
    elif z_a is not None:
        t_add = t_add + 0.0 * z_a
    if t_spritz_override is not None:
        t_spritz = t_spritz_override
    elif cc.opt_spritz and z_s is not None:
        t_spritz = grids.t_end * jax.nn.sigmoid(z_s)
    elif z_s is not None:
        t_spritz = t_spritz + 0.0 * z_s
    if cc.opt_spritz and z_d is not None:
        dt_spritz = cc.dt_spritz_min + (cc.dt_spritz_max - cc.dt_spritz_min) * (
            jax.nn.sigmoid(z_d)
        )
    elif z_d is not None:
        dt_spritz = dt_spritz + 0.0 * z_d
    if cc.opt_spritz and z_m is not None:
        spritz_dose = cc.m_spritz_min + (cc.m_spritz_max - cc.m_spritz_min) * (
            jax.nn.sigmoid(z_m)
        )
    elif z_m is not None:
        spritz_dose = spritz_dose + 0.0 * z_m
    # the M1 decisions: dry-brine time and wood-split interval -- bounded sigmoid maps like the spritz interval
    if cc.opt_salt and z_ts is not None:
        t_salt = cc.t_salt_max * jax.nn.sigmoid(z_ts)
    elif z_ts is not None:
        t_salt = t_salt + 0.0 * z_ts
    if cc.opt_splits and z_da is not None:
        dt_add = cc.dt_add_min + (cc.dt_add_max - cc.dt_add_min) * jax.nn.sigmoid(z_da)
    elif z_da is not None:
        dt_add = dt_add + 0.0 * z_da
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
        t_add=t_add,
        t_spritz=t_spritz,
        dt_spritz=dt_spritz,
        spritz_dose=spritz_dose,
        t_salt=t_salt,
        dt_add=dt_add,
        **tols,
    )

    return score_outputs(
        sim,
        jnp.asarray(grids.tknots),
        u,
        t_pull,
        t_add,
        cc,
        dt_add=dt_add,
        leth_banked=leth_banked,
        cut_banked_h=cut_banked_h,
        leth_banked_core=leth_banked_core,
        cut_banked_core_h=cut_banked_core_h,
    )


def score_outputs(
    sim,
    t_s,
    u,
    t_pull,
    t_add,
    cc: ControlConfig,
    dt_add=0.0,
    leth_banked: float = 0.0,
    cut_banked_h: float = 0.0,
    leth_banked_core: float = 0.0,
    cut_banked_core_h: float = 0.0,
):
    """Score a simulated cook's outputs (the quality objective; lower is better).

    Factored out of :func:`objective` so the seconds-mode and the
    duration-mode (:func:`objective_tf`) paths score identically; ``t_s``
    is the save grid in seconds (``s_save * t_f`` in duration mode, where
    it carries the d/dt_f stretching terms).
    """
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
    # docs/model.md; w_retained remains available for the scale story
    taste = smoke_band + cc.w_dry * (1.0 - sim["juiciness"]) ** 2

    # operational terms; fuel normalized by the FIXED reference load so
    # replans price burning identically to the whole-cook objective. The
    # fire keeps burning after the pull, so price the fuel AT the pull
    # (jnp.interp clamps: t_pull = never reads the end of the horizon
    # exactly) and cap bark-burn in the pit only ((1 - pull blend) weight)
    surf_over = jax.nn.softplus((sim["tsurf"] - cc.t_surf_max) / cc.dT_soft)
    in_pit = 1.0 - jax.nn.sigmoid((t_s - t_pull) / DELTA_PULL_S)
    surf_term = jnp.sum(in_pit * surf_over**2) / jnp.sum(in_pit)
    m_at_pull = jnp.interp(jnp.minimum(t_pull, t_s[-1]), t_s, sim["m_fuel"])
    # fuel burned = load + additions before the pull - fuel at the pull.
    # A finite dt_add prices the whole wood-split pulse train (one
    # m_chimney per split, each counted by its own smooth before-the-pull
    # gate); a static dt_add <= 0 reduces to the single-chimney term.
    if isinstance(dt_add, (int, float)) and dt_add <= 0:
        m_added = cc.m_chimney * jax.nn.sigmoid((t_pull - t_add) / TAU_ADD_S)
    else:
        ks = jnp.arange(cc.n_split_cap)
        m_added = cc.m_chimney * jnp.sum(
            jax.nn.sigmoid((t_pull - (t_add + ks * dt_add)) / TAU_ADD_S)
        )
    fuel_frac = (sim["m_fuel"][0] + m_added - m_at_pull) / cc.m0_ref

    # food safety (unconditional): lethality curve on the surface and on
    # the core node (the slab's cold spot), through the rest as well; the
    # come-up time is integrated the same way and penalized only when the
    # cut makes it critical (never for brisket, FSIS Gap 6)
    F_s, c_s = lethality_cut(sim["tsurf"], t_s, cc, jnp)
    F_c, c_c = lethality_cut(sim["tcore"], t_s, cc, jnp)
    safety = safety_penalty(
        leth_banked + F_s, cut_banked_h + c_s, cc, jnp
    ) + safety_penalty(leth_banked_core + F_c, cut_banked_core_h + c_c, cc, jnp)

    return (
        cc.w_appearance * appearance
        + cc.w_taste * taste
        + cc.w_tenderness * tenderness
        + cc.w_surf * surf_term
        + cc.w_fuel * fuel_frac
        + cc.w_smooth * jnp.sum(jnp.diff(u) ** 2)
        + safety
    )


def objective_tf(
    z,
    z_w,
    z_p,
    z_tf,
    theta,
    phi,
    tx,
    s_u,
    s_save,
    cc: ControlConfig,
    tols,
    t_lo: float,
    t_hi: float,
    rest: dict | None = None,
    ics: dict | None = None,
):
    """Joint duration x schedule quality objective (differentiable duration).

    The cook length is a decision variable: ``t_f = t_lo + (t_hi − t_lo) ·
    sigmoid(z_tf)``, with the damper schedule on the fractional grid
    ``s_u`` and the wrap/pull as fractions of the cook — the monolith's
    fractional-time mode (s = t/t_f), so ``jax.grad`` sees d/d(t_f)
    including the horizon-stretching terms. Chimney and spritz stay off
    (the scan optimum uses neither); scored by :func:`score_outputs`
    exactly as the seconds-mode :func:`objective`.
    """
    from .coupled import FRAC_NEVER, coupled_cook_tf

    u = jax.nn.sigmoid(z)
    frac_wrap = jax.nn.sigmoid(z_w) if cc.opt_wrap else FRAC_NEVER + 0.0 * z_w
    frac_pull = jax.nn.sigmoid(z_p) if cc.opt_pull else FRAC_NEVER + 0.0 * z_p
    t_f = t_lo + (t_hi - t_lo) * jax.nn.sigmoid(z_tf)
    sim = coupled_cook_tf(
        theta,
        phi,
        u,
        t_f,
        tx,
        s_u,
        s_save,
        frac_wrap=frac_wrap,
        frac_pull=frac_pull,
        rest=rest,
        ics=ics,
        **tols,
    )
    t_s = jnp.asarray(s_save) * t_f
    return score_outputs(sim, t_s, u, frac_pull * t_f, FRAC_NEVER * t_f, cc)


def _mask_decision_grads(grads, cc: ControlConfig):
    """Zero the gradients of decisions that are switched off.

    Order: (z, z_w, z_p, z_a, z_s, z_d, z_m, z_ts, z_da).
    """
    g, g_w, g_p, g_a, g_s, g_d, g_m, g_ts, g_da = grads
    if not cc.opt_wrap:
        g_w = jnp.zeros_like(g_w)
    if not cc.opt_pull:
        g_p = jnp.zeros_like(g_p)
    if not cc.opt_add:
        g_a = jnp.zeros_like(g_a)
    if not cc.opt_spritz:
        g_s, g_d, g_m = (jnp.zeros_like(g_s), jnp.zeros_like(g_d), jnp.zeros_like(g_m))
    if not cc.opt_salt:
        g_ts = jnp.zeros_like(g_ts)
    if not cc.opt_splits:
        g_da = jnp.zeros_like(g_da)
    return [g, g_w, g_p, g_a, g_s, g_d, g_m, g_ts, g_da]


def _decisions(
    cc: ControlConfig,
    grids,
    z_a,
    z_s,
    z_d,
    t_add,
    t_spritz,
    dt_spritz,
    t_add_override=None,
    t_spritz_override=None,
    z_m=None,
    spritz_dose=0.05,
):
    """Map the optimized chimney / spritz variables back to times [s] and a dose [kg/m²]."""
    if t_add_override is not None:
        ta = float(t_add_override)
    elif cc.opt_add:
        ta = float(grids.t_end * jax.nn.sigmoid(z_a))
    else:
        ta = float(t_add)
    if t_spritz_override is not None:
        ts = float(t_spritz_override)
    elif cc.opt_spritz:
        ts = float(grids.t_end * jax.nn.sigmoid(z_s))
    else:
        ts = float(t_spritz)
    if cc.opt_spritz:
        td = float(
            cc.dt_spritz_min
            + (cc.dt_spritz_max - cc.dt_spritz_min) * jax.nn.sigmoid(z_d)
        )
    else:
        td = float(dt_spritz)
    if cc.opt_spritz and z_m is not None:
        tm = float(
            cc.m_spritz_min + (cc.m_spritz_max - cc.m_spritz_min) * jax.nn.sigmoid(z_m)
        )
    else:
        tm = float(spritz_dose)
    return ta, ts, td, tm


def _decisions_m1(cc: ControlConfig, z_ts, z_da, t_salt, dt_add):
    """Map the optimized salt / split variables back to values."""
    tsalt = (
        float(cc.t_salt_max * jax.nn.sigmoid(z_ts)) if cc.opt_salt else float(t_salt)
    )
    tda = (
        float(cc.dt_add_min + (cc.dt_add_max - cc.dt_add_min) * jax.nn.sigmoid(z_da))
        if cc.opt_splits
        else float(dt_add)
    )
    return tsalt, tda


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
    z_a_init: float | None = None,
    z_s_init: float | None = None,
    z_d_init: float | None = None,
    t_add: float = 1.0e9,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    z_m_init: float | None = None,
    spritz_dose: float = 0.05,
    z_ts_init: float | None = None,
    z_da_init: float | None = None,
    t_salt: float = 0.0,
    dt_add: float = 0.0,
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

    def expected_obj(z, z_w, z_p, z_a, z_s, z_d, z_m, z_ts, z_da, ths, phs):
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
                    t_add=t_add,
                    z_a=z_a,
                    t_spritz=t_spritz,
                    dt_spritz=dt_spritz,
                    z_s=z_s,
                    z_d=z_d,
                    spritz_dose=spritz_dose,
                    z_m=z_m,
                    t_salt=t_salt,
                    z_ts=z_ts,
                    dt_add=dt_add,
                    z_da=z_da,
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
    z_a = jnp.asarray(ADD_Z_INIT if z_a_init is None else z_a_init)
    z_s = jnp.asarray(SPRITZ_Z_INIT if z_s_init is None else z_s_init)
    z_d = jnp.asarray(SPRITZ_DT_Z_INIT if z_d_init is None else z_d_init)
    z_m = jnp.asarray(SPRITZ_DOSE_Z_INIT if z_m_init is None else z_m_init)
    z_ts = jnp.asarray(SALT_Z_INIT if z_ts_init is None else z_ts_init)
    z_da = jnp.asarray(SPLIT_Z_INIT if z_da_init is None else z_da_init)
    opt = optax.adam(cc.lr)
    vars_ = (z, z_w, z_p, z_a, z_s, z_d, z_m, z_ts, z_da)
    state = opt.init(vars_)
    val_grad = jax.jit(
        jax.value_and_grad(
            expected_obj, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        )
    )

    losses = []
    best_loss = np.inf
    best = vars_
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        loss, (*grads, _, _) = val_grad(*vars_, thetas, phis)
        lf = float(loss)
        losses.append(lf)
        if np.isfinite(lf) and lf < best_loss:
            best_loss, best = lf, vars_
        grads = _mask_decision_grads(grads, cc)
        updates, state = opt.update(tuple(grads), state, vars_)
        vars_ = optax.apply_updates(vars_, updates)
        z = vars_[0]
        if callback and (i % log_every == 0 or i == cc.n_iter - 1):
            callback(i, lf, np.asarray(jax.nn.sigmoid(z)))
        if _plateau_stop(losses, cc):
            break
    (best_z, best_z_w, best_z_p, best_z_a, best_z_s, best_z_d, best_z_m,
     best_z_ts, best_z_da) = best
    tsalt, tda = _decisions_m1(cc, best_z_ts, best_z_da, t_salt, dt_add)
    tw = float(grids.t_end * jax.nn.sigmoid(best_z_w)) if cc.opt_wrap else 1.0e9
    tp = float(grids.t_end * jax.nn.sigmoid(best_z_p)) if cc.opt_pull else t_pull
    ta, ts, td, tm = _decisions(
        cc,
        grids,
        best_z_a,
        best_z_s,
        best_z_d,
        t_add,
        t_spritz,
        dt_spritz,
        z_m=best_z_m,
        spritz_dose=spritz_dose,
    )
    return np.asarray(jax.nn.sigmoid(best_z)), {
        "losses": np.asarray(losses),
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
        "t_wrap": tw,
        "t_pull": tp,
        "t_add": ta,
        "t_spritz": ts,
        "dt_spritz": td,
        "spritz_dose": tm,
        "t_salt": tsalt,
        "dt_add": tda,
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
    cut_banked_h: float = 0.0,
    leth_banked_core: float = 0.0,
    cut_banked_core_h: float = 0.0,
    t_wrap_override: float | None = None,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    z_p_init: float | None = None,
    z_a_init: float | None = None,
    z_s_init: float | None = None,
    z_d_init: float | None = None,
    t_add: float = 1.0e9,
    t_add_override: float | None = None,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    t_spritz_override: float | None = None,
    z_m_init: float | None = None,
    spritz_dose: float = 0.05,
    z_ts_init: float | None = None,
    z_da_init: float | None = None,
    t_salt: float = 0.0,
    dt_add: float = 0.0,
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
    z_a = jnp.asarray(ADD_Z_INIT if z_a_init is None else z_a_init)
    z_s = jnp.asarray(SPRITZ_Z_INIT if z_s_init is None else z_s_init)
    z_d = jnp.asarray(SPRITZ_DT_Z_INIT if z_d_init is None else z_d_init)
    z_m = jnp.asarray(SPRITZ_DOSE_Z_INIT if z_m_init is None else z_m_init)
    z_ts = jnp.asarray(SALT_Z_INIT if z_ts_init is None else z_ts_init)
    z_da = jnp.asarray(SPLIT_Z_INIT if z_da_init is None else z_da_init)
    opt = optax.adam(cc.lr)
    vars_ = (z, z_w, z_p, z_a, z_s, z_d, z_m, z_ts, z_da)
    state = opt.init(vars_)
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
            lambda zz, zw, zp, za, zs, zd, zm, zts, zda, th, ph: objective(
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
                cut_banked_h=cut_banked_h,
                leth_banked_core=leth_banked_core,
                cut_banked_core_h=cut_banked_core_h,
                t_wrap_override=t_wrap_override,
                t_pull=t_pull,
                rest=rest,
                z_p=zp,
                t_add=t_add,
                z_a=za,
                t_add_override=t_add_override,
                t_spritz=t_spritz,
                dt_spritz=dt_spritz,
                z_s=zs,
                z_d=zd,
                t_spritz_override=t_spritz_override,
                spritz_dose=spritz_dose,
                z_m=zm,
                t_salt=t_salt,
                z_ts=zts,
                dt_add=dt_add,
                z_da=zda,
            ),
            argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        )
    )

    losses, schedules = [], []
    best_loss = np.inf
    best = vars_
    t0 = time.perf_counter()
    for i in range(cc.n_iter):
        # Single retry: a rare timing-dependent race in the local-client
        # dispatch can hand one call a degenerate input (shape guard in the
        # tesseract api raises a clear error). One retry has always sufficed;
        # two consecutive failures is a real bug worth surfacing.
        try:
            loss, (*grads, _g_theta, _g_phi) = val_grad(*vars_, theta, phi)
        except RuntimeError as e:
            if "DEGENERATE" not in str(e) and "BoundsError" not in str(e):
                raise
            print(f"  iter {i}: degenerate-input race, retrying once ({e})", flush=True)
            loss, (*grads, _g_theta, _g_phi) = val_grad(*vars_, theta, phi)
        lf = float(loss)
        losses.append(lf)
        if np.isfinite(lf) and lf < best_loss:
            best_loss, best = lf, vars_
        grads = _mask_decision_grads(grads, cc)
        updates, state = opt.update(tuple(grads), state, vars_)
        vars_ = optax.apply_updates(vars_, updates)
        if i % log_every == 0 or i == cc.n_iter - 1:
            schedules.append(np.asarray(jax.nn.sigmoid(best[0])))
            if callback:
                callback(i, lf, np.asarray(jax.nn.sigmoid(best[0])))
        if _plateau_stop(losses, cc):
            break
    (best_z, best_z_w, best_z_p, best_z_a, best_z_s, best_z_d, best_z_m,
     best_z_ts, best_z_da) = best
    tsalt, tda = _decisions_m1(cc, best_z_ts, best_z_da, t_salt, dt_add)
    if t_wrap_override is not None:
        tw = float(t_wrap_override)
    elif cc.opt_wrap:
        tw = float(grids.t_end * jax.nn.sigmoid(best_z_w))
    else:
        tw = 1.0e9
    tp = float(grids.t_end * jax.nn.sigmoid(best_z_p)) if cc.opt_pull else t_pull
    ta, ts, td, tm = _decisions(
        cc,
        grids,
        best_z_a,
        best_z_s,
        best_z_d,
        t_add,
        t_spritz,
        dt_spritz,
        t_add_override,
        t_spritz_override,
        best_z_m,
        spritz_dose,
    )
    return np.asarray(jax.nn.sigmoid(best_z)), {
        "losses": np.asarray(losses),
        "schedules": np.asarray(schedules),
        "t_wrap": tw,
        "t_pull": tp,
        "t_add": ta,
        "t_spritz": ts,
        "dt_spritz": td,
        "spritz_dose": tm,
        "t_salt": tsalt,
        "dt_add": tda,
        "best_loss": best_loss,
        "wall_s": time.perf_counter() - t0,
        "grids": grids,
        "converged": _plateau_stop(losses, cc),
        "n_iter_run": len(losses),
    }
