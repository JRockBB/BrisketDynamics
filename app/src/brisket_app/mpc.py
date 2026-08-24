"""Receding-horizon MPC over the coupled cook.

Plant and controller are both the calibrated coupled model; the *plant* may
carry a disturbance (e.g. an ambient cold front) the controller's model does
not know about. Every `dt_replan` the controller re-optimizes the remaining
schedule from the plant's current state (full-field handoff via the
Tesseracts' state IO), then the plant executes the first interval.

This is state-feedback MPC (the controller reads the plant state directly);
output-feedback with an estimator is future work, noted in the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from .control import ControlConfig, optimize_schedule
from .coupled import HR, CookGrids, coupled_cook


@dataclass
class CookState:
    """Plant state carried across receding-horizon intervals.

    Fields are the pit state (temperature, fuel, ignited fraction) plus the
    full meat fields for the state handoff; empty field arrays mean uniform
    default initial conditions.
    """

    t: float  # elapsed cook time [s]
    tpit0: float
    m0: float
    a0: float
    t_field0: np.ndarray  # empty array = uniform default ICs
    w_field0: np.ndarray
    c_field0: np.ndarray
    f_field0: np.ndarray
    t0_meat: float = 281.15
    bark_b0: float = float("nan")  # NaN = fresh (0)
    bark_s0: float = float("nan")
    leth_f: float = 0.0  # banked lethality [60C-equivalent s] of executed cook
    danger_h: float = 0.0  # banked danger-zone dwell [h] of executed cook

    @classmethod
    def fresh(cls, tpit0=293.15, m0=7.0, a0=1.0, t0_meat=281.15):
        """Cold-start state: ambient pit, full fuel ring, fridge meat."""
        z = np.zeros(0)
        return cls(0.0, tpit0, m0, a0, z, z, z, z, t0_meat=t0_meat)


def _ics(state: CookState, t_amb: float) -> dict:
    """Assemble the coupled_cook `ics` dict from a CookState."""
    ics = {
        "tpit0": state.tpit0,
        "m0": state.m0,
        "t_amb": t_amb,
        "t_field0": state.t_field0,
        "w_field0": state.w_field0,
        "c_field0": state.c_field0,
        "f_field0": state.f_field0,
        "bark_b0": state.bark_b0,
        "bark_s0": state.bark_s0,
    }
    if state.t_field0.size == 0:
        ics["t0_meat"] = state.t0_meat
    return ics


def _with_a0(phi: np.ndarray, a0: float) -> np.ndarray:
    """Return phi with its ignition entry replaced by the plant's current a0."""
    out = np.asarray(phi, dtype=np.float64).copy()
    out[6] = np.clip(a0, 1e-3, 1.0)
    return out


def advance_plant(
    state: CookState,
    u_seg: np.ndarray,
    dt: float,
    theta,
    phi,
    t_amb: float,
    tols=None,
    t_wrap_seg: float = 1.0e9,
) -> tuple[CookState, dict]:
    """Run the plant one interval; return (new state, trajectories).

    ``t_wrap_seg`` is the wrap moment in SEGMENT-relative time: 1e9 = no
    wrap this segment, a value inside [0, dt) executes the crutch here,
    and a negative value (~-3600) means the cook arrived already wrapped
    (blend saturated on from the segment start).
    """
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    grids = CookGrids(t_end=dt, dt_exchange=min(600.0, dt / 2), n_u=len(u_seg))
    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(_with_a0(phi, state.a0)),
        jnp.asarray(u_seg),
        *_CLIENTS,
        grids,
        K=2,
        ics=_ics(state, t_amb),
        t_wrap=t_wrap_seg,
        **tols,
    )
    new = CookState(
        t=state.t + dt,
        tpit0=float(sim["tpit"][-1]),
        m0=float(sim["m_fuel"][-1]),
        a0=float(sim["ignited"][-1]),
        t_field0=np.asarray(sim["t_field_final"]),
        w_field0=np.asarray(sim["w_field_final"]),
        c_field0=np.asarray(sim["c_field_final"]),
        f_field0=np.asarray(sim["f_field_final"]),
        bark_b0=float(sim["bark_b"][-1]),
        bark_s0=float(sim["bark_s"][-1]),
        # safety banking (leth_f / danger_h) is applied by run_cook, which
        # owns the ControlConfig and therefore the correct constants
        leth_f=state.leth_f,
        danger_h=state.danger_h,
    )
    return new, {
        k: np.asarray(sim[k]) for k in ("tpit", "tcore", "tsurf", "m_fuel", "ignited")
    }


_CLIENTS = None  # set by set_clients()


@dataclass
class AmbientKF:
    """Scalar Kalman filter for the unmodeled ambient temperature.

    State: T_amb as a random walk (process intensity ``q`` [K^2/s]).
    Measurement: the segment-end pit temperature; the innovation is
    measured-minus-predicted, where the prediction is one forward solve of
    the controller model under the current belief. The measurement
    sensitivity is the pit's first-order-lag response to a constant
    ambient offset over the segment,

        H = 1 - exp(-UA * dt / C_th),

    linearized from the model phi each update — no hand-tuned gain: the
    Kalman gain K = P H / (H^2 P + R) weighs prior confidence against the
    calibrated pit-noise variance R.
    """

    x: float  # ambient belief [K]
    p: float  # belief variance [K^2]
    q: float  # process intensity [K^2/s]
    r: float  # measurement variance [K^2]

    def update(self, innovation: float, ua: float, c_th: float, dt: float) -> None:
        """One predict-correct cycle over an executed segment."""
        self.p += self.q * dt  # random-walk predict
        h = 1.0 - np.exp(-ua * dt / c_th)
        k = self.p * h / (h * h * self.p + self.r)
        self.x = float(np.clip(self.x + k * innovation, 250.0, 330.0))
        self.p *= 1.0 - k * h


def executed_objective(out, cc: ControlConfig, t_f: float) -> dict:
    """Evaluate the quality objective J on an EXECUTED cook.

    Works from run_cook's trajectories + terminal quality (no re-simulation,
    so it is exact for the disturbed plant the cook actually experienced) —
    with ONE approximation: the smoothness term penalizes changes between
    executed per-segment damper MEANS, whereas the optimizer penalizes
    differences between schedule knots. All other terms mirror
    control.objective exactly.

    Returns
    -------
    dict
        J plus its named components.
    """
    sp = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)  # softplus
    t = np.asarray(out["t"])
    tsurf = np.asarray(out["tsurf"])
    u = np.asarray(out["u"])

    c0 = out["doneness"]
    under = (sp(20.0 * (cc.c_target - c0)) / 20.0) / 0.05
    band_o = 1.0 - cc.c_overdone
    k_ov = 4.0 / band_o
    over = (sp(k_ov * (c0 - cc.c_overdone)) / k_ov) / band_o
    tenderness = under**2 + cc.over_rel * over**2

    lst = out["l_star_end"]
    appearance = (
        sp((lst - cc.l_mahogany_hi) / cc.dl_soft) ** 2
        + sp((cc.l_mahogany_lo - lst) / cc.dl_soft) ** 2
    )
    s_end = out["smoke_end"]
    smoke_band = (
        sp((cc.s_smoke_lo - s_end) / cc.ds_soft) ** 2
        + sp((s_end - cc.s_smoke_hi) / cc.ds_soft) ** 2
    )
    taste = smoke_band + cc.w_dry * (1.0 - out["juiciness"]) ** 2

    surf_over = sp((tsurf - cc.t_surf_max) / cc.dT_soft)
    fuel_frac = out.get("fuel_frac", 0.0)

    F = np.trapezoid(10.0 ** ((tsurf - cc.leth_t_ref) / cc.leth_z), t)
    leth = (
        cc.w_lethality * sp((np.log10(cc.leth_f_req) - np.log10(F + 1e-9)) / 0.2) ** 2
    )
    in_zone = 1.0 / (1.0 + np.exp(-(tsurf - cc.t_danger_lo) / 2.0))
    in_zone *= 1.0 / (1.0 + np.exp(-(cc.t_danger_hi - tsurf) / 2.0))
    danger_h = np.trapezoid(in_zone, t / 3600.0)
    dwell = cc.w_dwell * sp((danger_h - cc.danger_cap_h) / 0.5) ** 2

    # smoothness over the executed per-segment means (hourly steps)
    u_steps = u[np.r_[0, np.nonzero(np.diff(u))[0] + 1]]
    smooth = cc.w_smooth * float(np.sum(np.diff(u_steps) ** 2))

    comp = {
        "appearance": cc.w_appearance * float(appearance),
        "taste": cc.w_taste * float(taste),
        "tenderness": cc.w_tenderness * float(tenderness),
        "surf": cc.w_surf * float(np.mean(surf_over**2)),
        "fuel": cc.w_fuel * float(fuel_frac),
        "smooth": smooth,
        "safety": float(leth + dwell),
    }
    comp["J"] = sum(comp.values())
    return comp


def set_clients(meat_tx, pit_tx):
    """Install the module-level Tesseract clients used by plan/advance."""
    global _CLIENTS
    _CLIENTS = (meat_tx, pit_tx)


def plan(
    theta,
    phi,
    state: CookState,
    t_remaining: float,
    cc: ControlConfig,
    model_t_amb: float,
    u_init=0.5,
    iters=150,
    z_w_init: float = 0.0,
    z_w_pin: float = 8.0,
    t_wrap_override: float | None = None,
):
    """Re-optimize the remaining-horizon plan from the current state.

    Returns (u_opt, grids, t_wrap_rel): the schedule, its grids, and the
    planned wrap moment relative to NOW (t_end*sigmoid(z_w); meaningful
    only when ``cc.opt_wrap``). With opt_wrap off, z_w stays pinned at
    ``z_w_pin`` (8 = never; -8 = already wrapped).
    """
    n_u = max(2, round(t_remaining / (0.75 * HR)))
    grids = CookGrids(
        t_end=t_remaining, dt_exchange=min(600.0, t_remaining / 4), n_u=n_u
    )
    meat_tx, pit_tx = _CLIENTS
    u_opt, hist = optimize_schedule(
        theta,
        _with_a0(phi, state.a0),
        meat_tx,
        pit_tx,
        grids=grids,
        # iters is a CAP; the plateau rule in optimize_schedule stops the
        # warm-started replan when converged (usually well before the cap)
        cc=replace(cc, n_iter=iters),
        u_init=u_init,
        ics=_ics(state, model_t_amb),
        z_w_init=z_w_init,
        z_w_pin=z_w_pin,
        leth_banked=state.leth_f,
        danger_banked_h=state.danger_h,
        t_wrap_override=t_wrap_override,
    )
    return u_opt, grids, hist["t_wrap"]


def run_cook(
    theta,
    phi_plant,
    plant_t_amb_fn,
    t_f=12 * HR,
    dt_replan=1 * HR,
    cc: ControlConfig | None = None,
    mode="mpc",
    u_openloop=None,
    u_openloop_grids=None,
    model_t_amb=300.0,
    phi_model=None,
    log=print,
    t_wrap_init_abs: float | None = None,
    u_incumbent=None,
    u_incumbent_grids=None,
    wrap_deadband: float = 0.5 * HR,
):
    """Simulate a full cook under MPC or open-loop control.

    mode='mpc' replans each dt_replan interval from the advanced plant
    state (full-field handoff); mode='openloop' executes u_openloop
    blindly. plant_t_amb_fn(t) is the ambient-temperature disturbance the
    controller's model does NOT know. MPC mode carries an integral
    ambient Kalman filter (AmbientKF; tuning in ControlConfig): after each
    executed segment, one forward solve of the controller model predicts
    the segment-end pit temperature under the current belief, and the
    filter assimilates the innovation. Open-loop mode is blind by
    definition — it executes a fixed plan and estimates nothing.
    ``wrap_deadband`` suppresses wrap-decision drift along the flat
    J-vs-t_wrap ridge: a replan moves the planned wrap only when it
    differs from the incumbent plan by more than the dead-band.
    ``t_wrap_init_abs`` seeds the
    planned wrap moment (absolute [s]): open-loop executes it verbatim;
    MPC (with cc.opt_wrap) treats it as the incumbent plan and re-times
    it each replan until executed. ``u_incumbent`` (+ its grids) is the
    incumbent SCHEDULE: the first replan warm-starts from it instead of
    a flat 0.5, so absent disturbances MPC refines the assessed plan
    rather than re-deriving one (principle-of-optimality fidelity up to
    optimizer convergence and knot re-gridding).

    Returns
    -------
    dict
        Trajectories t/tpit/tcore/tsurf/u plus final core doneness.
    """
    cc = cc or ControlConfig()
    phi_model = phi_plant if phi_model is None else phi_model
    state = CookState.fresh(t0_meat=281.15)
    m0_start = state.m0
    traj = {k: [] for k in ("t", "tpit", "tcore", "tsurf", "u")}
    # incumbent plan (if given) is "a plan made at t = 0": the standard
    # shift-and-interpolate warm start then applies from the first replan
    u_prev_plan = None if u_incumbent is None else np.asarray(u_incumbent)
    _prev_grids = u_incumbent_grids
    prev_plan_t0 = 0.0
    # Texas-crutch state machine (active when cc.opt_wrap): each replan
    # re-times the wrap over the remaining horizon; once a plant segment
    # executes past the planned moment the cook IS wrapped and irreversible
    # -- both plant and replans then model the blend CONTINUOUSLY from the
    # true execution moment (t_wrap_exec_abs).
    kf = AmbientKF(x=float(model_t_amb), p=cc.amb_p0, q=cc.amb_q, r=cc.amb_r)
    t_amb_hat_hist: list[tuple[float, float, float]] = []  # (t, belief, var)
    t_wrap_abs: float | None = t_wrap_init_abs  # planned wrap, absolute [s]
    wrapped = False
    t_wrap_exec_abs: float | None = None  # when the crutch ACTUALLY fired
    wrap_history: list[tuple[float, float]] = []  # (replan t, planned abs wrap)

    # final segment shrinks to the remainder, so arbitrary horizons execute
    # exactly t_f (round(t_f/dt) previously over/under-shot)
    while state.t < t_f - 1e-6:
        dt_seg = min(dt_replan, t_f - state.t)
        t_remaining = t_f - state.t
        if mode == "mpc":
            if u_prev_plan is None:
                u_init = 0.5
            else:
                # shift-and-interpolate the previous plan onto the new
                # remaining-horizon knots (standard MPC warm start): each
                # replan refines the last instead of restarting.
                prev_t = _prev_grids.u_tknots + prev_plan_t0
                n_u_new = max(2, round(t_remaining / (0.75 * HR)))
                new_t = state.t + np.linspace(0, t_remaining, n_u_new)
                u_init = np.clip(np.interp(new_t, prev_t, u_prev_plan), 1e-3, 1 - 1e-3)
            tw_override = None
            if cc.opt_wrap and not wrapped:
                # warm-start z_w from the previous ABSOLUTE plan, mapped to
                # the new remaining horizon (logit of the horizon fraction)
                if t_wrap_abs is None:
                    zw0 = 0.0
                else:
                    frac = np.clip((t_wrap_abs - state.t) / t_remaining, 1e-3, 1 - 1e-3)
                    zw0 = float(np.log(frac / (1 - frac)))
                cc_seg = cc
                zw_pin = 8.0
            else:
                # wrapped (or wrap disabled): plan the schedule only. An
                # executed wrap is modeled CONTINUOUSLY: the replan gets the
                # true (negative) wrap time relative to now, so its blend
                # state matches the plant exactly (a z_w pin cannot express
                # a wrap in the past — sigmoid maps to (0, t_end)).
                cc_seg = replace(cc, opt_wrap=False)
                zw0 = 0.0
                zw_pin = 8.0
                if wrapped:
                    tw_override = t_wrap_exec_abs - state.t
            u_plan, plan_grids, t_wrap_rel = plan(
                theta,
                phi_model,
                state,
                t_remaining,
                cc_seg,
                kf.x,
                u_init=u_init,
                z_w_init=zw0,
                z_w_pin=zw_pin,
                t_wrap_override=tw_override,
            )
            if cc.opt_wrap and not wrapped:
                cand = state.t + float(t_wrap_rel)
                # dead-band: J vs t_wrap is a flat ridge in forecast, so
                # plateau-stopped replans wander it; only adopt a move that
                # exceeds the dead-band (drift is noise, retiming is signal)
                if t_wrap_abs is None or abs(cand - t_wrap_abs) > wrap_deadband:
                    t_wrap_abs = cand
                wrap_history.append((state.t, t_wrap_abs))
            u_prev_plan, _prev_grids = u_plan, plan_grids
            prev_plan_t0 = state.t
            # execute the first dt_seg of the plan, sampled at exchange
            # resolution — a 2-point interp coarsened every plan into pit
            # overshoot (reference executed-J 1.162 vs its model-J 1.005,
            # mostly a surface-cap violation the plan never had)
            n_exec = max(2, round(dt_seg / 600.0) + 1)
            tt = np.linspace(0, dt_seg, n_exec)
            u_seg = np.interp(tt, plan_grids.u_tknots, u_plan)
        else:
            n_exec = max(2, round(dt_seg / 600.0) + 1)
            tt = state.t + np.linspace(0, dt_seg, n_exec)
            u_seg = np.interp(tt, u_openloop_grids.u_tknots, u_openloop)

        t_amb = plant_t_amb_fn(state.t + dt_seg / 2)
        if wrapped:
            # blend continues from the TRUE wrap moment (negative relative
            # time), bitwise-continuous across segment boundaries — a fixed
            # saturated constant would jump a partial blend at the joint
            tw_seg = t_wrap_exec_abs - state.t
        elif t_wrap_abs is not None and t_wrap_abs < state.t + dt_seg:
            tw_seg = max(0.0, t_wrap_abs - state.t)  # crutch executes NOW
            t_wrap_exec_abs = state.t + tw_seg
        else:
            tw_seg = 1.0e9
        t_seg_start = state.t
        state_before = state
        state, seg_traj = advance_plant(
            state, u_seg, dt_seg, theta, phi_plant, t_amb, t_wrap_seg=tw_seg
        )
        if mode == "mpc":
            # one forward solve: what the controller EXPECTED this segment's
            # pit end to be under its current ambient belief
            pred_grids = CookGrids(
                t_end=dt_seg, dt_exchange=min(600.0, dt_seg / 2), n_u=len(u_seg)
            )
            pred = coupled_cook(
                jnp.asarray(theta),
                jnp.asarray(_with_a0(phi_model, state_before.a0)),
                jnp.asarray(u_seg),
                *_CLIENTS,
                pred_grids,
                K=2,
                ics=_ics(state_before, kf.x),
                t_wrap=tw_seg,
            )
            innov = float(seg_traj["tpit"][-1]) - float(np.asarray(pred["tpit"])[-1])
            # phi = [C_th, UA, ...]: the lag linearization uses the model's
            # own thermal mass and loss coefficient
            kf.update(
                innov, ua=float(phi_model[1]), c_th=float(phi_model[0]), dt=dt_seg
            )
            t_amb_hat_hist.append((state.t, kf.x, kf.p))
        if tw_seg < 1.0e8 and not wrapped:
            wrapped = True  # irreversible from here on
        # safety banking with the CONFIGURED constants (hard-coded 60C/6K
        # here would silently disagree with a customized ControlConfig)
        nseg_t = len(seg_traj["tpit"])
        tseg = np.linspace(0.0, dt_seg, nseg_t)
        ts_seg = np.asarray(seg_traj["tsurf"])
        state.leth_f += float(
            np.trapezoid(10.0 ** ((ts_seg - cc.leth_t_ref) / cc.leth_z), tseg)
        )
        in_zone = 1.0 / (1.0 + np.exp(-(ts_seg - cc.t_danger_lo) / 2.0))
        in_zone *= 1.0 / (1.0 + np.exp(-(cc.t_danger_hi - ts_seg) / 2.0))
        state.danger_h += float(np.trapezoid(in_zone, tseg / 3600.0))
        traj["t"].extend((t_seg_start + np.linspace(0, dt_seg, nseg_t)).tolist())
        for k in ("tpit", "tcore", "tsurf"):
            traj[k].extend(seg_traj[k].tolist())
        traj["u"].extend([float(np.mean(u_seg))] * nseg_t)
        log(
            f"  [{mode}] t={state.t / HR:4.1f} h  Tpit={state.tpit0 - 273.15:5.1f} C  "
            f"core={seg_traj['tcore'][-1] - 273.15:5.1f} C  "
            f"C={state.c_field0[0]:.3f}  u~{np.mean(u_seg):.2f}  "
            f"T_amb={t_amb - 273.15:.0f} C  "
            f"T_amb_hat={kf.x - 273.15:.1f}±{np.sqrt(kf.p):.1f} C"
        )

    doneness = float(state.c_field0[0])
    fuel_frac = (m0_start - state.m0) / m0_start
    # terminal quality vector from the final plant state (constants mirror
    # MeatConfig: W0/F0/beta_fat/eta_drain and the L* map)
    W0, F0, BETA_FAT, ETA_DRAIN = 2.33, 0.22, 1.8, 0.5
    L0_BARK, DL_B, DL_S = 52.0, 20.0, 10.0

    def _fv_mean(v):
        n = len(v)
        return (v[0] + v[-1] + 2.0 * np.sum(v[1:-1])) / (2.0 * (n - 1))

    w_ret = (
        float(_fv_mean(state.w_field0)) / W0 if state.w_field0.size else float("nan")
    )
    f_mean = float(_fv_mean(state.f_field0)) if state.f_field0.size else float("nan")
    fat_rendered = (F0 - f_mean) / F0
    juiciness = w_ret + BETA_FAT * (1 - ETA_DRAIN) * (F0 - f_mean) / W0
    b_end = 0.0 if np.isnan(state.bark_b0) else float(state.bark_b0)
    s_end = 0.0 if np.isnan(state.bark_s0) else float(state.bark_s0)
    l_star = L0_BARK - DL_B * b_end - DL_S * s_end
    out = {k: np.asarray(v) for k, v in traj.items()} | {
        "final_state": state,
        "doneness": doneness,
        "juiciness": juiciness,
        "fat_rendered": fat_rendered,
        "l_star_end": l_star,
        "smoke_end": s_end,
        "fuel_frac": fuel_frac,
    }
    if cc.opt_wrap or t_wrap_init_abs is not None:
        out["wrapped"] = wrapped
        out["t_wrap_abs"] = t_wrap_abs
        out["wrap_history"] = wrap_history
    out["t_amb_hat_hist"] = t_amb_hat_hist
    return out
