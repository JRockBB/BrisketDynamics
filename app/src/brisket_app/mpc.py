"""Receding-horizon MPC over the coupled cook.

Plant and controller are both the calibrated coupled model; the *plant* may
carry a disturbance (e.g. an ambient cold front) the controller's model does
not know about. Every `dt_replan` the controller re-optimizes the remaining
schedule from the plant's current state (full-field handoff via the
Tesseracts' state IO), then the plant executes the first interval.

This is state-feedback MPC (the controller reads the plant state directly);
output-feedback with an estimator is future work, noted in the roadmap.

The cook runs to serve (``t_f``): every replan optimizes the schedule, the
wrap and the pull over the remaining time, scored at serve; the pull
executes like the wrap (candidate -> commit -> irreversible) and the plant
then rests to serve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from .control import (
    ControlConfig,
    lethality_dwell,
    optimize_schedule,
    safety_penalty,
)
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
    ring0: float = float("nan")  # smoke-ring depth, material coordinate [m]
    leth_f: float = 0.0  # banked surface lethality [60C-equivalent s] of executed cook
    danger_h: float = 0.0  # banked surface danger-zone dwell [h] of executed cook
    leth_f_core: float = 0.0  # the same two on the core node (cold spot)
    danger_h_core: float = 0.0

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
        "ring0": state.ring0,
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
    t_pull_seg: float = 1.0e9,
    rest: dict | None = None,
    t_add_seg: float = 1.0e9,
    t_spritz_seg: float = 1.0e9,
    dt_spritz_seg: float = 3600.0,
    spritz_dose_seg: float = 0.05,
) -> tuple[CookState, dict]:
    """Run the plant one interval; return (new state, trajectories).

    ``t_wrap_seg`` / ``t_pull_seg`` are the wrap and pull moments in
    SEGMENT-relative time: 1e9 = not this segment, a value inside [0, dt)
    executes the event here, and a negative value (~-3600) means the cook
    arrived already wrapped / pulled (blend saturated from the segment
    start; after the pull the meat rests under the ``rest`` preset).
    """
    tols = tols or {"abstol": 1e-6, "reltol": 1e-6}
    grids = CookGrids(t_end=dt, dt_exchange=min(600.0, dt / 2), n_u=len(u_seg))
    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(_with_a0(phi, state.a0)),
        jnp.asarray(u_seg),
        _CLIENTS,
        grids,
        ics=_ics(state, t_amb),
        t_wrap=t_wrap_seg,
        t_pull=t_pull_seg,
        rest=rest,
        t_add=t_add_seg,
        t_spritz=t_spritz_seg,
        dt_spritz=dt_spritz_seg,
        spritz_dose=spritz_dose_seg,
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
        ring0=float(sim["ring"]),
        # safety banking (leth_f / danger_h) is applied by run_cook, which
        # owns the ControlConfig and therefore the correct constants
        leth_f=state.leth_f,
        danger_h=state.danger_h,
        leth_f_core=state.leth_f_core,
        danger_h_core=state.danger_h_core,
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
    # bark-burn cap and damper smoothness apply in the pit only; safety
    # integrates through the rest, like control.objective
    tp = out.get("t_pull_exec_abs")
    in_pit = np.ones(t.shape, dtype=bool) if tp is None else t <= float(tp)

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

    tcore = np.asarray(out["tcore"])
    safety = safety_penalty(*lethality_dwell(tsurf, t, cc), cc) + safety_penalty(
        *lethality_dwell(tcore, t, cc), cc
    )

    # smoothness over the executed per-segment means (hourly steps)
    u_pit = u[in_pit]
    u_steps = u_pit[np.r_[0, np.nonzero(np.diff(u_pit))[0] + 1]]
    smooth = cc.w_smooth * float(np.sum(np.diff(u_steps) ** 2))

    comp = {
        "appearance": cc.w_appearance * float(appearance),
        "taste": cc.w_taste * float(taste),
        "tenderness": cc.w_tenderness * float(tenderness),
        "surf": cc.w_surf * float(np.mean(surf_over[in_pit] ** 2)),
        "fuel": cc.w_fuel * float(fuel_frac),
        "smooth": smooth,
        "safety": float(safety),
    }
    comp["J"] = sum(comp.values())
    return comp


def set_clients(tx):
    """Install the module-level Tesseract clients used by plan/advance."""
    global _CLIENTS
    _CLIENTS = tx


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
    z_p_init: float | None = None,
    rest: dict | None = None,
    t_pull: float = 1.0e9,
    z_a_init: float | None = None,
    t_add: float = 1.0e9,
    t_add_override: float | None = None,
    z_s_init: float | None = None,
    z_d_init: float | None = None,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    t_spritz_override: float | None = None,
    z_m_init: float | None = None,
    spritz_dose: float = 0.05,
):
    """Re-optimize the remaining-horizon plan from the current state.

    The horizon is the remaining time to serve; with ``cc.opt_pull`` the
    pull is optimized inside it (the rest is what follows) and quality is
    scored at serve. Returns (u_opt, grids, hist): the schedule, its
    grids, and the optimize_schedule history (``t_wrap`` / ``t_pull`` are
    remaining-horizon relative; ``best_loss`` / ``losses`` feed
    wrap_commit). With opt_wrap off, z_w stays pinned at ``z_w_pin``
    (8 = never; -8 = already wrapped).
    """
    n_u = max(2, round(t_remaining / (0.75 * HR)))
    grids = CookGrids(
        t_end=t_remaining, dt_exchange=min(600.0, t_remaining / 4), n_u=n_u
    )
    tx = _CLIENTS
    u_opt, hist = optimize_schedule(
        theta,
        _with_a0(phi, state.a0),
        tx,
        grids=grids,
        # iters is a CAP; the plateau rule in optimize_schedule stops the
        # warm-started replan when converged (usually well before the cap).
        cc=replace(cc, n_iter=iters),
        u_init=u_init,
        ics=_ics(state, model_t_amb),
        z_w_init=z_w_init,
        z_w_pin=z_w_pin,
        leth_banked=state.leth_f,
        danger_banked_h=state.danger_h,
        leth_banked_core=state.leth_f_core,
        danger_banked_core_h=state.danger_h_core,
        t_wrap_override=t_wrap_override,
        t_pull=t_pull,
        rest=rest,
        z_p_init=z_p_init,
        z_a_init=z_a_init,
        t_add=t_add,
        t_add_override=t_add_override,
        z_s_init=z_s_init,
        z_d_init=z_d_init,
        t_spritz=t_spritz,
        dt_spritz=dt_spritz,
        t_spritz_override=t_spritz_override,
        z_m_init=z_m_init,
        spritz_dose=spritz_dose,
    )
    return u_opt, grids, hist


def wrap_due(tw, t0, dt_seg) -> bool:
    """Return True if a planned event (wrap or pull) falls in this segment."""
    return tw is not None and np.isfinite(tw) and tw < t0 + dt_seg


def wrap_commit(cand, incumbent, t0, dt_seg, j_best, j_start, tol_rel) -> bool:
    """Whether to fire an irreversible event (crutch or pull) this segment.

    The planned moment always tracks the optimizer candidate. Execution is
    gated: wrap this hour only if the candidate is due *and* either the
    incumbent is also due (two consecutive 'wrap now' votes) or this
    replan improved J from its warm start by more than ``tol_rel``.
    Same-horizon ΔJ — not a |Δt| dead-band, and not J from last hour.
    """
    if not wrap_due(cand, t0, dt_seg):
        return False
    if wrap_due(incumbent, t0, dt_seg):
        return True
    if (
        j_start is None
        or j_best is None
        or not np.isfinite(j_start)
        or not np.isfinite(j_best)
    ):
        return False
    return float(j_best) < float(j_start) - float(tol_rel) * abs(float(j_start))


def _adopt_event(cand, last_cand, t0, dt_seg, hist, cc, candidates):
    """Adopt one replan's candidate moment for an irreversible event.

    Execution is gated by ``wrap_commit``; a candidate due this segment that
    does not commit is deferred to the next segment so ridge noise cannot
    fire it. Appends the (t, cand, incumbent, j_best, j_start, commit)
    record and returns the adopted absolute time.
    """
    losses = np.asarray(hist.get("losses", []), dtype=float)
    j_best = hist.get("best_loss")
    j_start = float(losses[0]) if losses.size else j_best
    commit = wrap_commit(cand, last_cand, t0, dt_seg, j_best, j_start, cc.tol_rel)
    candidates.append(
        (
            t0,
            cand,
            np.nan if last_cand is None else float(last_cand),
            np.nan if j_best is None else float(j_best),
            np.nan if j_start is None else float(j_start),
            float(commit),
        )
    )
    if commit:
        return cand
    if wrap_due(cand, t0, dt_seg):
        return t0 + dt_seg
    return cand


def _seed_event(active, last_cand, t_abs, t_now, t_remaining):
    """Warm-start (z0, fixed time) of an event for a replan: z0 from the last candidate when the decision is live, else the fixed remaining-horizon time."""
    if active and last_cand is not None:
        frac = np.clip((last_cand - t_now) / t_remaining, 1e-3, 1 - 1e-3)
        return float(np.log(frac / (1 - frac))), 1.0e9
    if t_abs is not None:
        return None, max(0.0, t_abs - t_now)
    return None, 1.0e9


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
    t_pull_init_abs: float | None = None,
    rest: dict | None = None,
    replan_iters: int = 200,
    t_add_init_abs: float | None = None,
    t_spritz_init_abs: float | None = None,
    dt_spritz_init: float | None = None,
    spritz_dose_init: float | None = None,
):
    """Simulate a full cook under MPC or open-loop control.

    ``t_f`` is the cook length, start to serve. The meat is pulled at
    ``t_pull_init_abs`` (open-loop: verbatim; MPC with cc.opt_pull: the
    incumbent, re-timed every replan and executed under ``wrap_commit``)
    and rests under the ``rest`` preset to serve, where the terminal
    quality is read. No pull = the meat stays in the pit to serve.
    ``replan_iters`` caps each replan's optimizer iterations; the plateau
    rule (ControlConfig.tol_rel / patience) is the convergence stop.

    mode='mpc' replans each dt_replan interval from the advanced plant
    state (full-field handoff); mode='openloop' executes u_openloop
    blindly. plant_t_amb_fn(t) is the ambient-temperature disturbance the
    controller's model does NOT know. MPC mode carries an integral
    ambient Kalman filter (AmbientKF; tuning in ControlConfig): after each
    executed segment, one forward solve of the controller model predicts
    the segment-end pit temperature under the current belief, and the
    filter assimilates the innovation. Open-loop mode is blind by
    definition — it executes a fixed plan and estimates nothing.
    Wrap *planning* follows each replan's candidate; wrap *execution*
    uses ``wrap_commit`` so ridge wander cannot fire an early crutch.
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
    replan_pull = mode == "mpc" and cc.opt_pull
    t_pull_abs: float | None = t_pull_init_abs
    last_pull_cand: float | None = t_pull_init_abs
    pulled = False
    t_pull_exec_abs: float | None = None
    m_at_pull: float | None = None
    pull_history: list[tuple[float, float]] = []
    pull_candidates: list[tuple] = []
    # the chimney (one, irreversible) and the spritz policy (its start is
    # irreversible; the interval is re-optimized every replan)
    replan_add = mode == "mpc" and cc.opt_add
    t_add_abs: float | None = t_add_init_abs
    last_add_cand: float | None = t_add_init_abs
    added = False
    t_add_exec_abs: float | None = None
    add_history: list[tuple[float, float]] = []
    add_candidates: list[tuple] = []
    replan_spritz = mode == "mpc" and cc.opt_spritz
    t_spritz_abs: float | None = t_spritz_init_abs
    last_spritz_cand: float | None = t_spritz_init_abs
    spritzing = False
    t_spritz_exec_abs: float | None = None
    dt_spritz_cur = float(dt_spritz_init) if dt_spritz_init else 3600.0
    spritz_dose_cur = float(spritz_dose_init) if spritz_dose_init else 0.05
    spritz_history: list[tuple[float, float, float, float]] = []
    spritz_candidates: list[tuple] = []

    def record_segment(seg_traj, t_seg_start, dt_seg, u_trace):
        # safety banking with the CONFIGURED constants (hard-coded 60C/6K
        # here would silently disagree with a customized ControlConfig)
        nseg_t = len(seg_traj["tpit"])
        tseg = np.linspace(0.0, dt_seg, nseg_t)
        F_s, d_s = lethality_dwell(np.asarray(seg_traj["tsurf"]), tseg, cc)
        F_c, d_c = lethality_dwell(np.asarray(seg_traj["tcore"]), tseg, cc)
        state.leth_f += float(F_s)
        state.danger_h += float(d_s)
        state.leth_f_core += float(F_c)
        state.danger_h_core += float(d_c)
        traj["t"].extend((t_seg_start + tseg).tolist())
        for k in ("tpit", "tcore", "tsurf"):
            traj[k].extend(seg_traj[k].tolist())
        traj["u"].extend(np.broadcast_to(u_trace, (nseg_t,)).tolist())

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
    t_wrap_abs: float | None = t_wrap_init_abs  # execution wrap, absolute [s]
    last_cand: float | None = t_wrap_init_abs  # last optimizer wrap (warm start)
    wrapped = False
    t_wrap_exec_abs: float | None = None  # when the crutch ACTUALLY fired
    wrap_history: list[tuple[float, float]] = []  # (replan t, adopted abs wrap)
    wrap_candidates: list[tuple] = []  # (t, cand, incumbent, j_best, j_start, commit)

    # final segment shrinks to the remainder, so arbitrary horizons execute
    # exactly t_f (round(t_f/dt) previously over/under-shot)
    while state.t < t_f - 1e-6 and not pulled:
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
                # warm-start z_w from the last CANDIDATE (not a deferred
                # execution time, which would pin wrap at "now")
                seed = last_cand if last_cand is not None else t_wrap_abs
                if seed is None:
                    zw0 = 0.0
                else:
                    frac = np.clip((seed - state.t) / t_remaining, 1e-3, 1 - 1e-3)
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
            # the pull: warm-start z_p from the last candidate; without
            # replanning it, the plan still scores at serve with the fixed pull
            zp0 = None
            tp_fixed = 1.0e9
            if replan_pull and last_pull_cand is not None:
                frac_p = np.clip(
                    (last_pull_cand - state.t) / t_remaining, 1e-3, 1 - 1e-3
                )
                zp0 = float(np.log(frac_p / (1 - frac_p)))
            elif t_pull_abs is not None:
                tp_fixed = max(0.0, t_pull_abs - state.t)
            za0, ta_fixed = _seed_event(
                replan_add and not added, last_add_cand, t_add_abs, state.t, t_remaining
            )
            ta_override = (t_add_exec_abs - state.t) if added else None
            zs0, ts_fixed = _seed_event(
                replan_spritz and not spritzing,
                last_spritz_cand,
                t_spritz_abs,
                state.t,
                t_remaining,
            )
            ts_override = (t_spritz_exec_abs - state.t) if spritzing else None
            fd = (dt_spritz_cur - cc.dt_spritz_min) / (
                cc.dt_spritz_max - cc.dt_spritz_min
            )
            fd = float(np.clip(fd, 1e-3, 1 - 1e-3))
            zd0 = float(np.log(fd / (1 - fd)))
            fm = (spritz_dose_cur - cc.m_spritz_min) / (
                cc.m_spritz_max - cc.m_spritz_min
            )
            fm = float(np.clip(fm, 1e-3, 1 - 1e-3))
            zm0 = float(np.log(fm / (1 - fm)))
            u_plan, plan_grids, hist = plan(
                theta,
                phi_model,
                state,
                t_remaining,
                replace(
                    cc_seg,
                    opt_pull=replan_pull,
                    opt_add=replan_add and not added,
                    opt_spritz=replan_spritz,
                ),
                kf.x,
                u_init=u_init,
                iters=replan_iters,
                z_w_init=zw0,
                z_w_pin=zw_pin,
                t_wrap_override=tw_override,
                z_p_init=zp0,
                rest=rest,
                t_pull=tp_fixed,
                z_a_init=za0,
                t_add=ta_fixed,
                t_add_override=ta_override,
                z_s_init=zs0,
                z_d_init=zd0,
                t_spritz=ts_fixed,
                dt_spritz=dt_spritz_cur,
                t_spritz_override=ts_override,
                z_m_init=zm0,
                spritz_dose=spritz_dose_cur,
            )
            # each live decision: the planned moment tracks the candidate,
            # execution is gated by wrap_commit (deferral on ridge noise)
            if cc.opt_wrap and not wrapped:
                cand = state.t + float(hist["t_wrap"])
                t_wrap_abs = _adopt_event(
                    cand, last_cand, state.t, dt_seg, hist, cc, wrap_candidates
                )
                last_cand = cand
                wrap_history.append((state.t, t_wrap_abs))
            if replan_pull:
                cand_p = state.t + float(hist["t_pull"])
                t_pull_abs = _adopt_event(
                    cand_p, last_pull_cand, state.t, dt_seg, hist, cc, pull_candidates
                )
                last_pull_cand = cand_p
                pull_history.append((state.t, t_pull_abs))
            if replan_add and not added:
                cand_a = state.t + float(hist["t_add"])
                t_add_abs = _adopt_event(
                    cand_a, last_add_cand, state.t, dt_seg, hist, cc, add_candidates
                )
                last_add_cand = cand_a
                add_history.append((state.t, t_add_abs))
            if replan_spritz:
                dt_spritz_cur = float(hist["dt_spritz"])
                spritz_dose_cur = float(hist["spritz_dose"])
                if not spritzing:
                    cand_s = state.t + float(hist["t_spritz"])
                    t_spritz_abs = _adopt_event(
                        cand_s,
                        last_spritz_cand,
                        state.t,
                        dt_seg,
                        hist,
                        cc,
                        spritz_candidates,
                    )
                    last_spritz_cand = cand_s
                spritz_history.append(
                    (state.t, t_spritz_abs, dt_spritz_cur, spritz_dose_cur)
                )
            u_prev_plan, _prev_grids = u_plan, plan_grids
            prev_plan_t0 = state.t

        # the pull fires this segment: the plant runs through the rest to
        # serve, damper shut after the pull
        pull_now = wrap_due(t_pull_abs, state.t, dt_seg)
        if pull_now:
            tp_seg = max(0.0, t_pull_abs - state.t)
            dt_seg = t_f - state.t
        else:
            tp_seg = 1.0e9
        # the chimney and the spritz policy: executed when due, then modelled
        # continuously from their true moments (negative segment-relative
        # times), like the wrap
        if added:
            ta_seg = t_add_exec_abs - state.t
        elif wrap_due(t_add_abs, state.t, dt_seg) and not (
            pull_now and t_add_abs > t_pull_abs
        ):
            ta_seg = max(0.0, t_add_abs - state.t)
        else:
            ta_seg = 1.0e9
        if spritzing:
            ts_seg = t_spritz_exec_abs - state.t
        elif wrap_due(t_spritz_abs, state.t, dt_seg):
            ts_seg = max(0.0, t_spritz_abs - state.t)
        else:
            ts_seg = 1.0e9
        # execute the first dt_seg of the plan, sampled at exchange
        # resolution — a 2-point interp coarsened every plan into pit
        # overshoot (reference executed-J 1.162 vs its model-J 1.005,
        # mostly a surface-cap violation the plan never had)
        n_exec = max(2, round(dt_seg / 600.0) + 1)
        tt = np.linspace(0, dt_seg, n_exec)
        if mode == "mpc":
            u_seg = np.interp(tt, plan_grids.u_tknots, u_plan)
        else:
            u_seg = np.interp(state.t + tt, u_openloop_grids.u_tknots, u_openloop)
        if pull_now:
            u_seg = np.where(tt < tp_seg, u_seg, 0.0)

        t_amb = plant_t_amb_fn(state.t + dt_seg / 2)
        if wrapped:
            # blend continues from the TRUE wrap moment (negative relative
            # time), bitwise-continuous across segment boundaries — a fixed
            # saturated constant would jump a partial blend at the joint
            tw_seg = t_wrap_exec_abs - state.t
        elif (
            t_wrap_abs is not None
            and t_wrap_abs < state.t + dt_seg
            and not (pull_now and t_wrap_abs > t_pull_abs)
        ):
            tw_seg = max(0.0, t_wrap_abs - state.t)  # crutch executes NOW
            t_wrap_exec_abs = state.t + tw_seg
        else:
            tw_seg = 1.0e9
        t_seg_start = state.t
        state_before = state
        state, seg_traj = advance_plant(
            state,
            u_seg,
            dt_seg,
            theta,
            phi_plant,
            t_amb,
            t_wrap_seg=tw_seg,
            t_pull_seg=tp_seg,
            rest=rest,
            t_add_seg=ta_seg,
            t_spritz_seg=ts_seg,
            dt_spritz_seg=dt_spritz_cur,
            spritz_dose_seg=spritz_dose_cur,
        )
        if pull_now:
            pulled = True
            t_pull_exec_abs = t_seg_start + tp_seg
            m_at_pull = float(
                np.interp(
                    tp_seg,
                    np.linspace(0.0, dt_seg, len(seg_traj["m_fuel"])),
                    seg_traj["m_fuel"],
                )
            )
        if mode == "mpc" and not pull_now:
            # one forward solve: what the controller EXPECTED this segment's
            # pit end to be under its current ambient belief
            pred_grids = CookGrids(
                t_end=dt_seg, dt_exchange=min(600.0, dt_seg / 2), n_u=len(u_seg)
            )
            pred = coupled_cook(
                jnp.asarray(theta),
                jnp.asarray(_with_a0(phi_model, state_before.a0)),
                jnp.asarray(u_seg),
                _CLIENTS,
                pred_grids,
                ics=_ics(state_before, kf.x),
                t_wrap=tw_seg,
                t_add=ta_seg,
                t_spritz=ts_seg,
                dt_spritz=dt_spritz_cur,
                spritz_dose=spritz_dose_cur,
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
        if ta_seg < 1.0e8 and not added:
            added = True
            t_add_exec_abs = t_seg_start + ta_seg
        if ts_seg < 1.0e8 and not spritzing:
            spritzing = True
            t_spritz_exec_abs = t_seg_start + ts_seg
        if pull_now:
            in_pit = tt < tp_seg
            u_pit = float(np.mean(u_seg[in_pit])) if np.any(in_pit) else 0.0
            tseg = np.linspace(0.0, dt_seg, len(seg_traj["tpit"]))
            record_segment(
                seg_traj, t_seg_start, dt_seg, np.where(tseg < tp_seg, u_pit, 0.0)
            )
        else:
            record_segment(seg_traj, t_seg_start, dt_seg, float(np.mean(u_seg)))
        log(
            f"  [{mode}] t={state.t / HR:4.1f} h  Tpit={state.tpit0 - 273.15:5.1f} C  "
            f"core={seg_traj['tcore'][-1] - 273.15:5.1f} C  "
            f"C={state.c_field0[0]:.3f}  u~{np.mean(u_seg):.2f}  "
            f"T_amb={t_amb - 273.15:.0f} C  "
            f"T_amb_hat={kf.x - 273.15:.1f}±{np.sqrt(kf.p):.1f} C"
            + (f"  PULLED at {t_pull_exec_abs / HR:.2f} h -> rest" if pull_now else "")
            + (
                f"  CHIMNEY at {t_add_exec_abs / HR:.2f} h"
                if added and t_add_exec_abs >= t_seg_start
                else ""
            )
            + (
                f"  SPRITZ from {t_spritz_exec_abs / HR:.2f} h every "
                f"{dt_spritz_cur / 60:.0f} min, {1e3 * spritz_dose_cur:.0f} g/m²"
                if spritzing
                and t_spritz_exec_abs >= t_seg_start
                and t_spritz_exec_abs
                < min(
                    t_wrap_exec_abs if t_wrap_exec_abs is not None else np.inf,
                    t_pull_exec_abs if t_pull_exec_abs is not None else np.inf,
                )
                else ""
            )
        )

    doneness = float(state.c_field0[0])
    m_end = state.m0 if m_at_pull is None else m_at_pull
    m_added = (
        cc.m_chimney
        if added and (t_pull_exec_abs is None or t_add_exec_abs < t_pull_exec_abs)
        else 0.0
    )
    fuel_frac = (m0_start + m_added - m_end) / m0_start
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
    ring = 0.0 if np.isnan(state.ring0) else float(state.ring0)
    out = {k: np.asarray(v) for k, v in traj.items()} | {
        "final_state": state,
        "ring": ring,
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
        out["t_wrap_exec_abs"] = t_wrap_exec_abs
        out["wrap_history"] = wrap_history
        out["wrap_candidates"] = wrap_candidates
    out["pulled"] = pulled
    out["t_pull_abs"] = t_pull_abs
    out["t_pull_exec_abs"] = t_pull_exec_abs
    out["pull_history"] = pull_history
    out["pull_candidates"] = pull_candidates
    out["added"] = added
    out["t_add_exec_abs"] = t_add_exec_abs
    out["add_history"] = add_history
    out["add_candidates"] = add_candidates
    out["spritzing"] = spritzing
    out["t_spritz_exec_abs"] = t_spritz_exec_abs
    out["dt_spritz"] = dt_spritz_cur
    out["spritz_dose"] = spritz_dose_cur
    out["spritz_history"] = spritz_history
    out["spritz_candidates"] = spritz_candidates
    out["t_amb_hat_hist"] = t_amb_hat_hist
    return out
