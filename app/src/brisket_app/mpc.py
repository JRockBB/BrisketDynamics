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

    @classmethod
    def fresh(cls, tpit0=293.15, m0=7.0, a0=1.0, t0_meat=281.15):
        """Cold-start state: ambient pit, full fuel ring, fridge meat."""
        z = np.zeros(0)
        s = cls(0.0, tpit0, m0, a0, z, z, z)
        s.t0_meat = t0_meat
        return s


def _ics(state: CookState, t_amb: float) -> dict:
    """Assemble the coupled_cook `ics` dict from a CookState."""
    ics = {
        "tpit0": state.tpit0,
        "m0": state.m0,
        "t_amb": t_amb,
        "t_field0": state.t_field0,
        "w_field0": state.w_field0,
        "c_field0": state.c_field0,
    }
    if state.t_field0.size == 0 and hasattr(state, "t0_meat"):
        ics["t0_meat"] = state.t0_meat
    return ics


def _with_a0(phi: np.ndarray, a0: float) -> np.ndarray:
    """Return phi with its ignition entry replaced by the plant's current a0."""
    out = np.asarray(phi, dtype=np.float64).copy()
    out[6] = np.clip(a0, 1e-3, 1.0)
    return out


def advance_plant(
    state: CookState, u_seg: np.ndarray, dt: float, theta, phi, t_amb: float, tols=None
) -> tuple[CookState, dict]:
    """Run the plant one interval; return (new state, trajectories)."""
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
    )
    return new, {
        k: np.asarray(sim[k]) for k in ("tpit", "tcore", "tsurf", "m_fuel", "ignited")
    }


_CLIENTS = None  # set by set_clients()


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
    iters=60,
):
    """Re-optimize the remaining-horizon schedule from the current state."""
    n_u = max(2, round(t_remaining / (0.75 * HR)))
    grids = CookGrids(
        t_end=t_remaining, dt_exchange=min(600.0, t_remaining / 4), n_u=n_u
    )
    meat_tx, pit_tx = _CLIENTS
    u_opt, _hist = optimize_schedule(
        theta,
        _with_a0(phi, state.a0),
        meat_tx,
        pit_tx,
        grids=grids,
        cc=replace(cc, n_iter=iters),
        u_init=u_init,
        ics=_ics(state, model_t_amb),
    )
    return u_opt, grids


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
):
    """Simulate a full cook under MPC or open-loop control.

    mode='mpc' replans each dt_replan interval from the advanced plant
    state (full-field handoff); mode='openloop' executes u_openloop
    blindly. plant_t_amb_fn(t) is the ambient-temperature disturbance the
    controller's model does NOT know.

    Returns
    -------
    dict
        Trajectories t/tpit/tcore/tsurf/u plus final core doneness.
    """
    cc = cc or ControlConfig()
    phi_model = phi_plant if phi_model is None else phi_model
    state = CookState.fresh(t0_meat=281.15)
    traj = {k: [] for k in ("t", "tpit", "tcore", "tsurf", "u")}
    n_seg = round(t_f / dt_replan)
    u_prev_plan, _prev_grids = None, None

    for _seg in range(n_seg):
        t_remaining = t_f - state.t
        if mode == "mpc":
            cc_mpc = cc  # the default objective already carries the band
            if u_prev_plan is None:
                u_init = 0.5
            else:
                # shift-and-interpolate the previous plan onto the new
                # remaining-horizon knots (standard MPC warm start): each
                # replan refines the last instead of restarting.
                prev_t = _prev_grids.u_tknots + (state.t - dt_replan)
                n_u_new = max(2, round(t_remaining / (0.75 * HR)))
                new_t = state.t + np.linspace(0, t_remaining, n_u_new)
                u_init = np.clip(np.interp(new_t, prev_t, u_prev_plan), 1e-3, 1 - 1e-3)
            u_plan, plan_grids = plan(
                theta, phi_model, state, t_remaining, cc_mpc, model_t_amb, u_init=u_init
            )
            u_prev_plan, _prev_grids = u_plan, plan_grids
            # execute the first dt_replan of the plan
            tt = np.linspace(0, dt_replan, 2)
            u_seg = np.interp(tt, plan_grids.u_tknots, u_plan)
        else:
            tt = state.t + np.linspace(0, dt_replan, 2)
            u_seg = np.interp(tt, u_openloop_grids.u_tknots, u_openloop)

        t_amb = plant_t_amb_fn(state.t + dt_replan / 2)
        state, seg_traj = advance_plant(
            state, u_seg, dt_replan, theta, phi_plant, t_amb
        )
        nseg_t = len(seg_traj["tpit"])
        traj["t"].extend(
            (state.t - dt_replan + np.linspace(0, dt_replan, nseg_t)).tolist()
        )
        for k in ("tpit", "tcore", "tsurf"):
            traj[k].extend(seg_traj[k].tolist())
        traj["u"].extend([float(np.mean(u_seg))] * nseg_t)
        log(
            f"  [{mode}] t={state.t / HR:4.1f} h  Tpit={state.tpit0 - 273.15:5.1f} C  "
            f"core={seg_traj['tcore'][-1] - 273.15:5.1f} C  "
            f"C={state.c_field0[0]:.3f}  u~{np.mean(u_seg):.2f}  "
            f"T_amb={t_amb - 273.15:.0f} C"
        )

    doneness = float(state.c_field0[0])
    return {k: np.asarray(v) for k, v in traj.items()} | {"doneness": doneness}
