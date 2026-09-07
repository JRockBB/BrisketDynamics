"""MPC receding-horizon correctness: the review-mandated invariants.

Covers: cumulative safety banking (lethality + come-up time), configured
(non-default) safety constants reaching the banking, wrap-blend continuity
across segment boundaries, and horizons not divisible by the replan
interval. Open-loop mode throughout (no optimization) so each test is a
handful of short coupled solves.
"""

import numpy as np
import pytest
from brisket_app import mpc
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig
from brisket_app.coupled import HR, CookGrids, coupled_cook

T_AMB = 300.0


@pytest.fixture(scope="module")
def clients():
    mpc.set_clients(monolith_client())
    return mpc._CLIENTS


@pytest.fixture(scope="module")
def theta_phi():
    from brisket_app.coupled import PHI_DEFAULT, THETA_DEFAULT

    return np.asarray(THETA_DEFAULT), np.asarray(PHI_DEFAULT)


def _openloop(theta, phi, t_f, cc=None, t_wrap=None, n_u=4):
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=n_u)
    u = np.full(n_u, 0.5)
    return mpc.run_cook(
        theta,
        phi,
        lambda t: T_AMB,
        t_f=t_f,
        dt_replan=1 * HR,
        cc=cc or ControlConfig(),
        mode="openloop",
        u_openloop=u,
        u_openloop_grids=grids,
        model_t_amb=T_AMB,
        t_wrap_init_abs=t_wrap,
        log=lambda *_: None,
    )


def test_cumulative_safety_banking(clients, theta_phi):
    """Banked lethality/come-up equal the whole-cook integrals of the
    executed surface trace (segment banking must not reset or drift)."""
    theta, phi = theta_phi
    cc = ControlConfig()
    out = _openloop(theta, phi, 2 * HR, cc=cc)
    st = out["final_state"]
    t = np.asarray(out["t"])
    ts = np.asarray(out["tsurf"])
    leth_ref = float(np.trapezoid(10.0 ** ((ts - cc.leth_t_ref) / cc.leth_z), t))
    in_zone = 1.0 / (1.0 + np.exp(-(ts - cc.cut_t_lo) / 2.0))
    in_zone *= 1.0 / (1.0 + np.exp(-(cc.cut_t_hi - ts) / 2.0))
    cut_ref = float(np.trapezoid(in_zone, t / 3600.0))
    assert st.leth_f > 0
    assert st.cut_h > 0
    # per-segment trapezoids vs whole-trace trapezoid differ only at the
    # segment joints
    assert st.leth_f == pytest.approx(leth_ref, rel=0.05)
    assert st.cut_h == pytest.approx(cut_ref, rel=0.05)
    # the cold spot is banked the same way and is the slower of the two
    tc = np.asarray(out["tcore"])
    leth_core_ref = float(np.trapezoid(10.0 ** ((tc - cc.leth_t_ref) / cc.leth_z), t))
    assert st.leth_f_core == pytest.approx(leth_core_ref, rel=0.05)
    assert st.leth_f_core <= st.leth_f
    assert st.cut_h_core >= st.cut_h


def test_cut_reported_not_binding_for_brisket():
    """FSIS Scientific Gap 6: a brisket's long come-up is reported, never a
    penalty or a rejection; a cut flagged critical is penalized and rejected."""
    from brisket_app.control import safety_penalty, safety_report

    cc = ControlConfig()
    t = np.linspace(0.0, 10 * HR, 601)
    # 8 h climbing through 10-52 C, then pasteurizing at 75 C
    T = np.where(t < 8 * HR, 283.15 + (325.0 - 283.15) * t / (8 * HR), 348.15)
    rep = safety_report(T, T, t, cc)
    assert rep["cut_h"] > cc.cut_cap_h and rep["cut_ok"] is False
    assert rep["lethality_ok"] and rep["ok"]
    assert safety_penalty(rep["F_s"], rep["cut_h"], cc) == pytest.approx(0.0, abs=1e-6)
    crit = ControlConfig(cut_critical=True)
    rep2 = safety_report(T, T, t, crit)
    assert rep2["ok"] is False and rep2["cut_critical"] is True
    assert safety_penalty(rep2["F_s"], rep2["cut_h"], crit) > 1.0
    # and the come-up window is the FSIS one: a trace parked at 70 C is out of
    # it (the smooth 2 K edge still leaks ~6 % at 60 C, by construction)
    hot = np.full_like(t, 343.15)
    assert safety_report(hot, hot, t, cc)["cut_h"] < 0.05


def test_custom_safety_constants_reach_banking(clients, theta_phi):
    """A customized ControlConfig must drive the banked integrals (a
    hard-coded 60C/6K banking would ignore it)."""
    theta, phi = theta_phi
    cc2 = ControlConfig(leth_t_ref=343.15, leth_z=8.0, cut_t_hi=343.15)
    out = _openloop(theta, phi, 2 * HR, cc=cc2)
    st = out["final_state"]
    t = np.asarray(out["t"])
    ts = np.asarray(out["tsurf"])
    leth_ref = float(np.trapezoid(10.0 ** ((ts - cc2.leth_t_ref) / cc2.leth_z), t))
    assert st.leth_f == pytest.approx(leth_ref, rel=0.05)
    # and it differs materially from the default-constant integral
    leth_default = float(np.trapezoid(10.0 ** ((ts - 333.15) / 6.0), t))
    assert abs(st.leth_f - leth_default) > 0.1 * leth_default


def test_wrap_continuity_across_segments(clients, theta_phi):
    """A wrap executed mid-segment must continue smoothly through later
    segment boundaries: the segmented plant matches a single whole-horizon
    solve with the same absolute wrap time."""
    theta, phi = theta_phi
    t_f, t_wrap = 2 * HR, 0.75 * HR
    out = _openloop(theta, phi, t_f, t_wrap=t_wrap)
    assert out["wrapped"] is True

    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=4)
    whole = coupled_cook(
        np.asarray(theta),
        np.asarray(phi),
        np.full(4, 0.5),
        mpc._CLIENTS,
        grids,
        ics={"tpit0": 293.15, "t_amb": T_AMB, "t0_meat": 281.15},
        t_wrap=t_wrap,
    )
    # tolerances cover segmentation (per-segment solves with a state handoff),
    # NOT wrap discontinuity — the old saturated-constant reset produced
    # multi-K jumps at the joint
    assert float(out["tcore"][-1]) == pytest.approx(
        float(np.asarray(whole["tcore"])[-1]), abs=1.0
    )
    assert float(out["tsurf"][-1]) == pytest.approx(
        float(np.asarray(whole["tsurf"])[-1]), abs=1.5
    )
    # no jump AT THE SEGMENT JOINTS (duplicated save times mark them):
    # the old saturated-constant wrap reset jumped the partial blend there,
    # while the initial heating ramp legitimately steps >10 K per save
    t = np.asarray(out["t"])
    ts = np.asarray(out["tsurf"])
    joints = np.nonzero(np.diff(t) == 0.0)[0]
    assert joints.size >= 1  # 2 h / 1 h replan => one interior joint
    for j in joints:
        assert abs(ts[j + 1] - ts[j]) < 0.5


def test_horizon_not_divisible_by_replan(clients, theta_phi):
    """t_f = 1.5 h with dt_replan = 1 h must execute exactly 1.5 h
    (round() previously ran 2 full hours)."""
    theta, phi = theta_phi
    t_f = 1.5 * HR
    out = _openloop(theta, phi, t_f)
    st = out["final_state"]
    assert st.t == pytest.approx(t_f, abs=1.0)
    assert float(np.asarray(out["t"])[-1]) == pytest.approx(t_f, rel=1e-6)


def test_cookgrids_endpoint_aligned():
    """Exchange grid includes t=0 and t=t_end exactly (arange previously drifted)."""
    g = CookGrids(t_end=1.5 * HR, dt_exchange=600.0, n_u=4)
    assert g.tknots[0] == 0.0
    assert g.tknots[-1] == pytest.approx(1.5 * HR)
    g2 = CookGrids(t_end=12 * HR, dt_exchange=600.0, n_u=16)
    assert g2.tknots[-1] == pytest.approx(12 * HR)
    assert np.allclose(np.diff(g2.tknots), g2.dt_exchange)


def test_nowrap_objective_matches_wrap_never(clients, theta_phi):
    """opt_wrap=False must be WRAP_NEVER, not a late-horizon sigmoid wrap."""
    import jax.numpy as jnp
    from brisket_app.control import ControlConfig, objective

    theta, phi = theta_phi
    tx = clients
    grids = CookGrids(t_end=2 * HR, dt_exchange=600.0, n_u=4)
    z = jnp.zeros(grids.n_u)
    z_w = jnp.asarray(8.0)
    tols = {"abstol": 1e-6, "reltol": 1e-6}
    cc = ControlConfig(opt_wrap=False)
    j_off = float(
        objective(
            z,
            z_w,
            jnp.asarray(theta),
            jnp.asarray(phi),
            tx,
            grids,
            cc,
            tols,
        )
    )
    j_never = float(
        objective(
            z,
            z_w,
            jnp.asarray(theta),
            jnp.asarray(phi),
            tx,
            grids,
            cc,
            tols,
            t_wrap_override=1.0e9,
        )
    )
    assert j_off == pytest.approx(j_never, rel=0, abs=1e-9)


def test_wrap_commit_agreement_or_dj():
    """Fire this hour only if the candidate is due and (incumbent is also
    due or same-horizon J improved past tol_rel)."""
    t0, dt = 4 * HR, 1 * HR
    due = 4.2 * HR
    later = 6.0 * HR
    # both due → commit even with a flat J
    assert mpc.wrap_commit(due, 4.1 * HR, t0, dt, j_best=1.0, j_start=1.0, tol_rel=1e-4)
    # candidate due, incumbent later, no improvement → hold
    assert not mpc.wrap_commit(
        due, later, t0, dt, j_best=1.0, j_start=1.0, tol_rel=1e-4
    )
    # candidate due, incumbent later, real same-horizon ΔJ → commit
    assert mpc.wrap_commit(due, later, t0, dt, j_best=0.8, j_start=1.0, tol_rel=1e-4)
    # candidate still in the future → never fire this hour
    assert not mpc.wrap_commit(
        5.5 * HR, 4.1 * HR, t0, dt, j_best=0.8, j_start=1.0, tol_rel=1e-4
    )
    # overdue incumbent still counts as due
    assert mpc.wrap_commit(due, 3.9 * HR, t0, dt, j_best=1.0, j_start=1.0, tol_rel=1e-4)
    assert mpc.wrap_due(3.9 * HR, t0, dt)
    assert not mpc.wrap_due(5.5 * HR, t0, dt)


def test_pull_rests_to_serve(clients, theta_phi):
    """A pulled cook keeps running to serve: the meat leaves the pit at the
    pull, the damper shuts, the surface cools in the cambro, and the
    executed objective scores the pit-only terms up to the pull."""
    theta, phi = theta_phi
    t_f, t_pull = 2 * HR, 1.25 * HR
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=4)
    out = mpc.run_cook(
        theta,
        phi,
        lambda t: T_AMB,
        t_f=t_f,
        dt_replan=1 * HR,
        cc=ControlConfig(),
        mode="openloop",
        u_openloop=np.full(4, 0.5),
        u_openloop_grids=grids,
        model_t_amb=T_AMB,
        t_pull_init_abs=t_pull,
        log=lambda *_: None,
    )
    t, ts, u = (np.asarray(out[k]) for k in ("t", "tsurf", "u"))
    assert out["pulled"] and out["t_pull_exec_abs"] == pytest.approx(t_pull, abs=1.0)
    assert t[-1] == pytest.approx(t_f, rel=1e-6)
    assert out["final_state"].t == pytest.approx(t_f, abs=1.0)
    assert np.all(u[t > t_pull + 1.0] == 0.0)
    i_pull = int(np.searchsorted(t, t_pull))
    assert ts[-1] < ts[i_pull] - 1.0
    comp = mpc.executed_objective(out, ControlConfig(), t_f)
    assert np.isfinite(comp["J"])
    # no pull: the meat stays in the pit to the end
    ref = _openloop(theta, phi, t_f)
    assert ref["pulled"] is False and ref["t_pull_exec_abs"] is None
    assert float(np.asarray(ref["tsurf"])[-1]) > ts[-1]


def test_plan_optimizes_pull(clients, theta_phi):
    """A replan with cc.opt_pull returns a pull inside the remaining horizon."""
    theta, phi = theta_phi
    state = mpc.CookState.fresh()
    _u, grids, hist = mpc.plan(
        theta,
        phi,
        state,
        2 * HR,
        ControlConfig(opt_wrap=False),
        T_AMB,
        iters=2,
        z_p_init=1.0,
    )
    assert grids.t_end == pytest.approx(2 * HR)
    assert 0.0 < hist["t_pull"] < 2 * HR


def test_fuel_add_executes_and_is_priced(clients, theta_phi):
    """An open-loop chimney at 1.5 h lands in the plant (fuel mass up by one
    chimney) and the executed fuel accounting counts the added mass."""
    theta, phi = theta_phi
    t_f = 3 * HR
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=4)
    kw = {
        "t_f": t_f,
        "dt_replan": 1 * HR,
        "cc": ControlConfig(),
        "mode": "openloop",
        "u_openloop": np.full(4, 0.5),
        "u_openloop_grids": grids,
        "model_t_amb": T_AMB,
        "log": lambda *_: None,
    }
    ref = mpc.run_cook(theta, phi, lambda t: T_AMB, **kw)
    fed = mpc.run_cook(theta, phi, lambda t: T_AMB, t_add_init_abs=1.5 * HR, **kw)
    assert fed["added"] and fed["t_add_exec_abs"] == pytest.approx(1.5 * HR, abs=1.0)
    assert ref["added"] is False
    gained = fed["final_state"].m0 - ref["final_state"].m0
    assert gained == pytest.approx(ControlConfig().m_chimney, rel=0.1)
    # fuel burned is what is priced: the same fire, so the same fraction
    assert fed["fuel_frac"] == pytest.approx(ref["fuel_frac"], abs=0.03)


def test_spritz_policy_executes(clients, theta_phi):
    """An open-loop spritz policy from 1 h every 30 min wets and cools the
    surface in the plant, continuously across segment boundaries."""
    theta, phi = theta_phi
    t_f = 3 * HR
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=4)
    kw = {
        "t_f": t_f,
        "dt_replan": 1 * HR,
        "cc": ControlConfig(),
        "mode": "openloop",
        "u_openloop": np.full(4, 0.5),
        "u_openloop_grids": grids,
        "model_t_amb": T_AMB,
        "log": lambda *_: None,
    }
    ref = mpc.run_cook(theta, phi, lambda t: T_AMB, **kw)
    sp = mpc.run_cook(
        theta,
        phi,
        lambda t: T_AMB,
        t_spritz_init_abs=1.0 * HR,
        dt_spritz_init=1800.0,
        spritz_dose_init=0.08,
        **kw,
    )
    assert sp["spritzing"] and sp["t_spritz_exec_abs"] == pytest.approx(
        1.0 * HR, abs=1.0
    )
    ts_ref, ts_sp = np.asarray(ref["tsurf"]), np.asarray(sp["tsurf"])
    t = np.asarray(sp["t"])
    assert np.mean(ts_sp[t > 1.5 * HR]) < np.mean(ts_ref[t > 1.5 * HR])


def test_plan_optimizes_add_and_spritz(clients, theta_phi):
    """A replan returns a chimney time and a spritz policy inside the horizon and bounds."""
    theta, phi = theta_phi
    state = mpc.CookState.fresh()
    cc = ControlConfig(opt_wrap=False)
    _u, _grids, hist = mpc.plan(theta, phi, state, 2 * HR, cc, T_AMB, iters=2)
    assert 0.0 < hist["t_add"] < 2 * HR
    assert 0.0 < hist["t_spritz"] < 2 * HR
    assert cc.dt_spritz_min <= hist["dt_spritz"] <= cc.dt_spritz_max
    assert cc.m_spritz_min <= hist["spritz_dose"] <= cc.m_spritz_max
