"""MPC receding-horizon correctness: the review-mandated invariants.

Covers: cumulative safety banking (lethality + danger dwell), configured
(non-default) safety constants reaching the banking, wrap-blend continuity
across segment boundaries, and horizons not divisible by the replan
interval. Open-loop mode throughout (no optimization) so each test is a
handful of short coupled solves.
"""

import numpy as np
import pytest

from brisket_app import mpc
from brisket_app.clients import meat_client, pit_client
from brisket_app.control import ControlConfig
from brisket_app.coupled import HR, CookGrids, coupled_cook

T_AMB = 300.0


@pytest.fixture(scope="module")
def clients():
    mpc.set_clients(meat_client(), pit_client())
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
    """Banked lethality/danger equal the whole-cook integrals of the
    executed surface trace (segment banking must not reset or drift)."""
    theta, phi = theta_phi
    cc = ControlConfig()
    out = _openloop(theta, phi, 2 * HR, cc=cc)
    st = out["final_state"]
    t = np.asarray(out["t"])
    ts = np.asarray(out["tsurf"])
    leth_ref = float(np.trapezoid(10.0 ** ((ts - cc.leth_t_ref) / cc.leth_z), t))
    in_zone = 1.0 / (1.0 + np.exp(-(ts - cc.t_danger_lo) / 2.0))
    in_zone *= 1.0 / (1.0 + np.exp(-(cc.t_danger_hi - ts) / 2.0))
    danger_ref = float(np.trapezoid(in_zone, t / 3600.0))
    assert st.leth_f > 0
    assert st.danger_h > 0
    # per-segment trapezoids vs whole-trace trapezoid differ only at the
    # segment joints
    assert st.leth_f == pytest.approx(leth_ref, rel=0.05)
    assert st.danger_h == pytest.approx(danger_ref, rel=0.05)


def test_custom_safety_constants_reach_banking(clients, theta_phi):
    """A customized ControlConfig must drive the banked integrals (a
    hard-coded 60C/6K banking would ignore it)."""
    theta, phi = theta_phi
    cc2 = ControlConfig(leth_t_ref=343.15, leth_z=8.0, t_danger_hi=343.15)
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
        *mpc._CLIENTS,
        grids,
        K=2,
        ics={"tpit0": 293.15, "t_amb": T_AMB, "t0_meat": 281.15},
        t_wrap=t_wrap,
    )
    # tolerances cover segmentation (per-segment Picard + exchange grids),
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
