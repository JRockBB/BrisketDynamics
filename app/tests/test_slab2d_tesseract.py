"""The 2-D Tesseract's endpoints and the shadow-scoring wrapper.

Small grids, short cook: apply shapes/finiteness and physical sanity
(the true cold spot sits below the axis core), a jacobian column against
finite differences, jvp-vs-jacobian consistency, and one shadow report.
"""

import numpy as np
import pytest
from brisket_app.clients import slab2d_client
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids
from brisket_app.shadow2d import THETA2_IDX, shadow_report

GEOM = {"radius": 0.05, "height": 0.05, "nr": 10, "nz": 10, "h_step": 30.0}


@pytest.fixture(scope="module")
def case():
    grids = CookGrids(t_end=2 * HR, dt_exchange=600.0, n_u=4)
    theta = np.asarray(THETA_DEFAULT)
    inputs = {
        "theta2": theta[list(THETA2_IDX)],
        "phi": np.asarray(PHI_DEFAULT),
        "u_knots": np.full(4, 0.5),
        "u_tknots": grids.u_tknots,
        "tsave": grids.tknots,
        "t_wrap": 1.0 * HR,
        "t_salt": 10.0 * HR,
        "t_amb": 300.0,
        **GEOM,
    }
    return slab2d_client(), inputs, grids


def test_apply_sane(case):
    tx, inputs, grids = case
    out = tx.apply(inputs)
    nt = grids.nt
    for k in ("tpit", "tcore", "tcold", "m_fuel", "pv_pit", "j_evap"):
        assert np.asarray(out[k]).shape == (nt,) and np.all(np.isfinite(out[k])), k
    tcold = np.asarray(out["tcold"])
    tcore = np.asarray(out["tcore"])
    assert np.all(tcold <= tcore + 1e-9)  # the TRUE cold spot
    assert float(out["w_retained"]) < 1.0
    assert 0.0 <= float(out["c_min"]) < 1.0
    assert float(out["salt_mean"]) > 0.0
    assert float(out["r_eff"]) < GEOM["radius"]  # shrink is on
    assert float(out["hz_eff"]) < GEOM["height"]


def test_jacobian_matches_fd(case):
    tx, inputs, _ = case
    J = tx.jacobian(inputs, ["t_wrap"], ["tcore", "w_retained"])
    h = 60.0
    fp = tx.apply({**inputs, "t_wrap": inputs["t_wrap"] + h})
    fm = tx.apply({**inputs, "t_wrap": inputs["t_wrap"] - h})
    for k in ("tcore", "w_retained"):
        fd = (np.asarray(fp[k], dtype=float) - np.asarray(fm[k], dtype=float)) / (
            2 * h
        )
        col = np.asarray(J[k]["t_wrap"], dtype=float)
        scale = np.max(np.abs(fd)) + 1e-12
        assert np.max(np.abs(col - fd)) / scale < 0.05, k
    assert np.max(np.abs(np.asarray(J["w_retained"]["t_wrap"]))) > 0


def test_jvp_and_vjp_match_jacobian(case):
    tx, inputs, _ = case
    ins = ["theta2", "phi", "t_wrap", "t_salt"]
    outs = ["tcore", "w_retained", "bark_mean"]
    J = tx.jacobian(inputs, ins, outs)
    dth = np.zeros(6)
    dth[0] = 1e-10
    dph = np.zeros(7)
    dph[1] = 1.0
    tangent = {
        "theta2": dth,
        "phi": dph,
        "t_wrap": np.asarray(120.0),
        "t_salt": np.asarray(1800.0),
    }
    d = tx.jacobian_vector_product(inputs, ins, outs, tangent)
    for k in outs:
        ref = (
            np.asarray(J[k]["theta2"]) @ dth
            + np.asarray(J[k]["phi"]) @ dph
            + np.asarray(J[k]["t_wrap"]) * 120.0
            + np.asarray(J[k]["t_salt"]) * 1800.0
        )
        got = np.asarray(d[k])
        assert np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12) < 5e-3, k
    ct = {"tcore": np.ones(len(inputs["tsave"])), "w_retained": np.asarray(3.0)}
    g = tx.vector_jacobian_product(inputs, ["phi", "t_salt"], list(ct), ct)
    ref_phi = np.asarray(J["tcore"]["phi"]).T @ ct["tcore"] + np.asarray(
        J["w_retained"]["phi"]
    ) * 3.0
    assert np.allclose(np.asarray(g["phi"]), ref_phi, rtol=1e-6,
                       atol=1e-9 * np.max(np.abs(ref_phi)) + 1e-30)
    assert np.isfinite(float(g["t_salt"]))


def test_shadow_report(case):
    tx, _, grids = case
    rep = shadow_report(
        np.asarray(THETA_DEFAULT),
        np.asarray(PHI_DEFAULT),
        np.full(4, 0.5),
        grids,
        t_wrap=1.0 * HR,
        t_salt=10.0 * HR,
        tx=tx,
        radius=GEOM["radius"],
        nr=GEOM["nr"],
        nz=GEOM["nz"],
        t_amb=300.0,
    )
    assert rep["F_cold"] <= rep["F_core"] + 1e-9  # the cold spot lags
    assert rep["cold_gap_K"] >= -1e-9
    assert 0 < rep["w_retained"] < 1
    assert rep["hz_eff_cm"] < 2.0 * float(THETA_DEFAULT[3]) * 100
    assert "cold_ok" in rep and isinstance(rep["cold_ok"], bool)
