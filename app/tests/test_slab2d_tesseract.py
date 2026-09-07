"""The 2-D Tesseract — app-layer contract only.

The solver and its gradients are proven at the Julia level
(test_slab2d.jl); here: apply shapes/physical sanity on a tiny grid, one
jvp-vs-jacobian remap check, and the shadow report.
"""

import numpy as np
import pytest
from brisket_app.clients import slab2d_client
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids
from brisket_app.shadow2d import THETA2_IDX, shadow_report

GEOM = {"radius": 0.05, "height": 0.05, "nr": 8, "nz": 8, "h_step": 30.0}


@pytest.fixture(scope="module")
def case():
    grids = CookGrids(t_end=1.5 * HR, dt_exchange=600.0, n_u=3)
    theta = np.asarray(THETA_DEFAULT)
    inputs = {
        "theta2": theta[list(THETA2_IDX)],
        "phi": np.asarray(PHI_DEFAULT),
        "u_knots": np.full(3, 0.5),
        "u_tknots": grids.u_tknots,
        "tsave": grids.tknots,
        "t_wrap": 0.75 * HR,
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
    assert np.all(np.asarray(out["tcold"]) <= np.asarray(out["tcore"]) + 1e-9)
    assert float(out["w_retained"]) < 1.0
    assert 0.0 <= float(out["c_min"]) < 1.0
    assert float(out["salt_mean"]) > 0.0
    assert float(out["r_eff"]) < GEOM["radius"]  # shrink is on
    assert float(out["hz_eff"]) < GEOM["height"]


def test_remap_and_shadow(case):
    tx, inputs, grids = case
    ins = ["t_wrap", "t_salt"]
    outs = ["w_retained", "bark_mean"]
    J = tx.jacobian(inputs, ins, outs)
    tangent = {"t_wrap": np.asarray(120.0), "t_salt": np.asarray(1800.0)}
    d = tx.jacobian_vector_product(inputs, ins, outs, tangent)
    for k in outs:
        ref = sum(np.asarray(J[k][n]) * float(tangent[n]) for n in ins)
        assert np.max(np.abs(np.asarray(d[k]) - ref)) / (
            np.max(np.abs(ref)) + 1e-12
        ) < 2e-2, k  # adaptive-step noise between Dual widths (see monolith tests)
    assert np.max(np.abs(np.asarray(J["w_retained"]["t_wrap"]))) > 0

    rep = shadow_report(
        np.asarray(THETA_DEFAULT), np.asarray(PHI_DEFAULT), np.full(3, 0.5),
        grids, t_wrap=0.75 * HR, t_salt=10.0 * HR, tx=tx,
        radius=GEOM["radius"], nr=GEOM["nr"], nz=GEOM["nz"], t_amb=300.0,
    )
    assert rep["F_cold"] <= rep["F_core"] + 1e-9  # the cold spot lags
    assert 0 < rep["w_retained"] < 1
    assert isinstance(rep["cold_ok"], bool)
