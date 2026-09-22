"""The monolith Tesseract's endpoints agree with each other.

apply → shapes and finiteness; jacobian_vector_product along a tangent equals
the jacobian's columns times that tangent; vector_jacobian_product equals the
transposed jacobian applied to a cotangent.
"""

import numpy as np
import pytest
from brisket_app.clients import monolith_client
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids


@pytest.fixture(scope="module")
def case():
    grids = CookGrids(t_end=2 * HR, dt_exchange=600.0, n_u=4)
    inputs = {
        "theta": np.asarray(THETA_DEFAULT),
        "phi": np.asarray(PHI_DEFAULT),
        "u_knots": np.full(4, 0.5),
        "u_tknots": grids.u_tknots,
        "tsave": grids.tknots,
        "t_wrap": 0.75 * HR,
        "t_pull": 1.5 * HR,
        "t_add": 1.0 * HR,
        "t_spritz": 0.5 * HR,
        "dt_spritz": 1800.0,
        "t_amb": 300.0,
    }
    return monolith_client(), inputs, grids


def test_apply_shapes(case):
    tx, inputs, grids = case
    out = tx.apply(inputs)
    nt = grids.nt
    for k in (
        "tpit",
        "tcore",
        "tsurf",
        "m_fuel",
        "bark_s",
        "l_star",
        "pv_pit",
        "m_pan",
        "t_pan",
        "t_firebox",
        "wrap_latch",
    ):
        assert np.asarray(out[k]).shape == (nt,) and np.all(np.isfinite(out[k]))
    assert np.asarray(out["c_field_final"]).shape == (40,)
    assert 0.0 < float(out["juiciness"]) < 2.0
    assert 0.0 <= float(out["ring"]) < 0.02
    assert float(out["tcore"][-1]) > float(out["tcore"][0]) + 20.0


def test_jvp_matches_jacobian(case):
    tx, inputs, _ = case
    # the sub-minute lid and spritz pulses make timing derivatives step-size
    # sensitive: compare the two endpoints at tight solver tolerances
    inputs = {**inputs, "abstol": 1e-8, "reltol": 1e-8}
    ins = ["theta", "u_knots", "t_pull", "t_add", "t_spritz", "dt_spritz"]
    J = tx.jacobian(inputs, ins, ["tcore", "juiciness"])
    dtheta = np.zeros(8)
    dtheta[0] = 1e-9
    du = np.array([0.1, 0.0, -0.05, 0.0])
    tangent = {
        "theta": dtheta,
        "u_knots": du,
        "t_pull": np.asarray(600.0),
        "t_add": np.asarray(600.0),
        "t_spritz": np.asarray(300.0),
        "dt_spritz": np.asarray(300.0),
    }
    d = tx.jacobian_vector_product(inputs, ins, ["tcore", "juiciness"], tangent)
    for k in ("tcore", "juiciness"):
        ref = (
            np.asarray(J[k]["theta"]) @ dtheta
            + np.asarray(J[k]["u_knots"]) @ du
            + np.asarray(J[k]["t_pull"]) * 600.0
            + np.asarray(J[k]["t_add"]) * 600.0
            + np.asarray(J[k]["t_spritz"]) * 300.0
            + np.asarray(J[k]["dt_spritz"]) * 300.0
        )
        got = np.asarray(d[k])
        assert np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12) < 5e-3


def test_vjp_matches_jacobian_transpose(case):
    tx, inputs, _ = case
    J = tx.jacobian(inputs, ["phi", "t_wrap"], ["tsurf", "w_retained"])
    rng = np.random.default_rng(0)
    ct = {
        "tsurf": rng.standard_normal(len(inputs["tsave"])),
        "w_retained": np.asarray(2.0),
    }
    g = tx.vector_jacobian_product(
        inputs, ["phi", "t_wrap"], ["tsurf", "w_retained"], ct
    )
    ref_phi = (
        np.asarray(J["tsurf"]["phi"]).T @ ct["tsurf"]
        + np.asarray(J["w_retained"]["phi"]) * 2.0
    )
    ref_tw = float(
        np.asarray(J["tsurf"]["t_wrap"]) @ ct["tsurf"]
        + np.asarray(J["w_retained"]["t_wrap"]) * 2.0
    )
    assert np.allclose(
        np.asarray(g["phi"]), ref_phi, rtol=1e-6, atol=1e-9 * np.max(np.abs(ref_phi))
    )
    assert np.isclose(float(g["t_wrap"]), ref_tw, rtol=1e-6)
