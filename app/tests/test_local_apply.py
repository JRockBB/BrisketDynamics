"""Each Tesseract through the LocalClient boundary: shapes, physics sanity,
and the stall reproduced through Python."""

import numpy as np
import pytest

from brisket_app.clients import meat_client, pit_client
from brisket_app.coupled import HR, THETA_DEFAULT, PHI_DEFAULT, CookGrids


@pytest.fixture(scope="module")
def grids():
    return CookGrids()


def test_meat_apply_stall(grids):
    out = meat_client().apply(
        {
            "theta": THETA_DEFAULT,
            "tpit_knots": np.full(grids.nt, 383.15),
            "tknots": grids.tknots,
            "tsave": grids.tknots,
            "abstol": 1e-6,
            "reltol": 1e-6,
        }
    )
    tcore = np.asarray(out["tcore"]) - 273.15
    assert out["tcore"].shape == (grids.nt,)
    assert out["c_field_final"].shape == (40,)
    assert np.all(np.asarray(out["j_evap"]) >= 0)
    # stall through the boundary: >2 h of slow rise between 60 and 82 °C
    dt_h = grids.dt_exchange / HR
    rate = np.diff(tcore) / dt_h
    in_stall = (rate < 1.5) & (tcore[:-1] >= 60) & (tcore[:-1] <= 82)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, in_stall.astype(int), 0])))[::2]
    assert runs.max(initial=0) * dt_h > 2.0


def test_pit_apply_hold(grids):
    out = pit_client().apply(
        {
            "phi": PHI_DEFAULT,
            "u_knots": np.full(grids.n_u, 0.4),
            "u_tknots": grids.u_tknots,
            "qmeat_knots": np.zeros(grids.nt),
            "q_tknots": grids.tknots,
            "tsave": grids.tknots,
        }
    )
    tpit_f = (np.asarray(out["tpit"]) - 273.15) * 9 / 5 + 32
    assert out["tpit"].shape == (grids.nt,)
    dt_h = grids.dt_exchange / HR
    assert np.sum((tpit_f >= 225) & (tpit_f <= 275)) * dt_h > 8.0


def test_meat_vjp_endpoint_runs(grids):
    """VJP endpoint contract: right keys and shapes, finite values."""
    inputs = {
        "theta": THETA_DEFAULT,
        "tpit_knots": np.full(grids.nt, 383.15),
        "tknots": grids.tknots,
        "tsave": grids.tknots,
        "abstol": 1e-6,
        "reltol": 1e-6,
    }
    cot = {"tcore": np.ones(grids.nt)}
    out = meat_client().vector_jacobian_product(
        inputs, vjp_inputs={"theta", "tpit_knots"}, vjp_outputs={"tcore"}, cotangent_vector=cot
    )
    assert set(out) == {"theta", "tpit_knots"}
    assert np.all(np.isfinite(out["theta"]))
    assert out["tpit_knots"].shape == (grids.nt,)
    # heating the pit heats the core: gradient w.r.t. pit knots must be ≥ 0 overall
    assert np.sum(out["tpit_knots"]) > 0
