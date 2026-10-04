"""M1 planner exposure through the Tesseract — the app-layer contract.

The derivative VALUES are proven at the Julia level (test_monolith.jl
FD-checks every new column); what the app layer adds is the schema, the
0-sentinels, the seconds/duration COLUMN REMAP, and the salt restart
handoff. That is all this file tests, on a short cook at production
tolerances.
"""

import numpy as np
import pytest
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig, _decisions_m1
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids

DECISIONS = ("t_salt", "dt_add")


@pytest.fixture(scope="module")
def case():
    grids = CookGrids(t_end=1.5 * HR, dt_exchange=600.0, n_u=3)
    inputs = {
        "theta": np.asarray(THETA_DEFAULT),
        "phi": np.asarray(PHI_DEFAULT),
        "u_knots": np.full(3, 0.5),
        "u_tknots": grids.u_tknots,
        "tsave": grids.tknots,
        # the splits ride on t_add
        "t_add": 0.5 * HR,
        "t_salt": 10.0 * HR,
        "dt_add": 1800.0,
        "t_amb": 300.0,
    }
    return monolith_client(), inputs, grids


def test_sentinels_inert_and_live(case):
    tx, inputs, _ = case
    off = {k: v for k, v in inputs.items() if k not in DECISIONS}
    base = tx.apply(off)
    zeros = tx.apply({**off, "t_salt": 0.0, "dt_add": 0.0})
    for k in ("tcore", "w_retained", "m_fuel"):
        assert np.array_equal(np.asarray(base[k]), np.asarray(zeros[k]))
    on = tx.apply(inputs)
    # splits feed the fire (more fuel mass), the brine salts the meat
    assert float(np.asarray(on["m_fuel"])[-1]) > float(np.asarray(base["m_fuel"])[-1])
    assert float(np.max(np.asarray(on["salt_field_final"]))) > 1e-3


def test_column_remap(case):
    # jvp against jacobian columns exercises the python->julia index
    # remap in BOTH time modes — the one thing this layer can get wrong
    tx, inputs, _ = case
    outs = ["w_retained", "m_fuel"]
    J = tx.jacobian(inputs, list(DECISIONS), outs)
    tangent = {"t_salt": np.asarray(1800.0), "dt_add": np.asarray(300.0)}
    d = tx.jacobian_vector_product(inputs, list(DECISIONS), outs, tangent)
    for k in outs:
        ref = sum(np.asarray(J[k][n]) * float(tangent[n]) for n in DECISIONS)
        assert np.max(np.abs(np.asarray(d[k]) - ref)) / (
            np.max(np.abs(ref)) + 1e-12
        ) < 2e-2, k  # adaptive-step noise between Dual widths (see monolith tests)
    assert np.max(np.abs(np.asarray(J["m_fuel"]["dt_add"]))) > 0  # live

    tf_inputs = {
        **{k: v for k, v in inputs.items()
           if k not in ("u_tknots", "tsave", "t_add")},
        "u_tknots": np.linspace(0.0, 1.0, 3),
        "tsave": np.linspace(0.0, 1.0, 10),
        "t_f": 1.5 * HR,
        "t_add": 0.4,  # fractions in duration mode
        "dt_add": 0.333,
    }
    # t_f sits LAST in the python layout but first in the julia tf vector;
    # t_salt / dt_add follow the event tail in both
    ins = ["t_f", "t_salt", "dt_add"]
    Jf = tx.jacobian(tf_inputs, ins, outs)
    tangent = {"t_f": np.asarray(600.0), "t_salt": np.asarray(1800.0),
               "dt_add": np.asarray(0.02)}
    df = tx.jacobian_vector_product(tf_inputs, ins, outs, tangent)
    for k in outs:
        ref = sum(np.asarray(Jf[k][n]) * float(tangent[n]) for n in ins)
        assert np.max(np.abs(np.asarray(df[k]) - ref)) / (
            np.max(np.abs(ref)) + 1e-12
        ) < 2e-2, k  # adaptive-step noise between Dual widths (see monolith tests)


def test_salt_restart_contract(case):
    # a salted cook split in half and restarted from the handed-off
    # fields (salt included) matches the unsplit cook. Kept out of the
    # comparison: the fuel add (its pulse would straddle the seam)
    tx, inputs0, _ = case
    inputs = dict(inputs0)
    inputs["t_add"] = 1.0e9
    inputs["dt_add"] = 0.0
    whole = tx.apply(inputs)
    gh = CookGrids(t_end=0.75 * HR, dt_exchange=450.0, n_u=3)
    seg1 = tx.apply({**inputs, "u_tknots": gh.u_tknots, "tsave": gh.tknots})
    seg2 = tx.apply(
        {
            **{k: v for k, v in inputs.items() if k != "t_salt"},
            "u_tknots": gh.u_tknots,
            "tsave": gh.tknots,
            "t_salt": 0.0,
            "tpit0": float(np.asarray(seg1["tpit"])[-1]),
            "m0": float(np.asarray(seg1["m_fuel"])[-1]),
            "phi": np.concatenate(
                [np.asarray(PHI_DEFAULT)[:6],
                 [float(np.asarray(seg1["ignited"])[-1])]]
            ),
            "t_field0": np.asarray(seg1["t_field_final"]),
            "w_field0": np.asarray(seg1["w_field_final"]),
            "f_field0": np.asarray(seg1["f_field_final"]),
            "s_field0": np.asarray(seg1["salt_field_final"]),
            "c_field0": np.asarray(seg1["c_field_final"]),
            "bark_b0": float(np.asarray(seg1["bark_b"])[-1]),
            "bark_s0": float(np.asarray(seg1["bark_s"])[-1]),
            "ring0": float(np.asarray(seg1["ring"])),
        }
    )
    assert np.max(np.abs(np.asarray(seg2["salt_field_final"]) -
                         np.asarray(whole["salt_field_final"]))) < 5e-3
    assert abs(float(np.asarray(seg2["tcore"])[-1]) -
               float(np.asarray(whole["tcore"])[-1])) < 2.0
    # dropping the salt field on restart would zero it: prove it carried
    assert float(np.max(np.asarray(seg2["salt_field_final"]))) > 1e-3


def test_planner_maps():
    cc = ControlConfig(opt_salt=True, opt_splits=True)
    tsalt, tda = _decisions_m1(cc, 0.0, 0.0, 0.0, 0.0)
    assert tsalt == pytest.approx(cc.t_salt_max / 2)
    assert tda == pytest.approx((cc.dt_add_min + cc.dt_add_max) / 2)
    off = ControlConfig()
    assert _decisions_m1(off, 5.0, 5.0, 0.0, 0.0) == (0.0, 0.0)
