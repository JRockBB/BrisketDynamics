"""M1 planner exposure: t_salt / t_trig / dt_add through the Tesseract.

The three decisions follow the t_f pattern: optional differentiable
scalar inputs, inert at their 0 sentinels, live derivative columns when
enabled — checked against finite differences in seconds mode and against
the jacobian endpoint (the column remap) in both modes. The planner maps
are unit-checked without an optimizer run.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig, _decisions_m1, objective
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids

DECISIONS = ("t_salt", "t_trig", "dt_add")


@pytest.fixture(scope="module")
def case():
    grids = CookGrids(t_end=2 * HR, dt_exchange=600.0, n_u=4)
    inputs = {
        "theta": np.asarray(THETA_DEFAULT),
        "phi": np.asarray(PHI_DEFAULT),
        "u_knots": np.full(4, 0.5),
        "u_tknots": grids.u_tknots,
        "tsave": grids.tknots,
        # the trigger is the ONLY wrap here (t_wrap never), so its column
        # is live; the splits ride on t_add
        "t_add": 1.0 * HR,
        "t_salt": 10.0 * HR,
        "t_trig": 305.0,
        "dt_add": 2700.0,
        "t_amb": 300.0,
        "abstol": 1e-8,
        "reltol": 1e-8,
    }
    return monolith_client(), inputs, grids


def test_sentinels_inert(case):
    tx, inputs, _ = case
    off = {k: v for k, v in inputs.items() if k not in DECISIONS}
    base = tx.apply(off)
    zeros = tx.apply({**off, "t_salt": 0.0, "t_trig": 0.0, "dt_add": 0.0})
    for k in ("tcore", "w_retained", "m_fuel"):
        assert np.array_equal(np.asarray(base[k]), np.asarray(zeros[k]))
    on = tx.apply(inputs)
    # trigger wraps (retention up), splits feed the fire (more fuel mass)
    assert float(on["w_retained"]) > float(base["w_retained"])
    assert float(np.asarray(on["m_fuel"])[-1]) > float(np.asarray(base["m_fuel"])[-1])


def test_jacobian_matches_fd(case):
    tx, inputs, _ = case
    J = tx.jacobian(inputs, list(DECISIONS), ["tcore", "w_retained", "m_fuel"])
    steps = {"t_salt": 1800.0, "t_trig": 0.25, "dt_add": 60.0}
    # each decision is checked on the output it actually drives: on this
    # short fresh-load fixture t_salt and dt_add have ~1e-15 temperature
    # derivatives (below the adaptive-solver FD noise floor) -- salt
    # moves retention, the split interval moves the fuel mass (split
    # feeding is a firebox-door event, no chamber lid lift)
    outs = {"t_salt": ("w_retained",), "t_trig": ("tcore", "w_retained"),
            "dt_add": ("m_fuel",)}
    for name, h in steps.items():
        fp = tx.apply({**inputs, name: inputs[name] + h})
        fm = tx.apply({**inputs, name: inputs[name] - h})
        for k in outs[name]:
            fd = (np.asarray(fp[k], dtype=float) - np.asarray(fm[k], dtype=float)) / (
                2 * h
            )
            col = np.asarray(J[k][name], dtype=float)
            scale = np.max(np.abs(fd)) + 1e-12
            assert np.max(np.abs(col - fd)) / scale < 0.05, (name, k)
            assert np.max(np.abs(fd)) > 0, (name, k)  # live


def test_jvp_matches_jacobian_including_decisions(case):
    tx, inputs, _ = case
    ins = ["theta", "t_add", *DECISIONS]
    J = tx.jacobian(inputs, ins, ["tcore", "w_retained"])
    dtheta = np.zeros(8)
    dtheta[0] = 1e-9
    tangent = {
        "theta": dtheta,
        "t_add": np.asarray(600.0),
        "t_salt": np.asarray(1800.0),
        "t_trig": np.asarray(1.0),
        "dt_add": np.asarray(300.0),
    }
    d = tx.jacobian_vector_product(inputs, ins, ["tcore", "w_retained"], tangent)
    for k in ("tcore", "w_retained"):
        ref = np.asarray(J[k]["theta"]) @ dtheta + sum(
            np.asarray(J[k][n]) * float(tangent[n])
            for n in ("t_add", *DECISIONS)
        )
        got = np.asarray(d[k])
        assert np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12) < 5e-3


def test_duration_mode_columns(case):
    tx, inputs, _ = case
    # fractional-time form of the same cook: events as fractions of t_f
    tf_inputs = {
        **{k: v for k, v in inputs.items() if k not in ("u_tknots", "tsave", "t_add")},
        "u_tknots": np.linspace(0.0, 1.0, 4),
        "tsave": np.linspace(0.0, 1.0, 13),
        "t_f": 2.0 * HR,
        "t_add": 0.5,  # fractions in duration mode
        "dt_add": 0.375,
    }
    ins = ["t_f", *DECISIONS]
    J = tx.jacobian(tf_inputs, ins, ["tcore", "w_retained"])
    for n in ins:
        assert np.all(np.isfinite(np.asarray(J["tcore"][n]))), n
    assert np.max(np.abs(np.asarray(J["tcore"]["t_trig"]))) > 0
    tangent = {
        "t_f": np.asarray(600.0),
        "t_salt": np.asarray(1800.0),
        "t_trig": np.asarray(1.0),
        "dt_add": np.asarray(0.01),
    }
    d = tx.jacobian_vector_product(tf_inputs, ins, ["tcore", "w_retained"], tangent)
    for k in ("tcore", "w_retained"):
        ref = sum(np.asarray(J[k][n]) * float(tangent[n]) for n in ins)
        got = np.asarray(d[k])
        assert np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12) < 5e-3


def test_planner_maps():
    cc = ControlConfig(opt_salt=True, opt_trig=True, opt_splits=True)
    tsalt, ttrig, tda = _decisions_m1(cc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert tsalt == pytest.approx(cc.t_salt_max / 2)
    assert ttrig == pytest.approx((cc.t_trig_min + cc.t_trig_max) / 2)
    assert tda == pytest.approx((cc.dt_add_min + cc.dt_add_max) / 2)
    off = ControlConfig()
    assert _decisions_m1(off, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_objective_sees_the_trigger(case):
    tx, _, grids = case
    cc = ControlConfig(
        opt_trig=True, opt_wrap=False, opt_pull=False, opt_add=False,
        opt_spritz=False, t_trig_min=300.0, t_trig_max=320.0,
    )
    tols = {"abstol": 1e-6, "reltol": 1e-6}
    args = (jnp.zeros(4), jnp.asarray(0.0))
    kw = dict(
        tx=tx, grids=grids, cc=cc, tols=tols,
        theta=jnp.asarray(THETA_DEFAULT), phi=jnp.asarray(PHI_DEFAULT),
    )
    j_lo = float(objective(args[0], args[1], kw["theta"], kw["phi"], tx, grids,
                           cc, tols, z_tg=jnp.asarray(-2.0)))
    j_hi = float(objective(args[0], args[1], kw["theta"], kw["phi"], tx, grids,
                           cc, tols, z_tg=jnp.asarray(2.0)))
    assert np.isfinite(j_lo) and np.isfinite(j_hi)
    assert j_lo != j_hi  # the trigger decision reaches the objective
