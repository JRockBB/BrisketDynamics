"""One objective, two implementations: the app scores the stored parity cook term by term as Julia recorded it.

data/mock/objective_parity.json (estimator/scripts/objective_parity.jl) holds one cook's surface, core and fuel
trajectories, its executed damper as hourly steps, the scalars the objective reads at serve, and the terms the
Julia port (estimator/src/horizon.jl) gave them. No physics runs here: the formula and the goal parameters are
what the two sides must agree on. Two terms are integrals whose windows differ by design and are held to a
tolerance: the bark-burn mean (Julia blends the pull and the serve smoothly, the executed score cuts at the
pull) and the lethality (Julia weights the last minutes before the serve by its smooth horizon).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from brisket_app import control, mpc
from brisket_app.control import ControlConfig

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "mock" / "objective_parity.json"

# the goal parameters, Julia's name -> the app's
GOALS = {
    "w_appearance": "w_appearance", "w_taste": "w_taste", "w_tenderness": "w_tenderness",
    "c_target": "c_target", "c_overdone": "c_overdone", "over_rel": "over_rel",
    "l_lo": "l_mahogany_lo", "l_hi": "l_mahogany_hi", "dl_soft": "dl_soft",
    "s_lo": "s_smoke_lo", "s_hi": "s_smoke_hi", "ds_soft": "ds_soft", "w_dry": "w_dry",
    "t_surf_max": "t_surf_max", "dT_soft": "dT_soft", "w_surf": "w_surf", "w_fuel": "w_fuel", "w_smooth": "w_smooth",
    "w_lethality": "w_lethality", "leth_t_ref": "leth_t_ref", "leth_z": "leth_z", "leth_f_req": "leth_f_req",
    "m_chimney": "m_chimney", "m0_ref": "m0_ref",
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_goal_parameters_are_the_apps():
    fx = _fixture()
    cc = ControlConfig()
    for jl_name, py_name in GOALS.items():
        assert getattr(cc, py_name) == fx["goals"][jl_name], (jl_name, py_name)
    assert set(fx["goals"]) == set(GOALS), "a goal parameter exists on one side only"


def test_terms_match_julia_on_the_stored_cook():
    fx = _fixture()
    cc = ControlConfig()
    t = np.asarray(fx["t"], dtype=float)
    out = {
        "t": t, "tsurf": np.asarray(fx["tsurf"], dtype=float), "tcore": np.asarray(fx["tcore"], dtype=float),
        "u": np.asarray(fx["u_steps"], dtype=float), "t_pull_exec_abs": float(fx["t_pull"]),
        "doneness": float(fx["doneness"]), "l_star_end": float(fx["l_star_end"]), "smoke_end": float(fx["smoke_end"]),
        "juiciness": float(fx["juiciness"]), "fuel_frac": float(fx["fuel_frac"]),
    }
    comp = mpc.executed_objective(out, cc, float(fx["t_serve"]))
    jl = fx["terms"]
    # the terms scored from the same scalars: identical formulas, identical numbers
    for k in ("tenderness", "appearance", "taste", "fuel", "smooth"):
        np.testing.assert_allclose(comp[k], jl[k], rtol=1e-9, atol=1e-11, err_msg=k)
    # the bark-burn mean: the same excess, windows that differ by the pull blend and the serve horizon
    surf = max(comp["surf"], jl["surf"])
    if surf > 1e-6:
        assert abs(comp["surf"] - jl["surf"]) <= 0.05 * surf, (comp["surf"], jl["surf"])
    else:
        assert comp["surf"] < 1e-6 and jl["surf"] < 1e-6
    # the lethality: the same trapezoid, Julia's smooth horizon at the serve
    f_s, _ = control.lethality_cut(np.asarray(fx["tsurf"], dtype=float), t, cc)
    f_c, _ = control.lethality_cut(np.asarray(fx["tcore"], dtype=float), t, cc)
    assert abs(f_s - fx["leth_s"]) <= 0.05 * fx["leth_s"] and abs(f_c - fx["leth_c"]) <= 0.05 * fx["leth_c"]
    assert comp["safety"] < 1e-9 and jl["safety"] < 1e-9          # a cook this long is far past the requirement
    assert abs(comp["J"] - jl["J"]) <= 1e-9 + 0.05 * surf + comp["safety"] + jl["safety"]
