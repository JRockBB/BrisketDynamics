"""Oven-roast validation against Papasidero et al. (2015), CET 43:151.

The second data-free validation of the generalized meat model (after
Baldwin sous vide), and the first with MEASURED WEIGHT LOSS: a ~1.4 kg
beef cylinder (18 x 10 cm) roasted in a domestic oven from a 22 C cold
start to 180 C, stopped at core 65 C (~87 min), with core / 5-mm-depth /
intermediate thermocouples and before/after dry-basis water content
(2.38 -> 1.75 kg/kg, digitized by scripts/digitize_cet2015.py into
data/validation/cet2015_roast.json).

Model setup: the pit drive is the oven (prescribed ramp 22 -> 180 C over
t_preheat, then flat — the paper says only "piecewise"), h_c = the
paper's 10 W/m2K, emissivity 0.91, radius L = 0.05 m. Geometry: an
infinite cylinder (geom_m = 1) plus an eigenvalue-matched equivalent for
the finite H/D = 1.8 cylinder (geom_m ~ 1.2): for Bi -> inf the finite
cylinder's first eigenvalue is (2.405/R)^2 + (pi/H)^2, ~13 % above the
infinite one, and the generalized m-solid interpolates lambda_1 between
2.405 (m=1) and pi (m=2).

Mass loss is read from the volume-weighted mean of the W field (the PDE
carries evaporation at the surface and the purge sink toward W_eq(T), so
mean-W decline is evaporation + drip together, which is what a scale
measures).

Run from app/:  uv run python -u ../scripts/run_oven_roast.py
Writes data/validation/oven_roast_cet2015.json and
figures/oven_roast_cet2015.png.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

DATA = json.loads((REPO / "data" / "validation" / "cet2015_roast.json").read_text())

# paper conditions
T_OVEN0 = 295.15  # 22 C cold oven
T_OVEN = 453.15  # 180 C setpoint
H_C = 10.0  # the paper's estimated convective coefficient [W/m2K]
EPS = 0.91  # the paper's meat surface emissivity
R_CYL = 0.05  # cylinder radius [m]
W0 = 2.379  # digitized initial dry-basis water content [kg/kg]
T_PREHEAT_MIN = 10.0  # assumed oven ramp (the paper: "piecewise"); +-5 in sensitivity
T_END_MIN = 115.0  # past the experiment's ~87 min so t65 always resolves

# repo-default meat parameter vector [D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv]
THETA = [3.8e-9, 0.0018, H_C, R_CYL, 1.0, 5.6e-4, 0.20, 0.6]


def exp_curve(name):
    d = DATA["curves"][name]
    return np.asarray(d["t_min"]), np.asarray(d["T_C"])


def run_case(jl, geom_m, t_preheat_min):
    """One forward solve; returns time series and summary metrics."""
    tsave = np.arange(0.0, T_END_MIN * 60.0 + 1.0, 60.0)
    tknots = np.array([0.0, t_preheat_min * 60.0, T_END_MIN * 60.0])
    tpit = np.array([T_OVEN0, T_OVEN, T_OVEN])
    t_exp0 = float(exp_curve("core_C")[1][0])  # ~6 C fridge start
    cfg = jl.MeatConfig(T0=t_exp0 + 273.15, W0=W0, geom_m=geom_m, ε=EPS)
    out = jl.meat_solve(
        jl.Vector([float(x) for x in THETA]),
        jl.Vector(tpit),
        jl.Vector(tknots),
        jl.Vector(tsave),
        cfg=cfg,
    )
    Tf = np.asarray(out.T_field).copy()  # (nx, nt) [K]
    Wf = np.asarray(out.W_field).copy()
    nx = Tf.shape[0]
    xi = np.linspace(0.0, 1.0, nx)
    # volume weights of the m-solid cells (half cells at the ends)
    edges = np.clip(np.concatenate([[0.0], (xi[:-1] + xi[1:]) / 2, [1.0]]), 0, 1)
    wvol = edges[1:] ** (geom_m + 1) - edges[:-1] ** (geom_m + 1)
    wvol = wvol / wvol.sum()
    tm = tsave / 60.0
    core = Tf[0] - 273.15
    # the paper's "surface" probe sits 5 mm below the surface: xi = 0.9
    i9 = np.searchsorted(xi, 0.9)
    f = (0.9 - xi[i9 - 1]) / (xi[i9] - xi[i9 - 1])
    depth5 = (1 - f) * Tf[i9 - 1] + f * Tf[i9] - 273.15
    w_mean = (wvol[:, None] * Wf).sum(axis=0)

    te, Te = exp_curve("core_C")
    t65_exp = float(np.interp(65.0, Te, te))
    t65_mod = (
        float(np.interp(65.0, core, tm)) if core.max() >= 65.0 else float("nan")
    )
    rms_core = float(
        np.sqrt(np.mean((np.interp(te, tm, core) - Te) ** 2))
    )
    td, Td = exp_curve("depth5mm_C")
    rms_d5 = float(np.sqrt(np.mean((np.interp(td, tm, depth5) - Td) ** 2)))
    pts = DATA["water_dry_basis"]["experiment_points"]
    w_end_exp = pts[-1]["W_db"]
    w_end_mod = float(np.interp(pts[-1]["t_min"], tm, w_mean))
    return {
        "geom_m": geom_m,
        "t_preheat_min": t_preheat_min,
        "t65_model_min": t65_mod,
        "t65_exp_min": t65_exp,
        "rms_core_C": rms_core,
        "rms_depth5mm_C": rms_d5,
        "W_end_model": w_end_mod,
        "W_end_exp": w_end_exp,
        "loss_ratio": (W0 - w_end_mod) / (W0 - w_end_exp),
        "series": {"t_min": tm, "core_C": core, "depth5mm_C": depth5, "W_db": w_mean},
    }


def main():
    from juliacall import Main as jl

    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics: meat_solve, MeatConfig")
    print("Julia ready", flush=True)

    t0 = time.perf_counter()
    cases = []
    for geom_m, tph in [(1.0, 10.0), (1.2, 10.0), (1.2, 5.0), (1.2, 15.0)]:
        c = run_case(jl, geom_m, tph)
        cases.append(c)
        print(
            f"  m={geom_m:.1f} preheat={tph:4.1f} min: t65 {c['t65_model_min']:5.1f} "
            f"vs exp {c['t65_exp_min']:5.1f} min (x{c['t65_model_min'] / c['t65_exp_min']:.2f}); "
            f"rms core {c['rms_core_C']:.1f} C, 5mm {c['rms_depth5mm_C']:.1f} C; "
            f"W_end {c['W_end_model']:.2f} vs {c['W_end_exp']:.2f} "
            f"(loss x{c['loss_ratio']:.2f})",
            flush=True,
        )
    wall = time.perf_counter() - t0

    primary = cases[1]  # m=1.2, 10 min preheat
    out = {
        "dataset": "data/validation/cet2015_roast.json",
        "theta": THETA,
        "conditions": {
            "T_oven_C": 180.0,
            "T_oven0_C": 22.0,
            "h_c": H_C,
            "emissivity": EPS,
            "radius_m": R_CYL,
            "W0_db": W0,
        },
        "cases": [
            {k: v for k, v in c.items() if k != "series"} for c in cases
        ],
        "primary": "geom_m=1.2 (finite-cylinder equivalent), 10 min preheat",
        "wall_s": wall,
    }
    (REPO / "data" / "validation" / "oven_roast_cet2015.json").write_text(
        json.dumps(out, indent=1)
    )
    print("wrote data/validation/oven_roast_cet2015.json")
    _figure(cases, primary)


def _figure(cases, primary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for name, label, color in [
        ("core_C", "core", "tab:blue"),
        ("depth5mm_C", "5 mm below surface", "tab:red"),
        ("intermediate_C", "intermediate", "tab:green"),
    ]:
        d = DATA["curves"][name]
        ax.plot(d["t_min"], d["T_C"], ".", ms=3, color=color, alpha=0.6,
                label=f"{label} (exp, digitized)")
    s = primary["series"]
    ax.plot(s["t_min"], s["core_C"], color="tab:blue", label="core (model)")
    ax.plot(s["t_min"], s["depth5mm_C"], color="tab:red",
            label="5 mm depth (model)")
    alt = cases[0]["series"]
    ax.plot(alt["t_min"], alt["core_C"], "--", color="tab:blue", alpha=0.5,
            label="core (model, infinite cyl)")
    ax.set(xlabel="time [min]", ylabel="temperature [C]",
           title="CET 43:151 roast: temperatures")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    pm = DATA["water_dry_basis"]["paper_model"]
    ax.plot(pm["t_min"], pm["W_db"], ":", color="gray", label="paper's model")
    for p in DATA["water_dry_basis"]["experiment_points"]:
        ax.plot(p["t_min"], p["W_db"], "s", color="tab:red", ms=8,
                label="experiment" if p["t_min"] == 0 else None)
    ax.plot(s["t_min"], s["W_db"], color="tab:blue", label="our model (mean W)")
    ax.set(xlabel="time [min]", ylabel="dry-basis water [kg/kg]",
           title="Weight loss: the paper overestimated; where do we land?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Oven-roast validation vs Papasidero et al. (2015) — no parameters fit to this data",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "oven_roast_cet2015.png", dpi=140)
    print("wrote figures/oven_roast_cet2015.png")


if __name__ == "__main__":
    main()
