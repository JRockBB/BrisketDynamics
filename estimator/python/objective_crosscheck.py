"""One objective, two evaluations: the port check on J.

The app's KCBS quality objective J is evaluated by re-running each
executed schedule (damper, wrap, pull, fuel add, spritz, serve)
open-loop through the app's own cook function (the Tesseract) and
scoring it with `mpc.executed_objective`, exactly as scripts/run_mpc.py
scores its variants; the Julia port of the same objective
(`estimator/src/horizon.jl`, `cook_outputs` and `objective_terms`) scored
each cook on the Julia plant in data/mock/pilot_comparison.json. The
two should agree term for term up to the port's plant differences.

Run from app/:
  uv run python ../estimator/python/objective_crosscheck.py --params map_mock1d.json --tpit0-K 362.4278 --t0-meat-K 299.7611
Writes data/mock/objective_crosscheck.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app" / "src"))
HR = 3600.0
T_AMB = 300.0


def app_cook(theta, phi, u_t_h, u, events_h, t_f_h, cc, args):
    """Run one executed schedule open-loop through the app's cook and score it."""
    from brisket_app import mpc
    from brisket_app.coupled import CookGrids

    t_f = t_f_h * HR
    n_u = int(round(t_f / 600.0)) + 1
    grids = CookGrids(t_end=t_f, dt_exchange=600.0, n_u=n_u)
    u_knots = np.interp(grids.u_tknots, np.asarray(u_t_h) * HR, np.asarray(u))
    kw = dict(t_wrap_init_abs=None if events_h.get("wrap") is None else events_h["wrap"] * HR,
              t_pull_init_abs=None if events_h.get("pull") is None else events_h["pull"] * HR,
              t_add_init_abs=None if events_h.get("add") is None else events_h["add"] * HR,
              t_spritz_init_abs=None if events_h.get("spritz") is None else events_h["spritz"] * HR,
              dt_spritz_init=None if events_h.get("spritz") is None else events_h["spritz_dt"] * HR,
              spritz_dose_init=None if events_h.get("spritz") is None else events_h.get("spritz_m", 0.05))
    r = mpc.run_cook(theta, phi, lambda t: T_AMB, t_f=t_f, mode="openloop", u_openloop=u_knots, u_openloop_grids=grids,
                     t_salt=args.t_salt_h * HR, tpit0=args.tpit0_K, t0_meat=args.t0_meat_K, log=lambda *a, **k: None, **kw)
    J = mpc.executed_objective(r, cc, t_f)
    return {"J": {k: float(v) for k, v in J.items()}, "doneness": float(r["doneness"]), "juiciness": float(r["juiciness"]),
            "l_star_end": float(r["l_star_end"]), "smoke_end": float(r["smoke_end"]),
            "t_wrap_exec_h": None if r.get("t_wrap_exec_abs") is None else float(r["t_wrap_exec_abs"]) / HR,
            "t_pull_exec_h": None if r.get("t_pull_exec_abs") is None else float(r["t_pull_exec_abs"]) / HR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="map_mock1d.json")
    ap.add_argument("--t-salt-h", type=float, default=0.0)
    ap.add_argument("--tpit0-K", type=float, default=362.4278)
    ap.add_argument("--t0-meat-K", type=float, default=299.7611)
    ap.add_argument("--mpc-trace", default="mpc_mock1d_traces_mpc_nominal_trace.json")
    args = ap.parse_args()
    from brisket_app import mpc
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import monolith_client
    from brisket_app.control import ControlConfig

    m = json.loads((REPO / "data" / "calibration" / args.params).read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(monolith_client())
    cc = ControlConfig(opt_wrap=True)

    cmp = json.loads((REPO / "data" / "mock" / "pilot_comparison.json").read_text(encoding="utf-8"))
    rec = json.loads((REPO / "data" / "mock" / "closed_loop_realcook.json").read_text(encoding="utf-8"))
    ex = rec["executed"]
    u10 = np.array([np.nan if v is None else v for v in ex["u"]], dtype=float)
    last = u10[~np.isnan(u10)][-1]
    u10 = np.where(np.isnan(u10), last, u10)
    sp = ex["spritz"]
    rx_events = {"wrap": ex["t_wrap_h"], "pull": ex["t_pull_h"], "add": ex["t_add_h"],
                 "spritz": None if sp is None else sp["t_h"], "spritz_dt": None if sp is None else sp["dt_h"], "spritz_m": None if sp is None else sp["m"]}
    print("RxInfer cook through the app's plant ...", flush=True)
    rx_app = app_cook(theta, phi, ex["t_h"], u10, rx_events, ex["t_serve_h"], cc, args)
    mp = json.loads((REPO / "data" / "control" / args.mpc_trace).read_text(encoding="utf-8"))
    es = mp["events_s"]
    mpc_events = {"wrap": None if es["wrap"] is None else es["wrap"] / HR, "pull": None if es["pull"] is None else es["pull"] / HR,
                  "add": None if es["add"] is None else es["add"] / HR, "spritz": None if es["spritz"] is None else es["spritz"] / HR,
                  "spritz_dt": None if es["dt_spritz"] is None else es["dt_spritz"] / HR, "spritz_m": es["spritz_dose"]}
    print("MPC cook through the app's plant ...", flush=True)
    mpc_app = app_cook(theta, phi, mp["t_h"], mp["u"], mpc_events, es["serve"] / HR, cc, args)

    out = {"rxinfer": {"julia_J": cmp["rxinfer"]["metrics"]["J_terms"], "app": rx_app, "julia": cmp["rxinfer"]["metrics"]},
           "mpc": {"julia_J": cmp["mpc"]["metrics"]["J_terms"], "app": mpc_app, "julia": cmp["mpc"]["metrics"]},
           "mpc_tesseract_own_run": mp["metrics"]}
    (REPO / "data" / "mock" / "objective_crosscheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    keys = ["tenderness", "appearance", "taste", "surf", "fuel", "smooth", "safety", "J"]
    print(f"\n{'':14s} {'RxInfer, Julia':>15s} {'RxInfer, app':>13s} {'MPC, Julia':>12s} {'MPC, app':>10s}")
    for k in keys:
        print(f"{k:14s} {out['rxinfer']['julia_J'][k]:15.3f} {rx_app['J'][k]:13.3f} {out['mpc']['julia_J'][k]:12.3f} {mpc_app['J'][k]:10.3f}")
    print(f"doneness / juiciness / L*   Julia {cmp['rxinfer']['metrics']['doneness_serve']:.3f} / {cmp['rxinfer']['metrics']['juiciness_serve']:.3f} / {cmp['rxinfer']['metrics']['L_serve']:.1f}"
          f"   app {rx_app['doneness']:.3f} / {rx_app['juiciness']:.3f} / {rx_app['l_star_end']:.1f}   |   MPC Julia {cmp['mpc']['metrics']['doneness_serve']:.3f} / {cmp['mpc']['metrics']['juiciness_serve']:.3f} / {cmp['mpc']['metrics']['L_serve']:.1f}"
          f"   app {mpc_app['doneness']:.3f} / {mpc_app['juiciness']:.3f} / {mpc_app['l_star_end']:.1f}")
    print("wrote data/mock/objective_crosscheck.json")


if __name__ == "__main__":
    main()
