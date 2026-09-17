"""One MPC cook on the mock's brisket under the mock's own start conditions,
for the pilot comparison: the app's receding-horizon MPC (state
feedback, the KCBS quality objective, hourly replans) seeded by the full
plan on the same parameters, no disturbance. Saves the traces in the
layout scripts/run_mpc.py uses (the `mpc_nominal` variant) so
estimator/python/export_mpc_trace.py can read them.

Run from app/:
  uv run python ../estimator/python/run_mpc_mock1d.py --params map_mock1d.json --plan fullplan_result_mock1d.json \
      --t-salt-h 0 --tpit0-K 362.4278 --t0-meat-K 299.7611 --out mpc_mock1d.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app" / "src"))
HR = 3600.0
T_AMB_BASE = 300.0
TRAJ_KEYS = ("t", "tcore", "tpit", "tsurf", "u")
METRIC_KEYS = ("doneness", "juiciness", "l_star_end", "smoke_end")


def _nan(v):
    return np.array(float("nan") if v is None else float(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="map_mock1d.json")
    ap.add_argument("--plan", default="fullplan_result_mock1d.json")
    ap.add_argument("--t-salt-h", type=float, default=0.0)
    ap.add_argument("--tpit0-K", type=float, default=293.15)
    ap.add_argument("--t0-meat-K", type=float, default=281.15)
    ap.add_argument("--replan-iters", type=int, default=200)
    ap.add_argument("--model-params", default=None, help="the controller's parameter file (default: the plant's, an oracle)")
    ap.add_argument("--out", default="mpc_mock1d.json")
    args = ap.parse_args()

    from brisket_app import mpc
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import monolith_client
    from brisket_app.control import ControlConfig
    from brisket_app.coupled import CookGrids

    m = json.loads((REPO / "data" / "calibration" / args.params).read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    if args.model_params:
        mm = json.loads((REPO / "data" / "calibration" / args.model_params).read_text())
        theta_model, phi_model = np.array([mm[n] for n, _ in THETA_SITES]), phi_full(mm)
    else:
        theta_model, phi_model = None, None
    mpc.set_clients(monolith_client())
    b = json.loads((REPO / "data" / "control" / args.plan).read_text())
    T_F = float(b["best_t_f_h"]) * HR
    t_wrap_star = float(b["best_t_wrap_h"]) * HR
    u0_plan = np.asarray(b["best_u"])
    T_PULL = float(b["best_t_pull_h"]) * HR if b.get("best_t_pull_h") else None
    T_ADD = float(b["best_t_add_h"]) * HR if b.get("best_t_add_h") else None
    T_SPRITZ = float(b["best_t_spritz_h"]) * HR if b.get("best_t_spritz_h") else None
    DT_SPRITZ = float(b["best_dt_spritz_min"]) * 60.0 if b.get("best_dt_spritz_min") else None
    SPRITZ_DOSE = float(b["best_spritz_dose_gm2"]) / 1e3 if b.get("best_spritz_dose_gm2") else None
    grids0 = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=len(u0_plan))
    print(f"plant {args.params}, plan {args.plan}: {T_F / HR:.1f} h cook, wrap seed {t_wrap_star / HR:.2f} h, "
          f"pull seed {T_PULL / HR if T_PULL else float('nan'):.2f} h; pit {args.tpit0_K - 273.15:.0f} C, meat {args.t0_meat_K - 273.15:.0f} C, brine {args.t_salt_h:.0f} h", flush=True)
    t0 = time.time()
    r = mpc.run_cook(
        theta, phi, lambda t: T_AMB_BASE, t_f=T_F, mode="mpc", model_t_amb=T_AMB_BASE, cc=ControlConfig(opt_wrap=True),
        t_wrap_init_abs=t_wrap_star, u_incumbent=u0_plan, u_incumbent_grids=grids0, replan_iters=args.replan_iters,
        t_pull_init_abs=T_PULL, t_add_init_abs=T_ADD, t_spritz_init_abs=T_SPRITZ, dt_spritz_init=DT_SPRITZ,
        spritz_dose_init=SPRITZ_DOSE, t_salt=args.t_salt_h * HR, tpit0=args.tpit0_K, t0_meat=args.t0_meat_K,
        theta_model=theta_model, phi_model=phi_model,
    )
    wall = time.time() - t0
    tag = "mpc_nominal"
    payload = {"t_dist_h": np.array(float("nan")), "wrap_mode": np.array(1), f"{tag}_wall_s": np.array(wall)}
    for k in TRAJ_KEYS:
        payload[f"{tag}_{k}"] = np.asarray(r[k])
    for k in METRIC_KEYS:
        payload[f"{tag}_{k}"] = np.array(float(r[k]))
    payload[f"{tag}_wrapped"] = np.array(int(bool(r.get("wrapped"))))
    for k in ("t_wrap_abs", "t_wrap_exec_abs", "t_pull_exec_abs", "t_add_exec_abs", "t_spritz_exec_abs", "dt_spritz", "spritz_dose"):
        payload[f"{tag}_{k}"] = _nan(r.get(k))
    stem = Path(args.out).stem
    tpath = REPO / "data" / "control" / f"{stem}_traces.npz"
    np.savez(tpath, **payload)
    summary = {"params": args.params, "model_params": args.model_params, "plan": args.plan, "cook_h": T_F / HR, "wall_s": wall,
               "t_wrap_exec_h": None if r.get("t_wrap_exec_abs") is None else float(r["t_wrap_exec_abs"]) / HR,
               "t_pull_exec_h": None if r.get("t_pull_exec_abs") is None else float(r["t_pull_exec_abs"]) / HR,
               "doneness": float(r["doneness"]), "juiciness": float(r["juiciness"]), "l_star_end": float(r["l_star_end"]),
               "n_replans": len(r.get("pull_history") or []), "tpit0_K": args.tpit0_K, "t0_meat_K": args.t0_meat_K, "t_salt_h": args.t_salt_h}
    (REPO / "data" / "control" / args.out).write_text(json.dumps(summary, indent=1))
    print(f"wrote {tpath.name} and {args.out}: wrap {summary['t_wrap_exec_h']} h, pull {summary['t_pull_exec_h']} h, "
          f"doneness {summary['doneness']:.3f}, juiciness {summary['juiciness']:.3f}, L* {summary['l_star_end']:.1f}; {wall:.0f} s", flush=True)


if __name__ == "__main__":
    main()
