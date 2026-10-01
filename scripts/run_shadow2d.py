"""Shadow-score a planned 1-D cook on the 2-D Tesseract.

Loads the calibrated MAP and a plan (default: the robust schedule), runs
the SAME schedule through the 2-D axisymmetric solve, and reports what
only the grid can see: the true cold-spot lethality (vs the symmetric
slab's core-node proxy), the bark and smoke distributions, and the
shrunk dimensions. The plan's fuel addition is replayed with its lid lift.

Usage:
    python scripts/run_shadow2d.py
    python scripts/run_shadow2d.py --plan schedule_robust.json --radius 0.12
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES
from brisket_app.coupled import HR, CookGrids
from brisket_app.shadow2d import shadow_report

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    """Load MAP + plan, run the 2-D shadow, print and save the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json",
                    help="plan json in data/control (robust schedule format)")
    ap.add_argument("--radius", type=float, default=None,
                    help="2-D surrogate radius [m]")
    ap.add_argument("--nr", type=int, default=16)
    ap.add_argument("--nz", type=int, default=14)
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the doubled-grid discretization check")
    ap.add_argument("--t-salt-h", type=float, default=0.0,
                    help="dry-brine hours before the cook")
    args = ap.parse_args()

    mp = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([mp[n] for n, _ in THETA_SITES], dtype=float)
    phi = np.array(
        [mp[n] for n, _ in PHI_SITES]
        + [mp.get(n, d.mean) for n, d in PHI_SITES_IGNITION],
        dtype=float,
    )

    plan = json.loads((REPO / "data" / "control" / args.plan).read_text())
    u = np.asarray(plan.get("u_robust", plan.get("u_opt")), dtype=float)
    t_end = float(plan.get("cook_h", 11.0)) * HR
    grids = CookGrids(t_end=t_end, dt_exchange=600.0, n_u=len(u))
    tw = plan.get("t_wrap_robust_h", plan.get("t_wrap_plan_h"))
    tsp = plan.get("t_spritz_robust_h", plan.get("t_spritz_plan_h"))
    dts = plan.get("dt_spritz_robust_min", plan.get("dt_spritz_plan_min"))
    tp = plan.get("t_pull_robust_h", plan.get("t_pull_plan_h"))
    ta = plan.get("t_add_robust_h", plan.get("t_add_plan_h"))

    rep = shadow_report(
        theta, phi, u, grids,
        t_wrap=(tw * HR if tw is not None else 1.0e9),
        t_salt=args.t_salt_h * HR,
        t_spritz=(tsp * HR if tsp is not None else 1.0e9),
        dt_spritz=(dts * 60.0 if dts is not None else 3600.0),
        t_pull=(tp * HR if tp is not None else 1.0e9),
        t_add=(ta * HR if ta is not None else 1.0e9),
        rest=plan.get("rest"),
        radius=args.radius, nr=args.nr, nz=args.nz,
        refine=not args.no_refine,
    )
    print("2-D shadow report for", args.plan)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    out = REPO / "data" / "control" / "shadow2d.json"
    out.write_text(json.dumps({"plan": args.plan, **rep}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
