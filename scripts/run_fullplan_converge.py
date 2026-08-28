"""Re-derive the 17 h optimum to CONVERGENCE (plateau-stopped).

The MPC-nominal discriminator showed the 60-iteration incumbent was
under-converged (a replanning controller found lower J at nominal
conditions, which is impossible against a true optimum). This re-derives
the chosen-duration plan under the convergence semantics: n_iter is a
high cap, the plateau rule stops it.

Run from app/:  uv run python -u ../scripts/run_fullplan_converge.py
Writes data/control/fullplan17_converged.json (and prints the delta vs
the 60-iter incumbent). Does NOT overwrite fullplan_result_ext.json —
promotion to incumbent is an explicit follow-up once inspected.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig, optimize_schedule
from brisket_app.coupled import HR
from brisket_app.timeplan import _grids_for

REPO = Path(__file__).resolve().parents[1]
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}


def main():
    """Converged re-optimization of the incumbent 17 h plan."""
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)

    best = json.loads(
        (REPO / "data" / "control" / "fullplan_result_ext.json").read_text()
    )
    t_f = best["best_t_f_h"] * HR
    u0 = np.asarray(best["best_u"])
    frac = np.clip(best["best_t_wrap_h"] * HR / t_f, 1e-3, 1 - 1e-3)
    grids = _grids_for(t_f)
    cc = ControlConfig(opt_wrap=True, n_iter=600)  # plateau rule stops it

    print(
        f"re-optimizing {best['best_t_f_h']:.0f} h plan to convergence "
        f"(cap {cc.n_iter}, patience {cc.patience}, tol_rel {cc.tol_rel})"
    )
    u_opt, hist = optimize_schedule(
        theta,
        phi,
        monolith_client(),
        grids=grids,
        cc=cc,
        u_init=u0,
        z_w_init=float(np.log(frac / (1 - frac))),
        ics=ICS,
        callback=lambda i, l, u: print(f"  iter {i:4d}  J={l:9.4f}", flush=True),
        log_every=25,
    )
    j0, j1 = float(hist["losses"][0]), float(hist["losses"][-1])
    print(
        f"J {j0:.4f} -> {j1:.4f} over {hist['n_iter_run']} iters "
        f"(converged={hist['converged']}); wrap "
        f"{best['best_t_wrap_h']:.2f} h -> {hist['t_wrap'] / HR:.2f} h"
    )
    (REPO / "data" / "control" / "fullplan17_converged.json").write_text(
        json.dumps(
            {
                "t_f_h": best["best_t_f_h"],
                "u_opt": u_opt.tolist(),
                "t_wrap_h": hist["t_wrap"] / HR,
                "J_incumbent_start": j0,
                "J_converged": j1,
                "n_iter_run": hist["n_iter_run"],
                "converged": bool(hist["converged"]),
            },
            indent=2,
        )
    )
    print("wrote data/control/fullplan17_converged.json")


if __name__ == "__main__":
    main()
