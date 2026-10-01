"""Shadow-score a planned 1-D cook on a 3-D brick of the cut.

The same schedule as the 2-D shadow (scripts/run_shadow2d.py) — damper, wrap, pull, spritz, the fuel addition —
on a flat of full thickness 2L whose top face carries half the 1-D exposed area (a 2:1 footprint), solved as one
eighth by symmetry inside the cook system (scripts/shadow3d.jl). Reports what only a 3-D cut shows: the true
cold spot (the grid minimum: the corners and edges heat first, the centre last), the bark, smoke and ring over
all six faces, and the cut's shrinkage, to serve. `--no-check` skips the coarser-grid discretization check.

Usage (from app/):
    uv run python ../scripts/run_shadow3d.py
    uv run python ../scripts/run_shadow3d.py --plan schedule_robust.json --nx 6 --ny 8 --nz 12
"""

import argparse
import json

import numpy as np
from brick3d_common import HR, REPO, simulate

from brisket_app.control import ControlConfig, lethality_cut


def report(sim, cc):
    t = sim["t"]
    tcold, tcore = sim["Tcold"], sim["Tcore"]
    F_cold, cut_cold = lethality_cut(tcold, t, cc)
    F_core, cut_core = lethality_cut(tcore, t, cc)
    binding = bool(cc.cut_critical)
    g = sim["geometry"]
    return {
        "F_cold": float(F_cold), "cut_h_cold": float(cut_cold),
        "F_core": float(F_core), "cut_h_core": float(cut_core), "cut_critical": binding,
        "cold_ok": bool(F_cold >= cc.leth_f_req and (cut_cold <= cc.cut_cap_h or not binding)),
        "t_pull_h": sim["decisions"]["t_pull"] / HR if sim["decisions"]["t_pull"] < 1e8 else None,
        "t_add_h": sim["decisions"]["t_add"] / HR if sim["decisions"]["t_add"] < 1e8 else None,
        "tcold_final_C": float(tcold[-1] - 273.15), "tcold_peak_C": float(np.max(tcold) - 273.15),
        "tcore_final_C": float(tcore[-1] - 273.15), "cold_gap_K": float(tcore[-1] - tcold[-1]),
        "w_retained": sim["w_retained"], "juiciness": sim["juiciness"], "fat_rendered": sim["fat_rendered"],
        "c_min": float(np.min(sim["C_field"][:, -1])),
        "bark_mean": float(sim["bark_b"][-1]), "bark_std": float(sim["bark_std"][-1]),
        "smoke_mean": float(sim["bark_s"][-1]), "ring_mean_mm": 1000.0 * float(sim["ring"][-1]),
        "l_star_mean": float(sim["l_star"][-1]), "bitterness": sim["bitterness"],
        "salt_mean": float(np.mean(sim["salt_field"][:, -1])),
        "mass_raw_kg": float(sim["mass"][0]), "mass_serve_kg": float(sim["mass"][-1]),
        "thickness_raw_cm": 100.0 * g["thickness"], "thickness_serve_cm": 200.0 * float(sim["l_thick"][-1]),
        "width_cm": 100.0 * g["width"], "length_cm": 100.0 * g["length"],
        "geometry_source": "thickness 2L; top face A_meat/2 at a 2:1 footprint; one eighth by symmetry",
        "grid": sim["grid"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nx", type=int, default=6, help="nodes through the half-thickness")
    ap.add_argument("--ny", type=int, default=8, help="nodes across the half-width")
    ap.add_argument("--nz", type=int, default=12, help="nodes along the half-length")
    ap.add_argument("--t-salt-h", type=float, default=0.0, help="dry-brine hours before the cook")
    ap.add_argument("--no-check", action="store_true", help="skip the coarser-grid check")
    args = ap.parse_args()

    cc = ControlConfig()
    sim = simulate(args.plan, args.nx, args.ny, args.nz, t_salt_s=args.t_salt_h * HR)
    rep = report(sim, cc)
    if not args.no_check:
        cnx, cny, cnz = max(args.nx * 2 // 3, 3), max(args.ny * 2 // 3, 3), max(args.nz * 2 // 3, 3)
        coarse = report(simulate(args.plan, cnx, cny, cnz, t_salt_s=args.t_salt_h * HR), cc)
        rep.update({
            "grid_coarse": coarse["grid"],
            "dtcold_serve_K": rep["tcold_final_C"] - coarse["tcold_final_C"],
            "dtcold_peak_K": rep["tcold_peak_C"] - coarse["tcold_peak_C"],
            "dtcore_serve_K": rep["tcore_final_C"] - coarse["tcore_final_C"],
            "dc_min": rep["c_min"] - coarse["c_min"],
            "dw_retained": rep["w_retained"] - coarse["w_retained"],
        })
    rep["assumption"] = (
        "rectangular flat (thickness 2L, 2:1 footprint carrying the 1-D exposed area on its top face), "
        "rested to serve after the planned pull; 1-D-calibrated constants, bark and smoke distributions "
        "uncalibrated (C2); come-up time reported, "
        + ("binding" if rep["cut_critical"] else "not binding for brisket (FSIS Gap 6)")
    )
    print("3-D shadow report for", args.plan)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    out = REPO / "data" / "control" / "shadow3d.json"
    out.write_text(json.dumps({"plan": args.plan, **rep}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
