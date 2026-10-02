"""Shadow-score a planned 1-D cook on a 3-D trimmed brisket flat.

The same schedule as the 2-D shadow (scripts/run_shadow2d.py) — damper, wrap, pull, spritz, the fuel addition —
on a flat tapering in thickness and width from its thick (point) end to its thin end with a fat cap on top (the
typical trimmed dimensions `BRISKET_FLAT_TYPICAL` until the cut is measured), half of it solved across its width
inside the cook system (scripts/shadow3d.jl). Reports what only a 3-D cut shows: the true cold spot (the grid
minimum: where the thick end's centre lags and the thin end's edges lead), the bark, smoke and ring over every
face, the fat rendered, to serve. `--no-check` skips the coarser-grid discretization check.

Usage (from app/):
    uv run python ../scripts/run_shadow3d.py
    uv run python ../scripts/run_shadow3d.py --plan schedule_robust.json --nl 12 --nw 6 --nt 8
"""

import argparse
import json

import numpy as np
from shadow3d_common import HR, REPO, coords_grid, simulate

from brisket_app.control import ControlConfig, lethality_cut


def report(sim, cc):
    t = sim["t"]
    tcold, tcore = sim["Tcold"], sim["Tcore"]
    F_cold, cut_cold = lethality_cut(tcold, t, cc)
    F_core, cut_core = lethality_cut(tcore, t, cc)
    binding = bool(cc.cut_critical)
    g = sim["geometry"]
    S, Y, Z = coords_grid(sim)
    cold = int(np.argmin(np.asarray(sim["T_field"])[:, -1]))
    sc, yc, zc = (float(a.reshape(-1, order="F")[cold]) for a in (S, Y, Z))
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
        "cold_spot_at_serve_cm": {"from_thick_end": sc, "from_centre_plane": yc, "above_grate": zc},
        "mass_raw_kg": float(sim["mass"][0]), "mass_serve_kg": float(sim["mass"][-1]),
        "core_to_surface_raw_cm": 100.0 * float(sim["l_thick"][0]),
        "core_to_surface_serve_cm": 100.0 * float(sim["l_thick"][-1]),
        "length_cm": 100.0 * g["length"], "width_cm": [100.0 * g["width_thick"], 100.0 * g["width_thin"]],
        "thickness_cm": [100.0 * g["thickness_thick"], 100.0 * g["thickness_thin"]], "fat_cap_mm": 1000.0 * g["cap"],
        "geometry_source": "BRISKET_FLAT_TYPICAL (typical trimmed flat; thick end first); half the cut by symmetry",
        "grid": sim["grid"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nl", type=int, default=12, help="nodes along the length")
    ap.add_argument("--nw", type=int, default=6, help="nodes across the half width")
    ap.add_argument("--nt", type=int, default=8, help="nodes up the thickness")
    ap.add_argument("--t-salt-h", type=float, default=0.0, help="dry-brine hours before the cook")
    ap.add_argument("--no-check", action="store_true", help="skip the coarser-grid check")
    args = ap.parse_args()

    cc = ControlConfig()
    sim = simulate(args.plan, args.nl, args.nw, args.nt, t_salt_s=args.t_salt_h * HR)
    rep = report(sim, cc)
    if not args.no_check:
        cl, cw, ct = max(args.nl * 2 // 3, 3), max(args.nw * 2 // 3, 3), max(args.nt * 2 // 3, 3)
        coarse = report(simulate(args.plan, cl, cw, ct, t_salt_s=args.t_salt_h * HR), cc)
        rep.update({
            "grid_coarse": coarse["grid"],
            "dtcold_serve_K": rep["tcold_final_C"] - coarse["tcold_final_C"],
            "dtcold_peak_K": rep["tcold_peak_C"] - coarse["tcold_peak_C"],
            "dtcore_serve_K": rep["tcore_final_C"] - coarse["tcore_final_C"],
            "dc_min": rep["c_min"] - coarse["c_min"],
            "dw_retained": rep["w_retained"] - coarse["w_retained"],
        })
    rep["assumption"] = (
        "a typical trimmed flat (tapering thickness and width, a fat cap), not yet measured; "
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
