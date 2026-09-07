"""Export the real-cook calibration problem for the Julia estimator gate.

Everything the Gauss-Newton MAP (scripts/run_gn_map.py) sees, in one JSON
the Julia side reads without re-deriving anything: the resampled cook log
and its masks, the whitening operator L of the masked AR(1) likelihood at
the MAP's noise parameters, the stacked observations, the prior
specification per site, the MAP in both spaces, the Gauss-Newton Hessian
and its inverse (the Fisher covariance every downstream script quotes),
and the cold-start point. The gate (estimator/gate_real_cook.jl) must
reproduce the MAP and that covariance from the same ingredients.

Run from app/:  uv run python -u ../estimator/export_gate_data.py
Writes data/calibration/estimator_gate_real.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

from brisket_app import cooklog  # noqa: E402
from brisket_app.calibrate import SIGMA_FLOOR_K  # noqa: E402
from brisket_app.monolith import (  # noqa: E402
    NAMES,
    SITES,
    Whitened,
    _is_beta,
    effective_sigma,
    objective,
    to_z,
)


def prior_spec():
    """Per-site prior as (family, p1, p2): lognormal (mu, sd of log) or beta (a, b)."""
    out = []
    for name, d in SITES:
        if _is_beta(d):
            out.append({"name": name, "family": "beta",
                        "p1": float(d.concentration1), "p2": float(d.concentration0)})
        else:
            out.append({"name": name, "family": "lognormal",
                        "p1": float(d.loc), "p2": float(d.scale)})
    return out


def prior_median_z():
    x = []
    for _, d in SITES:
        if _is_beta(d):
            a, b = float(d.concentration1), float(d.concentration0)
            x.append(a / (a + b))
        else:
            x.append(float(np.exp(d.loc)))
    return to_z(np.array(x))


def main():
    t0 = time.time()
    log = cooklog.load(dt_min=10.0)
    cal = REPO / "data" / "calibration"
    mp = json.loads((cal / "map_real.json").read_text())
    gn = json.loads((cal / "gn_map_real.json").read_text())
    noise = effective_sigma(mp)  # (sigma_pit_eff, rho_pit, sigma_core_eff, rho_core)
    wh = Whitened.build(log, *noise)
    x_map = np.array([mp[n] for n in NAMES], dtype=float)
    z_map = to_z(x_map)
    F, g, H, w = objective(z_map, wh, log)
    cov = np.linalg.inv(H)
    print(f"log: {len(log['t'])} points, {wh.L.shape[0]} whitened residuals; "
          f"F(MAP) = {F:.4f}, |grad| = {np.linalg.norm(g):.3g}, 0.5|w|^2 = {0.5 * w @ w:.4f} "
          f"({time.time() - t0:.0f} s)")
    out = {
        "log_name": "mache_2013",
        "t": np.asarray(log["t"], dtype=float).tolist(),
        "u": np.asarray(log["u"], dtype=float).tolist(),
        "obs_pit": np.asarray(log["obs_pit"], dtype=float).tolist(),
        "obs_core": np.asarray(log["obs_core"], dtype=float).tolist(),
        "mask_pit": np.asarray(log["mask_pit"], dtype=bool).tolist(),
        "mask_core": np.asarray(log["mask_core"], dtype=bool).tolist(),
        "meta": {
            "tpit0_K": float(log["meta"]["tpit0_K"]),
            "tcore0_K": float(log["meta"]["tcore0_K"]),
            "t_amb_K": float(log["meta"].get("t_amb_K", 300.0)),
            "m0_kg": 7.0,
            "abstol": 1e-6,
            "reltol": 1e-6,
        },
        "names": NAMES,
        "prior": prior_spec(),
        "noise": {
            "sigma_floor_K": SIGMA_FLOOR_K,
            "sigma_pit_eff": noise[0], "rho_pit": noise[1],
            "sigma_core_eff": noise[2], "rho_core": noise[3],
        },
        "L": wh.L.tolist(),
        "y": wh.y.tolist(),
        "z0": prior_median_z().tolist(),
        "x_map": x_map.tolist(),
        "z_map": z_map.tolist(),
        "F_map": float(F),
        "gnorm_map": float(np.linalg.norm(g)),
        "H_map": H.tolist(),
        "cov_map": cov.tolist(),
        "sd_z_fisher_gn": [float(gn["sd_z_fisher"][n]) for n in NAMES],
        "gn_iterations": int(gn["iterations"]),
        "gn_wall_s": float(gn["wall_s"]),
    }
    dst = cal / "estimator_gate_real.json"
    dst.write_text(json.dumps(out))
    print("wrote", dst, f"({dst.stat().st_size / 1e6:.1f} MB)")
    sd_here = np.sqrt(np.diag(cov))
    sd_gn = np.array(out["sd_z_fisher_gn"])
    print("Fisher sd here vs gn_map_real.json, max rel diff:",
          f"{np.max(np.abs(sd_here - sd_gn) / sd_gn):.2e}")


if __name__ == "__main__":
    main()
