"""The calibration's noise posterior as the estimator's noise prior, in data form.

The real-cook NUTS chain carries the AR(1) discrepancy sites of the pit
and core channels: rho and the excess sigma over the 1.2 K floor
(sigma_eff = sigma + floor). This writes their posterior moments, in
the estimator's parameterization (the marginal sd sigma_eff and rho),
to data/calibration/noise_prior_real.json, one record per channel.
Hypothetical channels carry no calibration; the estimator keeps their
assumed values with wide widths.

Run from app/:  uv run python ../estimator/export_noise_prior.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
gate = json.loads((REPO / "data" / "calibration" / "estimator_gate_real.json").read_text())
floor = float(gate["noise"]["sigma_floor_K"])
z = np.load(REPO / "data" / "calibration" / "nuts_mono_samples_real.npz")

out = {"source": "nuts_mono_samples_real.npz (the real 19 h cook), sigma_eff = sigma + floor",
       "sigma_floor_K": floor, "channels": {}}
for ch, s_key, r_key in (("pit", "sigma_pit", "rho_pit"), ("core", "sigma_core", "rho_core")):
    s = np.asarray(z[s_key]).ravel() + floor
    r = np.asarray(z[r_key]).ravel()
    out["channels"][ch] = {
        "sigma_mean": float(s.mean()), "sigma_sd": float(s.std(ddof=1)),
        "rho_mean": float(r.mean()), "rho_sd": float(r.std(ddof=1)),
        "n_draws": int(s.size),
        "sigma_eff_map": float(gate["noise"][f"sigma_{ch}_eff"]), "rho_map": float(gate["noise"][f"rho_{ch}"]),
    }
    print(f"{ch}: sigma_eff {s.mean():.3f} +- {s.std(ddof=1):.3f} K, rho {r.mean():.4f} +- {r.std(ddof=1):.4f}")
dst = REPO / "data" / "calibration" / "noise_prior_real.json"
dst.write_text(json.dumps(out, indent=1))
print("wrote", dst)
