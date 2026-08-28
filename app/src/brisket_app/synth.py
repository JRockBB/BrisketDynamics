"""Synthetic cook-log generation for calibration validation.

Simulates the coupled pit-brisket system at known "true" parameters and adds
observation noise, producing the dataset that Phase B's posterior-recovery
test calibrates against. The true parameters are deliberately offset from the
prior centers (see calibrate.PRIORS) so that recovery is informative.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .coupled import HR, CookGrids, coupled_cook

# Offset from THETA_DEFAULT / PHI_DEFAULT (the prior centers) by 20-40%:
# calibration has to *move* to find these, not just sit at the prior mode.
# (k_e/a_myosin truths offset likewise since joining theta 2026-08-21).
# Regenerating this dataset: from app/,
#   uv run python -c "from brisket_app.clients import monolith_client; from brisket_app import synth; synth.save(synth.generate(*[monolith_client()]))"
# then `uv run python ../scripts/run_map.py`. Do not mix with 5-parameter artifacts.
THETA_TRUE = np.array([1.0e-8, 0.0014, 12.5, 0.026, 1.15, 7.3e-4, 0.26])
# ignition entries inert (a0 = 1): synthetic cooks start fully lit
PHI_TRUE = np.array([18e3, 24.0, 2.5e-4, 1.35e7, 0.25, 3e-4, 1.0])

OBS_SIGMA_PIT = 1.5  # K, thermocouple-in-turbulent-pit scale
OBS_SIGMA_CORE = 0.8  # K, probe in thermal mass is quieter

# The synthetic cook: 12 h, damper stepped down after the ramp then up late —
# a realistic profile that also excites the pit dynamics beyond a constant.
U_COOK = np.array([0.8, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.55, 0.55])

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"


def generate(
    tx,
    grids: CookGrids | None = None,
    seed: int = 20260810,
    tols: dict | None = None,
) -> dict:
    """Generate one noisy synthetic cook at the known truths.

    Runs the coupled model at THETA_TRUE/PHI_TRUE under U_COOK (tight
    tolerances) and adds iid Gaussian probe noise.

    Returns
    -------
    dict
        t, u_knots, noisy obs_pit/obs_core, clean traces, the truths, the
        seed, and the grids.
    """
    grids = grids or CookGrids(t_end=12 * HR, dt_exchange=600.0, n_u=len(U_COOK))
    tols = tols or {"abstol": 1e-8, "reltol": 1e-8}  # data from a tighter solve
    out = coupled_cook(
        jnp.asarray(THETA_TRUE),
        jnp.asarray(PHI_TRUE),
        jnp.asarray(U_COOK),
        tx,
        grids,
        **tols,
    )
    rng = np.random.default_rng(seed)
    tpit_clean = np.asarray(out["tpit"])
    tcore_clean = np.asarray(out["tcore"])
    return {
        "t": np.asarray(grids.tknots),
        "u_knots": U_COOK,
        "obs_pit": tpit_clean + rng.normal(0.0, OBS_SIGMA_PIT, tpit_clean.shape),
        "obs_core": tcore_clean + rng.normal(0.0, OBS_SIGMA_CORE, tcore_clean.shape),
        "tpit_clean": tpit_clean,
        "tcore_clean": tcore_clean,
        "theta_true": THETA_TRUE,
        "phi_true": PHI_TRUE,
        "seed": seed,
        "grids": grids,
    }


def save(data: dict, name: str = "cook01") -> Path:
    """Write the array fields to data/synthetic/<name>.npz + a meta json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = DATA_DIR / f"{name}.npz"
    np.savez(
        npz_path,
        **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
    )
    meta = {
        "seed": data["seed"],
        "obs_sigma_pit": OBS_SIGMA_PIT,
        "obs_sigma_core": OBS_SIGMA_CORE,
        "t_end_h": float(data["t"][-1] / HR),
        "dt_exchange_s": float(data["t"][1] - data["t"][0]),
        "note": "Synthetic cook log from coupled_cook at THETA_TRUE/PHI_TRUE "
        "(tol 1e-8) + iid Gaussian noise. Generating code: synth.py.",
    }
    (DATA_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return npz_path


def load(name: str = "cook01") -> dict:
    """Load a saved synthetic cook as a plain dict of arrays."""
    with np.load(DATA_DIR / f"{name}.npz") as f:
        return dict(f)
