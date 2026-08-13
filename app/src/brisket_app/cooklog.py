"""Loader for the real cook log (data/cook_log/mache_brisket_cook.csv).

See data/cook_log/PROVENANCE.md: 18.5" WSM, HeaterMeter @ 225 F setpoint,
10.5 lb packer, unwrapped, single fuel load, ~19 h. Fan duty is the measured
control input; lid events are flagged. All temperatures converted to Kelvin,
resampled to a uniform grid, NaNs masked.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[3] / "data" / "cook_log" / "mache_brisket_cook.csv"

F2K = lambda f: (np.asarray(f, dtype=np.float64) - 32.0) * 5.0 / 9.0 + 273.15


def load(dt_min: float = 10.0, t_end_min: float | None = 1140.0) -> dict:
    """Load + resample to a uniform dt_min grid.

    Returns dict with t [s], u (fan duty 0-1, NaN-interpolated), obs_pit [K],
    obs_core [K], mask_pit/mask_core (True = usable observation: finite and
    not within a lid event), amb [K, qualitative], meta.
    """
    rows = list(csv.DictReader(open(DATA)))
    g = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                            for r in rows])
    t_min = g("t_min")
    pit = F2K(g("pit_F"))
    meat = F2K(g("meat_F"))
    amb = F2K(g("amb_F"))
    fan = g("fan_pct") / 100.0
    lid = np.array([r["lid_event"].strip().lower() == "true" for r in rows])

    if t_end_min is not None:
        keep = t_min <= t_end_min
        t_min, pit, meat, amb, fan, lid = (x[keep] for x in
                                           (t_min, pit, meat, amb, fan, lid))

    # uniform target grid
    tg = np.arange(t_min[0], t_min[-1] + 1e-9, dt_min)

    def resample(x):
        ok = np.isfinite(x)
        return np.interp(tg, t_min[ok], x[ok])

    # lid flag: a grid point is "lidded" if any raw lid sample within dt_min/2
    lid_g = np.zeros(len(tg), dtype=bool)
    for i, tc in enumerate(tg):
        near = np.abs(t_min - tc) <= dt_min / 2 + 1e-9
        lid_g[i] = bool(np.any(lid[near]))

    pit_g, meat_g, amb_g, fan_g = map(resample, (pit, meat, amb, fan))
    # masks: usable observations exclude lid events and originally-NaN spans
    raw_ok_meat = np.isfinite(meat)
    meat_has_data = np.interp(tg, t_min, raw_ok_meat.astype(float)) > 0.99
    mask_pit = ~lid_g
    mask_core = ~lid_g & meat_has_data

    return {
        "t": tg * 60.0,                    # [s]
        "u": np.clip(fan_g, 0.0, 1.0),
        "obs_pit": pit_g,
        "obs_core": meat_g,
        "mask_pit": mask_pit,
        "mask_core": mask_core,
        "amb": amb_g,
        "meta": {
            "tpit0_K": float(pit_g[0]),
            "tcore0_K": float(meat_g[meat_has_data.argmax()]),
            "t_end_h": float(tg[-1] / 60.0),
            "n_obs_pit": int(mask_pit.sum()),
            "n_obs_core": int(mask_core.sum()),
        },
    }
