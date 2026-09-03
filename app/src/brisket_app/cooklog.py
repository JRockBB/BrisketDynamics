"""Loaders for the real cook logs (registry below; see each PROVENANCE).

mache_2013 (the calibration reference): 18.5" WSM, HeaterMeter @ 225 F,
10.5 lb packer, unwrapped, single fuel load, ~19 h, ~85 F summer day.
phillip_2018: 15 lb packer, lump + hickory/sassafras, HeaterMeter @ 225 F,
~1 F winter night — a PARTIAL log (live-view screenshot, rise + stall entry,
no pull); the sub-freezing early core is sliced off (start_at_core_K) since
the model carries no thaw enthalpy. Fan duty is the measured control input;
lid events are flagged. Temperatures in Kelvin, uniform grid, NaNs masked.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]

F2K = lambda f: (np.asarray(f, dtype=np.float64) - 32.0) * 5.0 / 9.0 + 273.15

# per-log registry: csv path, default trim, ambient [K], and (optionally)
# the core temperature above which the fit starts (slices off a frozen or
# probe-settling early segment; t re-zeroed at the slice)
LOGS = {
    "mache_2013": {
        "csv": _REPO / "data" / "cook_log" / "mache_brisket_cook.csv",
        "t_end_min": 1140.0,
        "t_amb_K": 300.0,
        "start_at_core_K": None,
    },
    "phillip_2018": {
        "csv": _REPO / "tools" / "loghunt" / "logs" / "phillip_2018"
        / "phillip_2018.csv",
        "t_end_min": None,
        "t_amb_K": 255.9,  # ~1 F per the post
        "start_at_core_K": 274.8,  # ~35 F: above the phase-change regime
    },
    # Dale Ward's Big Steel Keg: two cooks on ONE vessel a year apart —
    # the same-smoker pooling pair (roadmap 5a). See each PROVENANCE.md.
    "keg_2013": {
        "csv": _REPO / "tools" / "loghunt" / "logs" / "keg_2013"
        / "keg_2013.csv",
        "t_end_min": None,
        "t_amb_K": 294.3,  # ~70 F: pit+meat probes flat for 3 h pre-relight
        "start_at_core_K": None,
    },
    "keg_2015": {
        "csv": _REPO / "tools" / "loghunt" / "logs" / "keg_2015"
        / "keg_2015.csv",
        "t_end_min": None,
        "t_amb_K": 294.0,  # ~69.5 F: pit probe flat all afternoon pre-fire
        "start_at_core_K": None,
    },
    "keg_2014": {
        "csv": _REPO / "tools" / "loghunt" / "logs" / "keg_2014"
        / "keg_2014.csv",
        "t_end_min": None,
        "t_amb_K": 311.0,  # ambient-probe median 100 F (hot day; may be
        # plume/sun-elevated — see PROVENANCE)
        "start_at_core_K": None,
    },
}


def load(
    dt_min: float = 10.0,
    t_end_min: float | None = 1140.0,
    log: str = "mache_2013",
) -> dict:
    """Load + resample to a uniform dt_min grid.

    Returns dict with t [s], u (fan duty 0-1, NaN-interpolated), obs_pit [K],
    obs_core [K], mask_pit/mask_core (True = usable observation: finite and
    not within a lid event), amb [K, qualitative], meta.
    """
    spec = LOGS[log]
    if log != "mache_2013":  # the mache default preserves historical calls
        t_end_min = spec["t_end_min"]
    with open(spec["csv"]) as f:
        rows = list(csv.DictReader(f))
    g = lambda k: np.array(
        [float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]
    )
    t_min = g("t_min")
    pit = F2K(g("pit_F"))
    meat = F2K(g("meat_F"))
    amb = F2K(g("amb_F"))
    fan = g("fan_pct") / 100.0
    lid = np.array([r["lid_event"].strip().lower() == "true" for r in rows])

    if t_end_min is not None:
        keep = t_min <= t_end_min
        t_min, pit, meat, amb, fan, lid = (
            x[keep] for x in (t_min, pit, meat, amb, fan, lid)
        )

    # uniform target grid
    tg = np.arange(t_min[0], t_min[-1] + 1e-9, dt_min)

    def resample(x):
        ok = np.isfinite(x)
        if not ok.any():  # e.g. an ambient probe that was off
            return np.full(len(tg), np.nan)
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
    usable_core = np.where(mask_core)[0]
    tcore0 = (
        float(meat_g[usable_core[0]])
        if usable_core.size
        else float(meat_g[meat_has_data.argmax()])
    )

    # optional slice: start the fit above a core threshold (frozen start)
    if spec["start_at_core_K"] is not None:
        ok = mask_core & (meat_g >= spec["start_at_core_K"])
        if not ok.any():
            raise ValueError(f"{log}: core never reaches start_at_core_K")
        i0 = int(np.argmax(ok))
        tg, pit_g, meat_g, amb_g, fan_g = (
            x[i0:] for x in (tg, pit_g, meat_g, amb_g, fan_g)
        )
        mask_pit, mask_core = mask_pit[i0:], mask_core[i0:]
        tg = tg - tg[0]
        tcore0 = float(meat_g[0])

    return {
        "t": tg * 60.0,  # [s]
        "u": np.clip(fan_g, 0.0, 1.0),
        "obs_pit": pit_g,
        "obs_core": meat_g,
        "mask_pit": mask_pit,
        "mask_core": mask_core,
        "amb": amb_g,
        "log": log,
        "meta": {
            "tpit0_K": float(pit_g[0]),
            "tcore0_K": tcore0,
            "t_amb_K": float(spec["t_amb_K"]),
            "t_end_h": float((tg[-1] - tg[0]) / 60.0),
            "n_obs_pit": int(mask_pit.sum()),
            "n_obs_core": int(mask_core.sum()),
        },
    }
