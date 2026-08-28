"""Open-loop replay of the real cook at default (uncalibrated) parameters.

Measured fan duty as u(t), default parameters, initial conditions from the
log. Shows how far the uncalibrated model sits from reality (and produces
the before-calibration figure).

Run from app/:  uv run python -u ../scripts/replay_cooklog.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.clients import monolith_client
from brisket_app.cooklog import load
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids, coupled_cook

REPO = Path(__file__).resolve().parents[1]

log = load(dt_min=10.0)
print(
    f"log: {log['meta']['t_end_h']:.1f} h, pit obs {log['meta']['n_obs_pit']}, "
    f"core obs {log['meta']['n_obs_core']}, Tpit0 {log['meta']['tpit0_K']:.1f} K, "
    f"Tcore0 {log['meta']['tcore0_K']:.1f} K"
)

grids = CookGrids(t_end=float(log["t"][-1]), dt_exchange=600.0, n_u=len(log["t"]))
ics = {
    "tpit0": log["meta"]["tpit0_K"],
    "t_amb": 300.0,
    "t0_meat": log["meta"]["tcore0_K"],
}
tx = monolith_client()
out = coupled_cook(
    jnp.asarray(THETA_DEFAULT),
    jnp.asarray(PHI_DEFAULT),
    jnp.asarray(log["u"]),
    tx,
    grids,
    ics=ics,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

th = log["t"] / HR
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
for ax, sim, obs, mask, name in [
    (axes[0], np.asarray(out["tpit"]), log["obs_pit"], log["mask_pit"], "pit"),
    (axes[1], np.asarray(out["tcore"]), log["obs_core"], log["mask_core"], "core"),
]:
    ax.plot(
        th[mask],
        obs[mask] - 273.15,
        ".",
        ms=2.5,
        color="k",
        alpha=0.5,
        label="observed (Mache/TVWBB)",
    )
    ax.plot(th, sim - 273.15, color="tab:orange", label="model @ defaults")
    ax.set(xlabel="time [h]", ylabel="temperature [C]", title=name)
    ax.legend(fontsize=8)
fig.suptitle("Real cook, open-loop fan-duty replay — BEFORE calibration")
fig.tight_layout()
fig.savefig(REPO / "figures" / "cooklog_replay_defaults.png", dpi=140)
print("wrote figures/cooklog_replay_defaults.png")
