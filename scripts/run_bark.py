"""Bark model forward validation (docs/bark_plan.md step 4).

Three cook styles through the full coupled pipeline at the MAP-real
smoker — low-and-slow, hot-and-fast, and low-and-slow with a 5 h wrap —
showing the two bark states and predicted surface lightness L*. The smoke
forcing is the pit's own burn rate (mdot), so this figure is a genuinely
two-Tesseract product.

Run from app/:  uv run python -u ../scripts/run_bark.py
Writes figures/bark_model.png + data/control/bark_forward.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp

from brisket_app.clients import meat_client, pit_client
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.coupled import HR, CookGrids, coupled_cook

REPO = Path(__file__).resolve().parents[1]
GRIDS = CookGrids(t_end=14 * HR, dt_exchange=600.0, n_u=8)
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}


def run(theta, phi, u, meat_tx, pit_tx, t_wrap=1.0e9):
    sim = coupled_cook(jnp.asarray(theta), jnp.asarray(phi),
                       jnp.asarray(np.full(GRIDS.n_u, u)), meat_tx, pit_tx,
                       GRIDS, K=2, ics=ICS, t_wrap=t_wrap)
    return {k: np.asarray(v) for k, v in sim.items()}


def main():
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    meat_tx, pit_tx = meat_client(), pit_client()
    th = np.asarray(GRIDS.tknots) / HR

    styles = {
        "low-and-slow (u=0.18)": dict(u=0.18),
        "hot-and-fast (u=0.55)": dict(u=0.55),
        "low-slow + wrap @ 5 h": dict(u=0.18, t_wrap=5 * HR),
    }
    sims = {name: run(theta, phi, **kw, meat_tx=meat_tx, pit_tx=pit_tx)
            for name, kw in styles.items()}

    out = {}
    for name, s in sims.items():
        out[name] = {"B_end": float(s["bark_b"][-1]),
                     "S_end": float(s["bark_s"][-1]),
                     "L_end": float(s["l_star"][-1]),
                     "pit_mean_C": float(np.mean(s["tpit"][len(th) // 4:]) - 273.15)}
        print(f"{name}: {out[name]}")
    (REPO / "data" / "control").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "control" / "bark_forward.json").write_text(
        json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED = "#256abf", "#eb6834", "#0b0b0b", "#52514e"
    COLORS = {list(styles)[0]: BLUE, list(styles)[1]: ORANGE,
              list(styles)[2]: MUTED}
    LS = {list(styles)[0]: "-", list(styles)[1]: "-", list(styles)[2]: "--"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, s in sims.items():
        c, ls = COLORS[name], LS[name]
        axes[0].plot(th, s["l_star"], ls, color=c, lw=2, label=name)
        axes[1].plot(th, s["bark_b"], ls, color=c, lw=2)
        axes[2].plot(th, s["bark_s"], ls, color=c, lw=2)
    axes[0].set(xlabel="time [h]", ylabel="predicted surface L*",
                title="bark color: lightness darkens and saturates")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set(xlabel="time [h]", ylabel="B (Maillard state)",
                title="browning: hot cooks Maillard-saturate")
    axes[2].set(xlabel="time [h]", ylabel="S (smoke deposition state)",
                title="smoke: deposits on the wet surface; wrap freezes it")
    wrap_name = list(styles)[2]
    for ax in axes:
        ax.axvline(5.0, color=MUTED, lw=1, ls=":", alpha=0.6)
    axes[2].annotate("wrap @ 5 h", (5.15, 0.05), color=MUTED, fontsize=8)

    fig.suptitle("Differentiable bark model (uncalibrated literature "
                 "constants): two states over the coupled cook", y=1.0)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "bark_model.png", dpi=140,
                bbox_inches="tight")
    print("wrote figures/bark_model.png")


if __name__ == "__main__":
    main()
