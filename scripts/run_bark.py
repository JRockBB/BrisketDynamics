"""Bark model forward validation (figures/model/bark_model.png).

Three cook styles through the full coupled pipeline at the MAP-real
smoker — low-and-slow, hot-and-fast, and low-and-slow with a 5 h wrap —
showing the two bark states and predicted surface lightness L*. The smoke
forcing is the pit's own burn rate (mdot), so this figure is a product of
the coupled solve.

Run from app/:  uv run python -u ../scripts/run_bark.py
Writes figures/model/bark_model.png + data/control/bark_forward.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.coupled import HR, CookGrids, coupled_cook

T_SALT = 12.0 * HR  # overnight dry brine, the standard cook

REPO = Path(__file__).resolve().parents[1]
GRIDS = CookGrids(t_end=15 * HR, dt_exchange=600.0, n_u=8)
# a forward comparison of fixed styles, not a plan: every style preheats the same hour, the meat then goes on
T_ON = 1.0 * HR
# the cook from lighting: the pit at ambient, the meat from the fridge (on at the plan's load)
ICS = {"tpit0": 300.0, "t_amb": 300.0, "t0_meat": 281.15}


def run(theta, phi, u, tx, t_wrap=1.0e9):
    """Simulate one cook style through the full coupled pipeline.

    Parameters
    ----------
    theta : array_like, shape (7,)
        Meat parameters (here the MAP-real fit).
    phi : array_like, shape (7,)
        Pit parameters (here the MAP-real fit).
    u : float
        Constant damper fraction in [0, 1], held on all GRIDS control knots.
    tx
        Tesseract clients for the meat and pit solvers.
    t_wrap : float, optional
        Texas-crutch wrap time [s]; the default 1e9 means never wrap.

    Returns
    -------
    dict of str to numpy.ndarray
        The ``coupled_cook`` outputs (tpit, tcore, bark_b, bark_s, l_star,
        ...) converted to NumPy arrays.
    """
    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.asarray(np.full(GRIDS.n_u, u)),
        tx,
        GRIDS,
        ics=ICS,
        t_wrap=t_wrap,
        t_salt=T_SALT,
        t_load=T_ON,
    )
    return {k: np.asarray(v) for k, v in sim.items()}


def main():
    """Run the three bark-validation cook styles and render the figure.

    Simulates low-and-slow, hot-and-fast, and low-and-slow with a 5 h wrap
    at the MAP-real smoker, prints and saves the final bark states and mean
    pit temperature per style (data/control/bark_forward.json), and writes
    figures/model/bark_model.png: three panels of predicted surface lightness L*,
    Maillard browning state B, and smoke-deposition state S versus time,
    with the wrap moment marked.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    tx = monolith_client()
    th = np.asarray(GRIDS.tknots) / HR

    styles = {
        "low-and-slow (u=0.18)": {"u": 0.18},
        "hot-and-fast (u=0.55)": {"u": 0.55},
        "low-slow + wrap @ 5 h": {"u": 0.18, "t_wrap": T_ON + 5 * HR},
    }
    sims = {name: run(theta, phi, **kw, tx=tx) for name, kw in styles.items()}

    out = {}
    for name, s in sims.items():
        out[name] = {
            "B_end": float(s["bark_b"][-1]),
            "S_end": float(s["bark_s"][-1]),
            "L_end": float(s["l_star"][-1]),
            "pit_mean_C": float(np.mean(s["tpit"][th >= T_ON / HR + 2.0]) - 273.15),
        }
        print(f"{name}: {out[name]}")
    (REPO / "data" / "control").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "control" / "bark_forward.json").write_text(
        json.dumps(out, indent=2)
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, MUTED = "#256abf", "#eb6834", "#52514e"
    names = list(styles)
    COLORS = {names[0]: BLUE, names[1]: ORANGE, names[2]: MUTED}
    LS = {names[0]: "-", names[1]: "-", names[2]: "--"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, s in sims.items():
        c, ls = COLORS[name], LS[name]
        axes[0].plot(th, s["l_star"], ls, color=c, lw=2, label=name)
        axes[1].plot(th, s["bark_b"], ls, color=c, lw=2)
        axes[2].plot(th, s["bark_s"], ls, color=c, lw=2)
    axes[0].set(
        xlabel="time from lighting [h]",
        ylabel="predicted surface L*",
        title="bark color: lightness darkens and saturates",
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set(
        xlabel="time from lighting [h]",
        ylabel="B (Maillard state)",
        title="browning: hot cooks Maillard-saturate",
    )
    axes[2].set(
        xlabel="time from lighting [h]",
        ylabel="S (smoke deposition state)",
        title="smoke: deposits on the wet surface; wrap freezes it",
    )
    t_w = T_ON / HR + 5.0
    for ax in axes:
        ax.axvline(T_ON / HR, color=MUTED, lw=1, ls="-.", alpha=0.6)
        ax.axvline(t_w, color=MUTED, lw=1, ls=":", alpha=0.6)
    axes[2].annotate("meat on", (T_ON / HR + 0.15, 0.72), color=MUTED, fontsize=8)
    axes[2].annotate("wrap 5 h after", (t_w + 0.15, 0.05), color=MUTED, fontsize=8)

    fig.suptitle(
        "Differentiable bark model (uncalibrated literature "
        "constants): two states over the coupled cook",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "model" / "bark_model.png", dpi=140, bbox_inches="tight")
    print("wrote figures/model/bark_model.png")


if __name__ == "__main__":
    main()
