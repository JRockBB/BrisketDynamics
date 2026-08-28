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
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.coupled import HR, CookGrids, coupled_cook

REPO = Path(__file__).resolve().parents[1]
GRIDS = CookGrids(t_end=14 * HR, dt_exchange=600.0, n_u=8)
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}


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
    )
    return {k: np.asarray(v) for k, v in sim.items()}


def main():
    """Run the three bark-validation cook styles and render the figure.

    Simulates low-and-slow, hot-and-fast, and low-and-slow with a 5 h wrap
    at the MAP-real smoker, prints and saves the final bark states and mean
    pit temperature per style (data/control/bark_forward.json), and writes
    figures/bark_model.png: three panels of predicted surface lightness L*,
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
        "low-slow + wrap @ 5 h": {"u": 0.18, "t_wrap": 5 * HR},
    }
    sims = {name: run(theta, phi, **kw, tx=tx) for name, kw in styles.items()}

    out = {}
    for name, s in sims.items():
        out[name] = {
            "B_end": float(s["bark_b"][-1]),
            "S_end": float(s["bark_s"][-1]),
            "L_end": float(s["l_star"][-1]),
            "pit_mean_C": float(np.mean(s["tpit"][len(th) // 4 :]) - 273.15),
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
        xlabel="time [h]",
        ylabel="predicted surface L*",
        title="bark color: lightness darkens and saturates",
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set(
        xlabel="time [h]",
        ylabel="B (Maillard state)",
        title="browning: hot cooks Maillard-saturate",
    )
    axes[2].set(
        xlabel="time [h]",
        ylabel="S (smoke deposition state)",
        title="smoke: deposits on the wet surface; wrap freezes it",
    )
    for ax in axes:
        ax.axvline(5.0, color=MUTED, lw=1, ls=":", alpha=0.6)
    axes[2].annotate("wrap @ 5 h", (5.15, 0.05), color=MUTED, fontsize=8)

    fig.suptitle(
        "Differentiable bark model (uncalibrated literature "
        "constants): two states over the coupled cook",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "bark_model.png", dpi=140, bbox_inches="tight")
    print("wrote figures/bark_model.png")


if __name__ == "__main__":
    main()
