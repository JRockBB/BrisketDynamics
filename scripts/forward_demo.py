"""Phase A demo: full 12-h coupled cook at default parameters.

Produces figures/forward_cook.png (pit + surface + core temperatures, the
stall, evaporative flux, fuel) and prints the Picard convergence residual.

Run:  uv run --project app python scripts/forward_demo.py
"""

import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brisket_app.clients import meat_client, pit_client
from brisket_app.coupled import HR, PHI_DEFAULT, THETA_DEFAULT, CookGrids, check_picard, coupled_cook


def main():
    """Run the 12-h coupled cook at default parameters and render the demo figure.

    Runs ``coupled_cook`` (K=2 Picard) at THETA_DEFAULT/PHI_DEFAULT with a
    flat damper of 0.4, prints the wall time and the Picard convergence
    residual max|dT_pit|, and writes figures/forward_cook.png: three stacked
    panels of pit/surface/core temperature (showing the stall), evaporative
    mass flux, and remaining fuel versus time. Also prints the final core
    temperature and core collagen conversion.
    """
    grids = CookGrids()
    meat_tx, pit_tx = meat_client(), pit_client()

    t0 = time.perf_counter()
    out = coupled_cook(
        jnp.asarray(THETA_DEFAULT),
        jnp.asarray(PHI_DEFAULT),
        jnp.full(grids.n_u, 0.4),
        meat_tx,
        pit_tx,
        grids,
        K=2,
    )
    dt = time.perf_counter() - t0
    print(f"coupled forward (K=2): {dt:.1f} s wall")
    check_picard(out)

    th = np.asarray(grids.tknots) / HR
    to_c = lambda x: np.asarray(x) - 273.15

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(th, to_c(out["tpit"]), label="pit", color="tab:red")
    ax.plot(th, to_c(out["tsurf"]), label="surface", color="tab:orange")
    ax.plot(th, to_c(out["tcore"]), label="core (probe)", color="tab:blue", lw=2)
    ax.set_ylabel("temperature [°C]")
    ax.legend()
    ax.set_title("Coupled 12-h cook, damper = 0.4 (K=2 Picard)")

    ax = axes[1]
    ax.plot(th, 1e6 * np.asarray(out["j_evap"]), color="tab:cyan")
    ax.set_ylabel("evaporative flux [mg/m²/s]")

    ax = axes[2]
    ax.plot(th, np.asarray(out["m_fuel"]), color="tab:brown")
    ax.set_ylabel("fuel [kg]")
    ax.set_xlabel("time [h]")

    figdir = REPO / "figures"
    figdir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(figdir / "forward_cook.png", dpi=150)
    print(f"wrote {figdir / 'forward_cook.png'}")

    core_c = to_c(out["tcore"])
    print(
        f"final core: {core_c[-1]:.1f} °C; final collagen (core): "
        f"{float(np.asarray(out['c_field_final'])[0]):.3f}"
    )


if __name__ == "__main__":
    main()
