"""Figures for calibration results: corner plot and posterior-predictive band.

Both take the raw sample dict from `MCMC.get_samples()` (or a dict of MAP
values expanded to shape (1,) arrays) so they work unchanged for NUTS samples
and for Laplace draws.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibrate import PHI_SITES, THETA_SITES
from .coupled import HR, coupled_cook

PARAM_NAMES = [n for n, _ in THETA_SITES] + [n for n, _ in PHI_SITES]
# log-scale for the strictly positive scale parameters; u_leak stays linear
LOG_PARAMS = {n for n in PARAM_NAMES if n != "u_leak"}


def corner_plot(samples: dict, out_path: Path, truths: dict | None = None):
    """Corner plot over the 10 physical parameters (log10 axes where apt)."""
    import corner as corner_mod

    cols = []
    labels = []
    truth_vals = [] if truths is not None else None
    for n in PARAM_NAMES:
        x = np.asarray(samples[n]).reshape(-1)
        if n in LOG_PARAMS:
            cols.append(np.log10(x))
            labels.append(f"log10 {n}")
            if truths is not None:
                truth_vals.append(np.log10(truths[n]))
        else:
            cols.append(x)
            labels.append(n)
            if truths is not None:
                truth_vals.append(truths[n])
    data = np.column_stack(cols)
    fig = corner_mod.corner(
        data, labels=labels, truths=truth_vals, truth_color="tab:red",
        show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )
    fig.suptitle("Posterior over pit + meat parameters (stall-informed)", y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def posterior_predictive_plot(samples: dict, data: dict, meat_tx, pit_tx, grids,
                              out_path: Path, n_draws: int = 30, seed: int = 0,
                              ics: dict | None = None, masks: dict | None = None):
    """Simulate n_draws posterior samples through the coupled pipeline and
    plot the predictive band over the observed pit/core trajectories.
    Sequential loop over draws — vmap over Tesseract calls needs an explicit
    vmap_method and buys nothing here.

    `data` needs u_knots/obs_pit/obs_core (synthetic dict or cooklog dict with
    u renamed); `ics`/`masks` are for real logs (initial conditions and
    observation masks)."""
    rng = np.random.default_rng(seed)
    n_total = len(np.asarray(samples[PARAM_NAMES[0]]).reshape(-1))
    idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

    th = np.asarray(grids.tknots) / HR
    cores, pits = [], []
    for i in idx:
        theta = jnp.asarray([float(np.asarray(samples[n]).reshape(-1)[i])
                             for n, _ in THETA_SITES])
        phi = jnp.asarray([float(np.asarray(samples[n]).reshape(-1)[i])
                           for n, _ in PHI_SITES])
        sim = coupled_cook(theta, phi, jnp.asarray(data["u_knots"]),
                           meat_tx, pit_tx, grids, K=2, ics=ics)
        cores.append(np.asarray(sim["tcore"]))
        pits.append(np.asarray(sim["tpit"]))
    cores = np.array(cores) - 273.15
    pits = np.array(pits) - 273.15

    mp = masks.get("pit") if masks else np.ones(len(th), dtype=bool)
    mc = masks.get("core") if masks else np.ones(len(th), dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, sims, obs, mk, name in [
        (axes[0], pits, data["obs_pit"] - 273.15, mp, "pit"),
        (axes[1], cores, data["obs_core"] - 273.15, mc, "core (probe)"),
    ]:
        lo, hi = np.percentile(sims, [5, 95], axis=0)
        ax.fill_between(th, lo, hi, alpha=0.3, label="posterior 5-95%")
        ax.plot(th, np.median(sims, axis=0), lw=1.2, label="posterior median")
        ax.plot(th[mk], obs[mk], ".", ms=3, color="k", alpha=0.5, label="observations")
        ax.set(xlabel="time [h]", ylabel="temperature [C]", title=name)
        ax.legend(fontsize=8)
    fig.suptitle("Posterior predictive vs observed cook log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
