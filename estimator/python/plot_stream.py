"""Figures for the streaming-estimator mock runs.

One figure per plant (data/mock/<name>_stream.json): per latent of
interest, the posterior width against cook hour (top) and the
standardized error against the plant's truth (bottom), one line per
channel set. Widths that stop shrinking are what the sensors cannot see;
standardized errors that leave the ±2 band are the model being wrong in a
way the extra sensors cannot absorb.

Run from app/:  uv run python ../estimator/python/plot_stream.py
Writes figures/estimator/stream_<name>.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SETS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (dataviz palette)
# color follows the channel set, not its position in a run
SET_COLOR = {"pit + core": SETS_COLORS[0], "pit + core + humidity + scale": SETS_COLORS[1],
             "+ rulers (thickness, footprint)": SETS_COLORS[2]}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SHOW = ["L", "h_c", "k_m", "a0", "A_meat", "R", "shape", "core_depth_fraction", "r_max", "a_myosin"]


def plot(path: Path):
    d = json.loads(path.read_text())
    names = d["names"]
    idx = {n: i for i, n in enumerate(names)}
    sets = d["sets"]
    cols = [n for n in SHOW if n in idx]
    # one error scale for the whole row: the panels are read against each other
    lim = max(3.0, 1.1 * float(np.nanmax(np.abs(np.array(
        [b["err_sd"][idx[n]] for n in cols for s in sets for b in s["batches"]])))))
    fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols) + 1.0, 5.6), sharex=True,
                             layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for c, n in enumerate(cols):
        j = idx[n]
        ax_sd, ax_err = axes[0, c], axes[1, c]
        for ax in (ax_sd, ax_err):
            ax.set_facecolor(SURFACE)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(colors=INK2, labelsize=8, length=2)
            for sp in ax.spines.values():
                sp.set_color("#c3c2b7")
        for k, s in enumerate(sets):
            th = np.array([b["t_h"] for b in s["batches"]])
            sd = np.array([b["sd"][j] for b in s["batches"]])
            err = np.array([b["err_sd"][j] for b in s["batches"]])
            col = SET_COLOR.get(s["label"], SETS_COLORS[k])
            ax_sd.plot(th, sd, color=col, lw=2, label=s["label"])
            ax_err.plot(th, err, color=col, lw=2)
        ax_sd.set_yscale("log")
        ax_sd.set_title(n, color=INK, fontsize=10)
        ax_err.axhspan(-2, 2, color="#f0efec", zorder=0)
        ax_err.axhline(0, color=INK2, lw=0.8)
        ax_err.set_xlabel("cook hour", color=INK2, fontsize=9)
        ax_err.set_ylim(-lim, lim)
    axes[0, 0].set_ylabel("posterior sd (unconstrained)", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("(estimate − truth) / sd", color=INK2, fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3,
               fontsize=9, frameon=False, labelcolor=INK2)
    fig.suptitle(f"Streaming estimator on a mock cook, plant: {d['plant']}; hourly full-history refits",
                 color=INK, fontsize=10.5)
    out = REPO / "figures" / "estimator" / ("stream_" + path.stem.replace("_stream", "") + ".png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted((REPO / "data" / "mock").glob("*_stream.json"))
    for p in paths:
        plot(p)
