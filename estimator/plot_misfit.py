"""The misfit check against cook hour on the two mock plants.

One figure: per channel (columns), the whitened residual power's χ²
z-score (top) and the lag-1 autocorrelation's z-score (bottom) at every
hourly refit, matched plant against the cylinder, solid for the
humidity-plus-scale set and dashed for pit + core. The ±3 band is where
a right model with right noise lives.

Run from app/:  uv run python ../estimator/plot_misfit.py
Reads data/mock/mock_1d_misfit.json and mock_2d_misfit.json,
writes figures/estimator_misfit.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
PLANT_COLORS = {"mock_1d": "#2a78d6", "mock_2d": "#eb6834"}  # categorical slots 1-2
PLANT_LABELS = {"mock_1d": "matched plant (monolith)", "mock_2d": "cylinder plant (2-D, mismatched)"}
SETS = {"pit + core + humidity + scale": "-", "pit + core": (0, (4, 2))}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
COLS = ["pit", "meat probe", "humidity", "scale", "total"]
STATS = [("z_chi2", "residual power, χ² z-score"), ("z_rho1", "lag-1 autocorrelation, z-score")]


def series(d, label, name, key):
    st = next(s for s in d["sets"] if s["label"] == label)
    th, val = [], []
    for b in st["batches"]:
        recs = b["channels"] + [b["total"]]
        rec = next((c for c in recs if c["name"] == name), None)
        if rec is None:
            continue
        th.append(b["t_h"])
        val.append(np.nan if rec[key] is None else rec[key])
    return np.array(th), np.array(val, dtype=float)


def main():
    data = {p: json.loads((REPO / "data" / "mock" / f"{p}_misfit.json").read_text()) for p in PLANT_COLORS}
    fig, axes = plt.subplots(2, len(COLS), figsize=(2.6 * len(COLS) + 0.8, 5.4), sharex=True,
                             layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    allv = [np.abs(series(d, lab, n, k)[1]) for d in data.values() for lab in SETS for n in COLS
            for k, _ in STATS]
    lim = max(6.0, 1.1 * float(np.nanmax(np.concatenate([v for v in allv if v.size]))))
    for c, name in enumerate(COLS):
        for r, (key, ylabel) in enumerate(STATS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(colors=INK2, labelsize=8, length=2)
            for sp in ax.spines.values():
                sp.set_color("#c3c2b7")
            ax.axhspan(-3, 3, color="#f0efec", zorder=0)
            ax.axhline(0, color=INK2, lw=0.8)
            for p, d in data.items():
                for lab, ls in SETS.items():
                    th, v = series(d, lab, name, key)
                    if th.size:
                        ax.plot(th, v, color=PLANT_COLORS[p], lw=2, ls=ls)
            ax.set_ylim(-lim, lim)
            if r == 0:
                ax.set_title(name, color=INK, fontsize=10)
            if r == 1:
                ax.set_xlabel("cook hour", color=INK2, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    handles = [Line2D([], [], color=PLANT_COLORS[p], lw=2, label=PLANT_LABELS[p]) for p in PLANT_COLORS]
    handles += [Line2D([], [], color=INK2, lw=2, ls=ls, label=lab) for lab, ls in SETS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=False,
               labelcolor=INK2)
    fig.suptitle("The misfit check at every hourly refit: what a refit reports next to its posterior",
                 color=INK, fontsize=10.5)
    out = REPO / "figures" / "estimator_misfit.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
