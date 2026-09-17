"""The noise identified as the cook goes: per channel, the AR node's sd
and correlation at every hourly refit with their posterior widths, the
truth and the prior it started from.

Run from app/:  uv run python ../estimator/python/plot_noise.py [record.json]
Default record data/mock/mock_1d_alt_noise_stream.json;
writes figures/estimator/noise_<name>.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SET_COLOR = {"pit + core": "#2a78d6", "pit + core + humidity + scale": "#eb6834"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
UNITS = {"pit": "K", "core": "K", "humidity": "Pa", "mass": "kg"}


def plot(path: Path):
    d = json.loads(path.read_text())
    chans = ["pit", "core", "humidity", "mass"]
    fig, axes = plt.subplots(2, len(chans), figsize=(2.9 * len(chans) + 0.8, 5.6), sharex=True,
                             layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for c, ch in enumerate(chans):
        for r, key in enumerate(("sigma", "rho")):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(colors=INK2, labelsize=8, length=2)
            for sp in ax.spines.values():
                sp.set_color("#c3c2b7")
            truth = d["noise_truth"][ch][key]
            prior = d["noise_prior"][ch][key]
            ax.axhline(truth, color=INK, lw=1.0, label="truth" if (r, c) == (0, 0) else None)
            ax.axhline(prior, color=INK2, lw=1.0, ls=(0, (2, 2)), label="prior mean" if (r, c) == (0, 0) else None)
            for s in d["sets"]:
                if ch not in s["channels"]:
                    continue
                th = np.array([b["t_h"] for b in s["batches"]])
                vals = np.array([b["noise"][ch]["σ" if key == "sigma" else "ρ"] for b in s["batches"]])
                sds = np.array([b["noise"][ch]["σ_sd" if key == "sigma" else "ρ_sd"] for b in s["batches"]])
                col = SET_COLOR[s["label"]]
                ax.fill_between(th, vals - sds, vals + sds, color=col, alpha=0.15, lw=0)
                ax.plot(th, vals, color=col, lw=2, label=s["label"] if (r, c) == (0, 0) else None)
            if r == 0:
                ax.set_title(ch, color=INK, fontsize=10)
                ax.set_ylabel(f"sd [{UNITS[ch]}]" if c == 0 else "", color=INK2, fontsize=9)
                if c == 0:
                    ax.set_ylabel("marginal sd (channel units)", color=INK2, fontsize=9)
            else:
                ax.set_xlabel("cook hour", color=INK2, fontsize=9)
                if c == 0:
                    ax.set_ylabel("lag-1 correlation ρ", color=INK2, fontsize=9)
                ax.set_ylim(0, 1)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9,
               frameon=False, labelcolor=INK2)
    fig.suptitle(f"Noise identified by the AR node at every hourly refit, plant: {d['plant']}", color=INK,
                 fontsize=10.5)
    out = REPO / "figures" / "estimator" / ("noise_" + path.stem.replace("_noise_stream", "") + ".png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "mock" / "mock_1d_alt_noise_stream.json"
    plot(p)
