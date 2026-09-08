"""The streaming estimator on the real-cook mock: the noise it identified
against the noise the plant generated, the pit's noise level as a
process against the fire regime that produced it, the misfit check per
channel, and the parameter errors, batch by batch.

Reads data/mock/mock_realcook_stream.json and data/mock/mock_realcook.json,
writes figures/estimator_realcook_stream.png.

Run from app/:  uv run python ../estimator/plot_realcook_stream.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
# one fixed hue per channel, never cycled
CHANNELS = [("pit", "pit", "#2a78d6"), ("core", "core", "#eb6834"), ("humidity", "humidity (censored)", "#1baf7a"),
            ("mass", "scale", "#8d5cd6")]
MISFIT_NAMES = {"pit": "pit", "meat probe": "core", "humidity (censored)": "humidity", "scale": "mass"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    s = json.loads((REPO / "data" / "mock" / "mock_realcook_stream.json").read_text(encoding="utf-8"))
    m = json.loads((REPO / "data" / "mock" / "mock_realcook.json").read_text(encoding="utf-8"))
    batches = s["sets"][0]["batches"]
    th = np.array([b["t_h"] for b in batches])
    truth = s["noise_truth"]
    t = np.array(m["t"]) / 3600.0
    nc = m["noise"]
    t_add = m["events"]["t_add"] / 3600.0

    fig = plt.figure(figsize=(12.5, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, left=0.07, right=0.98, top=0.93, bottom=0.08)

    # (a) the pit's noise level: inferred at the last batch against the plant's fire regime
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    level_true = np.where(t > t_add, 1 + nc["pit_add_bump"] * np.exp(-(t - t_add) * 3600 / nc["add_tau"]), 1.0)
    last = batches[-1]
    lv = np.array(last["level"]["pit"], dtype=float)
    mask = np.array(m["masks"]["pit"], dtype=bool) & (t <= last["t_h"] + 1e-9)
    ax.plot(t[t <= last["t_h"]], level_true[t <= last["t_h"]], color=INK2, lw=1.4, ls=(0, (3, 2)), label="the plant's level (fuel add at 12 h)")
    tt, ll = t[mask], lv[mask]
    # draw the inferred level run by run so the lid gaps stay open
    breaks = np.where(np.diff(np.where(mask)[0]) > 1)[0]
    for seg_t, seg_l in zip(np.split(tt, breaks + 1), np.split(ll, breaks + 1)):
        ax.plot(seg_t, seg_l, color="#2a78d6", lw=2)
    ax.plot([], [], color="#2a78d6", lw=2, label="inferred at the 14 h refit")
    ax.set_xlim(0, last["t_h"] + 0.3)
    ax.set_ylim(0.4, 3.4)
    ax.set_xlabel("hour of cook", color=INK2, fontsize=8.5)
    ax.set_ylabel("pit noise level (1 = the base sd)", color=INK2, fontsize=8.5)
    ax.set_title("The pit's noise level as a process", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper left")

    # (b) the identified noise sd per channel relative to the truth, batch by batch
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    ax.axhline(1.0, color=INK2, lw=0.9)
    for key, label, col in CHANNELS:
        ratio = np.array([b["noise"][key]["σ"] / truth[key]["sigma"] for b in batches])
        sd = np.array([b["noise"][key]["σ_sd"] / truth[key]["sigma"] for b in batches])
        ax.fill_between(th, ratio - sd, ratio + sd, color=col, alpha=0.13, lw=0)
        ax.plot(th, ratio, color=col, lw=2, label=label)
    ax.text(th[-1] + 0.4, batches[-1]["noise"]["core"]["σ"] / truth["core"]["sigma"], "at the\ncalibration's floor", color=INK2, fontsize=7.5, va="center")
    ax.set_xlim(th[0] - 0.5, th[-1] + 4.6)
    ax.set_xlabel("refit at hour", color=INK2, fontsize=8.5)
    ax.set_ylabel("identified sd / the plant's sd (band ± 1 sd)", color=INK2, fontsize=8.5)
    ax.set_title("Each channel's noise, identified at its own rate", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right", ncol=2)

    # (c) the misfit check per channel
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    ax.axhspan(-3, 3, color="#eeeeea", lw=0)
    ax.axhline(0, color=INK2, lw=0.9)
    for key, label, col in CHANNELS:
        z = [next(c["z_chi2"] for c in b["misfit"]["channels"] if MISFIT_NAMES[c["name"].replace(", levelled", "")] == key) for b in batches]
        ax.plot(th, z, color=col, lw=2, label=label)
    ax.set_xlabel("refit at hour", color=INK2, fontsize=8.5)
    ax.set_ylabel("residual power χ² z-score", color=INK2, fontsize=8.5)
    ax.set_title("The misfit check per channel (band: ±3)", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=2)

    # (d) the parameter errors in posterior sd
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    err = np.array([np.abs(b["err_sd"]) for b in batches])
    ax.fill_between(th, np.min(err, axis=1), np.max(err, axis=1), color="#2a78d6", alpha=0.13, lw=0, label="all seventeen latents")
    ax.plot(th, np.median(err, axis=1), color="#2a78d6", lw=2, label="median")
    ax.axhline(2.0, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(th[0], 2.05, "2 sd", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("refit at hour", color=INK2, fontsize=8.5)
    ax.set_ylabel("|estimate − truth| / posterior sd", color=INK2, fontsize=8.5)
    ax.set_title("Where the posterior stands against the truth", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    fig.suptitle("The streaming estimator on a plant read like a real cook", color=INK, fontsize=11.5)
    out = REPO / "figures" / "estimator_realcook_stream.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
