"""The innovation autocovariances the bench identifies the state's rates from, per channel and time lag: the
sample (the mean product of the standardized innovations at every pair of updates that far apart, pooled over
the tuning round's cooks, with the standard error from the products' own spread), the filter's own model at the
rates the round ran with (the predicted variance at lag zero, white beyond it) and the model at the rates the
autocovariance least-squares step identified (estimator/python/als_fit.py). A consistent filter's sample sits on
one at lag zero and on zero beyond; what the fit can and cannot explain is the gap that remains after.

Reads the tuning round's records (data/mock/<pattern>) for the pooled statistics and the rates file for the
fit; writes figures/estimator/als_autocovariance<tag>.png.

Run from app/:  uv run python ../estimator/python/plot_als_fit.py [pattern] [tag] [rates file]
"""
from __future__ import annotations

import glob
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from als_fit import lag_table, pool  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
SAMPLE_C, BEFORE_C, AFTER_C = "#1f2430", "#eb6834", "#2a78d6"
CHANNELS = {1: "pit probe", 2: "core probe", 3: "humidity", 4: "scale"}
MIN_PRODUCTS = 5        # a lag bin is drawn when this many products fall in it: fewer carry error bars that dwarf the panel


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "coverage_tune_*.json"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    rates_file = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data" / "calibration" / "state_noise.json"
    files = sorted(glob.glob(str(REPO / "data" / "mock" / pattern)))
    summaries = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        summaries += [c["als"] for c in d["cooks"] if c.get("als")]
    if not summaries:
        raise SystemExit(f"no innovation-autocovariance statistics in {pattern}")
    record = json.loads(rates_file.read_text(encoding="utf-8"))
    fit = record["bench"]["tune"]["als"]
    pooled = pool(summaries)
    n_bins = round(pooled["lag_s"] / pooled["bin_s"])
    rows = lag_table(pooled, fit, CHANNELS, list(range(n_bins + 1)))
    moved = [(g, fit["factors"][i]) for i, g in enumerate(fit["groups"]) if fit["identified"][i] and abs(fit["factors"][i] - 1) > 1e-6]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), facecolor=SURFACE)
    handles = None
    for j, (ax, lab) in enumerate(zip(axes.ravel(), CHANNELS.values())):
        style(ax)
        r = [x for x in rows if x["channel"] == lab and x["n"] >= MIN_PRODUCTS]
        if not r:
            ax.set_title(f"{lab}: fewer than {MIN_PRODUCTS} products at every lag", color=INK, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        lag = np.array([x["lag_s"] for x in r]) / 60; smp = np.array([x["sample"] for x in r]); se = np.array([x["se"] for x in r])
        before = np.array([x["model_before"] for x in r]); after = np.array([x["model_after"] for x in r])
        ax.axhline(0, color=INK2, lw=0.6)
        l1, = ax.plot(lag, before, color=BEFORE_C, lw=2.0, marker="o", ms=4, label="the model at the round's rates")
        l2, = ax.plot(lag, after, color=AFTER_C, lw=2.0, marker="o", ms=4, label="the model at the identified rates")
        l3 = ax.errorbar(lag, smp, yerr=se, fmt="o", ms=5, color=SAMPLE_C, ecolor=SAMPLE_C, elinewidth=1.0, capsize=2, label="the sample ± its standard error")
        handles = handles or [l1, l2, l3]
        n0 = r[0]["n"] if r[0]["bin"] == 0 else 0
        ax.set_title(f"{lab}: lag zero {smp[0]:.2f} ± {se[0]:.2f} ({n0} products); model {before[0]:.2f} before, {after[0]:.2f} after",
                     color=INK, fontsize=9, loc="left")
        ax.annotate("before", (lag[0], before[0]), xytext=(8, 6), textcoords="offset points", color=BEFORE_C, fontsize=8, va="bottom")
        ax.annotate("after", (lag[0], after[0]), xytext=(8, -6), textcoords="offset points", color=AFTER_C, fontsize=8, va="top")
        ax.set_xlim(-0.04 * lag[-1] - 1, lag[-1] * 1.04 + 1)
        if j % 2 == 0:
            ax.set_ylabel("autocovariance of the standardized innovations", color=INK2, fontsize=9)
        if j >= 2:
            ax.set_xlabel("lag between the two updates [min]", color=INK2, fontsize=9)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, labelcolor=INK2, bbox_to_anchor=(0.5, 0.045))
    moved_s = ", ".join(f"{g} ×{f:.2f}" for g, f in moved) if moved else "none"
    held_s = ", ".join(fit["held"]) if fit["held"] else "none"
    fig.suptitle("The innovation autocovariances per channel and lag: the sample against the filter's model before and after the identification\n"
                 f"{fit['n_cooks']} tuning cooks, {fit['n_updates']} updates, lags to {fit['lag_s'] / 3600:.1f} h; the fit's weighted cost {fit['rss_before']:.0f} before, {fit['rss_after']:.0f} after",
                 color=INK, fontsize=10.5)
    fig.text(0.5, 0.012, textwrap.fill(f"The rates moved: {moved_s}. The groups the innovations could not pin, kept at their numbers: {held_s}.", 170),
             ha="center", va="bottom", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    out = REPO / "figures" / "estimator" / f"als_autocovariance{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for x in rows:
        if x["bin"] in (0, 1, 2, 5, 10, 30, 60):
            print(f"  {x['channel']:10s} lag {x['lag_s'] / 60:4.0f} min  n {x['n']:6d}  sample {x['sample']:+.3f} ± {x['se']:.3f}  before {x['model_before']:+.3f}  after {x['model_after']:+.3f}")


if __name__ == "__main__":
    main()
