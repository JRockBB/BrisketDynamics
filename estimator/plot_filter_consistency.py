"""The filter's consistency tests through the cook, pooled over the
drifting cooks of estimator/coverage_filter.jl: the normalized
innovation squared per slice against its chi-squared bands (the
innovations at the predicted mean, before the update, whitened by the
predicted innovation covariance), the normalized estimation error
squared of the latents (16 degrees of freedom, a0 excluded) and of the
pit, centre and surface jointly (3) against the truth, and per channel
the standardized innovations' mean and lag-one autocorrelation with
their bands. A consistent filter sits inside the bands; too-narrow
widths push NIS and NEES above them.

Reads data/mock/coverage_filter_*.json, writes
figures/estimator/filter_consistency_drift.png and prints the numbers.

Run from app/:  uv run python ../estimator/plot_filter_consistency.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
COOK_C, MEAN_C, BAND = "#9aa5b4", "#2a78d6", "#efefec"
CHANNELS = {"1": "pit probe", "2": "core probe", "3": "humidity", "4": "scale"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    files = sorted(glob.glob(str(REPO / "data" / "mock" / "coverage_filter_*.json")))
    cooks = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        cooks += [c for c in d["cooks"] if c["temps"] and "nis" in c["temps"][0]]
    if not cooks:
        raise SystemExit("no coverage records with diagnostics")
    n = len(cooks)
    th = np.array([r["t_h"] for r in cooks[0]["temps"]])
    m = len(th)
    nis = np.full((n, m), np.nan); dof = np.zeros((n, m)); nz = np.full((n, m), np.nan); nT = np.full((n, m), np.nan)
    innov = {k: [] for k in CHANNELS}
    for i, c in enumerate(cooks):
        for j, r in enumerate(c["temps"][:m]):
            nis[i, j] = r["nis"] if r["dof"] > 0 else np.nan
            dof[i, j] = r["dof"]
            nz[i, j] = r["nees_z16"]; nT[i, j] = r["nees_T"]
        for k in CHANNELS:
            seq = [v for r in c["temps"] for v in r["innov"].get(k, [])]
            innov[k].append(np.array(seq))
    # the pooled NIS per slice: the sum over cooks is chi-squared with the summed dof
    nis_sum = np.nansum(nis, axis=0); dof_sum = np.sum(dof * np.isfinite(nis), axis=0)
    nis_ratio = np.where(dof_sum > 0, nis_sum / np.maximum(dof_sum, 1), np.nan)
    lo_nis = stats.chi2.ppf(0.025, dof_sum) / np.maximum(dof_sum, 1); hi_nis = stats.chi2.ppf(0.975, dof_sum) / np.maximum(dof_sum, 1)
    # the pooled NEES per slice: the mean over cooks of chi-squared(k) has bands chi2(n k) / (n k)
    def band(k):
        return stats.chi2.ppf(0.025, n * k) / (n * k), stats.chi2.ppf(0.975, n * k) / (n * k)
    nz_mean = np.nanmean(nz, axis=0) / 16; nT_mean = np.nanmean(nT, axis=0) / 3
    lo16, hi16 = band(16); lo3, hi3 = band(3)
    # whiteness per channel: the standardized innovations' mean and lag-one autocorrelation, pooled over cooks
    rows = []
    for k, lab in CHANNELS.items():
        seqs = [s for s in innov[k] if len(s) > 2]
        allv = np.concatenate(seqs) if seqs else np.array([])
        r1 = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in seqs if len(s) > 3]
        rows.append((lab, len(allv), float(np.mean(allv)) if len(allv) else np.nan, float(np.std(allv, ddof=1)) if len(allv) > 1 else np.nan,
                     float(np.mean(r1)) if r1 else np.nan, 2 / np.sqrt(np.mean([len(s) for s in seqs])) if seqs else np.nan))
    print(f"{n} cooks, {m} slices each; the filter's consistency at every slice, pooled:")
    inside_nis = np.mean((nis_ratio >= lo_nis) & (nis_ratio <= hi_nis))
    print(f"  NIS/dof: mean over slices {np.nanmean(nis_ratio):.2f}; inside the 95 % band on {100 * inside_nis:.0f} % of slices (a consistent filter: 95 %)")
    print(f"  NEES/16 of the latents: mean {np.nanmean(nz_mean):.2f} (band {lo16:.2f} to {hi16:.2f}); NEES/3 of the temperatures: mean {np.nanmean(nT_mean):.2f} (band {lo3:.2f} to {hi3:.2f})")
    print(f"  {'channel':12s} {'samples':>8s} {'mean':>7s} {'sd':>6s} {'lag-1 r':>8s} {'band':>6s}")
    for lab, cnt, mu, sd, r1, b in rows:
        print(f"  {lab:12s} {cnt:8d} {mu:7.2f} {sd:6.2f} {r1:8.2f} {b:6.2f}")

    fig = plt.figure(figsize=(12.5, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.8], hspace=0.45, left=0.08, right=0.98, top=0.93, bottom=0.07)
    ax = fig.add_subplot(gs[0]); style(ax)
    for i in range(n):
        ax.plot(th, nis[i] / np.maximum(dof[i], 1), color=COOK_C, lw=0.6, alpha=0.6)
    ax.fill_between(th, lo_nis, hi_nis, color=BAND, lw=0, label="the 95 % band of a consistent filter (pooled)")
    ax.plot(th, nis_ratio, color=MEAN_C, lw=2.0, label="pooled over the cooks")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_ylim(0, min(12, float(np.nanmax(nis_ratio)) * 1.6 + 0.5))
    ax.set_ylabel("NIS / dof", color=INK2, fontsize=9)
    ax.set_title(f"The normalized innovation squared per slice: every sample's residual at the predicted mean, whitened by the predicted innovation covariance; "
                 f"{n} cooks (grey) and the pool (blue); mean {np.nanmean(nis_ratio):.2f}, inside the band on {100 * inside_nis:.0f} % of slices", color=INK, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    ax = fig.add_subplot(gs[1], sharex=ax); style(ax)
    ax.fill_between(th, lo16, hi16, color=BAND, lw=0)
    ax.plot(th, nz_mean, color=MEAN_C, lw=2.0, label=f"the latents (16 dof, a0 excluded): mean {np.nanmean(nz_mean):.2f}")
    ax.plot(th, nT_mean, color="#eb6834", lw=2.0, label=f"the pit, centre and surface jointly (3 dof): mean {np.nanmean(nT_mean):.2f}")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.set_ylim(0, min(30, max(float(np.nanmax(nz_mean)), float(np.nanmax(nT_mean))) * 1.3 + 0.5))
    ax.set_ylabel("NEES / dof", color=INK2, fontsize=9)
    ax.set_title("The normalized estimation error squared against the truth in force, pooled over the cooks (a consistent filter sits in the grey band)", color=INK, fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2)
    ax = fig.add_subplot(gs[2]); style(ax)
    labs = [r[0] for r in rows]; x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[2] for r in rows], width=0.34, color=MEAN_C, label="mean of the standardized innovations")
    ax.bar(x + 0.18, [r[4] for r in rows], width=0.34, color="#eb6834", label="lag-one autocorrelation")
    for xi, r in zip(x, rows):
        ax.plot([xi + 0.18, xi + 0.18], [-r[5], r[5]], color=INK2, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labs, color=INK2)
    ax.set_ylim(-1, 1)
    ax.set_title("Whiteness per channel: the standardized innovations' mean and lag-one autocorrelation (the bar on the autocorrelation is ±2/√N)", color=INK, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_xlabel("cook hour (panels above)", color=INK2, fontsize=9)
    out = REPO / "figures" / "estimator" / "filter_consistency_drift.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
