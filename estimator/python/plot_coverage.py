"""The coverage test of the streaming estimator on the honest bench: many
mock cooks with truths drawn from the pilot's prior, each streamed with
its own noise draw (estimator/scripts/coverage_realcook.jl), the standardized
errors (estimate - truth) / sd pooled per site and per hour. A calibrated
posterior has them at mean zero and spread one at every hour; the figure
shows every cook's curve, the pooled mean with its standard error, and
the +-2 band; the table gives the pooled numbers at the last batch.

Reads data/mock/coverage_realcook*.json (every worker's file), writes
figures/estimator/coverage_realcook.png and data/mock/coverage_summary.json.

Run from app/:  uv run python ../estimator/python/plot_coverage.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
COOK_C, MEAN_C, BAND = "#9aa5b4", "#2a78d6", "#efefec"


def main():
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "coverage_realcook*.json"     # a round's manifest (bench_manifest_fresh.json) or a pattern, e.g. coverage_ridge*.json
    key = sys.argv[2] if len(sys.argv) > 2 else "err_sd"                          # err_sd (the Gaussian) or err_sd_ridge
    tag = sys.argv[3] if len(sys.argv) > 3 else ""                                # suffix of the figure and summary
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_manifest import resolve_inputs
    files, _ = resolve_inputs(pattern)
    cooks, names, prior_file, edges, drift_flag = [], None, None, None, None
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        names = d["names"]; prior_file = d["prior_file"]; edges = d["edges_h"]
        cooks += d["cooks"]
        dr = d.get("drift")
        if isinstance(dr, dict) and "on" in dr:                                    # the factor-graph backend's records say whether their plant drifted
            drift_flag = bool(dr["on"])
    n = len(cooks)
    if n == 0:
        raise SystemExit("no coverage records")
    hours = np.array(edges)
    nb = len(hours)
    # err[cook, batch, site]
    err = np.full((n, nb, len(names)), np.nan)
    for i, c in enumerate(cooks):
        for j, b in enumerate(c["batches"][:nb]):
            err[i, j, :] = b[key]
    ok = np.isfinite(err).all(axis=(1, 2))
    err = err[ok]; n_ok = int(ok.sum())

    mean = np.nanmean(err, axis=0); sd = np.nanstd(err, axis=0, ddof=1); se = sd / np.sqrt(n_ok)
    cover2 = np.mean(np.abs(err) <= 2, axis=0); cover1 = np.mean(np.abs(err) <= 1, axis=0)
    last = -1
    summary = {"n_cooks": n_ok, "n_drawn": n, "prior_file": prior_file, "hours": edges, "sites": {}}
    print(f"{n_ok} cooks (of {n} drawn) on prior {prior_file}; standardized errors at the last batch ({hours[last]:.0f} h):")
    print(f"{'site':12s} {'mean':>7s} {'z of mean':>10s} {'spread':>7s} {'within 1':>9s} {'within 2':>9s}")
    for k, nm in enumerate(names):
        z = mean[last, k] / se[last, k]
        summary["sites"][nm] = {"mean": float(mean[last, k]), "z_mean": float(z), "spread": float(sd[last, k]),
                                "cover1": float(cover1[last, k]), "cover2": float(cover2[last, k]),
                                "mean_by_hour": mean[:, k].tolist(), "spread_by_hour": sd[:, k].tolist()}
        print(f"{nm:12s} {mean[last, k]:7.2f} {z:10.2f} {sd[last, k]:7.2f} {cover1[last, k]:9.2f} {cover2[last, k]:9.2f}")
    pooled = err[:, last, :].ravel()
    summary["pooled_last"] = {"mean": float(np.mean(pooled)), "spread": float(np.std(pooled, ddof=1)),
                              "cover1": float(np.mean(np.abs(pooled) <= 1)), "cover2": float(np.mean(np.abs(pooled) <= 2))}
    print(f"{'all sites':12s} {np.mean(pooled):7.2f} {'':>10s} {np.std(pooled, ddof=1):7.2f} {np.mean(np.abs(pooled) <= 1):9.2f} {np.mean(np.abs(pooled) <= 2):9.2f}   (a calibrated posterior: 0, 1, 0.68, 0.95)")
    summary["key"] = key
    (REPO / "data" / "mock" / f"coverage_summary{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ncol = 6; nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.3 * nrow + 0.9), facecolor=SURFACE, sharex=True, sharey=True)
    for k, nm in enumerate(names):
        ax = axes.flat[k]
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.axhspan(-2, 2, color=BAND, zorder=0)
        ax.axhline(0, color=INK2, lw=0.6)
        for i in range(n_ok):
            ax.plot(hours, err[i, :, k], color=COOK_C, lw=0.6, alpha=0.6)
        ax.fill_between(hours, mean[:, k] - 2 * se[:, k], mean[:, k] + 2 * se[:, k], color=MEAN_C, alpha=0.18, lw=0)
        ax.plot(hours, mean[:, k], color=MEAN_C, lw=2.0)
        ax.set_title(f"{nm}: mean {mean[last, k]:+.2f}, spread {sd[last, k]:.2f}", fontsize=8.5, color=INK, loc="left")
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.grid(True, color=GRID_C, lw=0.5)
    for ax in axes.flat[len(names):]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("cook hour", color=INK2, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("(estimate − truth) / sd", color=INK2, fontsize=8)
    axes.flat[0].set_ylim(-5, 5)
    drifting = drift_flag if drift_flag is not None else ("drift" in pattern or key == "err_sd_graphfilter")
    what = {"err_sd": "the batch estimator (every latent a constant)" if drifting else "the Gaussian posterior",
            "err_sd_graphfilter": "the filter on the graph (the belief over the plant's state and the latents, updated at every arrival of a sample)",
            "err_sd_system": "the factor-graph backend (the square-root fixed-lag smoother over the system's state and the latents, updated at every arrival)",
            "err_sd_ridge": "the belief with the ridge chain",
            "err_sd_marg": "the Gaussian posterior with the noise marginalised",
            "err_sd_filter": "the filter (predict from a handed-off state, update on the window)",
            "err_sd_smoother": "the smoother (the moving sites piecewise constant over the history)"}.get(key, key)
    bench = "on the DRIFTING plant (the leak, the surface coefficients and the add's charge move; the error is against the truth in force at that hour)" if drifting else "on the honest bench"
    fig.suptitle(f"Coverage of {what} {bench}: {n_ok} cooks with truths drawn from the pilot's prior, "
                 f"each cook's standardized error (grey), the pooled mean ± 2 standard errors (blue), the ±2 band; "
                 f"pooled at {hours[last]:.0f} h: mean {np.mean(pooled):+.2f}, spread {np.std(pooled, ddof=1):.2f}, within 2 sd {100 * np.mean(np.abs(pooled) <= 2):.0f} %",
                 fontsize=9.5, color=INK, x=0.01, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO / "figures" / "estimator" / f"coverage_realcook{tag}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
