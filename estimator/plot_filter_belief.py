"""The filter's belief through a cook: the pit, centre and surface
temperatures as the filter believes them at every update (mean and
±2 sd), as the instruments' samples arrive, against what the plant did
and what the probes read; the latents that move on the plant (the leak,
the surface coefficients, the add's charge) as the filter follows them,
with the truth in force at each hour; and every channel's noise level
as the filter identifies it, with the spans where a sensor was
distrusted (its level over the threshold, or its reading frozen).

Reads data/mock/filter_drift.json (estimator/mock_filter_drift.jl),
writes figures/estimator/filter_belief_drift.png.

Run from app/:  uv run python ../estimator/plot_filter_belief.py [record.json] [figure stem]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
PIT, CORE, SURF, READ = "#eb6834", "#2a78d6", "#8b5a2b", "#9aa5b4"
EVENT_C = {"t_wrap": "#2a78d6", "t_add": "#eb6834", "t_spritz": "#1baf7a", "t_pull": "#8d5cd6"}
EVENT_L = {"t_wrap": "wrapped", "t_add": "fuel added", "t_spritz": "spritz begins", "t_pull": "pulled"}
CHANNELS = ["pit probe", "core probe", "humidity", "scale"]
CH_C = ["#eb6834", "#2a78d6", "#1baf7a", "#8d5cd6"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    rec_name = sys.argv[1] if len(sys.argv) > 1 else "filter_drift.json"
    stem = sys.argv[2] if len(sys.argv) > 2 else "filter_belief_drift"
    d = json.loads((REPO / "data" / "mock" / rec_name).read_text(encoding="utf-8"))
    th = np.array(d["t_h"])
    t_end = th[-1]
    pl = d["plant"]
    tp = np.array(pl["t_h"])
    keep = tp <= t_end + 1e-9
    names = d["names"]
    moving = [n for n in ("u_leak", "k_e", "h_c", "r_max", "dH_eff", "C_th") if n in names]
    fault_h = float(d.get("fault_h", 0.0) or 0.0)

    fig = plt.figure(figsize=(12.5, 13.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 0.75], hspace=0.38, left=0.08, right=0.98, top=0.9, bottom=0.05)

    # --- 1. the belief of the temperatures, update by update ------------------------------------
    ax = fig.add_subplot(gs[0])
    style(ax)
    for key, col, label in (("pit", PIT, "pit"), ("surface", SURF, "surface"), ("centre", CORE, "centre")):
        mu, sd = np.array(d[key]["mean"]), np.array(d[key]["sd"])
        ax.fill_between(th, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.25, lw=0, label=f"{label}: the belief, ±2 sd")
        ax.plot(th, mu, color=col, lw=1.0, ls=(0, (3, 2)))
    rd = d["readings"]
    for key, col in (("pit", PIT), ("core", CORE)):
        y = np.array([np.nan if v is None else v for v in rd[key]])
        ax.plot(tp[keep], y[keep], ".", color=col, ms=2.0, alpha=0.35, label=f"{key} probe readings")
    for key, col, lab in (("pit", PIT, "pit, the plant"), ("surface", SURF, "surface, the plant"), ("centre", CORE, "centre, the plant")):
        ax.plot(tp[keep], np.array(pl[key])[keep], color=col, lw=1.4, label=lab)
    faulty = np.array(d["faulty"], dtype=bool)
    for c in range(faulty.shape[1]):
        f = faulty[:, c]
        if f.any():
            edges = np.diff(np.concatenate(([0], f.astype(int), [0])))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                ax.axvspan(th[a], th[min(b, len(th) - 1)], color=CH_C[c], alpha=0.10, lw=0)
            ax.plot([], [], color=CH_C[c], lw=6, alpha=0.3, label=f"the {CHANNELS[c]} distrusted")
    above = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (key, col) in enumerate(EVENT_C.items()):
        te = d["events_h"].get(key)
        if te is None or te > t_end:
            continue
        ax.axvline(te, color=col, lw=1.0)
        right = te > 0.9 * t_end
        ax.text(te + (-0.1 if right else 0.1), 1.02 + 0.07 * (i % 3), f"{EVENT_L[key]} {te:.1f} h", color=col, fontsize=8.5, va="bottom",
                ha="right" if right else "left", transform=above)
    if fault_h > 0:
        ax.axvline(fault_h, color="#d6336c", lw=1.0, ls=(0, (4, 2)))
        ax.text(fault_h + 0.1, 1.02 + 0.07 * 2, f"core probe sticks {fault_h:.1f} h", color="#d6336c", fontsize=8.5, va="bottom", transform=above)
    ax.set_xlim(0, t_end)
    top = max(float(np.max(np.array(pl["pit"])[keep])), float(np.max(np.array(d["pit"]["mean"]) + 2 * np.array(d["pit"]["sd"])))) + 8
    ax.set_ylim(15, top + 55)
    ax.set_ylabel("temperature [°C]", color=INK2, fontsize=9)
    ax.set_title(f"The filter's belief of the cook, updated at every one of {len(th)} arrivals of a sample: the pit, surface and centre temperatures, "
                 f"mean and ±2 sd, against the plant", color=INK, fontsize=10.5, pad=86)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=2)

    # --- 2. the moving latents, followed ---------------------------------------------------------
    ax = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    z = np.array(d["z"]); zsd = np.array(d["z_sd"]); zt = np.array(d["z_true"])
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8d5cd6", "#c9a227", "#d6336c"]
    for j, nm in enumerate(moving):
        k = names.index(nm)
        e = z[:, k] - zt[0, k]
        et = zt[:, k] - zt[0, k]
        ax.fill_between(th, e - 2 * zsd[:, k], e + 2 * zsd[:, k], color=cols[j], alpha=0.15, lw=0)
        ax.plot(th, e, color=cols[j], lw=1.2, label=f"{nm}: the belief (±2 sd shaded)")
        ax.step(th, et, where="post", color=cols[j], lw=1.0, ls=(0, (2, 2)))
    ax.axhline(0, color=INK2, lw=0.6)
    lo = min(float(np.min((z[:, names.index(nm)] - zt[0, names.index(nm)]) - 2 * zsd[:, names.index(nm)])) for nm in moving)
    hi = max(float(np.max((z[:, names.index(nm)] - zt[0, names.index(nm)]) + 2 * zsd[:, names.index(nm)])) for nm in moving)
    ax.set_ylim(lo - 0.1, hi + 0.45 * (hi - lo) + 0.1)
    ax.set_ylabel("latent − its value at the start [z]", color=INK2, fontsize=9)
    ax.set_title("The latents that move on the plant: the belief (solid, ±2 sd) and the truth in force (dashed steps), as departures from the start",
                 color=INK, fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=3)

    # --- 3. every channel's noise level, and the sensor the belief stopped trusting -----------------
    ax = fig.add_subplot(gs[2], sharex=ax)
    style(ax)
    lv = np.array(d["level"])
    for c in range(lv.shape[1]):
        ax.plot(th, lv[:, c], color=CH_C[c], lw=1.2, label=f"{CHANNELS[c]}: level")
    ax.axhline(1, color=INK2, lw=0.6)
    ax.axhline(3, color="#d6336c", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.15, 3.1, "distrust threshold: three times the identified noise (a frozen reading is flagged on its own)", color="#d6336c", fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.2, max(10.0, float(np.nanmax(lv)) * 1.5))
    ax.set_ylabel("level (1 = the identified noise)", color=INK2, fontsize=9)
    ax.set_xlabel("cook hour", color=INK2, fontsize=9)
    ax.set_title("Each channel's noise level from the volatility smoother on its innovations, re-estimated at every sample the channel delivers",
                 color=INK, fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=4)

    out = REPO / "figures" / "estimator" / f"{stem}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
