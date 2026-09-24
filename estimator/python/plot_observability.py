"""What the innovations see: the block's finite-horizon observability on the real-cook mock, from
estimator/scripts/observability.jl. Left, the singular values of the observability matrix over the horizon
(how many sds of innovation each direction of the initial belief moves), with the count of directions above
one sd, a tenth and a hundredth. Right, per coordinate, the innovations' response to a one-sd move of that
coordinate alone: the latents by name, each state group by its most responsive node, each channel's
discrepancy; a group the innovations do not see at all sits at the axis floor, marked.

Reads scratchpad/observability.json (or the file given), writes figures/estimator/observability<tag>.png.

Run from app/:  uv run python ../estimator/python/plot_observability.py [observability.json] [tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
LATENT_C, STATE_C, DISC_C = "#2a78d6", "#eb6834", "#1f2430"
GROUP_LABEL = {"T": "temperature field", "W": "water field", "C": "collagen field", "F": "fat field", "S": "salt field", "bark": "bark and ring",
               "T_pit": "pit temperature", "m_fuel": "fuel mass", "m_pan": "pan mass", "T_pan": "pan temperature", "T_fb": "firebox temperature", "ignited": "ignited fraction"}
CHANNELS = ["pit probe", "core probe", "humidity", "scale"]
FLOOR = 1e-3


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "scratchpad" / "observability.json"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    d = json.loads(src.read_text(encoding="utf-8"))
    S = np.array(d["singular_values"], dtype=float)
    hours = d["hours"]; n = d["n"]; n_up = d["n_updates"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 8.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.35]})
    # left: the spectrum
    style(ax1)
    idx = np.arange(1, len(S) + 1)
    ax1.semilogy(idx, np.maximum(S, FLOOR / 10), color=LATENT_C, lw=2.0)
    ax1.scatter(idx, np.maximum(S, FLOOR / 10), color=LATENT_C, s=12, zorder=3)
    for thr, lab in ((1.0, "one sd"), (0.1, "a tenth"), (0.01, "a hundredth")):
        k = int(np.sum(S >= thr))
        ax1.axhline(thr, color=INK2, lw=0.6, ls="--")
        ax1.text(len(S) * 0.99, thr * 1.25, f"{k} directions above {lab}", ha="right", va="bottom", color=INK2, fontsize=8)
    ax1.set_ylim(FLOOR / 10, max(S) * 3)
    ax1.set_xlim(0, len(S) + 1)
    ax1.set_xlabel(f"direction of the initial belief, ordered ({len(S)} of {n} can be nonzero: one per reading)", color=INK2, fontsize=9)
    ax1.set_ylabel(f"innovation moved over {hours:g} h by a one-sd move along the direction [sd]", color=INK2, fontsize=9)
    ax1.set_title(f"the observability matrix's singular values: {int(np.sum(S >= FLOOR))} directions move the innovations at all", color=INK, fontsize=9.5, loc="left")
    # right: per coordinate
    style(ax2)
    rows = []
    lat = d["weakest"]["latents"]
    for name in sorted(lat, key=lambda k: -lat[k]):
        rows.append((name, lat[name], LATENT_C))
    grp = d["weakest"]["groups"]
    for g in ["T_pit", "T", "W", "F", "m_fuel", "C", "S", "bark", "m_pan", "T_pan", "T_fb", "ignited"]:
        rows.append((GROUP_LABEL[g], grp[g]["max"], STATE_C))
    for c, v in enumerate(d["weakest"]["discrepancies"]):
        rows.append((f"{CHANNELS[c]} discrepancy", v, DISC_C))
    y = np.arange(len(rows))[::-1]
    seen = np.array([r[1] >= FLOOR for r in rows])
    vals = np.array([r[1] if r[1] >= FLOOR else 0.0 for r in rows])                 # an unseen group draws no bar
    ax2.barh(y, vals, color=[r[2] for r in rows], height=0.7, left=FLOOR / 10)
    ax2.set_xscale("log")
    ax2.set_xlim(FLOOR / 10, 500)
    ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    for yi, v, s in zip(y, vals, seen):
        ax2.text(v * 1.15 if s else FLOOR / 8, yi, f"{v:.3g}" if s else "not seen in 2 h", va="center", ha="left", color=INK2, fontsize=7.5)
    ax2.axvline(1.0, color=INK2, lw=0.6, ls="--")
    ax2.set_xlabel(f"innovation moved over {hours:g} h by a one-sd move of the coordinate alone [sd]", color=INK2, fontsize=9)
    ax2.set_title("per coordinate: latents (blue), state groups by most responsive node (orange), discrepancies (dark)", color=INK, fontsize=8.5, loc="left")
    cp = d["core_pair"]; pp = d["pit_pair"]
    fig.suptitle(f"What the innovations see: the filter's block on the first {hours:g} h of the real-cook mock ({n_up} updates, {n} coordinates)\n"
                 f"the core's field node against its discrepancy: {cp['responses'][0]:.2f} and {cp['responses'][1]:.1f} sd, {cp['angle_deg']:.0f} degrees apart, the pair's second singular value {cp['singular'][1]:.2f}\n"
                 f"the pit against its discrepancy: {pp['responses'][0]:.1f} and {pp['responses'][1]:.1f} sd, {pp['angle_deg']:.0f} degrees apart, the second singular value {pp['singular'][1]:.1f}",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = REPO / "figures" / "estimator" / f"observability{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
