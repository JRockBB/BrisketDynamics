"""The closed loop on the mock plant: what the pilot planned at every
replan and what it committed.

Top: the planned event times (wrap, spritz start, fuel add, pull, serve)
against the replan time, each line ending where the event commits (the
marker), with the diagonal "now" for reference. Middle: the predicted
outcomes over the posterior draws (mean ± sd) against their goals, and
the plant's actual outcome at the end. Bottom: the posterior widths of
the thickness, the surface transfer and the area, and the misfit z.

Run from app/:  uv run python ../estimator/plot_closed_loop.py
Reads data/mock/closed_loop.json, writes figures/estimator_closed_loop.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

REPO = Path(__file__).resolve().parents[1]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
EVENTS = [("t_wrap_h", "wrap", "#2a78d6"), ("spritz", "spritz start", "#1baf7a"), ("t_add_h", "fuel add", "#eb6834"),
          ("t_pull_h", "pull", "#8c4bd6"), ("t_serve_h", "serve", "#c9a227")]
OUT = [("core at the pull [°C]", 0, 273.15, "T_pull", "sd_pull", "T_pull"),
       ("retained water", 1, 0.0, "W", "sd_W", "W"),
       ("bark L*", 2, 0.0, "L", "sd_L", "L"),
       ("core at serve [°C]", 3, 273.15, "T_serve_floor", None, "T_serve")]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(colors=INK2, labelsize=8, length=2)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")


def main():
    d = json.loads((REPO / "data" / "mock" / "closed_loop.json").read_text())
    reps = d["replans"]
    th = np.array([r["t_h"] for r in reps])
    fig = plt.figure(figsize=(12.5, 9.2), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1.0, 0.9])
    # --- the plan
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.plot(th, th, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(th[len(th) // 2] + 0.4, th[len(th) // 2] - 0.6, "now", color=INK2, fontsize=8, ha="left", va="top")
    stacked = {}          # labels of events committed at the same replan stack downward
    for key, label, col in EVENTS:
        vals = []
        for r in reps:
            p = r["plan"]
            v = p["spritz"]["t_h"] if key == "spritz" and p["spritz"] else (p.get(key) if key != "spritz" else np.nan)
            vals.append(np.nan if v is None else v)
        vals = np.array(vals, dtype=float)
        ax.plot(th, vals, color=col, lw=2, label=label)
        name = {"t_wrap_h": "wrap", "spritz": "spritz", "t_add_h": "add", "t_pull_h": "pull"}.get(key)
        if name:
            k = next((i for i, r in enumerate(reps) if name in r["committed"]), None)
            if k is not None:
                ax.plot([th[k]], [vals[k]], "o", color=col, ms=8, mec=SURFACE, mew=1.5)
                # the label at the end of the held line, past the last replan, where nothing else is drawn;
                # two events committed at the same hour stack above and below their (coincident) lines
                row = stacked.get(round(vals[k], 1), 0)
                stacked[round(vals[k], 1)] = row + 1
                ax.annotate(f"{label} committed at {vals[k]:.2f} h", (th[-1], vals[k]), textcoords="offset points",
                            xytext=(6, 3 if row == 0 else -3), fontsize=8, color=INK2, ha="left",
                            va="bottom" if row == 0 else "top")
    ax.set_xlim(th[0] - 0.5, th[-1] + 6.5)
    ax.set_ylabel("planned event time [h of cook]", color=INK2, fontsize=9)
    ax.set_xlabel("replan at hour", color=INK2, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=5)
    ax.set_title("What the pilot planned at every replan, and where it committed", color=INK, fontsize=10.5)
    # --- the predicted outcomes
    g = d["goals"]; actual = d["actual"]
    for c, (name, k, off, gkey, sdkey, akey) in enumerate(OUT):
        ax = fig.add_subplot(gs[1, c])
        style(ax)
        mean = np.array([r["predicted"]["mean"][k] for r in reps]) - off
        sd = np.array([r["predicted"]["sd"][k] for r in reps])
        ax.fill_between(th, mean - sd, mean + sd, color="#2a78d6", alpha=0.15, lw=0)
        ax.plot(th, mean, color="#2a78d6", lw=2)
        goal = g[gkey] - off
        if sdkey:
            ax.axhspan(goal - g[sdkey], goal + g[sdkey], color="#f0efec", zorder=0)
            ax.axhline(goal, color=INK, lw=0.8)
        else:
            ax.axhline(goal, color=INK, lw=1.2)
            ax.text(th[0], goal, "  floor", color=INK2, fontsize=8, va="bottom")
        ax.plot([th[-1]], [actual[akey]], "s", color="#eb6834", ms=7, mec=SURFACE, mew=1.2)
        ax.set_title(name, color=INK, fontsize=9.5)
        ax.set_xlabel("replan at hour", color=INK2, fontsize=8)
        if c == 0:
            ax.set_ylabel("predicted over draws (band ± sd); square: the plant", color=INK2, fontsize=8)
    # --- the posterior and the misfit
    names = d["names"]
    idx = {n: i for i, n in enumerate(names)}
    ax = fig.add_subplot(gs[2, :2])
    style(ax)
    for nm, col in (("L", "#2a78d6"), ("h_c", "#eb6834"), ("A_meat", "#1baf7a")):
        ax.plot(th, [r["posterior_sd"][idx[nm]] for r in reps], color=col, lw=2, label=nm)
    ax.set_yscale("log")
    ax.set_yticks([0.06, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0.06", "0.1", "0.2", "0.4"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("posterior sd (unconstrained)", color=INK2, fontsize=8)
    ax.set_xlabel("replan at hour", color=INK2, fontsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=3)
    ax = fig.add_subplot(gs[2, 2:])
    style(ax)
    mz = np.array([np.nan if r["misfit_z"] is None else r["misfit_z"] for r in reps])
    ax.axhspan(-3, 3, color="#f0efec", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(th, mz, color="#2a78d6", lw=2)
    ax.set_ylim(-6, 6)
    ax.set_ylabel("misfit check, total χ² z", color=INK2, fontsize=8)
    ax.set_xlabel("replan at hour", color=INK2, fontsize=8)
    out = REPO / "figures" / "estimator_closed_loop.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
