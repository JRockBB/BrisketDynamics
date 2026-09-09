"""The pilot on the real-cook plant, replanning when it should: what it
planned at every hourly check and where it committed, the standing
plan's expected loss against the loss it was made at (the replan
trigger), the predicted outcomes at each replan against the plant, the
misfit check, and the pit's effective noise sd as the fire regime
moves.

Reads data/mock/closed_loop_realcook.json, writes
figures/estimator_closed_loop_realcook.png.

Run from app/:  uv run python ../estimator/plot_closed_loop_realcook.py
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
EVENTS = [("t_wrap_h", "wrap", "#2a78d6"), ("spritz", "spritz start", "#1baf7a"), ("t_add_h", "fuel add", "#eb6834"),
          ("t_pull_h", "pull", "#8d5cd6"), ("t_serve_h", "serve", "#c9a227")]
# the reasons a replan fired: one fixed hue each
REASONS = [("first", "the first plan", "#5b6472"), ("event_due", "an event about to commit", "#2a78d6"),
           ("loss_rise", "the standing plan's loss rose", "#eb6834"), ("misfit", "the misfit check fired", "#d6336c"),
           ("floor", "the floor interval", "#8d5cd6")]
OUTCOMES = [("core at the pull [°C]", 0, lambda v: v - 273.15), ("retained water", 1, lambda v: v),
            ("bark L*", 2, lambda v: v), ("core at serve [°C]", 3, lambda v: v - 273.15)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    d = json.loads((REPO / "data" / "mock" / "closed_loop_realcook.json").read_text(encoding="utf-8"))
    checks = d["checks"]
    th = np.array([c["t_h"] for c in checks])
    replans = [c for c in checks if c["replanned"]]
    goals, actual, pol = d["goals"], d["actual"], d["policy"]

    fig = plt.figure(figsize=(13, 12.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.0, 1.0], hspace=0.5, wspace=0.34, left=0.06, right=0.985, top=0.95, bottom=0.05)

    # --- what it planned at every check, where it committed, and when it replanned -------------
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.plot(th, th, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(th[len(th) // 2] + 0.4, th[len(th) // 2] - 0.6, "now", color=INK2, fontsize=8, ha="left", va="top")
    committed = []
    for key, label, col in EVENTS:
        vals = np.array([(c["plan"]["spritz"]["t_h"] if c["plan"]["spritz"] else np.nan) if key == "spritz"
                         else (np.nan if c["plan"].get(key) is None else c["plan"][key]) for c in checks], dtype=float)
        ax.plot(th, vals, color=col, lw=2, label=label)
        name = {"t_wrap_h": "wrap", "spritz": "spritz", "t_add_h": "add", "t_pull_h": "pull"}.get(key)
        if name:
            k = next((i for i, c in enumerate(checks) if name in c["committed"]), None)
            if k is not None:
                ax.plot([th[k]], [vals[k]], "o", color=col, ms=8, mec=SURFACE, mew=1.5)
                committed.append((vals[k], f"{label} committed at {vals[k]:.2f} h"))
    # the labels at the end of the held lines, dodged apart where the lines are close, a leader to each
    committed.sort()
    ys = []
    for v, _ in committed:
        ys.append(v if not ys else max(v, ys[-1] + 0.8))
    for (v, text), y in zip(committed, ys):
        ax.annotate(text, (th[-1], v), xytext=(th[-1] + 0.9, y), textcoords="data", fontsize=8, color=INK2,
                    ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=2))
    # a tick at the bottom for every replan, coloured by its reason
    y0 = -1.2
    for key, label, col in REASONS:
        hs = [c["t_h"] for c in replans if c["reason"] == key]
        if hs:
            ax.plot(hs, [y0] * len(hs), "^", color=col, ms=7, mec=SURFACE, mew=0.8, ls="none", label=label)
    ax.set_xlim(th[0] - 0.5, th[-1] + 6.5)
    ax.set_ylim(y0 - 1.0, max(c["plan"]["t_serve_h"] for c in checks) + 1.5)
    ax.set_ylabel("planned event time [h of cook]", color=INK2, fontsize=9)
    ax.set_xlabel("check at hour (triangles: the replans, by reason)", color=INK2, fontsize=9)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.09), fontsize=8, frameon=False, labelcolor=INK2, ncol=2)
    ax.set_title(f"What the pilot planned at every hourly check, where it committed, and the {len(replans)} replans it chose to make",
                 color=INK, fontsize=10.5)

    # --- the predicted outcomes at each replan against the plant --------------------------------
    for j, (title, idx, conv) in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[1, j])
        style(ax)
        rh = np.array([c["t_h"] for c in replans])
        mean = np.array([conv(c["predicted"]["mean"][idx]) for c in replans])
        sd = np.array([c["predicted"]["sd"][idx] for c in replans])
        if idx == 3:
            floor = goals["T_serve_floor"] - 273.15
            ax.axhline(floor, color=INK, lw=1)
            ax.text(rh[0], floor, "  floor", color=INK2, fontsize=8, va="bottom")
        else:
            goal = [goals["T_pull"] - 273.15, goals["W"], goals["L"]][idx]
            gsd = [goals["sd_pull"], goals["sd_W"], goals["sd_L"]][idx]
            ax.axhspan(goal - gsd, goal + gsd, color="#eeeeea", lw=0)
            ax.axhline(goal, color=INK, lw=1)
        ax.fill_between(rh, mean - sd, mean + sd, color="#2a78d6", alpha=0.18, lw=0)
        ax.plot(rh, mean, color="#2a78d6", lw=2)
        act = conv(actual[["T_pull", "W", "L", "T_serve"][idx]] + (273.15 if idx in (0, 3) else 0))
        ax.plot([th[-1]], [act], "s", color="#eb6834", ms=7, mec=SURFACE, mew=1)
        ax.set_title(title, color=INK, fontsize=9.5)
        ax.set_xlabel("replan at hour", color=INK2, fontsize=8)
        if j == 0:
            ax.set_ylabel("predicted over draws (band ± sd); square: the plant", color=INK2, fontsize=8)

    # --- the trigger: the standing plan's loss against the loss it was made at ------------------
    ax = fig.add_subplot(gs[2, :2])
    style(ax)
    L_now = np.array([np.nan if c["L_now"] is None else c["L_now"] for c in checks])
    L_plan = np.array([c["L_plan"] for c in checks])
    ax.step(th, L_plan + pol["loss_rise"], where="post", color="#eb6834", lw=1.2, ls=(0, (3, 2)), label="the standing plan's loss when made, plus the threshold")
    ax.step(th, L_plan, where="post", color=INK2, lw=1.4, label="the standing plan's loss when made")
    ax.plot(th, L_now, color="#2a78d6", lw=2, label="the standing plan's loss under the current posterior")
    y_tri = np.nanmin(np.r_[L_now[1:], L_plan]) * 0.45          # the replan markers along the bottom, clear of the legend
    for key, label, col in REASONS:
        hs = [c["t_h"] for c in replans if c["reason"] == key]
        if hs:
            ax.plot(hs, [y_tri] * len(hs), "^", color=col, ms=7, mec=SURFACE, mew=0.8, ls="none")
    ax.set_yscale("log")
    ax.set_ylim(y_tri * 0.6, np.nanmax(np.r_[L_now[1:], L_plan]) * 8)
    ax.set_xlabel("check at hour", color=INK2, fontsize=8)
    ax.set_ylabel("expected loss (half-squared-sd units)", color=INK2, fontsize=8)
    ax.set_title("The replan trigger: the standing plan's expected loss, then and now", color=INK, fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper right")

    # --- the misfit check ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    style(ax)
    ax.axhspan(-pol["misfit_z"], pol["misfit_z"], color="#eeeeea", lw=0)
    ax.axhline(0, color=INK2, lw=0.9)
    mz = [(c["t_h"], c["misfit_z"]) for c in checks if c["misfit_z"] is not None]
    ax.plot([m[0] for m in mz], [m[1] for m in mz], color="#2a78d6", lw=2)
    ax.set_xlabel("check at hour", color=INK2, fontsize=8)
    ax.set_ylabel("misfit check, total χ² z", color=INK2, fontsize=8)
    ax.set_title("The misfit check (band: the policy's ±3)", color=INK, fontsize=9.5)

    # --- the pit's effective noise sd -------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 3])
    style(ax)
    eff = [(c["t_h"], c["noise"]["pit"]["sigma"] * c["pit_level_now"]) for c in checks if "pit" in c["noise"]]
    base = [(c["t_h"], c["noise"]["pit"]["sigma"]) for c in checks if "pit" in c["noise"]]
    ax.plot([b[0] for b in base], [b[1] for b in base], color=INK2, lw=1.4, ls=(0, (3, 2)), label="base sd, identified")
    ax.plot([e[0] for e in eff], [e[1] for e in eff], color="#2a78d6", lw=2, label="base sd × level now")
    ax.axhline(d["noise_truth"]["pit_sd"], color=INK, lw=0.9)
    ax.text(7.3, d["noise_truth"]["pit_sd"] - 0.02, "the plant's base sd", color=INK2, fontsize=7.5, va="top")
    if d["executed"]["t_add_h"] is not None:
        ax.axvline(d["executed"]["t_add_h"], color="#eb6834", lw=1)
        ax.text(d["executed"]["t_add_h"], ax.get_ylim()[1], "fuel add ", color="#eb6834", fontsize=7.5, va="top", ha="right")
    ax.set_xlabel("check at hour", color=INK2, fontsize=8)
    ax.set_ylabel("pit noise sd [°C]", color=INK2, fontsize=8)
    ax.set_title("The pit's noise level follows the fire", color=INK, fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper left")

    out = REPO / "figures" / "estimator_closed_loop_realcook.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
