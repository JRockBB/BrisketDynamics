"""The pilot on the real-cook plant, read like a cook: what the pit and
the core did, when the pilot wrapped, added fuel, spritzed, pulled and
served, when it chose to replan and why, and how the outcome landed
against the goals.

Reads data/mock/closed_loop_realcook.json and
data/mock/closed_loop_realcook_trace.json (the plant's history under the
executed schedule, from estimator/closed_loop_trace.jl), writes
figures/estimator/closed_loop_realcook.png.

Run from app/:  uv run python ../estimator/plot_closed_loop_realcook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
PIT, CORE, SURF = "#eb6834", "#2a78d6", "#8b5a2b"
EVENT_C = {"wrap": "#2a78d6", "add": "#eb6834", "spritz": "#1baf7a", "pull": "#8d5cd6", "serve": "#c9a227"}
REASONS = [("first", "the first plan", "#5b6472"), ("event_due", "an event about to commit", "#2a78d6"),
           ("loss_rise", "the plan's value changed", "#eb6834"), ("misfit", "the misfit check", "#d6336c"),
           ("floor", "the floor interval", "#8d5cd6")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    rec_name = sys.argv[1] if len(sys.argv) > 1 else "closed_loop_realcook.json"     # the loop record; the figure takes its suffix
    d = json.loads((REPO / "data" / "mock" / rec_name).read_text(encoding="utf-8"))
    tr = json.loads((REPO / "data" / "mock" / rec_name.replace(".json", "_trace.json")).read_text(encoding="utf-8"))
    stem = rec_name.replace(".json", "")
    suffix = stem.replace("closed_loop_realcook", "") if stem.startswith("closed_loop_realcook") else "_" + stem.replace("closed_loop_", "")
    checks = d["checks"]
    goals, actual = d["goals"], d["actual"]
    ev = tr["events_h"]
    t = np.array(tr["t_h"])
    t_end = ev["serve"] + 0.5
    keep = t <= t_end

    fig = plt.figure(figsize=(12.5, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.0, 0.9], hspace=0.42, left=0.08, right=0.98, top=0.895, bottom=0.06)

    # --- 1. the cook as it happened ----------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    style(ax)
    # the pilot's belief as it held it: at every hourly check the posterior's pit, core and surface temperatures
    # with the covariance carried through the plan physics (closed_loop_trace.jl), each check's belief drawn from
    # that hour to the next check (its forecast over the coming hour), the last check's to the serve; one track
    # per temperature, updated where the checks are
    import matplotlib.patches as mpatches
    band_handles = []
    preds = sorted([pr for pr in tr.get("predictive", []) if pr["t_h"] <= t_end], key=lambda pr: pr["t_h"])
    for j, pr in enumerate(preds):
        g = np.array(pr["grid_h"])
        t_next = preds[j + 1]["t_h"] if j + 1 < len(preds) else t_end
        m = (g >= pr["t_h"] - 1e-9) & (g <= t_next + 1e-9)
        if not m.any():
            continue
        for key, col in (("Tpit", PIT), ("Tcore", CORE), ("Tsurf", SURF)):
            mu, sd = np.array(pr[key]["mean"]), np.array(pr[key]["sd"])
            ax.fill_between(g[m], (mu - 2 * sd)[m], (mu + 2 * sd)[m], color=col, alpha=0.22, lw=0)
            ax.plot(g[m], mu[m], color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.8)
    if preds:
        band_handles.append(mpatches.Patch(color=INK2, alpha=0.35, label="the pilot's belief: mean (dashed) and ±2 sd, updated at each hourly check"))
    ax.plot(t[keep], np.array(tr["T_pit_C"])[keep], color=PIT, lw=1.6, label="pit, the plant")
    ax.plot(t[keep], np.array(tr["T_surf_C"])[keep], color=SURF, lw=1.2, label="surface, the plant")
    ax.plot(t[keep], np.array(tr["T_core_C"])[keep], color=CORE, lw=2.2, label="core, the plant")
    labels = [("wrap", ev["wrap"], "wrapped"), ("add", ev["add"], "fuel added"), ("spritz", ev["spritz"], "spritz begins"),
              ("pull", ev["pull"], "pulled"), ("serve", ev["serve"], "served")]
    ymax = 140
    import matplotlib.transforms as mtransforms
    above = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)   # x in hours, y in axes units
    for i, (key, th, text) in enumerate(labels):
        if th is None:
            continue
        ax.axvline(th, color=EVENT_C[key], lw=1.2)
        right = key == "serve"        # the serve line is at the edge: its label sits to the left
        # the event labels sit above the axes in two rows, where no curve can reach them
        ax.text(th + (-0.15 if right else 0.15), 1.02 + 0.07 * (i % 3), f"{text} {th:.1f} h", color=EVENT_C[key], fontsize=8.5,
                va="bottom", ha="right" if right else "left", transform=above)
    if ev["spritz"] is not None:
        for k in range(1, 12):
            ts = ev["spritz"] + k * ev["spritz_dt"]
            if ts < min(ev["pull"], ev["wrap"] if ev["wrap"] is not None else ev["pull"]):   # no spritz on a wrapped brisket: the plant gates the train by the wrap
                ax.axvline(ts, color=EVENT_C["spritz"], lw=0.6, alpha=0.5)
    ax.set_xlim(0, t_end)
    # headroom above the hottest trace or band for the legend, which sits in the upper left where no event line runs
    top = max(ymax, float(np.max(np.array(tr["T_pit_C"])[keep])) + 8)
    for j, pr in enumerate(preds):
        g = np.array(pr["grid_h"]); t_next = preds[j + 1]["t_h"] if j + 1 < len(preds) else t_end
        m = (g >= pr["t_h"] - 1e-9) & (g <= t_next + 1e-9)
        m.any() and (top := max(top, float(np.max((np.array(pr["Tpit"]["mean"]) + 2 * np.array(pr["Tpit"]["sd"]))[m])) + 4))
    ax.set_ylim(15, top + 60)
    ax.set_ylabel("temperature [°C]", color=INK2, fontsize=9)
    core = np.array(tr["T_core_C"])
    pit_mean = float(np.mean(np.array(tr["T_pit_C"])[t <= ev["pull"]]))
    t70 = t[np.argmax(core >= 70)] if (core >= 70).any() else float("nan")
    ax.set_title(f"The cook as it happened: a {pit_mean * 9 / 5 + 32:.0f} °F pit, the core at 70 °C by {t70:.1f} h, "
                 f"wrapped at {ev['wrap']:.1f} h, pulled at {ev['pull']:.1f} h at {actual['T_pull']:.0f} °C, served at {ev['serve']:.1f} h"
                 + ("\nand the pilot's belief of it: the pit, surface and core under the posterior at each hourly check, carried to the next" if preds else ""),
                 color=INK, fontsize=10.5, pad=60)
    hs, ls_ = ax.get_legend_handles_labels()
    ax.legend(hs + band_handles, ls_ + [h_.get_label() for h_ in band_handles], loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=1,
              borderaxespad=0.4, handlelength=1.6)      # one column, clear of the first spritz line

    # --- 2. what the pilot decided, hour by hour -------------------------------------------------
    ax = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    th = np.array([c["t_h"] for c in checks])
    pull = np.array([c["plan"]["t_pull_h"] for c in checks])
    wrap = np.array([np.nan if c["plan"]["t_wrap_h"] is None else c["plan"]["t_wrap_h"] for c in checks])
    ax.plot(th, th, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(th[-1] + 0.2, th[-1] - 1.5, "now", color=INK2, fontsize=8, ha="left")
    ax.plot(th, pull, color=EVENT_C["pull"], lw=2, label="the planned pull time")
    ax.plot(th, wrap, color=EVENT_C["wrap"], lw=2, label="the planned wrap time")
    k_pull = next(i for i, c in enumerate(checks) if "pull" in c["committed"])
    k_wrap = next(i for i, c in enumerate(checks) if "wrap" in c["committed"])
    ax.plot([th[k_pull]], [pull[k_pull]], "o", color=EVENT_C["pull"], ms=8, mec=SURFACE, mew=1.5)
    ax.plot([th[k_wrap]], [wrap[k_wrap]], "o", color=EVENT_C["wrap"], ms=8, mec=SURFACE, mew=1.5)
    ax.text(th[k_pull] - 0.3, pull[k_pull] + 0.8, "pull committed", color=EVENT_C["pull"], fontsize=8, ha="right")
    ax.text(th[k_wrap] - 0.3, wrap[k_wrap] + 0.8, "wrap committed", color=EVENT_C["wrap"], fontsize=8, ha="right")
    # one marker per hourly check: hollow for a hold, filled and coloured for a replan
    y0 = 1.5
    holds = [c["t_h"] for c in checks if not c["replanned"]]
    ax.plot(holds, [y0] * len(holds), "o", ms=7, mfc=SURFACE, mec=INK2, mew=1.2, ls="none", label="checked, kept the plan")
    for key, label, col in REASONS:
        hs = [c["t_h"] for c in checks if c["replanned"] and c["reason"] == key]
        if hs:
            ax.plot(hs, [y0] * len(hs), "o", ms=7, color=col, mec=SURFACE, mew=1.0, ls="none", label=f"replanned: {label}")
    ax.set_ylim(-0.5, max(34.0, float(np.nanmax(pull)) + 9.0))   # headroom for the legend above the committed-pull label
    ax.set_ylabel("hour of cook", color=INK2, fontsize=9)
    n_re = sum(c["replanned"] for c in checks)
    ax.set_title(f"What the pilot decided at each hourly check: {n_re} replans, {len(checks) - n_re} holds", color=INK, fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2, ncol=2)

    # --- 3. the outcome against the goals, in goal units -----------------------------------------
    ax = fig.add_subplot(gs[2])
    style(ax)
    last = next(c for c in reversed(checks) if c["replanned"])
    pm, ps = last["predicted"]["mean"], last["predicted"]["sd"]
    if preds:                                  # the belief's covariance through the plan physics, at the last replan drawn
        pm, ps = preds[-1]["y"]["mean"], preds[-1]["y"]["sd"]
    # each goal is a band (or a floor): the row's shaded span is the band in the goal's sd, measured from its center
    # (for a floor, from the floor upward); the markers are the prediction and the plant on the same scale
    rows = [("tenderness: collagen at the center at serve", 0, goals["c_target"], goals["c_overdone"], 0.05, actual["C_serve"], "{:.3f}"),
            ("appearance: bark L* at serve", 2, goals["l_lo"], goals["l_hi"], goals["dl_soft"], actual["L"], "{:.1f}"),
            ("taste: smoke deposition at serve", 4, goals["s_lo"], goals["s_hi"], goals["ds_soft"], actual["S"], "{:.2f}")]
    for i, (name, idx, lo, hi, sd, act, fmt) in enumerate(rows):
        y = len(rows) - i
        center = lo if hi is None else 0.5 * (lo + hi)
        span = (0.0, 5.0) if hi is None else (-(hi - lo) / 2 / sd, (hi - lo) / 2 / sd)
        ax.fill_between(span, [y - 0.42, y - 0.42], [y + 0.08, y + 0.08], color="#eeeeea", lw=0)
        ax.axhline(y, color=GRID_C, lw=0.8)
        if ps[idx] is not None and np.isfinite(ps[idx]):
            ax.errorbar([(pm[idx] - center) / sd], [y], xerr=[2 * ps[idx] / sd], fmt="none", ecolor=CORE, elinewidth=1.4, capsize=3, alpha=0.8)
        ax.plot([(pm[idx] - center) / sd], [y], "o", color=CORE, ms=8, mec=SURFACE, mew=1.2)
        ax.plot([(act - center) / sd], [y], "s", color=PIT, ms=8, mec=SURFACE, mew=1.2)
        band = f"at least {lo:g}" if hi is None else f"{lo:g} to {hi:g}"
        ax.text(-4.9, y + 0.12, f"{name}: {band}, softness {sd:g}", color=INK2, fontsize=8.5, va="bottom")
        ax.text(4.9, y + 0.12, f"the plant: {fmt.format(act)}", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.axvline(0, color=INK, lw=1)
    ax.plot([], [], "o", color=CORE, label="predicted at the last replan (±2 sd of the belief)" if preds else "predicted at the last replan")
    ax.plot([], [], "s", color=PIT, label="the plant")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.2, len(rows) + 0.6)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("distance from the band's center, in the band's softness; shaded: inside the band", color=INK2, fontsize=9)
    ax.set_title(f"The outcome under the app's objective: J {actual['J']:.2f} (tenderness {actual['tenderness']:.2f}, appearance {actual['appearance']:.2f}, taste {actual['taste']:.2f}, "
                 f"burn {actual['surf']:.2f}, fuel {actual['fuel']:.2f}); collagen {actual['C_serve']:.2f}, juiciness {actual['juiciness']:.2f}, L* {actual['L']:.0f}", color=INK, fontsize=9.5)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK2, ncol=2)

    out = REPO / "figures" / "estimator" / f"closed_loop_realcook{suffix}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
