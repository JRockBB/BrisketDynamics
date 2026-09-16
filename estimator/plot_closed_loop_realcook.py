"""The pilot on the real-cook plant, read like a cook: what the pit, the
surface and the core did and what the filter believed of them at every
update (one band per temperature, mean and ±2 sd, updated as the samples
arrived), the probes' readings, the instrument the belief stopped
trusting, when the pilot wrapped, added fuel, spritzed, pulled and
served, when it chose to replan and why, and how the outcome landed
against the goals.

Reads data/mock/<record>.json (estimator/mock_closed_loop_realcook.jl),
writes figures/estimator/closed_loop_realcook<suffix>.png.

Run from app/:  uv run python ../estimator/plot_closed_loop_realcook.py [record.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
PIT, CORE, SURF, READ = "#eb6834", "#2a78d6", "#8b5a2b", "#9aa5b4"
EVENT_C = {"wrap": "#2a78d6", "add": "#eb6834", "spritz": "#1baf7a", "pull": "#8d5cd6", "serve": "#c9a227"}
REASONS = [("first", "the first plan", "#5b6472"), ("event_due", "an event about to commit", "#2a78d6"),
           ("loss_rise", "the plan's value fell", "#eb6834"), ("loss_high", "the plan's value low outright", "#c9a227"),
           ("misfit", "the misfit check", "#d6336c")]
CHANNELS = ["pit", "core", "humidity", "mass"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    rec_name = sys.argv[1] if len(sys.argv) > 1 else "closed_loop_realcook.json"
    d = json.loads((REPO / "data" / "mock" / rec_name).read_text(encoding="utf-8"))
    stem = rec_name.replace(".json", "")
    suffix = stem.replace("closed_loop_realcook", "") if stem.startswith("closed_loop_realcook") else "_" + stem.replace("closed_loop_", "")
    rows = d["checks"]
    goals, actual, ex, pl = d["goals"], d["actual"], d["executed"], d["plant"]
    ev = {"wrap": ex["t_wrap_h"], "pull": ex["t_pull_h"], "serve": ex["t_serve_h"], "add": ex["t_add_h"],
          "spritz": None if ex["spritz"] is None else ex["spritz"]["t_h"], "spritz_dt": None if ex["spritz"] is None else ex["spritz"]["dt_h"]}
    th = np.array([r["t_h"] for r in rows])
    t_end = ev["serve"] + 0.5
    tp = np.array(pl["t_h"])
    keep = tp <= t_end

    fig = plt.figure(figsize=(12.5, 11.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1.0, 0.9], hspace=0.42, left=0.08, right=0.98, top=0.9, bottom=0.06)

    # --- 1. the cook as it happened, and the filter's belief of it at every update -----------------------
    ax = fig.add_subplot(gs[0])
    style(ax)
    for key, col, label in (("T_pit", PIT, "pit"), ("T_surface", SURF, "surface"), ("T_centre", CORE, "centre")):
        mu = np.array([r[key][0] for r in rows]); sd = np.array([r[key][1] for r in rows])
        ax.fill_between(th, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.25, lw=0)
        ax.plot(th, mu, color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.85)
    for key, col in (("pit", PIT), ("core", CORE)):
        y = np.array([np.nan if r["readings"][key] is None else r["readings"][key] for r in rows])
        ax.plot(th, y, ".", color=col, ms=2.0, alpha=0.35)
    ax.plot(tp[keep], np.array(pl["pit"])[keep], color=PIT, lw=1.4, label="pit, the plant")
    ax.plot(tp[keep], np.array(pl["surface"])[keep], color=SURF, lw=1.2, label="surface, the plant")
    ax.plot(tp[keep], np.array(pl["centre"])[keep], color=CORE, lw=2.0, label="centre, the plant")
    # the instrument the belief stopped trusting: the spans where a channel's level stayed high
    faulty = np.array([r["faulty"] for r in rows], dtype=bool)
    for c, name in enumerate(CHANNELS):
        f = faulty[:, c]
        if f.any():
            on = np.flatnonzero(np.diff(np.concatenate(([0], f.astype(int), [0]))) == 1)
            off = np.flatnonzero(np.diff(np.concatenate(([0], f.astype(int), [0]))) == -1)
            for a, b in zip(on, off):
                ax.axvspan(th[a], th[min(b, len(th) - 1)], color="#d6336c", alpha=0.08, lw=0)
            ax.plot([], [], color="#d6336c", lw=6, alpha=0.3, label=f"the {name} sensor distrusted (its level high)")
    labels = [("wrap", ev["wrap"], "wrapped"), ("add", ev["add"], "fuel added"), ("spritz", ev["spritz"], "spritz begins"),
              ("pull", ev["pull"], "pulled"), ("serve", ev["serve"], "served")]
    above = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (key, t_ev, text) in enumerate(labels):
        if t_ev is None:
            continue
        ax.axvline(t_ev, color=EVENT_C[key], lw=1.2)
        right = key == "serve"
        ax.text(t_ev + (-0.15 if right else 0.15), 1.02 + 0.07 * (i % 3), f"{text} {t_ev:.1f} h", color=EVENT_C[key], fontsize=8.5,
                va="bottom", ha="right" if right else "left", transform=above)
    if ev["spritz"] is not None:
        for k in range(1, 12):
            ts = ev["spritz"] + k * ev["spritz_dt"]
            if ts < min(ev["pull"], ev["wrap"] if ev["wrap"] is not None else ev["pull"]):   # no spritz on a wrapped brisket
                ax.axvline(ts, color=EVENT_C["spritz"], lw=0.6, alpha=0.5)
    ax.set_xlim(0, t_end)
    top = max(float(np.max(np.array(pl["pit"])[keep])), float(np.max(np.array([r["T_pit"][0] + 2 * r["T_pit"][1] for r in rows])))) + 8
    ax.set_ylim(15, top + 60)
    ax.set_ylabel("temperature [°C]", color=INK2, fontsize=9)
    core = np.array(pl["centre"])
    pit_mean = float(np.mean(np.array(pl["pit"])[tp <= ev["pull"]]))
    t70 = tp[np.argmax(core >= 70)] if (core >= 70).any() else float("nan")
    n_up = sum(r["updated"] for r in rows)
    ax.set_title(f"The cook as it happened: a {pit_mean * 9 / 5 + 32:.0f} °F pit, the core at 70 °C by {t70:.1f} h, "
                 f"wrapped at {ev['wrap']:.1f} h, pulled at {ev['pull']:.1f} h at {actual['T_pull']:.0f} °C, served at {ev['serve']:.1f} h\n"
                 f"and the filter's belief of it: the pit, surface and centre (mean dashed, ±2 sd shaded) at every one of {n_up} updates, as the samples arrived (dots)",
                 color=INK, fontsize=10.5, pad=60)
    hs, ls_ = ax.get_legend_handles_labels()
    hs.append(mpatches.Patch(color=INK2, alpha=0.3, label="the filter's belief, ±2 sd (one band per temperature)"))
    ax.legend(hs, ls_ + [hs[-1].get_label()], loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=1, borderaxespad=0.4, handlelength=1.6)

    # --- 2. what the pilot decided, and when ---------------------------------------------------------
    ax = fig.add_subplot(gs[1], sharex=ax)
    style(ax)
    pull = np.array([r["plan"]["t_pull_h"] for r in rows])
    wrap = np.array([np.nan if r["plan"]["t_wrap_h"] is None else r["plan"]["t_wrap_h"] for r in rows])
    ax.plot(th, th, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(th[-1] + 0.2, th[-1] - 1.5, "now", color=INK2, fontsize=8, ha="left")
    ax.plot(th, pull, color=EVENT_C["pull"], lw=2, label="the planned pull time")
    ax.plot(th, wrap, color=EVENT_C["wrap"], lw=2, label="the planned wrap time")
    k_pull = next((i for i, r in enumerate(rows) if "pull" in r["committed"]), None)
    k_wrap = next((i for i, r in enumerate(rows) if "wrap" in r["committed"]), None)
    for kc, arr, key, text in ((k_pull, pull, "pull", "pull committed"), (k_wrap, wrap, "wrap", "wrap committed")):
        if kc is None:
            continue
        ax.plot([th[kc]], [arr[kc]], "o", color=EVENT_C[key], ms=8, mec=SURFACE, mew=1.5)
        ax.text(th[kc] - 0.3, arr[kc] + 0.8, text, color=EVENT_C[key], fontsize=8, ha="right")
    # one marker per replan, coloured by its reason; the updates that held the plan are the baseline
    y0 = 1.5
    ax.plot([r["t_h"] for r in rows if r["updated"] and not r["replanned"]], [y0] * sum(1 for r in rows if r["updated"] and not r["replanned"]),
            "|", ms=6, color=GRID_C, mew=0.8, ls="none", label="updated, kept the plan")
    for key, label, col in REASONS:
        hs_ = [r["t_h"] for r in rows if r["replanned"] and r["reason"] == key]
        if hs_:
            ax.plot(hs_, [y0] * len(hs_), "o", ms=7, color=col, mec=SURFACE, mew=1.0, ls="none", label=f"replanned: {label}")
    kept = [r["t_h"] for r in rows if r["replanned"] and r["kept_standing"]]
    if kept:
        ax.plot(kept, [y0] * len(kept), "x", ms=6, color=INK, mew=1.0, ls="none", label="the replan was worse: the standing plan kept")
    ax.set_ylim(-0.5, max(34.0, float(np.nanmax(pull)) + 9.0))
    ax.set_ylabel("hour of cook", color=INK2, fontsize=9)
    n_re = sum(r["replanned"] for r in rows)
    ax.set_title(f"What the pilot decided: {n_re} replans over {n_up} updates, each on the freshest belief, none on a clock", color=INK, fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2, ncol=2)

    # --- 3. the outcome against the goals, in goal units -----------------------------------------
    ax = fig.add_subplot(gs[2])
    style(ax)
    last = next(r for r in reversed(rows) if r["replanned"])
    pm, ps = last["predicted"]["mean"], last["predicted"]["sd"]
    rows3 = [("tenderness: collagen at the center at serve", 0, goals["c_target"], goals["c_overdone"], 0.05, actual["C_serve"], "{:.3f}"),
             ("appearance: bark L* at serve", 2, goals["l_lo"], goals["l_hi"], goals["dl_soft"], actual["L"], "{:.1f}"),
             ("taste: smoke deposition at serve", 4, goals["s_lo"], goals["s_hi"], goals["ds_soft"], actual["S"], "{:.2f}")]
    for i, (name, idx, lo, hi, sd, act, fmt) in enumerate(rows3):
        y = len(rows3) - i
        center = 0.5 * (lo + hi)
        span = (-(hi - lo) / 2 / sd, (hi - lo) / 2 / sd)
        ax.fill_between(span, [y - 0.42, y - 0.42], [y + 0.08, y + 0.08], color="#eeeeea", lw=0)
        ax.axhline(y, color=GRID_C, lw=0.8)
        if ps[idx] is not None and np.isfinite(ps[idx]):
            ax.errorbar([(pm[idx] - center) / sd], [y], xerr=[2 * ps[idx] / sd], fmt="none", ecolor=CORE, elinewidth=1.4, capsize=3, alpha=0.8)
        ax.plot([(pm[idx] - center) / sd], [y], "o", color=CORE, ms=8, mec=SURFACE, mew=1.2)
        ax.plot([(act - center) / sd], [y], "s", color=PIT, ms=8, mec=SURFACE, mew=1.2)
        ax.text(-4.9, y + 0.12, f"{name}: {lo:g} to {hi:g}, softness {sd:g}", color=INK2, fontsize=8.5, va="bottom")
        ax.text(4.9, y + 0.12, f"the plant: {fmt.format(act)}", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.axvline(0, color=INK, lw=1)
    ax.plot([], [], "o", color=CORE, label=f"predicted at the last replan ({last['t_h']:.1f} h), ±2 sd of the belief through the plan")
    ax.plot([], [], "s", color=PIT, label="the plant")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.2, len(rows3) + 0.6)
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
