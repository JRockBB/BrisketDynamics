"""The pilots on the same brisket under the one objective: the RxInfer
receding-horizon pilot (a posterior from readings, replans when it
should) and the app's MPC in the variants compared (the oracle with the
plant's own parameters and state; the MPC with the calibration's
parameters as its model), each cook re-run through the same plant
(estimator/scripts/pilot_comparison_trace.jl). Top: the cooks as they happened,
pit and core with the executed events. Bottom: the scoreboard.

Reads data/mock/pilot_comparison.json (and pilot_comparison_calmodel.json when present), writes
figures/estimator/pilot_comparison.png.

Run from app/:  uv run python ../estimator/python/plot_pilot_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INK, INK2, GRID_C, SURFACE = "#1f2430", "#5b6472", "#e4e7ec", "#fbfbfa"
PIT, CORE = "#eb6834", "#2a78d6"
EVENT_C = {"wrap": "#2a78d6", "add": "#eb6834", "spritz": "#1baf7a", "pull": "#8d5cd6", "serve": "#c9a227"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def cook_panel(ax, run, title, t_end):
    style(ax)
    t = np.array(run["t_h"])
    # the pilot's belief of its own cook at every update, one band per temperature (the plant's traces over it);
    # the MPC's panels have no belief: an oracle reads the plant's state
    b = run.get("belief")
    if b is not None:
        tb = np.array(b["t_h"])
        for key, col in (("T_pit_C", PIT), ("T_core_C", CORE)):
            mu = np.array([v[0] for v in b[key]]); sd = np.array([v[1] for v in b[key]])
            ax.fill_between(tb, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.22, lw=0, label=f"{'pit' if key == 'T_pit_C' else 'core'}: the pilot's belief, ±2 sd")
    ax.plot(t, run["T_pit_C"], color=PIT, lw=1.5, label="pit, the plant")
    ax.plot(t, run["T_core_C"], color=CORE, lw=2.2, label="core, the plant")
    ev = run["events_h"]
    above = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    labels = [("wrap", ev["wrap"], "wrapped"), ("add", ev["add"], "fuel added"), ("spritz", ev["spritz"], "spritz begins"),
              ("pull", ev["pull"], "pulled"), ("serve", ev["serve"], "served")]
    row = 0
    for key, th, text in labels:
        if th is None:
            continue
        ax.axvline(th, color=EVENT_C[key], lw=1.2)
        right = key == "serve"
        ax.text(th + (-0.12 if right else 0.12), 1.02 + 0.09 * (row % 3), f"{text} {th:.1f} h", color=EVENT_C[key], fontsize=8,
                va="bottom", ha="right" if right else "left", transform=above)
        row += 1
    ax.set_xlim(0, t_end)
    ax.set_ylim(15, 190)
    ax.set_ylabel("temperature [°C]", color=INK2, fontsize=8.5)
    ax.text(0.01, 0.97, title, color=INK, fontsize=10, va="top", ha="left", transform=ax.transAxes)   # the top-left is empty
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    return ax


def main():
    # the pilot against each MPC variant compared: the oracle (the plant's own parameters and state) and, when it
    # exists, the MPC flown with the calibration's parameters as its model (what runs without an estimator)
    files = [("pilot_comparison.json", "the MPC, an oracle: the plant's parameters and state", "MPC, oracle", "the plant's parameters and state"),
             ("pilot_comparison_calmodel.json", "the MPC with the calibration's parameters, the plant's state", "MPC, calibration model", "the calibration's parameters, the plant's state")]
    runs = []
    for name, label, col, knows in files:
        f = REPO / "data" / "mock" / name
        if f.exists():
            runs.append((json.loads(f.read_text(encoding="utf-8")), label, col, knows))
    d = runs[0][0]
    rx = d["rxinfer"]
    mpcs = [(r["mpc"], label, r) for r, label, _, _ in runs]
    t_end = max([rx["events_h"]["serve"]] + [m["events_h"]["serve"] for m, _, _ in mpcs]) + 0.5
    n_rows = 1 + len(mpcs)

    fig = plt.figure(figsize=(12.5, 4.6 * n_rows + 4.2), facecolor=SURFACE)
    gs = fig.add_gridspec(n_rows + 1, 1, height_ratios=[1.0] * n_rows + [1.25], hspace=0.62, left=0.07, right=0.98, top=0.95, bottom=0.03)
    mr = rx["metrics"]
    cook_panel(fig.add_subplot(gs[0]), rx, f"The RxInfer pilot: {mr['cook_h']:.1f} h cook, pit {mr['pit_mean_C']:.0f} °C mean, pulled at {mr['T_pull_C']:.0f} °C", t_end)
    for i, (mc, label, _) in enumerate(mpcs):
        mm = mc["metrics"]
        ax = cook_panel(fig.add_subplot(gs[1 + i]), mc, f"{label[0].upper() + label[1:]}: {mm['cook_h']:.1f} h cook, pit {mm['pit_mean_C']:.0f} °C mean (max {mm['pit_max_C']:.0f}), pulled at {mm['T_pull_C']:.0f} °C", t_end)
    plt.gca().set_xlabel("hour of cook", color=INK2, fontsize=8.5)

    # --- the scoreboard: the one objective, every cook ------------------------------------------
    ax = fig.add_subplot(gs[n_rows])
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    fmt_h = lambda v: "—" if v is None else f"{v:.1f} h"
    cols = [("RxInfer pilot", mr, d["rxinfer_replans"], d["rxinfer_checks"])] + [(col, r["mpc"]["metrics"], None, None) for r, _, col, _ in runs]
    def row(name, f):
        return (name, [f(mx, re, ch) for _, mx, re, ch in cols])
    rows = [
        ("the objective J (the app's, every cook through one plant)", None),
        row("J, lower is better", lambda m, *_: f"{m['J_terms']['J']:.3f}"),
        row("tenderness (collagen 0.90 to 0.95 at serve)", lambda m, *_: f"{m['J_terms']['tenderness']:.3f}  ({m['doneness_serve']:.3f})"),
        row("appearance (bark L* 28 to 34)", lambda m, *_: f"{m['J_terms']['appearance']:.3f}  ({m['L_serve']:.1f})"),
        row("taste (smoke 0.55 to 0.90; juiciness)", lambda m, *_: f"{m['J_terms']['taste']:.3f}  ({m['S_serve']:.2f}, {m['juiciness_serve']:.2f})"),
        row("bark burn, fuel, smoothness", lambda m, *_: f"{m['J_terms']['surf']:.3f}, {m['J_terms']['fuel']:.3f}, {m['J_terms']['smooth']:.3f}"),
        row("safety", lambda m, *_: f"{m['J_terms']['safety']:.3f}"),
        ("the cook", None),
        row("wrap / pull / serve", lambda m, *_: f"{fmt_h(m['wrap_h'])} / {m['pull_h']:.1f} h / {m['cook_h']:.1f} h"),
        row("core at the pull / at serve", lambda m, *_: f"{m['T_pull_C']:.0f} °C / {m['T_serve_C']:.0f} °C"),
        row("pit, mean to the pull / max", lambda m, *_: f"{m['pit_mean_C']:.0f} °C ({m['pit_mean_C'] * 9 / 5 + 32:.0f} °F) / {m['pit_max_C']:.0f} °C"),
        row("water W/W0 at serve", lambda m, *_: f"{m['W_serve']:.3f}  ({100 * m['wet_basis_water_serve']:.0f} %)"),
        ("what it knows", ["the filter's belief: the state, 17 latents, the sensors' discrepancies"] + [knows for _, _, _, knows in runs]),
        ("when it replans", [f"a decision: {d['rxinfer_replans']} replans over {d['rxinfer_checks']} updates"] + ["every hour"] * len(mpcs)),
    ]
    n_c = len(cols)
    x0, x1 = 0.0, 0.30 if n_c > 2 else 0.36
    xs = [x1 + i * (1.0 - x1) / n_c for i in range(n_c)]
    y = 1.0
    for x, (name, _, _, _) in zip(xs, cols):
        ax.text(x, y, name, color=INK, fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
    y -= 0.075
    for name, vals in rows:
        if vals is None:
            ax.text(x0, y, name, color=INK, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
            y -= 0.06
            continue
        ax.text(x0 + 0.02, y, name, color=INK2, fontsize=8.5, va="top", transform=ax.transAxes)
        for x, v in zip(xs, vals):
            ax.text(x, y, v, color=INK, fontsize=8.5, va="top", transform=ax.transAxes)
        y -= 0.058
    ax.set_title(f"Every cook under the one objective (one plant; port check on the oracle MPC's core: {d['port_check_max_core_C']:.1f} °C)",
                 color=INK, fontsize=10, loc="left")

    out = REPO / "figures" / "estimator" / "pilot_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
