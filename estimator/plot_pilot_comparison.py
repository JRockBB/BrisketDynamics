"""Two pilots on the same brisket: the RxInfer receding-horizon pilot
(posterior draws, noise identified, replans when it should) and the
app's MPC (state feedback, KCBS quality objective), each cook re-run
through the same plant (estimator/pilot_comparison_trace.jl). Top: the
two cooks as they happened, pit and core with the executed events.
Bottom: the scoreboard, each cook under both objectives.

Reads data/mock/pilot_comparison.json, writes
figures/estimator_pilot_comparison.png.

Run from app/:  uv run python ../estimator/plot_pilot_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
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
    ax.plot(t, run["T_pit_C"], color=PIT, lw=1.5, label="pit")
    ax.plot(t, run["T_core_C"], color=CORE, lw=2.2, label="core")
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
    ax.set_title(title, color=INK, fontsize=10, pad=38, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)


def main():
    d = json.loads((REPO / "data" / "mock" / "pilot_comparison.json").read_text(encoding="utf-8"))
    rx, mc, g = d["rxinfer"], d["mpc"], d["goals"]
    mr, mm = rx["metrics"], mc["metrics"]
    t_end = max(rx["events_h"]["serve"], mc["events_h"]["serve"]) + 0.5

    fig = plt.figure(figsize=(12.5, 12), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.62, left=0.07, right=0.98, top=0.93, bottom=0.04)

    cook_panel(fig.add_subplot(gs[0]),
               rx, f"The RxInfer pilot: {mr['cook_h']:.1f} h cook, pit {mr['pit_mean_C']:.0f} °C mean, pulled at {mr['T_pull_C']:.0f} °C", t_end)
    ax = fig.add_subplot(gs[1])
    cook_panel(ax, mc, f"The MPC: {mm['cook_h']:.1f} h cook, pit {mm['pit_mean_C']:.0f} °C mean (max {mm['pit_max_C']:.0f}), pulled at {mm['T_pull_C']:.0f} °C", t_end)
    ax.set_xlabel("hour of cook", color=INK2, fontsize=8.5)

    # --- the scoreboard: each cook under both objectives --------------------------------------
    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    fmt_h = lambda v: "—" if v is None else f"{v:.1f} h"
    rows = [
        ("the pilot's goals", None, None, None),
        ("probe-tender at the pull (collagen at the center)", f"{g.get('C_pull', 0.92):.2f} ± {g.get('sd_C', 0.02):.2f}", f"{mr['doneness_pull']:.3f}  (core {mr['T_pull_C']:.0f} °C)", f"{mm['doneness_pull']:.3f}  (core {mm['T_pull_C']:.0f} °C)"),
        ("retained water W/W0 at serve", f"{g['W']:.2f} ± {g['sd_W']:.2f}", f"{mr['W_serve']:.3f}  ({100 * mr['wet_basis_water_serve']:.0f} % wet basis)",
         f"{mm['W_serve']:.3f}  ({100 * mm['wet_basis_water_serve']:.0f} % wet basis)"),
        ("bark L* at serve", f"{g['L']:.0f} ± {g['sd_L']:.0f}", f"{mr['L_serve']:.1f}", f"{mm['L_serve']:.1f}"),
        ("core at serve", f"floor {g['T_serve_floor'] - 273.15:.0f} °C", f"{mr['T_serve_C']:.1f} °C", f"{mm['T_serve_C']:.1f} °C"),
        ("the MPC's quality terms", None, None, None),
        ("collagen conversion at the center at serve (tenderness)", "0.90 to 0.95", f"{mr['doneness_serve']:.3f}", f"{mm['doneness_serve']:.3f}"),
        ("bark L* (mahogany band)", "28 to 34", f"{mr['L_serve']:.1f}", f"{mm['L_serve']:.1f}"),
        ("juiciness (water + retained fat)", "higher is better", f"{mr['juiciness_serve_approx']:.3f}", f"{mm['juiciness_serve_approx']:.3f}"),
        ("the cook", None, None, None),
        ("wrap / pull / serve", "", f"{fmt_h(mr['wrap_h'])} / {mr['pull_h']:.1f} h / {mr['cook_h']:.1f} h", f"{fmt_h(mm['wrap_h'])} / {mm['pull_h']:.1f} h / {mm['cook_h']:.1f} h"),
        ("core reaches 70 / 80 / 90 °C", "", f"{fmt_h(mr['core_70_h'])} / {fmt_h(mr['core_80_h'])} / {fmt_h(mr['core_90_h'])}",
         f"{fmt_h(mm['core_70_h'])} / {fmt_h(mm['core_80_h'])} / {fmt_h(mm['core_90_h'])}"),
        ("pit, mean to the pull / max", "", f"{mr['pit_mean_C']:.0f} °C ({mr['pit_mean_C'] * 9 / 5 + 32:.0f} °F) / {mr['pit_max_C']:.0f} °C",
         f"{mm['pit_mean_C']:.0f} °C ({mm['pit_mean_C'] * 9 / 5 + 32:.0f} °F) / {mm['pit_max_C']:.0f} °C"),
        ("what it knows", "", "a 17-latent posterior, noise per channel", "the plant state, an ambient filter"),
        ("when it replans", "", f"when worth it: {d['rxinfer_replans']} of {d['rxinfer_checks']} hourly checks", "every interval"),
    ]
    x_cols = [0.0, 0.40, 0.55, 0.78]
    y = 1.0
    ax.text(x_cols[2], y, "RxInfer pilot", color=INK, fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(x_cols[3], y, "MPC", color=INK, fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(x_cols[1], y, "goal / band", color=INK, fontsize=9.5, fontweight="bold", va="top", transform=ax.transAxes)
    y -= 0.085
    for name, goal, a, b in rows:
        if goal is None:
            ax.text(x_cols[0], y, name, color=INK, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
            y -= 0.07
            continue
        ax.text(x_cols[0] + 0.02, y, name, color=INK2, fontsize=8.5, va="top", transform=ax.transAxes)
        ax.text(x_cols[1], y, goal, color=INK2, fontsize=8.5, va="top", transform=ax.transAxes)
        ax.text(x_cols[2], y, a, color=INK, fontsize=8.5, va="top", transform=ax.transAxes)
        ax.text(x_cols[3], y, b, color=INK, fontsize=8.5, va="top", transform=ax.transAxes)
        y -= 0.065
    plant = "the same brisket" if d["same_plant"] else "different briskets: the MPC flew " + d["mpc_params_file"]
    ax.set_title(f"Each cook under both objectives ({plant}; port check on the MPC's core: {d['port_check_max_core_C']:.1f} °C)",
                 color=INK, fontsize=10, loc="left")

    out = REPO / "figures" / "estimator_pilot_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
