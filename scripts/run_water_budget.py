"""Water-budget sensitivity on the calibrated real cook.

Which (h_c, D_eff, L) retain 50-60 % of the water while still stalling?
docs/roadmap.md (Water budget): the MAP cook ends with ~4 % of its water
retained while a real flat loses ~40 %. Either the heat input to the meat
is ~2x too high or the slab is thermally thin for the water it carries.
This scan re-runs the calibrated real cook (the log's fan duty and
ambient, MAP parameters for everything else) over multipliers on h_c
(meat-side heat input, with k_m riding on the same draft law), D_eff
(interior supply to the surface) and L (thermal thickness), and maps the
retained-water fraction at the end of the cook together with the stall
duration (hours the core spends inside the 65-78 C plateau band, the
band the calibrated cook sits in for ~10 h). If a 50-60 % point exists
inside the 2-sigma prior band with a stall of several hours, the scale
experiment (roadmap C2) has a predicted target; if no such point exists,
the Fickian slab is the wrong model (docs/review.md, T1), not a wrong
k_m. No new data is needed for the map itself.

Run from app/:  uv run python -u ../scripts/run_water_budget.py
Writes data/validation/water_budget_scan.json and
figures/water_budget_scan.png.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES
from brisket_app.clients import monolith_client
from brisket_app.cooklog import load
from brisket_app.coupled import HR, CookGrids, coupled_cook

# the scan: multipliers on the MAP values (log-spaced), thickness in steps
H_MULT = np.geomspace(0.25, 1.0, 7)
D_MULT = np.geomspace(0.1, 3.0, 7)
L_MULT = np.array([1.0, 1.5, 2.0, 2.5])
STALL_BAND_C = (65.0, 78.0)  # the calibrated cook's plateau band
STALL_MIN_H = 4.0  # "still stalling": at least this long in the band
RET_TARGET = (0.50, 0.60)  # what a real flat retains


def prior_z(name, value):
    """Standardized log-distance of a value from its (LogNormal) prior."""
    d = dict(THETA_SITES)[name]
    return (np.log(value) - float(d.loc)) / float(d.scale)


def main():
    """Scan the calibrated cook, write the JSON and the figure."""
    log = load(dt_min=10.0)
    grids = CookGrids(
        t_end=float(log["t"][-1]),
        dt_exchange=float(log["t"][1] - log["t"][0]),
        n_u=len(log["t"]),
    )
    ics = {
        "tpit0": log["meta"]["tpit0_K"],
        "t_amb": float(log["meta"].get("t_amb_K", 300.0)),
        "t0_meat": log["meta"]["tcore0_K"],
    }
    mp = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([mp[n] for n, _ in THETA_SITES], dtype=float)
    phi = np.array(
        [mp[n] for n, _ in PHI_SITES]
        + [mp.get(n, d.mean) for n, d in PHI_SITES_IGNITION],
        dtype=float,
    )
    u = jnp.asarray(log["u"])
    tx = monolith_client()
    t_h = np.asarray(log["t"]) / HR
    dt_h = float(t_h[1] - t_h[0])

    def cook(th):
        out = coupled_cook(jnp.asarray(th), jnp.asarray(phi), u, tx, grids, ics=ics)
        tcore = np.asarray(out["tcore"]) - 273.15
        in_band = (tcore >= STALL_BAND_C[0]) & (tcore <= STALL_BAND_C[1])
        return {
            "w_retained": float(out["w_retained"]),
            "stall_h": float(np.sum(in_band) * dt_h),
            "tcore_end_C": float(tcore[-1]),
            "tcore_max_C": float(np.max(tcore)),
        }

    t0 = time.time()
    ref = cook(theta)
    print(
        f"MAP cook ({log['meta']['t_end_h']:.1f} h): retained {ref['w_retained']:.3f}, "
        f"stall {ref['stall_h']:.1f} h in {STALL_BAND_C} C, core end {ref['tcore_end_C']:.1f} C"
    )
    W = np.zeros((len(L_MULT), len(D_MULT), len(H_MULT)))
    S = np.zeros_like(W)
    Z = np.zeros_like(W)  # max |z| over the three scaled parameters (prior distance)
    n = 0
    for iL, mL in enumerate(L_MULT):
        for iD, mD in enumerate(D_MULT):
            for iH, mH in enumerate(H_MULT):
                th = theta.copy()
                th[0] *= mD
                th[2] *= mH
                th[3] *= mL
                r = cook(th)
                W[iL, iD, iH] = r["w_retained"]
                S[iL, iD, iH] = r["stall_h"]
                Z[iL, iD, iH] = max(
                    abs(prior_z("D_eff", th[0])),
                    abs(prior_z("h_c", th[2])),
                    abs(prior_z("L", th[3])),
                )
                n += 1
        print(f"  L x{mL:.1f} done ({n} solves, {time.time() - t0:.0f} s)")

    target = (W >= RET_TARGET[0]) & (W <= RET_TARGET[1])
    stalling = S >= STALL_MIN_H
    inside = Z <= 2.0
    feasible = target & stalling & inside
    cells = []
    for iL, iD, iH in zip(*np.nonzero(target & stalling), strict=True):
        cells.append(
            {
                "L_mult": float(L_MULT[iL]),
                "L_cm": float(100 * theta[3] * L_MULT[iL]),
                "D_mult": float(D_MULT[iD]),
                "h_mult": float(H_MULT[iH]),
                "h_c": float(theta[2] * H_MULT[iH]),
                "w_retained": float(W[iL, iD, iH]),
                "stall_h": float(S[iL, iD, iH]),
                "prior_zmax": float(Z[iL, iD, iH]),
                "inside_prior_2sigma": bool(inside[iL, iD, iH]),
            }
        )
    print(
        f"cells retaining {RET_TARGET[0]:.0%}-{RET_TARGET[1]:.0%} with a >= {STALL_MIN_H:.0f} h stall: "
        f"{len(cells)} ({int(feasible.sum())} inside the 2-sigma prior)"
    )
    for c in sorted(cells, key=lambda c: c["prior_zmax"])[:8]:
        print(
            f"  L x{c['L_mult']:.1f} ({c['L_cm']:.1f} cm)  D_eff x{c['D_mult']:.2f}  "
            f"h_c x{c['h_mult']:.2f} ({c['h_c']:.1f} W/m2K): retained {c['w_retained']:.2f}, "
            f"stall {c['stall_h']:.1f} h, prior |z|max {c['prior_zmax']:.1f}"
        )

    out = REPO / "data" / "validation" / "water_budget_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "log": log["meta"].get("name", "real cook"),
                "map": {"D_eff": theta[0], "h_c": theta[2], "L": theta[3]},
                "reference": ref,
                "h_mult": H_MULT.tolist(),
                "D_mult": D_MULT.tolist(),
                "L_mult": L_MULT.tolist(),
                "w_retained": W.tolist(),
                "stall_h": S.tolist(),
                "prior_zmax": Z.tolist(),
                "stall_band_C": STALL_BAND_C,
                "stall_min_h": STALL_MIN_H,
                "retention_target": RET_TARGET,
                "feasible_cells": cells,
                "n_feasible_inside_prior": int(feasible.sum()),
                "wall_s": time.time() - t0,
            },
            indent=1,
        )
    )
    print("wrote", out)
    figure(theta, W, S, Z, ref)


def figure(theta, W, S, Z, ref):
    """Draw the small multiples over L.

    Retained water on a one-hue ramp with the stall boundary, the 50 / 60 %
    target lines and the prior's 2-sigma edge, all explained in one figure
    legend (no in-plot labels to collide).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    ink, ink2, surface, accent = "#0b0b0b", "#52514e", "#fcfcfb", "#eb6834"
    ramp = LinearSegmentedColormap.from_list(
        "blue_seq",
        ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    )
    fig, axes = plt.subplots(
        1, len(L_MULT), figsize=(3.4 * len(L_MULT) + 1.2, 4.9), sharey=True,
        layout="constrained",
    )
    fig.patch.set_facecolor(surface)
    hh = np.log(H_MULT)
    dd = np.log(D_MULT)
    he = np.concatenate([[hh[0] - (hh[1] - hh[0]) / 2], (hh[1:] + hh[:-1]) / 2,
                         [hh[-1] + (hh[-1] - hh[-2]) / 2]])
    de = np.concatenate([[dd[0] - (dd[1] - dd[0]) / 2], (dd[1:] + dd[:-1]) / 2,
                         [dd[-1] + (dd[-1] - dd[-2]) / 2]])
    HH, DD = np.meshgrid(hh, dd)
    for ax, mL, Wl, Sl, Zl in zip(axes, L_MULT, W, S, Z, strict=True):
        ax.set_facecolor(surface)
        im = ax.pcolormesh(he, de, Wl, cmap=ramp, vmin=0.0, vmax=0.8,
                           edgecolors=surface, linewidth=0.8)
        if Sl.min() < STALL_MIN_H < Sl.max():
            ax.contour(HH, DD, Sl, levels=[STALL_MIN_H], colors=[ink], linewidths=1.8)
        for lev, ls in zip(RET_TARGET, ["--", "-."], strict=True):
            if Wl.min() < lev < Wl.max():
                ax.contour(HH, DD, Wl, levels=[lev], colors=[accent], linewidths=1.8,
                           linestyles=[ls])
        for iD in range(len(D_MULT)):
            for iH in range(len(H_MULT)):
                if Zl[iD, iH] > 2.0:
                    ax.add_patch(Rectangle((he[iH], de[iD]), he[iH + 1] - he[iH],
                                           de[iD + 1] - de[iD], fill=False, hatch="/",
                                           edgecolor=ink2, linewidth=0.0, alpha=0.7))
        if mL == 1.0:
            ax.plot([0.0], [0.0], marker="o", ms=9, mfc=surface, mec=ink, mew=1.8, ls="none")
            ax.annotate("MAP", (0.0, 0.0), xytext=(-8, 7), textcoords="offset points",
                        fontsize=8, color=ink, ha="right")
        ax.set_xticks(hh[::2])
        ax.set_xticklabels([f"{m:.2g}" for m in H_MULT[::2]], color=ink2, fontsize=8)
        ax.set_yticks(dd[::2])
        ax.set_yticklabels([f"{m:.2g}" for m in D_MULT[::2]], color=ink2, fontsize=8)
        ax.set_xlabel("h_c multiplier (heat input to the meat)", color=ink2, fontsize=9)
        ax.set_title(f"L x{mL:.1f} = {100 * theta[3] * mL:.1f} cm half-thickness",
                     color=ink, fontsize=10)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax.tick_params(colors=ink2, length=2)
    axes[0].set_ylabel("D_eff multiplier (interior water supply)", color=ink2, fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("water retained at the end of the calibrated cook", color=ink2, fontsize=9)
    cb.ax.tick_params(colors=ink2)
    handles = [
        Line2D([], [], color=ink, lw=1.8, label=f"stall boundary: {STALL_MIN_H:.0f} h in "
               f"{STALL_BAND_C[0]:.0f}\u2013{STALL_BAND_C[1]:.0f} \u00b0C (stalling side: right / up)"),
        Line2D([], [], color=accent, lw=1.8, ls="--", label=f"{RET_TARGET[0]:.0%} retained"),
        Line2D([], [], color=accent, lw=1.8, ls="-.", label=f"{RET_TARGET[1]:.0%} retained"),
        Patch(facecolor=surface, edgecolor=ink2, hatch="//", label="outside the 2-sigma prior"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5,
               frameon=False, labelcolor=ink2)
    fig.suptitle(
        f"Water budget on the calibrated real cook: the MAP retains {ref['w_retained']:.0%} "
        f"after a {ref['stall_h']:.0f} h stall; a real flat retains 50\u201360 %",
        color=ink, fontsize=10.5,
    )
    out = REPO / "figures" / "water_budget_scan.png"
    fig.savefig(out, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def replot():
    """Re-draw the figure from the saved scan (no solves)."""
    d = json.loads((REPO / "data" / "validation" / "water_budget_scan.json").read_text())
    theta = np.zeros(8)
    theta[3] = d["map"]["L"]
    figure(theta, np.array(d["w_retained"]), np.array(d["stall_h"]),
           np.array(d["prior_zmax"]), d["reference"])


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
