"""The full quality composition: COOK LENGTH x WRAP x PULL x SCHEDULE.

Outer scan over cook lengths (start to serve); inside each, the damper
schedule, the Texas-crutch wrap time and the pull are optimized jointly
against the full quality objective — doneness band (undershoot
catastrophic, overdone penalty above 0.95), juiciness, mechanistic bark
color L*, bark-burn surface constraint, fuel — scored at serve. The rest is
what follows the pull inside the cook.

Run from app/:  uv run python -u ../scripts/run_fullplan.py
Writes figures/fullplan.png + data/control/fullplan_result.json.
``--replot`` redraws the figure from the saved json (one forward solve).
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig, safety_report
from brisket_app.coupled import HR, coupled_cook
from brisket_app.timeplan import _grids_for, optimal_duration, plan_duration

REPO = Path(__file__).resolve().parents[1]
# CLI: run_fullplan.py [hours...] [--params FILE] [--tag SUFFIX] [--replot].
# The objective is the default ControlConfig (KCBS-weighted quality +
# safety); override weights there.
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}
CC = ControlConfig(opt_wrap=True)


def records_from_json(d: dict) -> tuple[list, dict]:
    """Rebuild the scan records and the chosen cook from fullplan_result.json."""
    n = len(d["durations_h"])
    cols = {
        "t_f_h": d["durations_h"],
        "j_quality": d["j_quality"],
        "t_wrap_h": d["t_wrap_h"],
        "t_pull_h": d["t_pull_h"],
        "safety_ok": d.get("safety_ok", [True] * n),
        "t_add_h": d.get("t_add_h", [None] * n),
        "t_spritz_h": d.get("t_spritz_h", [None] * n),
        "dt_spritz_min": d.get("dt_spritz_min", [None] * n),
        "spritz_dose_gm2": d.get("spritz_dose_gm2", [None] * n),
    }
    records = [{k: v[i] for k, v in cols.items()} for i in range(n)]
    best = {
        "t_f_h": d["best_t_f_h"],
        "t_wrap_h": d["best_t_wrap_h"],
        "t_pull_h": d["best_t_pull_h"],
        "t_pull": d["best_t_pull_h"] * HR if d["best_t_pull_h"] else 1.0e9,
        "t_add_h": d.get("best_t_add_h"),
        "t_add": d["best_t_add_h"] * HR if d.get("best_t_add_h") else 1.0e9,
        "t_spritz_h": d.get("best_t_spritz_h"),
        "t_spritz": d["best_t_spritz_h"] * HR if d.get("best_t_spritz_h") else 1.0e9,
        "dt_spritz_min": d.get("best_dt_spritz_min"),
        "dt_spritz": (
            d["best_dt_spritz_min"] * 60.0 if d.get("best_dt_spritz_min") else 3600.0
        ),
        "spritz_dose_gm2": d.get("best_spritz_dose_gm2"),
        "spritz_dose": (
            d["best_spritz_dose_gm2"] / 1e3 if d.get("best_spritz_dose_gm2") else 0.05
        ),
        "u_opt": np.asarray(d["best_u"]),
        "grids": _grids_for(d["best_t_f_h"] * HR),
        "rest": None,
    }
    return records, best


def main():
    """Scan cook lengths, jointly optimizing schedule, wrap and pull.

    Runs ``plan_duration`` over the requested hours at ``--params``, picks
    the best cook with ``optimal_duration``, and writes
    data/control/fullplan_result{tag}.json plus figures/fullplan{tag}.png:
    the quality objective versus cook length (each point annotated with its
    own wrap moment) and the chosen cook's core/surface trajectories with
    the wrap and pull marked, through the rest to serve.
    """
    # startup self-check: prove the RUNNING process imports the band we
    # think it does (stale-import debugging cost us a scan once)
    import jax

    _k = 4.0 / (1.0 - CC.c_overdone)
    _ov = float(
        (jax.nn.softplus(_k * (0.9995 - CC.c_overdone)) / _k) / (1.0 - CC.c_overdone)
    )
    print(
        f"band self-check: tenderness overshoot at c=0.9995 -> "
        f"{CC.w_tenderness * CC.over_rel * _ov**2:.3f} "
        f"(w_tenderness {CC.w_tenderness} x over_rel {CC.over_rel})",
        flush=True,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "durations",
        nargs="*",
        type=float,
        help="cook lengths [h], start to serve; default 8-20",
    )
    ap.add_argument(
        "--params",
        default="map_real.json",
        help="point-estimate json under data/calibration (MAP or VI mean)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for result/figure files (e.g. _vi); leaves MAP artifacts alone",
    )
    ap.add_argument(
        "--replot",
        action="store_true",
        help="redraw the figure from the saved json instead of scanning",
    )
    ap.add_argument(
        "--n-iter",
        type=int,
        default=CC.n_iter,
        help="iteration cap of the inner schedule optimizer (default %(default)s)",
    )
    ap.add_argument(
        "--t-salt-h",
        type=float,
        default=12.0,
        help="dry-brine hours before the cook (overnight rub; 0 = unsalted)",
    )
    args = ap.parse_args()
    cc = replace(CC, n_iter=args.n_iter)
    durations_h = args.durations or [float(h) for h in range(8, 21)]
    result_name = f"fullplan_result{args.tag}.json"
    fig_name = f"fullplan{args.tag}.png"
    result_path = REPO / "data" / "control" / result_name

    params_path = REPO / "data" / "calibration" / args.params
    m = json.loads(params_path.read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    tx = monolith_client()
    print(f"params: {params_path.name}", flush=True)

    if args.replot:
        records, best = records_from_json(json.loads(result_path.read_text()))
        print(f"replotting {result_name}", flush=True)
    else:
        print("full composition scan: cook length x wrap x pull x schedule")
        records = plan_duration(theta, phi, tx, durations_h, cc=cc, ics=ICS,
                                t_salt=args.t_salt_h * HR)
        best = optimal_duration(records)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "durations_h": [r["t_f_h"] for r in records],
                    "j_quality": [r["j_quality"] for r in records],
                    "t_wrap_h": [r["t_wrap_h"] for r in records],
                    "t_pull_h": [r["t_pull_h"] for r in records],
                    "t_add_h": [r["t_add_h"] for r in records],
                    "t_spritz_h": [r["t_spritz_h"] for r in records],
                    "dt_spritz_min": [r["dt_spritz_min"] for r in records],
                    "spritz_dose_gm2": [r["spritz_dose_gm2"] for r in records],
                    "doneness": [r["doneness"] for r in records],
                    "w_retained": [r["w_retained"] for r in records],
                    "juiciness": [r["juiciness"] for r in records],
                    "l_star_end": [r["l_star_end"] for r in records],
                    "ring_mm": [r["ring_mm"] for r in records],
                    "safety_ok": [bool(r["safety_ok"]) for r in records],
                    "F_core": [r["F_core"] for r in records],
                    "danger_h_core": [r["danger_h_core"] for r in records],
                    "best_t_f_h": best["t_f_h"],
                    "best_t_wrap_h": best["t_wrap_h"],
                    "best_t_pull_h": best["t_pull_h"],
                    "best_spritz_dose_gm2": best["spritz_dose_gm2"]
                    if best["t_spritz_h"]
                    else None,
                    "best_t_add_h": best["t_add_h"],
                    "best_t_spritz_h": best["t_spritz_h"],
                    "best_dt_spritz_min": best["dt_spritz_min"]
                    if best["t_spritz_h"]
                    else None,
                    "best_u": best["u_opt"].tolist(),
                },
                indent=2,
            )
        )

    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.asarray(best["u_opt"]),
        tx,
        best["grids"],
        ics=ICS,
        t_wrap=best["t_wrap_h"] * HR,
        t_add=best.get("t_add", 1.0e9),
        t_spritz=best.get("t_spritz", 1.0e9),
        dt_spritz=best.get("dt_spritz", 3600.0),
        spritz_dose=best.get("spritz_dose", 0.05),
        t_pull=best["t_pull"],
        rest=best.get("rest"),
        t_salt=args.t_salt_h * HR,
    )
    sim = {k: np.asarray(v) for k, v in sim.items()}
    doneness = float(sim["c_field_final"][0])
    juiciness = float(sim["juiciness"])
    l_end = float(sim["l_star"][-1])
    cook_h = best["t_f_h"]
    t_wrap_h = best["t_wrap_h"]
    t_pull_h = best["t_pull_h"] if best.get("t_pull_h") else cook_h
    # a chimney after the pull, or a spritz after the wrap or the pull, never fires
    adds = best.get("t_add_h") is not None and best["t_add_h"] < t_pull_h
    spritzes = best.get("t_spritz_h") is not None and best["t_spritz_h"] < min(
        t_wrap_h, t_pull_h
    )
    print(
        f"THE COOK: {cook_h:.0f} h — wrap at {t_wrap_h:.2f} h, pull at "
        f"{t_pull_h:.2f} h, rest {cook_h - t_pull_h:.2f} h, serve at {cook_h:.1f} h "
        f"— doneness {doneness:.3f}, juiciness {juiciness:.3f}, bark L* {l_end:.1f}, "
        f"smoke ring {1e3 * float(sim['ring']) * float(sim['l_thick'][-1]) / float(theta[3]):.1f} mm"
    )
    print(
        "fuel and water: "
        + (
            f"chimney at {best['t_add_h']:.2f} h"
            if adds
            else "no chimney (the load lasts)"
        )
        + "; "
        + (
            f"spritz every {best['dt_spritz_min']:.0f} min from {best['t_spritz_h']:.2f} h, "
            f"{best['spritz_dose_gm2']:.0f} g/m² a spritz"
            if spritzes
            else "no spritz"
        )
    )
    sr = safety_report(sim["tsurf"], sim["tcore"], best["grids"].tknots, CC)
    print(
        f"safety (soft penalties, not a certification): surface F={sr['F_s']:.3g}s "
        f"dwell={sr['danger_h']:.2f}h; core F={sr['F_core']:.3g}s "
        f"dwell={sr['danger_h_core']:.2f}h; ok={sr['ok']}"
    )
    th = np.asarray(best["grids"].tknots) / HR

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, MUTED = "#256abf", "#eb6834", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    cooks = np.array([r["t_f_h"] for r in records])
    js = np.array([r["j_quality"] for r in records])
    ax.plot(cooks, js, "o-", color=BLUE, lw=2)
    for i, r in enumerate(records):
        # the first point sits on a steep segment: label it above
        ax.annotate(
            f"wrap\n{r['t_wrap_h']:.1f}h",
            (r["t_f_h"], r["j_quality"]),
            textcoords="offset points",
            xytext=(0, 6) if i == 0 else (0, -22),
            va="bottom" if i == 0 else "baseline",
            color=MUTED,
            fontsize=7,
            ha="center",
        )
    ylim = (
        float(js.min()) - abs(float(js.min())) * 0.35 - 0.2,
        float(js.max()) + 0.3,
    )
    # marker line from the chosen point upward, clear of its wrap label
    j_best = float(js[np.argmin(np.abs(cooks - cook_h))])
    ax.plot([cook_h, cook_h], [j_best, ylim[1]], ":", color=ORANGE, lw=1.5)
    ax.annotate(
        f"the cook: {cook_h:.0f} h",
        (cook_h - 0.12, float(js.max()) * 0.85),
        color=ORANGE,
        fontsize=10,
        ha="right",
    )
    ax.set(
        xlabel="cook length, start to serve [h]",
        ylabel="quality objective J",
        ylim=ylim,
        title="cook length, wrap and pull chosen together\n"
        "(gray: each cook's own optimal wrap moment)",
    )

    ax = axes[1]
    ax.plot(th, sim["tcore"] - 273.15, "-", color=BLUE, lw=2, label="core")
    ax.plot(
        th, sim["tsurf"] - 273.15, "-", color=BLUE, lw=1, alpha=0.5, label="surface"
    )
    ax.axvline(t_wrap_h, color=ORANGE, lw=1.5, ls=":")
    ax.annotate(
        f"wrap at {t_wrap_h:.1f} h",
        (t_wrap_h - 0.15, 25),
        color=ORANGE,
        fontsize=10,
        ha="right",
    )
    if best.get("t_pull_h"):
        # the optimized pull: the trace continues through the rest
        # (carryover, then the slow cambro decline) to serve
        ax.axvline(t_pull_h, color=MUTED, lw=1.2, ls="--")
        ax.annotate(
            f"pull at {t_pull_h:.1f} h → rest",
            (t_pull_h - 0.15, 45),
            color=MUTED,
            fontsize=10,
            ha="right",
        )
    add_note = " · no chimney"
    if adds:
        add_note = f" · chimney {best['t_add_h']:.1f} h"
        ax.axvline(best["t_add_h"], color=MUTED, lw=1.0, ls=":")
        ax.annotate(
            f"chimney {best['t_add_h']:.1f} h",
            (best["t_add_h"] + 0.15, 15),
            color=MUTED,
            fontsize=9,
            ha="left",
        )
    spritz_note = " · no spritzing"
    if spritzes:
        # every pulse of the policy is a tick on the time axis (the wrap or
        # the pull ends the train); the title carries the policy
        pulses = np.arange(
            best["t_spritz_h"], min(t_wrap_h, t_pull_h), best["dt_spritz_min"] / 60.0
        )
        ax.plot(
            pulses,
            np.full(pulses.shape, 0.03),
            "|",
            color=MUTED,
            ms=9,
            mew=1.2,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        spritz_note = (
            f" · spritz ×{len(pulses)} · {best['dt_spritz_min']:.0f} min · "
            f"{best['spritz_dose_gm2']:.0f} g/m²"
        )
    ax.set(
        xlabel="time [h]",
        ylabel="temperature [°C]",
        xlim=(0, cook_h),
        title=f"the {cook_h:.0f} h cook: wrap {t_wrap_h:.1f} h · pull {t_pull_h:.1f} h "
        f"· serve {cook_h:.0f} h\n"
        f"doneness {doneness:.3f} · juiciness {juiciness:.2f} · bark L* {l_end:.0f}"
        + add_note
        + spritz_note,
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "The perfect brisket picks its own length, wrap and pull "
        f"({params_path.stem} smoker; bark constants uncalibrated)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / fig_name, dpi=140, bbox_inches="tight")
    print(f"wrote figures/{fig_name}")


if __name__ == "__main__":
    main()
