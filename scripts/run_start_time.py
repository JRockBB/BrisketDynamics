"""The serving deadline positions the start: when to light the fire.

The second half of the differentiable-duration item. The quality-optimal
cook (run_duration_opt.py) fixes the plan in fractional time; extending it
is the one-parameter family t_f -> t (fractions held). The PLANNED duration
is the shortest t >= t* whose empirical done-fraction over N posterior
draws reaches the target — a Monte Carlo frequency, not a certification.
The margin is one-sided because finishing early is nearly free (the resting
hold exists for this); light the fire at (serve deadline − planned).

Run from app/:  uv run python -u ../scripts/run_start_time.py [--serve 18:00]
Writes data/control/start_time.json and figures/control/start_time.png.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax.numpy as jnp
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.coupled import HR, coupled_cook_tf

REPO = Path(__file__).resolve().parents[1]
# the same cook conditions fullplan and the robust hedge plan under
# the cook from lighting: the pit at ambient, the meat from the fridge (on at the plan's load)
ICS = {"tpit0": 300.0, "t_amb": 300.0, "t0_meat": 281.15}


def _draws(cal: Path, n: int, seed: int, samples: str = "laplace_samples_real.npz"):
    """N posterior (theta, phi) draws from a sample file under data/calibration."""
    s = dict(np.load(cal / samples))
    n_total = len(np.asarray(s["a0"]).reshape(-1))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(n, n_total), replace=False)
    thetas, phis = [], []
    for i in idx:
        row = {k: float(np.asarray(v).reshape(-1)[i]) for k, v in s.items()}
        thetas.append(np.array([row[nm] for nm, _ in THETA_SITES]))
        phis.append(phi_full(row))
    return thetas, phis


def main():
    """Sweep durations >= the quality optimum until P(done) clears the target."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-target", type=float, default=0.9)
    ap.add_argument(
        "--samples",
        default="laplace_samples_real.npz",
        help="posterior sample file under data/calibration (the real cook from lighting)",
    )
    ap.add_argument(
        "--c-serveable",
        type=float,
        default=0.85,
        help="core collagen conversion counted as serveably done",
    )
    ap.add_argument("--step", type=float, default=0.5, help="coarse sweep step [h]")
    ap.add_argument("--refine", type=float, default=0.1, help="refinement step [h]")
    ap.add_argument("--t-max", type=float, default=20.0, help="give-up duration [h]")
    ap.add_argument(
        "--serve",
        default=None,
        help="serve deadline as HH:MM (next occurrence); prints when to light the fire",
    )
    ap.add_argument(
        "--plan",
        choices=("nominal", "robust"),
        default="nominal",
        help="nominal: the joint gradient optimum (duration_opt.json); robust: "
        "the posterior-hedged schedule (--robust-json)",
    )
    ap.add_argument(
        "--robust-json",
        default="schedule_robust_tf.json",
        help="hedged-schedule json under data/control (fullplan-era default: "
        "schedule_robust.json)",
    )
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    ctl = REPO / "data" / "control"

    if args.plan == "robust":
        rb = json.loads((ctl / args.robust_json).read_text())
        t_star = float(rb["cook_h"])
        fw = float(rb["t_wrap_robust_h"]) / t_star
        fp = float(rb["t_pull_robust_h"]) / t_star
        u = jnp.asarray(rb["u_robust"])
        s_u = np.linspace(0.0, 1.0, len(rb["u_robust"]))
        T_load = float(rb["T_load_robust"])
    else:
        plan = json.loads((ctl / "duration_opt.json").read_text())
        t_star = plan["t_f_h"]
        fw, fp = plan["frac_wrap"], plan["frac_pull"]
        u = jnp.asarray(plan["u"])
        s_u = np.asarray(plan["s_u"])
        T_load = float(plan["T_load"])
    s_save = np.linspace(0.0, 1.0, 67)
    thetas, phis = _draws(cal, args.draws, args.seed, args.samples)
    tx = monolith_client()
    print(
        f"{args.plan} plan: t*={t_star:.2f} h from lighting, meat on at a pit of "
        f"{T_load - 273.15:.0f} C, wrap {fw:.3f}, pull {fp:.3f} "
        f"(fractions held while extending); {len(thetas)} posterior draws "
        f"({args.samples}), done = C_core >= {args.c_serveable}, "
        f"target P >= {args.p_target}",
        flush=True,
    )

    cache: dict[float, dict] = {}

    def p_done(t_h: float) -> dict:
        key = round(t_h, 3)
        if key in cache:
            return cache[key]
        t0 = time.perf_counter()
        sims = [
            coupled_cook_tf(
                jnp.asarray(th),
                jnp.asarray(ph),
                u,
                t_h * HR,
                tx,
                s_u,
                s_save,
                frac_wrap=fw,
                frac_pull=fp,
                ics=ICS,
                T_load=T_load,
            )
            for th, ph in zip(thetas, phis, strict=True)
        ]
        c = np.array([float(np.asarray(sm["c_field_final"])[0]) for sm in sims])
        t_on = np.array([float(sm["t_load_used"]) / HR for sm in sims])
        rec = {
            "t_h": t_h,
            "t_load_med_h": float(np.median(t_on)),
            "t_load_p5_h": float(np.percentile(t_on, 5)),
            "t_load_p95_h": float(np.percentile(t_on, 95)),
            "p": float(np.mean(c >= args.c_serveable)),
            "c_p5": float(np.percentile(c, 5)),
            "c_med": float(np.median(c)),
            "c_p95": float(np.percentile(c, 95)),
        }
        cache[key] = rec
        print(
            f"  t={t_h:5.2f} h  P(done)={rec['p']:.2f}  "
            f"doneness p5/med/p95 {rec['c_p5']:.3f}/{rec['c_med']:.3f}/"
            f"{rec['c_p95']:.3f}  ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
        return rec

    # coarse upward sweep from the quality optimum, then one refinement pass
    t = t_star
    last_fail = None
    while t <= args.t_max + 1e-9:
        if p_done(t)["p"] >= args.p_target:
            break
        last_fail = t
        t += args.step
    else:
        raise SystemExit(
            f"P(done) never reached {args.p_target} by {args.t_max} h — "
            "the plan cannot buy this margin by extending"
        )
    if last_fail is not None:
        tr = last_fail + args.refine
        while tr < t - 1e-9:
            if p_done(tr)["p"] >= args.p_target:
                t = tr
                break
            tr += args.refine
    t_plan = t
    margin_h = t_plan - t_star
    print(
        f"planned duration {t_plan:.2f} h = quality-optimal {t_star:.2f} h "
        f"+ {margin_h:.2f} h one-sided margin "
        f"(P(done)={cache[round(t_plan, 3)]['p']:.2f} over {len(thetas)} draws; "
        "a Monte Carlo frequency, not a certification)",
        flush=True,
    )

    serve_at = start_at = None
    if args.serve:
        hh, mm = (int(x) for x in args.serve.split(":"))
        now = datetime.now().astimezone()  # local wall clock, tz-aware
        serve = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if serve <= now:
            serve += timedelta(days=1)
        start = serve - timedelta(hours=t_plan)
        serve_at, start_at = serve.isoformat(" ")[:16], start.isoformat(" ")[:16]
        on = cache[round(t_plan, 3)]
        meat_at = (start + timedelta(hours=on["t_load_med_h"])).isoformat(" ")[:16]
        print(
            f"serve at {serve_at} -> light the fire at {start_at}; the meat goes on when "
            f"the pit reaches {T_load - 273.15:.0f} C (median {meat_at}, "
            f"{on['t_load_p5_h']:.2f}-{on['t_load_p95_h']:.2f} h after lighting)",
            flush=True,
        )

    records = sorted(cache.values(), key=lambda r: r["t_h"])
    out = {
        "plan": args.plan,
        "t_star_h": t_star,
        "t_plan_h": t_plan,
        "margin_h": margin_h,
        "p_target": args.p_target,
        "c_serveable": args.c_serveable,
        "n_draws": len(thetas),
        "draw_source": args.samples,
        "frac_wrap": fw,
        "frac_pull": fp,
        "serve_at": serve_at,
        "light_fire_at": start_at,
        "T_load": T_load,
        "records": records,
    }
    tag = "" if args.plan == "nominal" else f"_{args.plan}"
    (ctl / f"start_time{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote data/control/start_time{tag}.json")
    _figure(out, REPO, tag)
    other = ctl / f"start_time{'_robust' if args.plan == 'nominal' else ''}.json"
    if other.exists():
        a = json.loads((ctl / "start_time.json").read_text())
        b = json.loads((ctl / "start_time_robust.json").read_text())
        _compare_figure(a, b, REPO)


def _figure(out: dict, repo: Path, tag: str = ""):
    """P(done) and the doneness band vs extended duration; t* and t_plan marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = out["records"]
    th = [x["t_h"] for x in r]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.8, 5.6), sharex=True, height_ratios=[1.0, 1.0]
    )
    ax.plot(th, [x["p"] for x in r], "o-", color="tab:blue", ms=4)
    ax.axhline(out["p_target"], color="tab:red", lw=1.0, ls="--")
    ax.text(
        0.25, out["p_target"] - 0.008, f"target {out['p_target']:.2f}",
        transform=ax.get_yaxis_transform(), ha="center", va="top", fontsize=8, color="tab:red",
    )
    ax.set_ylabel(f"P(C_core ≥ {out['c_serveable']:.2f} at serve)")
    ps = [x["p"] for x in r]
    ax.set_ylim(min(min(ps), out["p_target"]) - 0.05, min(1.02, max(max(ps), out["p_target"]) + 0.03))
    if len(r) > 1:
        ax2.fill_between(
            th, [x["c_p5"] for x in r], [x["c_p95"] for x in r],
            alpha=0.25, color="tab:blue", label="doneness p5–p95",
        )
        ax2.plot(th, [x["c_med"] for x in r], "-", color="tab:blue", label="median")
    else:  # the plan needs no margin: one duration, its spread as an error bar
        x0 = r[0]
        ax2.errorbar(th, [x0["c_med"]], yerr=[[x0["c_med"] - x0["c_p5"]], [x0["c_p95"] - x0["c_med"]]],
                     fmt="o", color="tab:blue", capsize=4, label="median, p5–p95")
        ax2.set_xlim(th[0] - 0.5, th[0] + 0.5)
        lo = min(x0["c_p5"], out["c_serveable"])
        ax2.set_ylim(lo - 0.03, 1.02)
    ax2.axhline(out["c_serveable"], color="tab:red", lw=1.0, ls="--")
    ax2.set(xlabel="lighting to serve, fractions held [h]", ylabel="doneness C_core")
    ax2.legend(fontsize=8, loc="lower right")
    for a in (ax, ax2):
        a.axvline(out["t_star_h"], color="0.5", lw=1.0)
        a.axvline(out["t_plan_h"], color="tab:orange", lw=1.4)
        a.grid(alpha=0.3)
    if out["margin_h"] > 1e-9:
        ax.text(out["t_star_h"], 0.92, " t* (quality)", fontsize=8, color="0.4",
                transform=ax.get_xaxis_transform())
        ax.text(out["t_plan_h"], 0.02, " planned", fontsize=8, color="tab:orange",
                transform=ax.get_xaxis_transform())
    else:
        ax.text(out["t_plan_h"], 0.02, "   t* = planned (no margin needed)", fontsize=8, color="tab:orange",
                transform=ax.get_xaxis_transform())
    ax.set_title(
        f"One-sided margin: planned {out['t_plan_h']:.2f} h = "
        f"t* {out['t_star_h']:.2f} h + {out['margin_h']:.2f} h",
        fontsize=11,
    )
    one = len(r) == 1  # one duration: the plan's line sits mid-axis, the note goes top left
    ax.text(
        0.02 if one else 0.62, 0.92 if one else 0.06,
        f"{out['n_draws']} posterior draws ({out.get('draw_source', 'posterior')})",
        transform=ax.transAxes, ha="left" if one else "right", fontsize=8, color="0.4",
    )
    fig.tight_layout()
    fig.savefig(repo / "figures" / "control" / f"start_time{tag}.png", dpi=140)
    plt.close(fig)
    print(f"wrote figures/control/start_time{tag}.png")


def _compare_figure(nominal: dict, robust: dict, repo: Path):
    """P(done) vs stretched duration for the nominal and the hedged plan."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for out, color, name in (
        (nominal, "tab:blue", "nominal plan (quality optimum)"),
        (robust, "tab:green", "robust plan (posterior-hedged schedule)"),
    ):
        r = out["records"]
        ax.plot(
            [x["t_h"] for x in r],
            [x["p"] for x in r],
            "o-",
            color=color,
            ms=4,
            label=f"{name}: planned {out['t_plan_h']:.2f} h",
        )
        ax.axvline(out["t_plan_h"], color=color, lw=1.1, alpha=0.5)
    ax.axhline(nominal["p_target"], color="tab:red", lw=1.0, ls="--")
    ax.text(
        0.5, nominal["p_target"] - 0.008, f"target {nominal['p_target']:.2f}",
        transform=ax.get_yaxis_transform(), ha="center", va="top", fontsize=8, color="tab:red",
    )
    ax.set(
        xlabel="lighting to serve, fractions held [h]",
        ylabel=f"P(C_core ≥ {nominal['c_serveable']:.2f} at serve)",
        title="Buying assurance: stretch the nominal plan, or hedge the schedule",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(repo / "figures" / "control" / "start_time_compare.png", dpi=140)
    plt.close(fig)
    print("wrote figures/control/start_time_compare.png")


if __name__ == "__main__":
    main()
