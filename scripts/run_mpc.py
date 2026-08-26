"""Receding-horizon MPC defending THE assessed optimal cook.

The plan under defense is the fullplan optimum (17 h, wrap at ~3.5 h —
data/control/fullplan_result_ext.json), not a freshly derived schedule.
The calibrated smoker (MAP-real parameters) is the plant. At t = 2 h —
BEFORE the planned wrap — a cold front drops ambient by 15 K, which the
controller's model does not know. Compare:
  reference: the assessed plan, no disturbance
  open-loop: the assessed plan executed blindly through the disturbance
             (wrap fires at its planned moment regardless)
  MPC:       hourly re-plans schedule AND wrap moment from measured state
             (full-field handoff; wrap irreversible once executed)

Run from app/:
    uv run python -u ../scripts/run_mpc.py
    uv run python -u ../scripts/run_mpc.py --replot --out mpc_result_vi.json --figure mpc_demo_vi.png

Writes figures/mpc_demo.png, data/control/mpc_result.json, and
data/control/mpc_result_traces.npz (time series + wrap + J so the figure
can be redrawn without another cook).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

REPO = Path(__file__).resolve().parents[1]
HR = 3600.0
T_DIST = 2 * HR  # before the planned ~3.5 h wrap: exercises wrap re-timing
T_AMB_BASE, T_AMB_COLD = 300.0, 285.0
SERIES = (
    ("reference (no cold front)", ":", "C0"),
    ("open-loop", "--", "C1"),
    ("MPC (no front)", "-.", "C2"),
    ("MPC", "-", "C3"),
)
COOK_TAGS = ("reference", "openloop", "mpc_nominal", "mpc")
TRAJ_KEYS = ("t", "tcore", "tpit", "tsurf", "u")
METRIC_KEYS = ("doneness", "juiciness", "l_star_end", "smoke_end")


def traces_path(out_name: str) -> Path:
    """NPZ sidecar next to the result json (mpc_result.json → mpc_result_traces.npz)."""
    p = Path(out_name)
    return REPO / "data" / "control" / f"{p.stem}_traces.npz"


def _nan(x):
    return np.array(np.nan if x is None else float(x))


def save_mpc_traces(path, ref, ol, mp0, mp, t_dist_h, wrap_mode):
    """Write everything render_mpc_figure needs: trajectories, wrap, J, metrics."""
    payload = {
        "t_dist_h": np.array(float(t_dist_h)),
        "wrap_mode": np.array(int(bool(wrap_mode))),
    }
    for tag, r in zip(COOK_TAGS, (ref, ol, mp0, mp), strict=True):
        for k in TRAJ_KEYS:
            if k in r:
                payload[f"{tag}_{k}"] = np.asarray(r[k])
        for k in METRIC_KEYS:
            if k in r:
                payload[f"{tag}_{k}"] = np.array(float(r[k]))
        payload[f"{tag}_wrapped"] = np.array(int(bool(r.get("wrapped"))))
        payload[f"{tag}_t_wrap_abs"] = _nan(r.get("t_wrap_abs"))
        payload[f"{tag}_t_wrap_exec_abs"] = _nan(r.get("t_wrap_exec_abs"))
        hist = np.asarray(r.get("wrap_history") or [], dtype=float)
        if hist.size == 0:
            hist = np.zeros((0, 2))
        else:
            hist = np.reshape(hist, (-1, 2))
        payload[f"{tag}_wrap_history"] = hist
        cands = np.asarray(r.get("wrap_candidates") or [], dtype=float)
        if cands.size == 0:
            cands = np.zeros((0, 6))
        else:
            cands = np.reshape(cands, (-1, 6))
        payload[f"{tag}_wrap_candidates"] = cands
        jj = r.get("J_executed") or {}
        for jk, jv in jj.items():
            payload[f"{tag}_J_{jk}"] = np.array(float(jv))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_mpc_traces(path):
    """Rebuild the four cook dicts plus (t_dist_h, wrap_mode) from a traces npz."""
    z = np.load(path, allow_pickle=False)
    names = set(z.files)
    cooks = []
    for tag in COOK_TAGS:
        r = {}
        for k in TRAJ_KEYS:
            key = f"{tag}_{k}"
            if key in names:
                r[k] = z[key]
        for k in METRIC_KEYS:
            key = f"{tag}_{k}"
            if key in names:
                r[k] = float(z[key])
        r["wrapped"] = bool(int(z[f"{tag}_wrapped"])) if f"{tag}_wrapped" in names else False
        tw = float(z[f"{tag}_t_wrap_abs"]) if f"{tag}_t_wrap_abs" in names else np.nan
        r["t_wrap_abs"] = None if np.isnan(tw) else tw
        te = (
            float(z[f"{tag}_t_wrap_exec_abs"])
            if f"{tag}_t_wrap_exec_abs" in names
            else np.nan
        )
        r["t_wrap_exec_abs"] = None if np.isnan(te) else te
        if f"{tag}_wrap_history" in names:
            r["wrap_history"] = [tuple(row) for row in np.asarray(z[f"{tag}_wrap_history"])]
        if f"{tag}_wrap_candidates" in names:
            r["wrap_candidates"] = [
                tuple(row) for row in np.asarray(z[f"{tag}_wrap_candidates"])
            ]
        prefix = f"{tag}_J_"
        r["J_executed"] = {
            k[len(prefix) :]: float(z[k]) for k in names if k.startswith(prefix)
        }
        cooks.append(r)
    meta = {
        "t_dist_h": float(z["t_dist_h"]) if "t_dist_h" in names else T_DIST / HR,
        "wrap_mode": bool(int(z["wrap_mode"])) if "wrap_mode" in names else True,
    }
    return cooks, meta


def render_mpc_figure(ref, ol, mp0, mp, figure_path, t_dist_h, wrap_mode=True):
    """Core / pit / damper comparison. Event labels live only on the core panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cooks = (ref, ol, mp0, mp)
    shorts = ("ref", "OL", "MPC₀", "MPC")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for r, (name, ls, color) in zip(cooks, SERIES, strict=True):
        axes[0].plot(r["t"] / HR, r["tcore"] - 273.15, ls, color=color, label=name)
        axes[1].plot(r["t"] / HR, r["tpit"] - 273.15, ls, color=color, label=name)
        axes[2].step(r["t"] / HR, r["u"], ls, where="post", color=color, label=name)
    t_end_h = max(float(np.asarray(r["t"])[-1]) for r in cooks) / HR
    wraps = []
    for r, (_name, ls, color), short in zip(cooks, SERIES, shorts, strict=True):
        tw = r.get("t_wrap_exec_abs", r.get("t_wrap_abs"))
        if wrap_mode and r.get("wrapped") and tw is not None and np.isfinite(tw):
            wraps.append((float(tw) / HR, color, ls, short))
    for ax in axes:
        ax.set_xlim(0.0, t_end_h)
    for ax in axes[:2]:
        ax.axvline(t_dist_h, color="steelblue", lw=1, ls="-.")
    ax0 = axes[0]
    ax0.set_ylim(0.0, 100.0)
    # Sit above the axes: a horizontal box at t = 2 h is ~3 h wide and covers
    # the rising traces. Offset points keep this off the data entirely.
    ax0.annotate(
        "cold front (−15 K)",
        xy=(t_dist_h, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="steelblue",
        annotation_clip=False,
        bbox={
            "boxstyle": "round,pad=0.2",
            "fc": "white",
            "ec": "steelblue",
            "lw": 0.8,
            "alpha": 0.95,
        },
    )
    # Empty band under the stall. Stacked, series-colored, may sit on the wrap
    # vlines; no leader lines so the boxes cannot cross.
    for i, (th, color, ls, short) in enumerate(wraps):
        ax0.axvline(th, color=color, lw=1.2, ls=ls)
        ax0.text(
            th,
            12.0 + i * 16.0,
            f"{short} crutch  {th:.1f} h",
            ha="left",
            va="center",
            fontsize=8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": color,
                "lw": 0.8,
                "alpha": 0.95,
            },
        )
    axes[0].set(xlabel="time [h]", ylabel="core [°C]")
    axes[1].set(xlabel="time [h]", ylabel="pit [°C]", title="pit temperature")
    axes[2].set(
        xlabel="time [h]",
        ylabel="damper fraction",
        ylim=(0, 1),
        title="executed schedules (MPC re-plans hourly)",
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[2].legend(fontsize=8, loc="center right")
    retimed = (
        wrap_mode
        and len(wraps) >= 2
        and (max(w[0] for w in wraps) - min(w[0] for w in wraps)) > 0.15
    )
    fig.suptitle(
        "Receding-horizon MPC absorbs an unmodeled disturbance"
        + (" — and re-times the crutch" if retimed else ""),
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    # one self-contained column per series, colored like the traces
    for i, (r, (name, _ls, color)) in enumerate(zip(cooks, SERIES, strict=True)):
        fig.text(
            0.14 + i * 0.24,
            0.80,
            f"{name}\n"
            f"J = {r['J_executed']['J']:.2f}\n"
            f"doneness  {r['doneness']:.2f}\n"
            f"juiciness  {r['juiciness']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            linespacing=1.35,
        )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)


def main():
    """Run the MPC-defends-the-plan demo and render the comparison figure.

    Loads the ASSESSED optimal cook (fullplan: 17 h, wrap ~3.5 h) as the
    plan under defense, then runs three cooks against the plant: a
    reference with no cold front, the plan executed open-loop through the
    t = 2 h ambient drop, and MPC re-planning schedule + wrap hourly from
    measured state. Prints the final doneness of each, writes
    data/control/mpc_result.json, and renders figures/mpc_demo.png.
    """
    wrap_mode = True  # the assessed plan wraps; the demo is wrap-aware

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params",
        default="map_real.json",
        help="plant point-estimate json under data/calibration",
    )
    ap.add_argument(
        "--plan",
        default="",
        help="plan json under data/control; default is fullplan17_converged.json "
        "if present else fullplan_result_ext.json",
    )
    ap.add_argument("--figure", default="mpc_demo.png")
    ap.add_argument("--out", default="mpc_result.json")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="rebuild the figure from data/control/<out-stem>_traces.npz; "
        "no cooks, no Tesseract",
    )
    args = ap.parse_args()
    tpath = traces_path(args.out)

    if args.replot:
        if not tpath.exists():
            raise SystemExit(
                f"no traces at {tpath}; run without --replot once to write them"
            )
        (ref, ol, mp0, mp), meta = load_mpc_traces(tpath)
        render_mpc_figure(
            ref,
            ol,
            mp0,
            mp,
            REPO / "figures" / args.figure,
            t_dist_h=meta["t_dist_h"],
            wrap_mode=meta["wrap_mode"],
        )
        print(f"replotted {args.figure} from {tpath.name}")
        return

    from brisket_app import mpc
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import meat_client, pit_client
    from brisket_app.control import ControlConfig, safety_report
    from brisket_app.coupled import CookGrids

    params_path = REPO / "data" / "calibration" / args.params
    m = json.loads(params_path.read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(meat_client(), pit_client())
    print(f"plant: {params_path.name}", flush=True)

    cold_front = lambda t: T_AMB_COLD if t >= T_DIST else T_AMB_BASE
    no_dist = lambda t: T_AMB_BASE
    cc = ControlConfig(opt_wrap=True)

    # THE plan under defense: --plan wins; else the CONVERGED optimum when
    # present (see run_fullplan_converge.py), else the scan incumbent.
    # A VI rerun must pass --plan so the MAP converged file is not reused.
    conv = REPO / "data" / "control" / "fullplan17_converged.json"
    plan_path = REPO / "data" / "control" / args.plan if args.plan else None
    if plan_path is not None:
        b = json.loads(plan_path.read_text())
        # Scan json has both best_* scalars and t_wrap_h as a per-duration
        # list; prefer best_*. Converged json has only t_f_h / t_wrap_h / u_opt.
        T_F = float(b["best_t_f_h"] if "best_t_f_h" in b else b["t_f_h"]) * HR
        t_wrap_star = float(
            b["best_t_wrap_h"] if "best_t_wrap_h" in b else b["t_wrap_h"]
        ) * HR
        u0_plan = np.asarray(b["best_u"] if "best_u" in b else b["u_opt"])
        print(f"using plan {plan_path.name}")
    elif conv.exists():
        b = json.loads(conv.read_text())
        T_F = b["t_f_h"] * HR
        t_wrap_star = b["t_wrap_h"] * HR
        u0_plan = np.asarray(b["u_opt"])
        print("using CONVERGED incumbent (fullplan17_converged.json)")
    else:
        best = json.loads(
            (REPO / "data" / "control" / "fullplan_result_ext.json").read_text()
        )
        T_F = best["best_t_f_h"] * HR
        t_wrap_star = best["best_t_wrap_h"] * HR
        u0_plan = np.asarray(best["best_u"])
    grids0 = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=len(u0_plan))
    print(
        f"plan under defense: {T_F / HR:.0f} h cook, "
        f"wrap at {t_wrap_star / HR:.2f} h, "
        f"u in [{u0_plan.min():.2f}, {u0_plan.max():.2f}]"
    )

    print("reference (no disturbance):")
    ref = mpc.run_cook(
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
    )
    print("open-loop through the cold front:")
    ol = mpc.run_cook(
        theta,
        phi,
        cold_front,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
    )
    print("MPC, NO disturbance (discriminator: plan-refinement vs weather):")
    mp0 = mpc.run_cook(
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="mpc",
        model_t_amb=T_AMB_BASE,
        cc=cc,
        t_wrap_init_abs=t_wrap_star,
        u_incumbent=u0_plan,
        u_incumbent_grids=grids0,
    )
    print("MPC through the cold front:")
    mp = mpc.run_cook(
        theta,
        phi,
        cold_front,
        t_f=T_F,
        mode="mpc",
        model_t_amb=T_AMB_BASE,
        cc=cc,
        t_wrap_init_abs=t_wrap_star,
        u_incumbent=u0_plan,
        u_incumbent_grids=grids0,
    )
    if wrap_mode and mp.get("wrapped"):
        tw_exec = mp.get("t_wrap_exec_abs", mp.get("t_wrap_abs"))
        n_cand = len(mp.get("wrap_candidates") or [])
        print(
            f"  crutch executed at {tw_exec / HR:.2f} h "
            f"(replanned {len(mp.get('wrap_history') or [])}x, "
            f"{n_cand} wrap candidates logged)"
        )

    for name, r in [
        ("reference", ref),
        ("open-loop", ol),
        ("MPC-nominal", mp0),
        ("MPC", mp),
    ]:
        jj = mpc.executed_objective(r, cc, T_F)
        r["J_executed"] = jj
        print(
            f"  {name}: J={jj['J']:.3f}  doneness {r['doneness']:.3f} (band "
            f"{cc.c_target}-{cc.c_overdone})  juiciness {r['juiciness']:.3f}  "
            f"L* {r['l_star_end']:.1f}  smoke {r['smoke_end']:.2f}"
        )
        print(f"    components: {jj}")
        sr = safety_report(r["tsurf"], r["t"], cc)
        print(
            f"    safety (surface, intact-muscle): "
            f"F={sr['F_s']:.1f}s ok={sr['lethality_ok']}  "
            f"dwell={sr['danger_h']:.2f}h ok={sr['dwell_ok']}"
        )

    out = REPO / "data" / "control"
    out.mkdir(parents=True, exist_ok=True)
    (out / args.out).write_text(
        json.dumps(
            {
                **{
                    f"{k}_{tag}": r[k]
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                    for k in ("doneness", "juiciness", "l_star_end", "smoke_end")
                },
                **{
                    f"J_{tag}": r["J_executed"]
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                },
                "t_disturbance_h": T_DIST / HR,
                "dT_amb": T_AMB_BASE - T_AMB_COLD,
                "params": args.params,
                "plan": args.plan or (
                    "fullplan17_converged.json"
                    if conv.exists()
                    else "fullplan_result_ext.json"
                ),
                **(
                    {
                        "wrapped": bool(mp.get("wrapped")),
                        "t_wrap_abs_h": (
                            mp["t_wrap_abs"] / HR
                            if mp.get("t_wrap_abs") is not None
                            else None
                        ),
                        "wrap_history_h": [
                            [t / HR, w / HR] for t, w in mp.get("wrap_history", [])
                        ],
                        "wrap_candidates_h": [
                            [
                                row[0] / HR,
                                row[1] / HR,
                                row[2] / HR,
                                row[3],
                                row[4],
                                row[5],
                            ]
                            for row in mp.get("wrap_candidates", [])
                        ],
                        "t_wrap_exec_abs_h": (
                            mp["t_wrap_exec_abs"] / HR
                            if mp.get("t_wrap_exec_abs") is not None
                            else None
                        ),
                    }
                    if wrap_mode
                    else {}
                ),
            },
            indent=2,
        )
    )

    traces = traces_path(args.out)
    save_mpc_traces(traces, ref, ol, mp0, mp, t_dist_h=T_DIST / HR, wrap_mode=wrap_mode)
    print(f"wrote {traces.relative_to(REPO)}")
    render_mpc_figure(
        ref,
        ol,
        mp0,
        mp,
        REPO / "figures" / args.figure,
        t_dist_h=T_DIST / HR,
        wrap_mode=wrap_mode,
    )
    print(f"wrote figures/{args.figure}")
    print(
        f"replot: uv run python -u ../scripts/run_mpc.py --replot "
        f"--out {args.out} --figure {args.figure}"
    )


if __name__ == "__main__":
    main()
