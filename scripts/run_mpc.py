"""Receding-horizon MPC defending THE assessed optimal cook.

The plan under defense is the fullplan optimum (data/control/
fullplan_result.json: pit horizon, wrap, pull, rest), not a freshly derived
schedule. The calibrated smoker is the plant. Two hours after the plan's
meat-on time a cold front drops ambient by 15 K, which the controller's
model does not know. Every cook runs to SERVE (pull + rest) and quality is
read there. Compare:
  reference: the assessed plan, no disturbance
  open-loop: the assessed plan executed blindly through the disturbance
             (wrap and pull fire at their planned moments regardless)
  MPC:       hourly re-plans schedule, wrap AND pull from measured state
             (full-field handoff; wrap and pull irreversible once executed)

Run from app/:
    uv run python -u ../scripts/run_mpc.py
    uv run python -u ../scripts/run_mpc.py --replot --out mpc_result_vi.json --figure mpc_demo_vi.png

Writes figures/control/mpc_demo.png, data/control/mpc_result.json, and
data/control/mpc_result_traces.npz (time series + wrap + J so the figure
can be redrawn without another cook).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

REPO = Path(__file__).resolve().parents[1]
HR = 3600.0
COLD_FRONT_DELAY = 2 * HR  # measured from the plan's meat-on, not from lighting
T_AMB_BASE, T_AMB_COLD = 300.0, 285.0


def _fired_add(r):
    """Tell whether the chimney fires: only before the pull, never after."""
    ta, tp = r.get("t_add_exec_abs"), r.get("t_pull_exec_abs")
    return ta is not None and np.isfinite(ta) and (tp is None or ta < tp)


def _fired_spritz(r):
    """Tell whether the spritz train fires: only before the wrap and the pull."""
    ts = r.get("t_spritz_exec_abs")
    tw = r.get("t_wrap_exec_abs", r.get("t_wrap_abs"))
    tp = r.get("t_pull_exec_abs")
    stop = min([x for x in (tw, tp) if x is not None], default=np.inf)
    return ts is not None and np.isfinite(ts) and ts < stop


SERIES = (
    ("reference (no cold front)", ":", "C0"),
    ("open-loop", "--", "C1"),
    ("MPC (no front)", "-.", "C2"),
    ("MPC", "-", "C3"),
)
COOK_TAGS = ("reference", "openloop", "mpc_nominal", "mpc")
TRAJ_KEYS = ("t", "tcore", "tpit", "tsurf", "u")
METRIC_KEYS = ("doneness", "juiciness", "l_star_end", "smoke_end")


def planned_disturbance_time(plan):
    """One common weather event, two hours after the reference plan's load."""
    load_h = plan.get("best_t_load_h", plan.get("t_load_h"))
    if load_h is None:
        if plan.get("best_T_load") or plan.get("T_load"):
            raise ValueError(
                "A preheating plan must record its meat-on time to place the cold front"
            )
        load_h = 0.0  # legacy plans start with the meat already on
    if not np.isfinite(load_h):
        raise ValueError("The plan's meat-on time must be finite")
    return float(load_h) * HR + COLD_FRONT_DELAY


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def cached_cook(path, fingerprint, run, fresh=False):
    """Save each completed cook atomically; reuse only matching inputs/code."""
    if path.exists() and not fresh:
        saved = json.loads(path.read_text())
        if saved["fingerprint"] == fingerprint:
            print(f"resuming {path.stem} from completed checkpoint", flush=True)
            result = saved["result"]
            for key in TRAJ_KEYS:
                result[key] = np.asarray(result[key])
            return result
    t0 = time.perf_counter()
    result = run()
    # The demo uses terminal metrics, trajectories and event histories. The
    # dataclass final_state is only needed to continue a live run_cook.
    saved = {
        "fingerprint": fingerprint,
        "wall_s": time.perf_counter() - t0,
        "result": {k: v for k, v in result.items() if k != "final_state"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(saved, default=_json_default))
    temp.replace(path)
    print(f"saved {path.stem} ({saved['wall_s']:.1f} s)", flush=True)
    return result


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
        payload[f"{tag}_t_pull_exec_abs"] = _nan(r.get("t_pull_exec_abs"))
        payload[f"{tag}_t_load_exec_abs"] = _nan(r.get("t_load_exec_abs"))
        payload[f"{tag}_T_load"] = _nan(r.get("T_load"))
        payload[f"{tag}_t_add_exec_abs"] = _nan(r.get("t_add_exec_abs"))
        payload[f"{tag}_t_spritz_exec_abs"] = _nan(r.get("t_spritz_exec_abs"))
        payload[f"{tag}_dt_spritz"] = _nan(r.get("dt_spritz"))
        payload[f"{tag}_spritz_dose"] = _nan(r.get("spritz_dose"))
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
        r["wrapped"] = (
            bool(int(z[f"{tag}_wrapped"])) if f"{tag}_wrapped" in names else False
        )
        tw = float(z[f"{tag}_t_wrap_abs"]) if f"{tag}_t_wrap_abs" in names else np.nan
        r["t_wrap_abs"] = None if np.isnan(tw) else tw
        te = (
            float(z[f"{tag}_t_wrap_exec_abs"])
            if f"{tag}_t_wrap_exec_abs" in names
            else np.nan
        )
        r["t_wrap_exec_abs"] = None if np.isnan(te) else te
        tl = (
            float(z[f"{tag}_t_load_exec_abs"])
            if f"{tag}_t_load_exec_abs" in names
            else np.nan
        )
        r["t_load_exec_abs"] = None if np.isnan(tl) else tl
        tp = (
            float(z[f"{tag}_t_pull_exec_abs"])
            if f"{tag}_t_pull_exec_abs" in names
            else np.nan
        )
        r["t_pull_exec_abs"] = None if np.isnan(tp) else tp
        for key in ("t_add_exec_abs", "t_spritz_exec_abs", "dt_spritz", "spritz_dose"):
            v = float(z[f"{tag}_{key}"]) if f"{tag}_{key}" in names else np.nan
            r[key] = None if np.isnan(v) else v
        if f"{tag}_wrap_history" in names:
            r["wrap_history"] = [
                tuple(row) for row in np.asarray(z[f"{tag}_wrap_history"])
            ]
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
        "t_dist_h": float(z["t_dist_h"])
        if "t_dist_h" in names
        else COLD_FRONT_DELAY / HR,
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
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
    # Sit above the axes: a horizontal box at the front covers
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
    # the pulls: labels right-aligned at the pull, stacked like the crutch
    # labels; the traces after the pull are the rest (cambro) to serve
    pulls = []
    for r, (_name, ls, color), short in zip(cooks, SERIES, shorts, strict=True):
        tp = r.get("t_pull_exec_abs")
        if tp is not None and np.isfinite(tp):
            pulls.append((float(tp) / HR, color, ls, short))
    for i, (th, color, ls, short) in enumerate(pulls):
        ax0.axvline(th, color=color, lw=1.2, ls=ls)
        ax0.text(
            th,
            20.0 + i * 16.0,  # rows interleave with the crutch rows
            f"{short} pull  {th:.1f} h ",
            ha="right",
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
    # the chimney (pit panel) and the spritz policy (damper panel)
    for i, (r, (_name, _ls, color), short) in enumerate(
        zip(cooks, SERIES, shorts, strict=True)
    ):
        ta = r.get("t_add_exec_abs")
        if _fired_add(r):
            axes[1].axvline(ta / HR, color=color, lw=1.0, ls=":")
            axes[1].text(
                ta / HR + 0.15,
                0.97 - 0.07 * i,
                f"{short} chimney {ta / HR:.1f} h",
                transform=axes[1].get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.5,
                color=color,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1},
            )
        ts = r.get("t_spritz_exec_abs")
        if _fired_spritz(r):
            axes[2].axvline(ts / HR, color=color, lw=1.0, ls=":")
            axes[2].text(
                ts / HR + 0.15,
                0.97 - 0.07 * i,
                f"{short} spritz {ts / HR:.1f} h / {r.get('dt_spritz', 0.0) / 60:.0f} min",
                transform=axes[2].get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.5,
                color=color,
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1},
            )
    axes[0].set(xlabel="time from lighting [h]", ylabel="core [°C]")
    axes[1].set(
        xlabel="time from lighting [h]", ylabel="pit [°C]", title="pit temperature"
    )
    axes[2].set(
        xlabel="time from lighting [h]",
        ylabel="damper fraction",
        ylim=(0, 1),
        title="executed schedules (MPC re-plans hourly)",
    )
    # one legend, on the pit panel: the core panel carries the event labels
    # and the damper panel the spritz labels (same colors and line styles)
    axes[1].legend(fontsize=8, loc="lower center")
    retimed = (
        wrap_mode
        and len(wraps) >= 2
        and (max(w[0] for w in wraps) - min(w[0] for w in wraps)) > 0.15
    )
    repulled = (
        len(pulls) >= 2 and (max(p[0] for p in pulls) - min(p[0] for p in pulls)) > 0.15
    )
    moved = [s for s, f in (("crutch", retimed), ("pull", repulled)) if f]
    fig.suptitle(
        f"Receding-horizon MPC absorbs an unmodeled disturbance on the "
        f"{t_end_h:.0f} h cook"
        + (f" — and re-times the {' and '.join(moved)}" if moved else ""),
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    # one self-contained column per series, colored like the traces
    for i, (r, (name, _ls, color)) in enumerate(zip(cooks, SERIES, strict=True)):
        tp = r.get("t_pull_exec_abs")
        tl = r.get("t_load_exec_abs")
        load_label = f"meat on {tl / HR:.1f} h" if tl is not None else "meat on —"
        ta, ts = r.get("t_add_exec_abs"), r.get("t_spritz_exec_abs")
        fig.text(
            0.14 + i * 0.24,
            0.765,
            f"{name}\n"
            f"J = {r['J_executed']['J']:.2f}\n"
            f"doneness  {r['doneness']:.2f}\n"
            f"juiciness  {r['juiciness']:.2f}"
            + f"\n{load_label}"
            + (f" / pull {tp / HR:.1f} h" if tp is not None else "")
            + (f"\nchimney  {ta / HR:.1f} h" if _fired_add(r) else "\nchimney  none")
            + (
                f"\nspritz  from {ts / HR:.1f} h, every {(r.get('dt_spritz') or 0.0) / 60:.0f} min"
                if _fired_spritz(r)
                else "\nspritz  none"
            ),
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

    Loads the ASSESSED optimal cook (fullplan json) as the plan under
    defense, then runs four cooks against the plant, each pulled and rested
    to serve: a reference with no cold front, the plan executed open-loop
    through the ambient drop two hours after meat-on, and MPC (with and without the front)
    re-planning schedule + wrap + pull hourly from measured state. Prints
    the quality at serve of each, writes data/control/mpc_result.json, and
    renders figures/control/mpc_demo.png.
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
        help="plan json under data/control (default fullplan_result.json)",
    )
    ap.add_argument(
        "--t-salt-h",
        type=float,
        default=12.0,
        help="dry-brine hours before the cook (overnight rub; 0 = unsalted)",
    )
    ap.add_argument(
        "--tpit0-K",
        type=float,
        default=300.0,
        help="pit temperature at lighting [K] (ambient)",
    )
    ap.add_argument(
        "--t0-meat-K",
        type=float,
        default=281.15,
        help="meat temperature at the start [K]",
    )
    ap.add_argument("--figure", default="mpc_demo.png")
    ap.add_argument("--out", default="mpc_result.json")
    ap.add_argument(
        "--fresh", action="store_true", help="ignore completed-cook checkpoints"
    )
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
            REPO / "figures" / "control" / args.figure,
            t_dist_h=meta["t_dist_h"],
            wrap_mode=meta["wrap_mode"],
        )
        print(f"replotted {args.figure} from {tpath.name}")
        return

    from brisket_app import mpc
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import monolith_client
    from brisket_app.control import ControlConfig, safety_report
    from brisket_app.coupled import CookGrids

    params_path = REPO / "data" / "calibration" / args.params
    m = json.loads(params_path.read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(monolith_client())
    print(f"plant: {params_path.name}", flush=True)

    no_dist = lambda t: T_AMB_BASE
    cc = ControlConfig(opt_wrap=True)

    # the plan under defense (fullplan scan json)
    plan_path = REPO / "data" / "control" / (args.plan or "fullplan_result.json")
    b = json.loads(plan_path.read_text())
    t_dist = planned_disturbance_time(b)
    cold_front = lambda t: T_AMB_COLD if t >= t_dist else T_AMB_BASE
    T_F = float(b["best_t_f_h"] if "best_t_f_h" in b else b["t_f_h"]) * HR
    t_wrap_star = (
        float(b["best_t_wrap_h"] if "best_t_wrap_h" in b else b["t_wrap_h"]) * HR
    )
    u0_plan = np.asarray(b["best_u"] if "best_u" in b else b["u_opt"])
    t_pull_star = b.get("best_t_pull_h")
    T_PULL = float(t_pull_star) * HR if t_pull_star else None
    T_ADD = float(b["best_t_add_h"]) * HR if b.get("best_t_add_h") else None
    T_SPRITZ = float(b["best_t_spritz_h"]) * HR if b.get("best_t_spritz_h") else None
    DT_SPRITZ = (
        float(b["best_dt_spritz_min"]) * 60.0 if b.get("best_dt_spritz_min") else None
    )
    SPRITZ_DOSE = (
        float(b["best_spritz_dose_gm2"]) / 1e3
        if b.get("best_spritz_dose_gm2")
        else None
    )
    print(f"using plan {plan_path.name}")
    print(f"cold front: {t_dist / HR:.2f} h from lighting (2 h after planned meat-on)")
    ics_kw = {"tpit0": args.tpit0_K, "t0_meat": args.t0_meat_K}
    # the plan's load temperature: open-loop loads when its pit reaches it, MPC re-decides it until the meat is on
    T_LOAD = b.get("best_T_load") or b.get("T_load")
    T_LOAD = float(T_LOAD) if T_LOAD else None
    # a chimney after the pull, or a spritz after the wrap or the pull, never fires
    adds = T_ADD is not None and T_ADD < (T_PULL or T_F)
    spritzes = T_SPRITZ is not None and T_SPRITZ < min(t_wrap_star, T_PULL or T_F)
    # every cook pulls (open-loop at the plan's pull, MPC when it decides)
    # and rests to serve at T_F
    grids0 = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=len(u0_plan))
    print(
        f"plan under defense: the {T_F / HR:.0f} h cook — "
        f"wrap at {t_wrap_star / HR:.2f} h"
        + (f", pull at {T_PULL / HR:.2f} h" if T_PULL else "")
        + (f", chimney at {T_ADD / HR:.2f} h" if adds else ", no chimney")
        + (
            f", spritz every {DT_SPRITZ / 60:.0f} min from {T_SPRITZ / HR:.2f} h"
            if spritzes
            else ", no spritz"
        )
        + (f", meat on at a pit of {T_LOAD - 273.15:.0f} C" if T_LOAD else "")
        + f", serve at {T_F / HR:.1f} h from lighting, "
        f"u in [{u0_plan.min():.2f}, {u0_plan.max():.2f}]"
    )
    pull_kw = {
        "t_pull_init_abs": T_PULL,
        "t_add_init_abs": T_ADD,
        "t_spritz_init_abs": T_SPRITZ,
        "dt_spritz_init": DT_SPRITZ,
        "spritz_dose_init": SPRITZ_DOSE,
        "t_salt": args.t_salt_h * HR,
        "T_load_init": T_LOAD,
    }
    digest = hashlib.sha256()
    for path in [
        Path(__file__),
        plan_path,
        params_path,
        *sorted((REPO / "src").rglob("*.jl")),
        *sorted((REPO / "app/src/brisket_app").glob("*.py")),
        REPO / "tesseracts/monolith/tesseract_api.py",
    ]:
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {"pull": pull_kw, "ics": ics_kw, "t_dist": t_dist}, sort_keys=True
        ).encode()
    )
    fingerprint = digest.hexdigest()

    def run_one(tag, *pos, **kw):
        path = REPO / "scratchpad/regen" / Path(args.out).stem / f"{tag}.json"
        return cached_cook(
            path, fingerprint, lambda: mpc.run_cook(*pos, **kw), fresh=args.fresh
        )

    print("reference (no disturbance):")
    ref = run_one(
        "reference",
        theta,
        phi,
        no_dist,
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
        **pull_kw,
        **ics_kw,
    )
    print("open-loop through the cold front:")
    ol = run_one(
        "openloop",
        theta,
        phi,
        cold_front,
        plant_tstops=(t_dist,),
        t_f=T_F,
        mode="openloop",
        u_openloop=u0_plan,
        u_openloop_grids=grids0,
        t_wrap_init_abs=t_wrap_star,
        **pull_kw,
        **ics_kw,
    )
    print("MPC, NO disturbance (discriminator: plan-refinement vs weather):")
    mp0 = run_one(
        "mpc_nominal",
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
        **pull_kw,
        **ics_kw,
    )
    print("MPC through the cold front:")
    mp = run_one(
        "mpc",
        theta,
        phi,
        cold_front,
        plant_tstops=(t_dist,),
        t_f=T_F,
        mode="mpc",
        model_t_amb=T_AMB_BASE,
        cc=cc,
        t_wrap_init_abs=t_wrap_star,
        u_incumbent=u0_plan,
        u_incumbent_grids=grids0,
        **pull_kw,
        **ics_kw,
    )
    if wrap_mode and mp.get("wrapped"):
        tw_exec = mp.get("t_wrap_exec_abs", mp.get("t_wrap_abs"))
        n_cand = len(mp.get("wrap_candidates") or [])
        print(
            f"  crutch executed at {tw_exec / HR:.2f} h "
            f"(replanned {len(mp.get('wrap_history') or [])}x, "
            f"{n_cand} wrap candidates logged)"
        )
    if mp.get("pulled"):
        print(
            f"  pull executed at {mp['t_pull_exec_abs'] / HR:.2f} h "
            f"(plan {T_PULL / HR:.2f} h; {len(mp.get('pull_candidates') or [])} "
            f"pull candidates logged), served at {T_F / HR:.1f} h"
        )

    for name, r in [
        ("reference", ref),
        ("open-loop", ol),
        ("MPC-nominal", mp0),
        ("MPC", mp),
    ]:
        jj = mpc.executed_objective(r, cc, T_F)
        r["J_executed"] = jj
        tp = r.get("t_pull_exec_abs")
        print(
            f"  {name}: J={jj['J']:.3f}  doneness {r['doneness']:.3f} (band "
            f"{cc.c_target}-{cc.c_overdone})  juiciness {r['juiciness']:.3f}  "
            f"L* {r['l_star_end']:.1f}  smoke {r['smoke_end']:.2f}"
            + (f"  pull {tp / HR:.2f} h" if tp is not None else "")
            + (
                f"  chimney {r['t_add_exec_abs'] / HR:.2f} h"
                if _fired_add(r)
                else "  no chimney"
            )
            + (
                f"  spritz from {r['t_spritz_exec_abs'] / HR:.2f} h every "
                f"{(r.get('dt_spritz') or 0.0) / 60:.0f} min"
                if _fired_spritz(r)
                else "  no spritz"
            )
        )
        print(f"    components: {jj}")
        if r.get("t_load_exec_abs") is not None:
            print(
                f"    meat on at {r['t_load_exec_abs'] / HR:.2f} h "
                f"(load temperature {r['T_load'] - 273.15:.0f} C)"
            )
        else:
            print(
                "    the meat never went on (the pit never reached the load temperature)"
            )
        sr = safety_report(
            r["tsurf"], r["tcore"], r["t"], cc, t_load=r.get("t_load_exec_abs")
        )
        print(
            f"    safety: surface F={sr['F_s']:.3g}s come-up={sr['cut_h']:.2f}h; "
            f"core F={sr['F_core']:.3g}s come-up={sr['cut_h_core']:.2f}h; ok={sr['ok']}"
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
                    f"t_load_h_{tag}": (
                        r["t_load_exec_abs"] / HR
                        if r.get("t_load_exec_abs") is not None
                        else None
                    )
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                },
                "T_load_plan": T_LOAD,
                "T_load_mpc": mp.get("T_load"),
                **{
                    f"J_{tag}": r["J_executed"]
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                },
                "t_disturbance_h": t_dist / HR,
                "replan_history_nominal": mp0.get("replan_history", []),
                "replan_history_mpc": mp.get("replan_history", []),
                "dT_amb": T_AMB_BASE - T_AMB_COLD,
                "params": args.params,
                "plan": args.plan or "fullplan_result.json",
                "cook_h": T_F / HR,
                "t_pull_plan_h": T_PULL / HR if T_PULL else None,
                **{
                    f"t_pull_exec_h_{tag}": (
                        r["t_pull_exec_abs"] / HR
                        if r.get("t_pull_exec_abs") is not None
                        else None
                    )
                    for tag, r in [
                        ("reference", ref),
                        ("openloop", ol),
                        ("mpc_nominal", mp0),
                        ("mpc", mp),
                    ]
                },
                "pull_history_h": [
                    [t / HR, p / HR] for t, p in mp.get("pull_history", [])
                ],
                "pull_candidates_h": [
                    [row[0] / HR, row[1] / HR, row[2] / HR, row[3], row[4], row[5]]
                    for row in mp.get("pull_candidates", [])
                ],
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
    save_mpc_traces(traces, ref, ol, mp0, mp, t_dist_h=t_dist / HR, wrap_mode=wrap_mode)
    print(f"wrote {traces.relative_to(REPO)}")
    render_mpc_figure(
        ref,
        ol,
        mp0,
        mp,
        REPO / "figures" / "control" / args.figure,
        t_dist_h=t_dist / HR,
        wrap_mode=wrap_mode,
    )
    print(f"wrote figures/control/{args.figure}")
    print(
        f"replot: uv run python -u ../scripts/run_mpc.py --replot "
        f"--out {args.out} --figure {args.figure}"
    )


if __name__ == "__main__":
    main()
