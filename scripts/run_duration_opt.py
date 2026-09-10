"""Differentiable duration: the cook length as a gradient decision.

The roadmap's s = t/t_f item: instead of the outer scan over cook lengths
(13 inner optimizations, ~95 min), optimize (t_f, wrap fraction, pull
fraction, damper schedule) JOINTLY by gradient — d(quality at serve)/d(t_f)
flows through the monolith's fractional-time mode, horizon-stretching terms
included. Warm-started from the scan optimum by default (--cold starts at
mid-range defaults); scored by the same `score_outputs` the scan used.

Run from app/:  uv run python -u ../scripts/run_duration_opt.py
Writes data/control/duration_opt.json and figures/control/duration_opt.png.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
import jax.numpy as jnp
import optax
from brisket_app.calibrate import THETA_SITES, phi_full
from brisket_app.clients import monolith_client
from brisket_app.control import ControlConfig, objective_tf
from brisket_app.coupled import HR

REPO = Path(__file__).resolve().parents[1]
HOURS = HR
# the same cook conditions fullplan plans under (run_fullplan.py ICS)
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}


def _logit(f: float) -> float:
    f = min(max(f, 1e-4), 1 - 1e-4)
    return float(np.log(f / (1 - f)))


def main():
    """Optimize (duration, wrap, pull, schedule) jointly by gradient."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-u", type=int, default=16, help="fractional damper knots")
    ap.add_argument("--n-save", type=int, default=67, help="fractional save points")
    ap.add_argument("--t-lo", type=float, default=8.0, help="duration lower bound [h]")
    ap.add_argument("--t-hi", type=float, default=20.0, help="duration upper bound [h]")
    ap.add_argument(
        "--cold",
        action="store_true",
        help="start at mid-range defaults instead of the scan optimum",
    )
    ap.add_argument("--seed-json", default="fullplan_result.json")
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    ctl = REPO / "data" / "control"

    m = json.loads((cal / "map_real.json").read_text())
    theta = jnp.asarray([m[n] for n, _ in THETA_SITES])
    phi = jnp.asarray(phi_full(m))
    tx = monolith_client()
    cc = ControlConfig(opt_wrap=True, opt_pull=True, opt_add=False, opt_spritz=False)
    tols = {"abstol": 1e-6, "reltol": 1e-6}
    t_lo, t_hi = args.t_lo * HOURS, args.t_hi * HOURS
    s_u = np.linspace(0.0, 1.0, args.n_u)
    s_save = np.linspace(0.0, 1.0, args.n_save)

    # warm start from the scan optimum (schedule interpolated to fractions)
    seed = None
    if not args.cold and (ctl / args.seed_json).exists():
        seed = json.loads((ctl / args.seed_json).read_text())
    if seed is not None:
        t_f0 = seed["best_t_f_h"] * HOURS
        fw0 = seed["best_t_wrap_h"] / seed["best_t_f_h"]
        fp0 = seed["best_t_pull_h"] / seed["best_t_f_h"]
        u_seed = np.asarray(seed["best_u"], dtype=float)
        s_seed = np.linspace(0.0, 1.0, len(u_seed))
        u0 = np.clip(np.interp(s_u, s_seed, u_seed), 1e-3, 1 - 1e-3)
        print(
            f"warm start from the scan: t_f={seed['best_t_f_h']:.1f} h, "
            f"wrap {fw0:.3f}, pull {fp0:.3f}",
            flush=True,
        )
    else:
        t_f0, fw0, fp0 = 12.0 * HOURS, 0.6, 0.75
        u0 = np.full(args.n_u, 0.5)
        print("cold start: t_f=12 h, wrap 0.6, pull 0.75, u=0.5", flush=True)
    z_tf0 = _logit((t_f0 - t_lo) / (t_hi - t_lo))
    params = {
        "z": jnp.asarray(np.log(u0 / (1 - u0))),
        "z_w": jnp.asarray(_logit(fw0)),
        "z_p": jnp.asarray(_logit(fp0)),
        "z_tf": jnp.asarray(z_tf0),
    }

    def loss(p):
        return objective_tf(
            p["z"], p["z_w"], p["z_p"], p["z_tf"], theta, phi, tx, s_u, s_save, cc,
            tols, t_lo, t_hi, ics=ICS,
        )

    vg = jax.value_and_grad(loss)
    opt = optax.adam(args.lr)
    state = opt.init(params)
    hist = []
    t0 = time.perf_counter()
    best = None
    for i in range(args.n_iter):
        val, g = vg(params)
        val = float(val)
        t_f_h = float(
            (t_lo + (t_hi - t_lo) * jax.nn.sigmoid(params["z_tf"])) / HOURS
        )
        hist.append({"iter": i, "J": val, "t_f_h": t_f_h})
        if best is None or val < best["J"]:
            best = {"J": val, "t_f_h": t_f_h, "params": params}
        if i % 10 == 0 or i == args.n_iter - 1:
            print(
                f"  it {i:3d}  J={val:.4f}  t_f={t_f_h:.2f} h  "
                f"({(time.perf_counter() - t0) / 60:.1f} min)",
                flush=True,
            )
        # plateau stop: no meaningful improvement over the last 30 iters
        if len(hist) > 40:
            recent = min(h["J"] for h in hist[-30:])
            earlier = min(h["J"] for h in hist[:-30])
            if earlier - recent < 1e-4:
                print(f"  plateau stop at iter {i}", flush=True)
                break
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)

    p = best["params"]
    t_f = float(t_lo + (t_hi - t_lo) * jax.nn.sigmoid(p["z_tf"]))
    fw = float(jax.nn.sigmoid(p["z_w"]))
    fp = float(jax.nn.sigmoid(p["z_p"]))
    u = np.asarray(jax.nn.sigmoid(p["z"]))
    wall = time.perf_counter() - t0
    print(
        f"joint optimum: J={best['J']:.4f}  cook {t_f / HOURS:.2f} h  "
        f"wrap {fw * t_f / HOURS:.2f} h  pull {fp * t_f / HOURS:.2f} h  "
        f"rest {(1 - fp) * t_f / HOURS:.2f} h  ({wall / 60:.1f} min, "
        f"{len(hist)} iters)",
        flush=True,
    )
    scan = None
    if seed is not None:
        scan = {
            "durations_h": seed["durations_h"],
            "j_quality": seed["j_quality"],
            "best_t_f_h": seed["best_t_f_h"],
            "best_j": float(np.interp(
                seed["best_t_f_h"], seed["durations_h"], seed["j_quality"]
            )),
        }
        print(
            f"scan reference: min J {min(seed['j_quality']):.4f} on the grid; "
            f"chosen {seed['best_t_f_h']:.1f} h",
            flush=True,
        )

    out = {
        "J": best["J"],
        "t_f_h": t_f / HOURS,
        "t_wrap_h": fw * t_f / HOURS,
        "t_pull_h": fp * t_f / HOURS,
        "frac_wrap": fw,
        "frac_pull": fp,
        "u": u.tolist(),
        "s_u": s_u.tolist(),
        "n_iter": len(hist),
        "wall_s": wall,
        "lr": args.lr,
        "t_lo_h": args.t_lo,
        "t_hi_h": args.t_hi,
        "cold": bool(args.cold),
        "history": hist,
        "scan": scan,
    }
    (ctl / "duration_opt.json").write_text(json.dumps(out, indent=2))
    print("wrote data/control/duration_opt.json")
    _figure(out, REPO)


def _figure(out: dict, repo: Path):
    """Scan curve J(t_f) with the gradient path overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if out.get("scan"):
        ax.plot(
            out["scan"]["durations_h"],
            out["scan"]["j_quality"],
            "o-",
            color="0.6",
            ms=4,
            lw=1.2,
            label="outer scan (13 inner optimizations, ~95 min)",
        )
    th = [h["t_f_h"] for h in out["history"]]
    js = [h["J"] for h in out["history"]]
    ax.plot(th, js, "-", color="tab:blue", lw=1.0, alpha=0.7,
            label="gradient path (t$_f$, wrap, pull, schedule jointly)")
    # the re-interpolated seed starts high; clip the axis to the region that
    # matters and let the descent enter from the top edge (the start marker
    # would be invisible, so the annotation carries it)
    j_all = list(out["scan"]["j_quality"]) if out.get("scan") else js
    hi = max(j_all) + 0.35
    if js[0] <= hi:
        ax.plot(th[0], js[0], "s", color="tab:blue", ms=6, mfc="none",
                label="gradient start")
    else:
        ax.set_ylim(min(min(js), min(j_all)) - 0.04, hi)
        ax.annotate(
            f"start J={js[0]:.2f} (seed re-interpolated to fractional knots)",
            xy=(th[0], hi),
            xytext=(th[0] + 0.4, hi - 0.12),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )
    ax.plot(
        out["t_f_h"],
        out["J"],
        "*",
        color="tab:orange",
        ms=14,
        label=f"joint gradient optimum ({out['wall_s'] / 60:.0f} min)",
    )
    ax.set(
        xlabel="cook length, start to serve [h]",
        ylabel="quality objective J",
        title="Duration as a differentiable decision (s = t/t$_f$)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(repo / "figures" / "control" / "duration_opt.png", dpi=140)
    plt.close(fig)
    print("wrote figures/control/duration_opt.png")


if __name__ == "__main__":
    main()
