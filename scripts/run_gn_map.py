"""Gauss-Newton MAP + Fisher covariance on the real cook from the monolith.

Levenberg-Marquardt over the 15 physics latents in the unconstrained space,
noise parameters (σ, ρ) fixed at the previous MAP's values, from a cold start
at the prior medians; then the Fisher information (Jᵀ Σ⁻¹ J + prior
curvature) at the optimum.

Run from app/:  uv run python -u ../scripts/run_gn_map.py [--warm]
Writes data/calibration/map_real.json (the MAP every downstream script
reads), data/calibration/gn_map_real.json (fit diagnostics, Fisher widths)
and figures/calibration/cooklog_fit_map.png.

The real cook is fitted from lighting (its documented ~90-min preheat): the
latents are the log's (`monolith.sites_for`: k_spread and the preheat
damper u_pre, a0 fixed at the chimney start's A_LIGHT); the MAP json
carries a0 = A_LIGHT so every φ consumer gets a cook from lighting.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import cooklog
from brisket_app.coupled import A_LIGHT
from brisket_app.monolith import (
    NOISE_NAMES,
    Whitened,
    effective_sigma,
    from_lighting,
    from_z,
    levenberg_marquardt,
    model_mean,
    names_for,
    preheat_trace,
    sites_for,
    to_z,
)

REPO = Path(__file__).resolve().parents[1]


def prior_median_z(sites):
    """Cold start: prior medians (LogNormal) / means (Beta), in z."""
    x = []
    for _, d in sites:
        if hasattr(d, "concentration1"):
            a, b = float(d.concentration1), float(d.concentration0)
            x.append(a / (a + b))
        else:
            x.append(float(np.exp(d.loc)))
    return to_z(np.array(x), sites)


def main():
    """Run LM, report the Fisher widths, write the MAP and the fit figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map",
        default="map_real.json",
        help="previous MAP json (noise params; warm start); overwritten",
    )
    ap.add_argument(
        "--warm",
        action="store_true",
        help="start from the previous MAP instead of the priors",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a warm-start value (constrained), e.g. k_spread=1e-3; repeatable",
    )
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    m = json.loads((cal / args.map).read_text())
    log = cooklog.load(dt_min=10.0)
    sites, NAMES = sites_for(log), names_for(log)
    lit = from_lighting(log)
    if lit:
        print(f"from lighting: {log['meta']['preheat_s'] / 60:.0f} min preheat, a0 = {A_LIGHT} (chimney start), "
              "the preheat damper u_pre a latent", flush=True)
    wh = Whitened.build(log, *effective_sigma(m))
    print(
        f"cook log: {len(log['t'])} points, {wh.L.shape[0]} whitened residuals",
        flush=True,
    )

    if args.warm:
        # the previous MAP's values where it has them (a log refitted from lighting starts u_pre at its prior mean)
        z_prior = prior_median_z(sites)
        z0 = np.array([to_z(np.array([m[n]]), [s])[0] if n in m else z_prior[i]
                       for i, (n, s) in enumerate(zip(NAMES, sites, strict=True))])
    else:
        z0 = prior_median_z(sites)
    for item in args.set:  # a start in the right basin (a 1-D scan of the come-up's latents)
        k, v = item.split("=")
        i = NAMES.index(k)
        z0[i] = to_z(np.array([float(v)]), [sites[i]])[0]
        print(f"start: {k} = {float(v):.4g}", flush=True)
    print(
        "LM from",
        "the previous MAP" if args.warm else "the prior medians (cold start)",
        flush=True,
    )
    t0 = time.perf_counter()
    z, F, H, hist = levenberg_marquardt(z0, wh, log, iters=args.iters)
    wall = time.perf_counter() - t0
    x = from_z(z, sites)
    print(f"LM done: {len(hist) - 1} iterations, {wall:.0f} s, F={F:.3f}", flush=True)

    cov = np.linalg.inv(H)
    sd = np.sqrt(np.diag(cov))
    print(f"{'site':10s} {'GN-MAP':>11s} {'sd(z) Fisher':>13s}", flush=True)
    for i, n in enumerate(NAMES):
        print(f"{n:10s} {x[i]:11.4g} {sd[i]:13.3f}")

    out = {
        "x": dict(zip(NAMES, map(float, x), strict=True)),
        "z": dict(zip(NAMES, map(float, z), strict=True)),
        "F": F,
        "iterations": len(hist) - 1,
        "wall_s": wall,
        "history": hist,
        "sd_z_fisher": dict(zip(NAMES, map(float, sd), strict=True)),
        "cold_start": not args.warm,
        "noise_from": args.map,
        "from_lighting": lit,
        "preheat_s": float(log["meta"].get("preheat_s", 0.0)),
        "a0_fixed": A_LIGHT if lit else None,
    }
    (cal / "gn_map_real.json").write_text(json.dumps(out, indent=2))
    print("wrote data/calibration/gn_map_real.json")

    params = dict(zip(NAMES, map(float, x), strict=True))
    if lit:
        params["a0"] = A_LIGHT  # the φ layout's lit fraction: a cook from lighting
    params.update({k: float(m[k]) for k in NOISE_NAMES})
    (cal / args.map).write_text(json.dumps(params, indent=2))
    print(f"wrote data/calibration/{args.map}")

    mu, _ = model_mean(x, log, jac=False)
    nt = len(log["t"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_pre = float(log["meta"].get("preheat_s", 0.0))
    th = (log["t"] - log["t"][0] + t_pre) / 3600.0                      # hours from lighting (or the log start)
    pre = preheat_trace(x, log) if lit else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, s, o, mk, name in [
        (axes[0], mu[:nt], log["obs_pit"], log["mask_pit"], "pit"),
        (axes[1], mu[nt:], log["obs_core"], log["mask_core"], "core"),
    ]:
        if pre is not None:
            k = pre["t"] <= t_pre
            ax.plot(pre["t"][k] / 3600.0, pre[f"t{name}"][k] - 273.15, ":", color="tab:orange",
                    label="model, preheat (not logged)")
            ax.axvline(t_pre / 3600.0, color="0.5", lw=1.0, ls="-.")
        ax.plot(
            th[mk],
            o[mk] - 273.15,
            ".",
            ms=2.5,
            color="k",
            alpha=0.5,
            label="observed (Mache/TVWBB)",
        )
        ax.plot(th, s - 273.15, color="tab:orange", label="model @ MAP")
        ax.set(xlabel="time from lighting [h]" if lit else "time [h]", ylabel="temperature [C]",
               title=name + (" (meat on at the dash-dot line)" if lit else ""))
        ax.legend(fontsize=8)
    fig.suptitle("Real cook - Gauss-Newton MAP fit")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "calibration" / "cooklog_fit_map.png", dpi=140)
    print("wrote figures/calibration/cooklog_fit_map.png")


if __name__ == "__main__":
    main()
