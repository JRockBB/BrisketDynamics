"""First REAL two-cook pooling: Mache 2013 x Phillip 2018.

Two briskets, two smokers, an 84 F ambient spread (summer WSM vs a 1 F
winter night). Per-log Gauss-Newton MAP + Laplace on the monolith, then
the two-stage pool: MEAT partially pooled (both packers — "briskets I
buy"), PIT and ignition per-cook (different smokers share nothing).

Caveats carried explicitly: phillip_2018 is a partial log (rise + stall
entry, no pull) sliced to start above the phase-change regime (the model
has no thaw enthalpy), its interior is assumed uniform at the slice-start
core temperature, and both logs borrow the Mache MAP's AR(1) noise scales
(same digitization process).

Run from app/:  uv run python -u ../scripts/run_pool_real.py
Writes data/calibration/pool_real.json and figures/pool_real.png.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app import cooklog
from brisket_app.approx import IGNITION_IID, MEAT_PARTIAL, PIT_SHARED, hierarchical_pool
from brisket_app.monolith import (
    ALL_NAMES_SORTED,
    NAMES,
    SITES,
    Whitened,
    _is_beta,
    effective_sigma,
    from_z,
    levenberg_marquardt,
    model_mean,
    to_z,
)

REPO = Path(__file__).resolve().parents[1]
LOG_NAMES = ("mache_2013", "phillip_2018")


def _prior_median_z():
    x = []
    for _, d in SITES:
        if hasattr(d, "concentration1"):
            a, b = float(d.concentration1), float(d.concentration0)
            x.append(a / (a + b))
        else:
            x.append(float(np.exp(d.loc)))
    return to_z(np.array(x))


def _embed_physics(z_phys, cov_phys):
    d = len(ALL_NAMES_SORTED)
    mean, cov = np.zeros(d), np.eye(d) * 0.05
    for i, ni in enumerate(NAMES):
        mean[ALL_NAMES_SORTED.index(ni)] = z_phys[i]
        for k, nk in enumerate(NAMES):
            cov[ALL_NAMES_SORTED.index(ni), ALL_NAMES_SORTED.index(nk)] = cov_phys[i, k]
    return mean, cov


def _x_from_z1(name, z1):
    j = NAMES.index(name)
    return 1 / (1 + np.exp(-z1)) if _is_beta(SITES[j][1]) else float(np.exp(z1))


def main():
    """Fit each log, pool the meat block, report and plot."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--tau2", default="0.04")
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    noise = {
        k: float(v)
        for k, v in json.loads((cal / "map_real.json").read_text()).items()
        if k in ("sigma_pit", "sigma_core", "rho_pit", "rho_core")
    }
    print(f"AR(1) noise from map_real.json: {noise} (shared digitization process)")

    fits, means, covs = {}, [], []
    for name in LOG_NAMES:
        log = cooklog.load(dt_min=10.0, log=name)
        m = log["meta"]
        print(
            f"\n=== {name}: {m['t_end_h']:.1f} h, t_amb {m['t_amb_K'] - 273.15:+.1f} C, "
            f"obs {m['n_obs_pit']}/{m['n_obs_core']}",
            flush=True,
        )
        wh = Whitened.build(log, *effective_sigma(noise))
        t0 = time.perf_counter()
        z, F, H, hist = levenberg_marquardt(
            _prior_median_z(), wh, log, iters=args.iters
        )
        wall = time.perf_counter() - t0
        H = 0.5 * (H + H.T)
        lam = np.linalg.eigvalsh(H)
        if lam.min() <= 0:
            H = H + (1e-4 * max(lam.max(), 1.0) - lam.min()) * np.eye(z.size)
        x = from_z(z)
        mu, _ = model_mean(x, log, jac=False)
        nt = len(log["t"])
        rms_pit = float(
            np.sqrt(np.mean((mu[:nt] - log["obs_pit"])[log["mask_pit"]] ** 2))
        )
        rms_core = float(
            np.sqrt(np.mean((mu[nt:] - log["obs_core"])[log["mask_core"]] ** 2))
        )
        print(
            f"LM: F={F:.1f} in {len(hist) - 1} iters ({wall:.0f} s); "
            f"rms pit {rms_pit:.2f} K, core {rms_core:.2f} K"
        )
        me, co = _embed_physics(z, np.linalg.inv(H))
        means.append(me)
        covs.append(co)
        fits[name] = {
            "x": dict(zip(NAMES, map(float, x), strict=True)),
            "F": float(F),
            "iters": len(hist) - 1,
            "wall_s": wall,
            "rms_pit_K": rms_pit,
            "rms_core_K": rms_core,
            "t_amb_K": m["t_amb_K"],
            "t_end_h": m["t_end_h"],
            "mu_pit": mu[:nt].tolist(),
            "mu_core": mu[nt:].tolist(),
        }

    # different smokers: nothing shared; meat partially pooled; pit + ignition
    # recorded per cook
    pool = hierarchical_pool(
        means,
        covs,
        names=list(ALL_NAMES_SORTED),
        shared=[],
        partial=MEAT_PARTIAL,
        independent=PIT_SHARED + IGNITION_IID,
        tau2="auto" if args.tau2 == "auto" else float(args.tau2),
    )
    print("\nmeat block: per-cook MAP -> partial-pool shrinkage -> hypermean")
    print(f"{'site':10s} {'mache':>12s} {'phillip':>12s} {'mache*':>12s} {'phillip*':>12s} {'hyper':>12s}")
    meat_rows = {}
    for i, n in enumerate(MEAT_PARTIAL):
        vals = [fits[k]["x"][n] for k in LOG_NAMES]
        shr = [_x_from_z1(n, float(pool.cook_partial_mean[j][i])) for j in range(2)]
        hyper = _x_from_z1(n, float(pool.partial_hyper_mean[i]))
        meat_rows[n] = {
            "map": vals,
            "shrunk": shr,
            "hyper": hyper,
            "hyper_sd_z": float(np.sqrt(pool.partial_hyper_cov[i, i])),
        }
        print(
            f"{n:10s} {vals[0]:12.4g} {vals[1]:12.4g} {shr[0]:12.4g} "
            f"{shr[1]:12.4g} {hyper:12.4g}"
        )
    print("\npit block (per cook; different smokers, nothing shared)")
    print(f"{'site':10s} {'mache':>12s} {'phillip':>12s}")
    for n in PIT_SHARED:
        print(
            f"{n:10s} {fits['mache_2013']['x'][n]:12.4g} "
            f"{fits['phillip_2018']['x'][n]:12.4g}"
        )

    out = {
        "logs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("mu_")}
                 for k, v in fits.items()},
        "meat_pool": meat_rows,
        "tau2": args.tau2,
        "noise_from": "map_real.json",
        "roles": {
            "shared": [],
            "partial": MEAT_PARTIAL,
            "independent": PIT_SHARED + IGNITION_IID,
        },
    }
    (cal / "pool_real.json").write_text(json.dumps(out, indent=2))
    print("\nwrote data/calibration/pool_real.json")
    _figure(fits, REPO)


def _figure(fits: dict, repo: Path):
    """Per-log MAP fits: pit and core against the observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    for row, name in enumerate(LOG_NAMES):
        log = cooklog.load(dt_min=10.0, log=name)
        th = log["t"] / 3600.0
        for col, (mu_key, obs_key, mask_key, label) in enumerate(
            [
                ("mu_pit", "obs_pit", "mask_pit", "pit"),
                ("mu_core", "obs_core", "mask_core", "core"),
            ]
        ):
            ax = axes[row, col]
            mk = log[mask_key]
            ax.plot(
                th[mk], np.asarray(log[obs_key])[mk] - 273.15,
                ".", ms=2.5, color="k", alpha=0.5, label="observed",
            )
            ax.plot(
                th, np.asarray(fits[name][mu_key]) - 273.15,
                color="tab:orange", label="model @ per-log MAP",
            )
            amb = fits[name]["t_amb_K"] - 273.15
            ax.set(
                xlabel="time [h]", ylabel="temperature [C]",
                title=f"{name} — {label} (ambient {amb:+.0f} C)",
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(
        "First real two-cook pooling set: per-log Gauss-Newton MAP fits", y=0.995
    )
    fig.tight_layout()
    fig.savefig(repo / "figures" / "pool_real.png", dpi=140)
    plt.close(fig)
    print("wrote figures/pool_real.png")


if __name__ == "__main__":
    main()
