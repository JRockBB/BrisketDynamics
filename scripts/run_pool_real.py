"""Real multi-cook pooling runs (two-stage: per-log GN-MAP + Laplace -> pool).

Two supported sets:

--set default   Mache 2013 x Phillip 2018 — two briskets, two DIFFERENT
                smokers, an 84 F ambient spread. MEAT partially pooled
                ("briskets I buy"), pit + ignition per-cook, nothing shared.

--set keg       Dale Ward's Big Steel Keg 2013/2014/2015 — the SAME vessel
                across three cooks (roadmap 5a): overnight 225 F brisket,
                hot-and-fast 325 F pork butt, and an overnight 225 F butt in
                the blower era. The vessel's thermal identity (C_th, UA,
                dH_eff) is fully shared; the actuator map (r_max, u_leak)
                stays per-cook (the airflow hardware evolved across the
                posts), and the meats are different cuts, so meat + ignition
                are per-cook too (butts on brisket priors are a stated
                placeholder).

--set wsm       mache_2013 x wbegg_2014 — two DIFFERENT UNITS of the
                same smoker model (WSM 18.5), brisket vs pork butt, a
                ~50 F ambient spread. The vessel identity is PARTIALLY
                pooled between units (same design, unit-to-unit spread);
                everything else per-cook.

Caveats carried explicitly: phillip_2018 is a partial log; keg_2014 is a
short hot-and-fast log; interiors start uniform at the first unmasked core
sample; all non-Mache logs borrow the Mache MAP's AR(1) noise scales (same
digitization process).

Run from app/:  uv run python -u ../scripts/run_pool_real.py [--set keg]
Writes data/calibration/pool_<set>.json and figures/calibration/pool_<set>.png
(the default set keeps its original pool_real.* names).
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

VESSEL = ["C_th", "UA", "dH_eff"]  # thermal identity of the cooker itself
ACTUATOR = ["r_max", "u_leak"]  # airflow map (hardware changed between posts)

SETS = {
    "default": {
        "logs": ("mache_2013", "phillip_2018"),
        "shared": [],
        "partial": MEAT_PARTIAL,
        "independent": PIT_SHARED + IGNITION_IID,
        "stem": "pool_real",
        "title": "First real two-cook pooling set: per-log Gauss-Newton MAP fits",
    },
    "wsm": {
        "logs": ("mache_2013", "wbegg_2014"),
        "shared": [],
        "partial": VESSEL,
        "independent": ACTUATOR + IGNITION_IID + MEAT_PARTIAL,
        "stem": "pool_wsm",
        "title": "Same-model pooling tier (WSM 18.5, two units): per-log MAP fits",
    },
    "keg": {
        "logs": ("keg_2013", "keg_2014", "keg_2015"),
        "shared": VESSEL,
        "partial": [],
        "independent": ACTUATOR + IGNITION_IID + MEAT_PARTIAL,
        "stem": "pool_keg",
        "title": "Same-smoker pooling set (Big Steel Keg '13/'14/'15): per-log MAP fits",
    },
}


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
    """Fit each log in the set, pool per its roles, report and plot."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", default="default", choices=SETS)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--tau2", default="0.04")
    args = ap.parse_args()
    spec = SETS[args.set_name]
    log_names = spec["logs"]
    cal = REPO / "data" / "calibration"
    noise = {
        k: float(v)
        for k, v in json.loads((cal / "map_real.json").read_text()).items()
        if k in ("sigma_pit", "sigma_core", "rho_pit", "rho_core")
    }
    print(f"AR(1) noise from map_real.json: {noise} (shared digitization process)")

    fits, means, covs = {}, [], []
    for name in log_names:
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

    pool = hierarchical_pool(
        means,
        covs,
        names=list(ALL_NAMES_SORTED),
        shared=spec["shared"],
        partial=spec["partial"],
        independent=spec["independent"],
        tau2="auto" if args.tau2 == "auto" else float(args.tau2),
    )
    short = [n.split("_")[-1] if len(set(n.split("_")[0] for n in log_names)) == 1
             else n.split("_")[0] for n in log_names]

    shared_rows = {}
    if spec["shared"]:
        print("\nshared block: one vessel, multiple operating points -> pooled identity")
        head = "".join(f"{s:>12s}" for s in short)
        print(f"{'site':10s}{head} {'pooled':>12s} {'sd(z)':>8s}")
        for i, n in enumerate(spec["shared"]):
            vals = [fits[k]["x"][n] for k in log_names]
            pooled = _x_from_z1(n, float(pool.shared_mean[i]))
            sd = float(np.sqrt(pool.shared_cov[i, i]))
            shared_rows[n] = {"map": vals, "pooled": pooled, "sd_z": sd}
            row = "".join(f"{v:12.4g}" for v in vals)
            print(f"{n:10s}{row} {pooled:12.4g} {sd:8.3f}")

    meat_rows = {}
    if spec["partial"]:
        print("\npartial block: per-cook MAP -> partial-pool shrinkage -> hypermean")
        head = "".join(f"{s:>12s}" for s in short) + "".join(
            f"{s + '*':>12s}" for s in short
        )
        print(f"{'site':10s}{head} {'hyper':>12s}")
        for i, n in enumerate(spec["partial"]):
            vals = [fits[k]["x"][n] for k in log_names]
            shr = [
                _x_from_z1(n, float(pool.cook_partial_mean[j][i]))
                for j in range(len(log_names))
            ]
            hyper = _x_from_z1(n, float(pool.partial_hyper_mean[i]))
            meat_rows[n] = {
                "map": vals,
                "shrunk": shr,
                "hyper": hyper,
                "hyper_sd_z": float(np.sqrt(pool.partial_hyper_cov[i, i])),
            }
            row = "".join(f"{v:12.4g}" for v in vals) + "".join(
                f"{v:12.4g}" for v in shr
            )
            print(f"{n:10s}{row} {hyper:12.4g}")

    label = "per cook" + (
        " (different smokers, nothing shared)" if args.set_name == "default"
        else " (actuator + meat + ignition)"
    )
    print(f"\nindependent block {label}")
    print(f"{'site':10s}" + "".join(f"{s:>12s}" for s in short))
    for n in spec["independent"]:
        print(f"{n:10s}" + "".join(f"{fits[k]['x'][n]:12.4g}" for k in log_names))

    out = {
        "set": args.set_name,
        "logs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("mu_")}
                 for k, v in fits.items()},
        "shared_pool": shared_rows,
        "meat_pool": meat_rows,
        "tau2": args.tau2,
        "noise_from": "map_real.json",
        "roles": {
            "shared": spec["shared"],
            "partial": spec["partial"],
            "independent": spec["independent"],
        },
    }
    (cal / f"{spec['stem']}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote data/calibration/{spec['stem']}.json")
    _figure(fits, log_names, spec, REPO)


def _figure(fits: dict, log_names, spec, repo: Path):
    """Per-log MAP fits: pit and core against the observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr = len(log_names)
    fig, axes = plt.subplots(nr, 2, figsize=(12.5, 3.6 * nr), squeeze=False)
    for row, name in enumerate(log_names):
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
    fig.suptitle(spec["title"], y=0.995)
    fig.tight_layout()
    fig.savefig(repo / "figures" / "calibration" / f"{spec['stem']}.png", dpi=140)
    plt.close(fig)
    print(f"wrote figures/calibration/{spec['stem']}.png")


if __name__ == "__main__":
    main()
