"""``brisket`` — the calibrate-from-your-log CLI (roadmap item 10).

One entry point over the supported chain:

  brisket logs                         what cook logs the repo knows
  brisket digitize <log.toml>          chart screenshot -> CSV (loghunt)
  brisket calibrate --log keg_2015     per-log GN-MAP + Laplace
  brisket calibrate --csv my.csv --t-amb-k 294 --name mycook
  brisket plan --params fit_mycook.json     the full schedule optimum
  brisket pilot --params fit_mycook.json --pit http://heatermeter ...
  brisket pool --set keg               hierarchical multi-cook pooling
  brisket mock-pit --port 8181         sim plant behind the HM REST API

``calibrate`` writes ``data/calibration/fit_<name>.json`` in the same
flat layout as ``map_real.json`` (site values + AR(1) noise), so
``plan`` and ``pilot`` consume it directly via ``--params``.
``plan``/``pilot``/``pool``/``digitize``/``mock-pit`` forward their
arguments verbatim to the canonical scripts in ``scripts/`` — the CLI
adds no second implementation of anything that already has one.

Your own CSV needs the cook-log column convention (see
tools/loghunt/README.md): ``t_min, pit_F, meat_F, amb_F, fan_pct,
setpoint_F, lid_event``, t = 0 at meat-on.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"

FORWARDED = {
    "digitize": REPO / "tools" / "loghunt" / "digitize.py",
    "plan": SCRIPTS / "run_fullplan.py",
    "pilot": SCRIPTS / "run_pilot.py",
    "pool": SCRIPTS / "run_pool_real.py",
    "mock-pit": SCRIPTS / "mock_heatermeter.py",
}


def _forward(script: Path, argv: list[str]) -> None:
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def _disclaimer() -> None:
    from .control import SAFETY_DISCLAIMER

    print(f"note: {SAFETY_DISCLAIMER}.")


def cmd_logs(_argv: list[str]) -> None:
    from . import cooklog

    print(f"{'name':15s} {'hours':>6s} {'t_amb':>7s}  csv")
    for name, spec in cooklog.LOGS.items():
        try:
            log = cooklog.load(dt_min=10.0, log=name)
            h = f"{log['meta']['t_end_h']:.1f}"
        except Exception as e:  # a missing CSV should not kill the listing
            h = f"({e.__class__.__name__})"
        amb = f"{spec['t_amb_K'] - 273.15:+.0f} C"
        print(f"{name:15s} {h:>6s} {amb:>7s}  {spec['csv'].relative_to(REPO)}")


def cmd_calibrate(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="brisket calibrate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--log", help="a registry log name (see: brisket logs)")
    g.add_argument("--csv", help="your own cook-log CSV (column convention above)")
    ap.add_argument("--name", help="output name (default: the log name)")
    ap.add_argument("--t-amb-k", type=float, default=300.0,
                    help="ambient [K] (your-CSV mode)")
    ap.add_argument("--start-core-k", type=float, default=None,
                    help="slice off a frozen/settling start below this core [K]")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--noise-from", default="map_real.json",
                    help="borrow AR(1) noise scales from this calibration")
    args = ap.parse_args(argv)

    from . import cooklog
    from .monolith import (
        NAMES,
        SITES,
        Whitened,
        effective_sigma,
        from_z,
        levenberg_marquardt,
        model_mean,
        to_z,
    )

    if args.csv:
        name = args.name or Path(args.csv).stem
        cooklog.LOGS[name] = {
            "csv": Path(args.csv).resolve(),
            "t_end_min": None,
            "t_amb_K": args.t_amb_k,
            "start_at_core_K": args.start_core_k,
        }
    else:
        name = args.name or args.log
    log = cooklog.load(dt_min=10.0, log=args.log or name)
    m = log["meta"]
    print(f"{name}: {m['t_end_h']:.1f} h, t_amb {m['t_amb_K'] - 273.15:+.1f} C, "
          f"obs {m['n_obs_pit']}/{m['n_obs_core']}")

    cal = REPO / "data" / "calibration"
    noise = {
        k: float(v)
        for k, v in json.loads((cal / args.noise_from).read_text()).items()
        if k in ("sigma_pit", "sigma_core", "rho_pit", "rho_core")
    }
    print(f"AR(1) noise from {args.noise_from} (shared digitization process)")

    x0 = []
    for _, d in SITES:
        if hasattr(d, "concentration1"):
            a, b = float(d.concentration1), float(d.concentration0)
            x0.append(a / (a + b))
        else:
            x0.append(float(np.exp(d.loc)))
    wh = Whitened.build(log, *effective_sigma(noise))
    t0 = time.perf_counter()
    z, F, H, hist = levenberg_marquardt(
        to_z(np.array(x0)), wh, log, iters=args.iters
    )
    wall = time.perf_counter() - t0
    H = 0.5 * (H + H.T)
    lam = np.linalg.eigvalsh(H)
    if lam.min() <= 0:
        H = H + (1e-4 * max(lam.max(), 1.0) - lam.min()) * np.eye(z.size)
    sd_z = np.sqrt(np.diag(np.linalg.inv(H)))
    x = from_z(z)
    mu, _ = model_mean(x, log, jac=False)
    nt = len(log["t"])
    rms_pit = float(np.sqrt(np.mean(
        (mu[:nt] - log["obs_pit"])[log["mask_pit"]] ** 2)))
    rms_core = float(np.sqrt(np.mean(
        (mu[nt:] - log["obs_core"])[log["mask_core"]] ** 2)))
    print(f"GN-MAP: F={F:.1f} in {len(hist) - 1} iters ({wall:.0f} s); "
          f"rms pit {rms_pit:.2f} K, core {rms_core:.2f} K")

    # map_real-compatible flat layout: plan/pilot consume it via --params
    out = dict(zip(NAMES, map(float, x), strict=True))
    out.update(noise)
    out["_fit"] = {
        "log": name,
        "F": float(F),
        "rms_pit_K": rms_pit,
        "rms_core_K": rms_core,
        "laplace_sd_z": dict(zip(NAMES, map(float, sd_z), strict=True)),
        "t_amb_K": m["t_amb_K"],
        "noise_from": args.noise_from,
    }
    path = cal / f"fit_{name}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(REPO)}")
    _fit_figure(log, mu, name)
    print(f"next:  brisket plan --params fit_{name}.json")


def _fit_figure(log, mu, name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nt = len(log["t"])
    th = log["t"] / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, sl, obs_key, mask_key, label in (
        (axes[0], slice(None, nt), "obs_pit", "mask_pit", "pit"),
        (axes[1], slice(nt, None), "obs_core", "mask_core", "core"),
    ):
        mk = log[mask_key]
        ax.plot(th[mk], np.asarray(log[obs_key])[mk] - 273.15, ".", ms=2.5,
                color="k", alpha=0.5, label="observed")
        ax.plot(th, np.asarray(mu[sl]) - 273.15, color="tab:orange",
                label="model @ MAP")
        ax.set(xlabel="time [h]", ylabel="temperature [C]", title=label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"GN-MAP fit: {name}")
    fig.tight_layout()
    out = REPO / "figures" / "calibration" / f"fit_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = ["logs", "calibrate", *FORWARDED]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands:", ", ".join(cmds))
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "logs":
        cmd_logs(rest)
    elif cmd == "calibrate":
        cmd_calibrate(rest)
    elif cmd in FORWARDED:
        _forward(FORWARDED[cmd], rest)
        if cmd in ("plan", "pilot"):
            _disclaimer()
    else:
        sys.exit(f"unknown command {cmd!r}; try one of: {', '.join(cmds)}")


if __name__ == "__main__":
    main()
