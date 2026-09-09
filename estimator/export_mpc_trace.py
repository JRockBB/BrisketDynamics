"""Export one variant of the MPC demo's traces (scripts/run_mpc.py's
<out-stem>_traces.npz) as JSON for the Julia plant to re-run: the
executed damper on its grid, the executed events, the serve time, the
Tesseract's own metrics, and the parameter set the cook was flown on.

Run from app/:
  uv run python ../estimator/export_mpc_trace.py --traces mpc_result_traces.npz --variant mpc_nominal --params map_real.json
Writes data/control/<traces-stem>_<variant>_trace.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NAMES = ["D_eff", "k_m", "h_c", "L", "k_mult", "k_e", "a_myosin", "m_conv",
         "C_th", "UA", "r_max", "dH_eff", "u_leak", "k_spread", "a0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="mpc_result_traces.npz")
    ap.add_argument("--variant", default="mpc_nominal")
    ap.add_argument("--params", default="map_real.json")
    args = ap.parse_args()
    z = np.load(REPO / "data" / "control" / args.traces)
    v = args.variant
    scalar = lambda k: (None if not np.isfinite(float(z[v + k])) else float(z[v + k]))
    params = json.loads((REPO / "data" / "calibration" / args.params).read_text())
    t_f = float(z[v + "_t"][-1])
    # the app's rule: a chimney fires only before the pull, a spritz only before the wrap and the pull
    wrap, pull = scalar("_t_wrap_exec_abs"), scalar("_t_pull_exec_abs")
    add = scalar("_t_add_exec_abs")
    add = add if (add is not None and (pull is None or add < pull)) else None
    spritz = scalar("_t_spritz_exec_abs")
    stop = min([x for x in (wrap, pull) if x is not None], default=float("inf"))
    spritz = spritz if (spritz is not None and spritz < stop) else None
    out = {
        "variant": v, "traces": args.traces, "params_file": args.params,
        "params": {n: params[n] for n in NAMES},
        "t_h": (z[v + "_t"] / 3600).tolist(), "u": z[v + "_u"].tolist(),
        "tcore_C": (z[v + "_tcore"] - 273.15).tolist(), "tpit_C": (z[v + "_tpit"] - 273.15).tolist(),
        "events_s": {"wrap": wrap, "pull": pull, "serve": t_f, "add": add, "spritz": spritz,
                     "dt_spritz": scalar("_dt_spritz"), "spritz_dose": scalar("_spritz_dose")},
        "metrics": {"doneness": float(z[v + "_doneness"]), "juiciness": float(z[v + "_juiciness"]),
                    "l_star_end": float(z[v + "_l_star_end"]), "smoke_end": float(z[v + "_smoke_end"])},
    }
    dst = REPO / "data" / "control" / f"{Path(args.traces).stem}_{v}_trace.json"
    dst.write_text(json.dumps(out), encoding="utf-8")
    ev = out["events_s"]
    print(f"wrote {dst.name}: wrap {ev['wrap'] and ev['wrap'] / 3600:.2f} h, pull {ev['pull'] and ev['pull'] / 3600:.2f} h, serve {t_f / 3600:.2f} h, "
          f"{len(out['t_h'])} damper points, doneness {out['metrics']['doneness']:.3f}")


if __name__ == "__main__":
    main()
