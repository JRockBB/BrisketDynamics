"""Fly a cook: the receding-horizon pilot pointed at a pit (roadmap item 1).

Backends (--pit):
  sim                       in-process SimPit (calibrated plant + PI)
  http://<heatermeter>      a real HeaterMeter's LinkMeter v14 API —
                            wrap/pull become console/journal ADVISORIES
  http://127.0.0.1:8181     scripts/mock_heatermeter.py — the same REST
                            client path, with sim event hooks

The incumbent plan is the fullplan optimum (data/control/
fullplan_result.json); the pilot re-plans schedule + wrap + pull each
--dt-replan from the estimated state, pushes the planned pit trajectory
as SETPOINTS (the pit's own PID does the fan), assimilates measurements
(model advance under measured fan duty; ambient Kalman filter on the pit
innovation, with an optional forecast as feedforward), and journals
every event as JSONL.

Forecast (--forecast): a JSON file [[hour, T_amb_K], ...] interpolated
over the cook, for the wind/weather feedforward; the KF corrects the
residual bias.

Examples (from app/):
    uv run python -u ../scripts/run_pilot.py --pit sim --t-f 8 \
        --front-t 2 --journal ../data/control/pilot_sim.jsonl
    uv run python -u ../scripts/run_pilot.py --pit http://127.0.0.1:8181 \
        --t-f 8 --journal ../data/control/pilot_http.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

HR = 3600.0


def load_incumbent(plan_path: Path):
    from brisket_app.coupled import CookGrids

    b = json.loads(plan_path.read_text())
    t_f = float(b.get("best_t_f_h", b.get("t_f_h"))) * HR
    u = np.asarray(b.get("best_u", b.get("u_opt")))
    inc = {
        "u": u,
        "grids": CookGrids(t_end=t_f, dt_exchange=600.0, n_u=len(u)),
        "t_wrap": float(b.get("best_t_wrap_h", b.get("t_wrap_h"))) * HR,
        "t_pull": float(b["best_t_pull_h"]) * HR if b.get("best_t_pull_h") else None,
        "t_add": float(b["best_t_add_h"]) * HR if b.get("best_t_add_h") else None,
        "t_spritz": (
            float(b["best_t_spritz_h"]) * HR if b.get("best_t_spritz_h") else None
        ),
        "dt_spritz": (
            float(b["best_dt_spritz_min"]) * 60.0
            if b.get("best_dt_spritz_min")
            else None
        ),
        "spritz_dose": (
            float(b["best_spritz_dose_gm2"]) / 1e3
            if b.get("best_spritz_dose_gm2")
            else None
        ),
    }
    return t_f, inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pit", default="sim", help="'sim' or the HeaterMeter base URL")
    ap.add_argument("--apikey", default="")
    ap.add_argument("--units", default="F", choices=("F", "C"))
    ap.add_argument("--params", default="map_real.json")
    ap.add_argument("--plan", default="fullplan_result.json")
    ap.add_argument("--t-f", type=float, default=None,
                    help="cook length [h] (default: the plan's)")
    ap.add_argument("--dt-replan", type=float, default=1.0, help="[h]")
    ap.add_argument("--dt-status", type=float, default=60.0, help="[s]")
    ap.add_argument("--replan-iters", type=int, default=200)
    ap.add_argument("--t-amb", type=float, default=300.0,
                    help="ambient belief seed [K]")
    ap.add_argument("--forecast", default=None,
                    help="JSON file [[hour, T_amb_K], ...] feedforward")
    ap.add_argument("--journal", default=None)
    ap.add_argument("--out", default=None, help="summary JSON path")
    ap.add_argument("--wall-pace", type=float, default=0.0,
                    help="wall s per sim s (0 = free-run; 1 = real time)")
    ap.add_argument("--advisory-only", action="store_true",
                    help="never execute events, even where the backend can")
    # sim-only plant options
    ap.add_argument("--front-t", type=float, default=None,
                    help="sim: cold-front onset [h]")
    ap.add_argument("--front-dk", type=float, default=15.0)
    args = ap.parse_args()

    from brisket_app import mpc
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import monolith_client
    from brisket_app.control import ControlConfig
    from brisket_app.pilot import Journal, Pilot, PilotConfig
    from brisket_app.pit_io import F2K, HeaterMeterClient, MockHMClient, SimPit

    m = json.loads((REPO / "data" / "calibration" / args.params).read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    mpc.set_clients(monolith_client())

    t_f_plan, inc = load_incumbent(REPO / "data" / "control" / args.plan)
    t_f = args.t_f * HR if args.t_f else t_f_plan

    if args.pit == "sim":
        base = args.t_amb

        def t_amb_fn(t):
            if args.front_t is not None and t >= args.front_t * HR:
                return base - args.front_dk
            return base

        pit = SimPit(theta, phi, t_amb_fn, setpoint_K=F2K(225.0))
        print("pit: in-process SimPit"
              + (f" (cold front -{args.front_dk:.0f} K at {args.front_t:.1f} h)"
                 if args.front_t is not None else ""))
    else:
        cls = MockHMClient if "127.0.0.1" in args.pit or "localhost" in args.pit \
            else HeaterMeterClient
        pit = cls(args.pit, apikey=args.apikey, units=args.units)
        v = pit._get("/luci/lm/api/version")
        print(f"pit: {args.pit} (api {v.get('api')}, ucid {v.get('ucid')}, "
              f"{cls.__name__})")

    forecast_fn = None
    if args.forecast:
        fc = np.asarray(json.loads(Path(args.forecast).read_text()), dtype=float)
        forecast_fn = lambda t: float(np.interp(t / HR, fc[:, 0], fc[:, 1]))

    cfg = PilotConfig(
        t_f=t_f,
        dt_replan=args.dt_replan * HR,
        dt_status=args.dt_status,
        replan_iters=args.replan_iters,
        model_t_amb=args.t_amb,
        auto_events=not args.advisory_only,
        wall_pace=args.wall_pace,
    )
    pilot = Pilot(
        pit, theta, phi, cfg,
        cc=ControlConfig(opt_wrap=True, opt_pull=inc["t_pull"] is not None),
        incumbent=inc,
        forecast_fn=forecast_fn,
        journal=Journal(args.journal),
    )
    print(f"cook: {t_f / HR:.1f} h, replan every {args.dt_replan:.1f} h, "
          f"incumbent wrap {inc['t_wrap'] / HR:.2f} h"
          + (f", pull {inc['t_pull'] / HR:.2f} h" if inc["t_pull"] else ""))
    summary = pilot.run()
    print()
    print("cook complete:")
    for k, v in summary.items():
        if isinstance(v, float) and k.startswith("t_") and v > 1e4:
            print(f"  {k}: {v / HR:.2f} h")
        else:
            print(f"  {k}: {v}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
