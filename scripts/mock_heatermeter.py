"""Mock HeaterMeter: the calibrated sim plant behind the LinkMeter v14 API.

Serves the same endpoints a real HeaterMeter does, backed by
brisket_app.pit_io.SimPit, so the pilot's REAL HTTP client path is
exercised with no hardware:

  GET  /luci/lm/api/status   -> v14 status JSON (Fahrenheit)
  GET  /luci/lm/api/version  -> {"api": 1, "ucid": "mock"}
  POST /luci/lm/api/config   -> form-encoded; honors ``sp`` (setpoint, F)
                                and requires ``apikey`` when one is set

plus sim-only extensions (a real pit has a human instead):

  POST /sim/wrap  /sim/pull  /sim/add  /sim/spritz
                              -> execute the modeled event now
  GET  /sim/traj             -> full plant trajectories (analysis)

The plant free-runs in lockstep: each status GET advances one dt_step.
A cold-front disturbance the controller does not know is available via
--front-t/--front-dk (defaults mirror scripts/run_mpc.py's scenario).

Run from app/:
    uv run python -u ../scripts/mock_heatermeter.py --port 8181
"""

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

HR = 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--apikey", default="")
    ap.add_argument("--dt-step", type=float, default=120.0)
    ap.add_argument("--t-amb", type=float, default=300.0, help="base ambient [K]")
    ap.add_argument("--front-t", type=float, default=None,
                    help="cold-front onset [h] (None = no front)")
    ap.add_argument("--front-dk", type=float, default=15.0,
                    help="cold-front ambient drop [K]")
    ap.add_argument("--setpoint-f", type=float, default=225.0)
    args = ap.parse_args()

    from brisket_app import mpc
    from brisket_app.clients import monolith_client
    from brisket_app.pit_io import F2K, SimPit

    mpc.set_clients(monolith_client())
    cal = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = [cal[k] for k in
             ("D_eff", "k_m", "h_c", "L", "k_mult", "k_e", "a_myosin", "m_conv")]
    phi = [cal[k] for k in
           ("C_th", "UA", "r_max", "dH_eff", "u_leak", "k_spread", "a0")]

    def t_amb(t):
        if args.front_t is not None and t >= args.front_t * HR:
            return args.t_amb - args.front_dk
        return args.t_amb

    pit = SimPit(
        theta, phi, t_amb,
        setpoint_K=F2K(args.setpoint_f),
        dt_step=args.dt_step,
    )
    pit._step_once()  # warm the Julia/XLA path so the first GET is fast
    print(f"mock HeaterMeter on :{args.port} (dt_step {args.dt_step:.0f} s, "
          f"ambient {args.t_amb:.0f} K"
          + (f", cold front -{args.front_dk:.0f} K at {args.front_t:.1f} h"
             if args.front_t is not None else "") + ")", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/luci/lm/api/status"):
                self._send(200, json.dumps(pit.status_dict()))
            elif self.path.startswith("/luci/lm/api/version"):
                self._send(200, json.dumps({"api": 1, "ucid": "mock"}))
            elif self.path.startswith("/sim/traj"):
                self._send(200, json.dumps(pit.traj))
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            form = urllib.parse.parse_qs(self.rfile.read(n).decode())
            if args.apikey and form.get("apikey", [""])[0] != args.apikey:
                self._send(403, "bad apikey", "text/plain")
                return
            if self.path.startswith("/luci/lm/api/config"):
                if "sp" in form:
                    pit.set_setpoint(F2K(float(form["sp"][0])))
                self._send(200, "OK", "text/plain")
            elif self.path.startswith("/sim/wrap"):
                pit.t_wrap_exec = pit.state.t
                self._send(200, "wrapped", "text/plain")
            elif self.path.startswith("/sim/pull"):
                pit.t_pull_exec = pit.state.t
                self._send(200, "pulled", "text/plain")
            elif self.path.startswith("/sim/add"):
                pit.execute_add(pit.state.t)
                self._send(200, "fuel added", "text/plain")
            elif self.path.startswith("/sim/spritz"):
                pit.execute_spritz(
                    pit.state.t,
                    dt_spritz=float(form.get("dt_spritz", [3600.0])[0]),
                    spritz_dose=float(form.get("spritz_dose", [0.05])[0]),
                )
                self._send(200, "spritzing", "text/plain")
            else:
                self._send(404, "not found", "text/plain")

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
