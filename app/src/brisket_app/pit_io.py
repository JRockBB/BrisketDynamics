"""Pit hardware interface: one contract for a real HeaterMeter and the sim.

The pilot (scripts/run_pilot.py) plans ABOVE the low-level fan controller:
it reads pit state and commands a SETPOINT; the pit's own PID turns that
into fan/damper action. This module gives that contract two backends:

- :class:`HeaterMeterClient` — the LinkMeter v14 REST API
  (github.com/CapnBry/HeaterMeter/wiki/Accessing-Raw-Data-Remotely):
  ``GET  /luci/lm/api/status``  -> JSON {time, set, lid, fan{c,a,f},
                                   temps[{n,c,dph,...}]}
  ``POST /luci/lm/api/config``  -> form-encoded, e.g. ``sp=225&apikey=X``
  HeaterMeter speaks Fahrenheit by default; this client converts to
  Kelvin at the boundary (``units="F"`` | ``"C"``).

- :class:`SimPit` — the calibrated coupled plant behind the same
  contract, with an HM-style PI controller turning the commanded setpoint
  into damper duty. Used directly in-process, or served over HTTP by
  scripts/mock_heatermeter.py so the REAL HeaterMeterClient code path is
  exercised end-to-end with no hardware.

Temperatures cross this interface in KELVIN; times in seconds.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np

F2K = lambda f: (float(f) - 32.0) * 5.0 / 9.0 + 273.15
K2F = lambda k: (float(k) - 273.15) * 9.0 / 5.0 + 32.0
C2K = lambda c: float(c) + 273.15
K2C = lambda k: float(k) - 273.15


@dataclass
class PitStatus:
    """One reading of the pit: everything the pilot consumes."""

    t: float  # pit-side timestamp [s, UNIX or sim]
    setpoint_K: float
    tpit_K: float
    tcore_K: float | None  # food probe (None if unplugged)
    fan_pct: float  # current PID output [0-100]
    fan_avg_pct: float  # trailing-average output [0-100]
    lid_open_s: float  # lid-detect countdown (0 = closed)


class HeaterMeterClient:
    """LinkMeter v14 REST client (see module docstring for the API)."""

    def __init__(
        self,
        base_url: str,
        apikey: str = "",
        units: str = "F",
        pit_probe: int = 0,
        food_probe: int = 1,
        timeout_s: float = 45.0,  # a mock's first solves can JIT-compile
    ):
        self.base = base_url.rstrip("/")
        self.apikey = apikey
        self.to_K = F2K if units.upper() == "F" else C2K
        self.from_K = K2F if units.upper() == "F" else K2C
        self.pit_probe = pit_probe
        self.food_probe = food_probe
        self.timeout = timeout_s

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def read(self) -> PitStatus:
        s = self._get("/luci/lm/api/status")
        temps = s.get("temps", [])

        def probe(i):
            if i < len(temps) and temps[i].get("c") is not None:
                return self.to_K(temps[i]["c"])
            return None

        tpit = probe(self.pit_probe)
        if tpit is None:
            raise RuntimeError("pit probe unplugged or out of range")
        fan = s.get("fan", {}) or {}
        return PitStatus(
            t=float(s["time"]),
            setpoint_K=self.to_K(s["set"]),
            tpit_K=tpit,
            tcore_K=probe(self.food_probe),
            fan_pct=float(fan.get("c") or 0.0),
            fan_avg_pct=float(fan.get("a") or 0.0),
            lid_open_s=float(s.get("lid") or 0.0),
        )

    def set_setpoint(self, sp_K: float) -> None:
        body = urllib.parse.urlencode(
            {"sp": round(self.from_K(sp_K)), "apikey": self.apikey}
        ).encode()
        req = urllib.request.Request(
            self.base + "/luci/lm/api/config", data=body, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            r.read()


class MockHMClient(HeaterMeterClient):
    """HeaterMeterClient plus the mock server's sim-only event hooks."""

    def _post(self, path: str):
        body = urllib.parse.urlencode({"apikey": self.apikey}).encode()
        req = urllib.request.Request(self.base + path, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            r.read()

    def execute_wrap(self, t_now: float) -> None:
        self._post("/sim/wrap")

    def execute_pull(self, t_now: float) -> None:
        self._post("/sim/pull")

    def traj(self) -> dict:
        return self._get("/sim/traj")


@dataclass
class _PI:
    """HM-flavored PI with anti-windup on the damper duty (u in [0, 1])."""

    kp: float = 0.03  # [1/K]
    ki: float = 6.0e-5  # [1/(K s)]
    i: float = 0.25  # integrator (starts mid-damper)

    def step(self, err_K: float, dt: float) -> float:
        self.i = float(np.clip(self.i + self.ki * err_K * dt, 0.0, 1.0))
        return float(np.clip(self.kp * err_K + self.i, 0.0, 1.0))


class SimPit:
    """The calibrated coupled plant behind the pit contract.

    The pilot commands a setpoint exactly as it would on hardware; the
    sim's PI turns it into damper duty and the plant advances in
    ``dt_step`` chunks. ``t_amb_fn(t)`` is the ambient disturbance the
    CONTROLLER does not know; ``clock_scale`` maps wall seconds to sim
    seconds (0 = free-running: each :meth:`read` advances one step).
    """

    def __init__(
        self,
        theta,
        phi,
        t_amb_fn,
        t0_meat_K: float = 281.15,
        setpoint_K: float = 380.0,
        dt_step: float = 120.0,
        clock_scale: float = 0.0,
        fan_avg_tau_s: float = 600.0,
    ):
        from .mpc import CookState, advance_plant

        self._advance = advance_plant
        self.theta = np.asarray(theta, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        self.t_amb_fn = t_amb_fn
        self.state = CookState.fresh(t0_meat=t0_meat_K)
        self.setpoint_K = float(setpoint_K)
        self.dt_step = float(dt_step)
        self.clock_scale = float(clock_scale)
        self._wall0 = time.monotonic()
        self.pi = _PI()
        self.u = 0.5
        self.fan_avg = 0.5
        self._avg_a = float(np.exp(-dt_step / fan_avg_tau_s))
        self.traj = {k: [] for k in ("t", "tpit", "tcore", "u", "t_amb")}
        # irreversible events, set by the pilot when the human acts
        self.t_wrap_exec: float | None = None
        self.t_pull_exec: float | None = None

    def _step_once(self):
        st = self.state
        err = self.setpoint_K - float(st.tpit0)
        self.u = self.pi.step(err, self.dt_step)
        self.fan_avg = self._avg_a * self.fan_avg + (1 - self._avg_a) * self.u
        tw = (self.t_wrap_exec - st.t) if self.t_wrap_exec is not None else 1.0e9
        tp = (self.t_pull_exec - st.t) if self.t_pull_exec is not None else 1.0e9
        u_seg = np.full(2, 0.0 if tp < 0 else self.u)
        self.state, seg = self._advance(
            st,
            u_seg,
            self.dt_step,
            self.theta,
            self.phi,
            float(self.t_amb_fn(st.t + self.dt_step / 2)),
            t_wrap_seg=tw,
            t_pull_seg=tp,
        )
        self.traj["t"].append(self.state.t)
        self.traj["tpit"].append(float(self.state.tpit0))
        self.traj["tcore"].append(float(seg["tcore"][-1]))
        self.traj["u"].append(self.u)
        self.traj["t_amb"].append(float(self.t_amb_fn(st.t)))

    def advance_to(self, t_sim: float):
        while self.state.t < t_sim - 1e-6:
            self._step_once()

    def read(self) -> PitStatus:
        if self.clock_scale > 0:
            self.advance_to((time.monotonic() - self._wall0) * self.clock_scale)
        else:
            self._step_once()
        st = self.state
        return PitStatus(
            t=float(st.t),
            setpoint_K=self.setpoint_K,
            tpit_K=float(st.tpit0),
            tcore_K=float(self.traj["tcore"][-1]),
            fan_pct=100.0 * self.u,
            fan_avg_pct=100.0 * self.fan_avg,
            lid_open_s=0.0,
        )

    def set_setpoint(self, sp_K: float) -> None:
        self.setpoint_K = float(sp_K)

    def execute_wrap(self, t_now: float) -> None:
        self.t_wrap_exec = self.state.t

    def execute_pull(self, t_now: float) -> None:
        self.t_pull_exec = self.state.t

    # --- v14-shaped dict for the mock HTTP server (Fahrenheit) ---------------
    def status_dict(self) -> dict:
        s = self.read()
        return {
            "time": int(s.t),
            "set": round(K2F(s.setpoint_K), 1),
            "lid": 0,
            "fan": {
                "c": round(s.fan_pct, 1),
                "a": round(s.fan_avg_pct, 1),
                "f": round(s.fan_pct, 1),
            },
            "temps": [
                {"n": "Pit", "c": round(K2F(s.tpit_K), 1), "dph": 0},
                {"n": "Food", "c": round(K2F(s.tcore_K), 1), "dph": 0},
            ],
        }
