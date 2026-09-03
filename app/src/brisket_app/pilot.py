"""The cook pilot: the receding-horizon re-planner pointed at a pit.

This is `mpc.run_cook`'s control loop lifted out of simulation and wired
to the :mod:`pit_io` contract: the pilot READS the pit (temps, fan duty,
setpoint) and COMMANDS setpoints; the pit's own PID does the fan. Wrap
and pull are re-planned live (chimney/spritz ride along at fixed given
times in this version); on hardware they are ADVISORIES a human executes
— the pilot journals the call and, where the backend supports it (the
sim, the mock), executes it itself.

State estimation, per executed segment:
- the model plant (CookState) advances under the MEASURED fan duty — the
  meat fields, fuel and bark states are model-propagated;
- the pit temperature state is then snapped to the MEASURED pit (the
  sensor is truth for the one state it measures);
- the ambient belief is a scalar Kalman filter (mpc.AmbientKF) on the
  pit innovation (measured minus model-predicted segment end). With a
  FORECAST the filter becomes feedforward + feedback: the predict step
  shifts the belief by the forecast delta, and the innovation corrects
  the residual bias (wind, rain, a wrong forecast).

Everything is journaled as JSONL (one event per line: status samples,
replans, advisories, filter updates) so a cook can be audited or
re-rendered afterwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import mpc
from .control import ControlConfig
from .coupled import HR, CookGrids
from .pit_io import PitStatus

K2F = lambda k: (float(k) - 273.15) * 9.0 / 5.0 + 32.0


@dataclass
class PilotConfig:
    t_f: float = 12 * HR  # cook length, start to serve [s]
    dt_replan: float = 1 * HR
    dt_status: float = 60.0  # pit poll / setpoint push cadence [s]
    replan_iters: int = 200
    model_t_amb: float = 300.0  # ambient belief seed [K]
    sp_min_K: float = 340.0  # setpoint command clamp
    sp_max_K: float = 420.0
    sp_push_eps_K: float = 0.6  # push only when the reference moved
    auto_events: bool = True  # execute wrap/pull where the backend can
    wall_pace: float = 0.0  # wall s per sim s (0 = free-run lockstep)


class Journal:
    """Append-only JSONL event log."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def write(self, kind: str, **fields):
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps({"kind": kind, **fields}) + "\n")


class Pilot:
    """Fly one cook on a pit backend (SimPit, mock HTTP, or HeaterMeter)."""

    def __init__(
        self,
        pit,
        theta,
        phi_model,
        cfg: PilotConfig,
        cc: ControlConfig | None = None,
        incumbent: dict | None = None,
        rest: dict | None = None,
        forecast_fn=None,
        journal: Journal | None = None,
        log=print,
    ):
        self.pit = pit
        self.theta = np.asarray(theta, dtype=float)
        self.phi = np.asarray(phi_model, dtype=float)
        self.cfg = cfg
        self.cc = cc or ControlConfig()
        self.rest = rest
        self.forecast = forecast_fn  # t [s] -> T_amb [K], or None
        self.j = journal or Journal(None)
        self.log = log
        inc = incumbent or {}
        self.u_prev = None if inc.get("u") is None else np.asarray(inc["u"])
        self.prev_grids = inc.get("grids")
        self.prev_t0 = 0.0
        self.t_wrap = inc.get("t_wrap")
        self.t_pull = inc.get("t_pull")
        self.t_add = inc.get("t_add")
        self.t_spritz = inc.get("t_spritz")
        self.dt_spritz = inc.get("dt_spritz") or 3600.0
        self.spritz_dose = inc.get("spritz_dose") or 0.05
        self.last_wrap_cand = self.t_wrap
        self.last_pull_cand = self.t_pull
        self.wrapped = False
        self.pulled = False
        self.t_wrap_exec = None
        self.t_pull_exec = None
        self.wrap_candidates: list[tuple] = []
        self.pull_candidates: list[tuple] = []
        s0 = self.pit.read()
        self.state = mpc.CookState.fresh(
            tpit0=s0.tpit_K,
            t0_meat=s0.tcore_K if s0.tcore_K else 281.15,
        )
        seed = self.cfg.model_t_amb if self.forecast is None else self.forecast(0.0)
        self.kf = mpc.AmbientKF(
            x=float(seed), p=self.cc.amb_p0, q=self.cc.amb_q, r=self.cc.amb_r
        )
        self._fc_last = float(seed)

    # --- ambient belief -------------------------------------------------------
    def ambient(self, t: float) -> float:
        """Belief now: forecast feedforward (delta) is applied at predict."""
        return float(self.kf.x)

    def _kf_predict_feedforward(self, t_new: float):
        if self.forecast is not None:
            fc = float(self.forecast(t_new))
            self.kf.x += fc - self._fc_last
            self._fc_last = fc

    # --- one replan -----------------------------------------------------------
    def replan(self, t_remaining: float):
        t0 = self.state.t
        if self.u_prev is None:
            u_init = 0.5
        else:
            prev_t = self.prev_grids.u_tknots + self.prev_t0
            n_u = max(2, round(t_remaining / (0.75 * HR)))
            new_t = t0 + np.linspace(0, t_remaining, n_u)
            u_init = np.clip(np.interp(new_t, prev_t, self.u_prev), 1e-3, 1 - 1e-3)

        def zseed(t_abs):
            if t_abs is None:
                return None
            frac = np.clip((t_abs - t0) / t_remaining, 1e-3, 1 - 1e-3)
            return float(np.log(frac / (1 - frac)))

        if self.cc.opt_wrap and not self.wrapped:
            zw0, tw_override, cc_seg = zseed(self.last_wrap_cand) or 0.0, None, self.cc
        else:
            cc_seg = replace(self.cc, opt_wrap=False)
            zw0 = 0.0
            tw_override = (self.t_wrap_exec - t0) if self.wrapped else None
        zp0 = zseed(self.last_pull_cand) if self.cc.opt_pull else None
        tp_fixed = 1.0e9 if self.cc.opt_pull else (
            max(0.0, self.t_pull - t0) if self.t_pull is not None else 1.0e9
        )
        ta_fixed = max(0.0, self.t_add - t0) if self.t_add is not None else 1.0e9
        ts_fixed = max(0.0, self.t_spritz - t0) if self.t_spritz is not None else 1.0e9
        u_plan, grids, hist = mpc.plan(
            self.theta,
            self.phi,
            self.state,
            t_remaining,
            replace(cc_seg, opt_add=False, opt_spritz=False),
            self.ambient(t0),
            u_init=u_init,
            iters=self.cfg.replan_iters,
            z_w_init=zw0,
            t_wrap_override=tw_override,
            z_p_init=zp0,
            rest=self.rest,
            t_pull=tp_fixed,
            t_add=ta_fixed,
            t_spritz=ts_fixed,
            dt_spritz=self.dt_spritz,
            spritz_dose=self.spritz_dose,
        )
        dt_seg = min(self.cfg.dt_replan, t_remaining)
        if self.cc.opt_wrap and not self.wrapped:
            cand = t0 + float(hist["t_wrap"])
            self.t_wrap = mpc._adopt_event(
                cand, self.last_wrap_cand, t0, dt_seg, hist, self.cc,
                self.wrap_candidates,
            )
            self.last_wrap_cand = cand
        if self.cc.opt_pull and not self.pulled:
            cand = t0 + float(hist["t_pull"])
            self.t_pull = mpc._adopt_event(
                cand, self.last_pull_cand, t0, dt_seg, hist, self.cc,
                self.pull_candidates,
            )
            self.last_pull_cand = cand
        self.u_prev, self.prev_grids, self.prev_t0 = u_plan, grids, t0
        self.j.write(
            "replan", t=t0, J=float(hist["best_loss"]),
            t_wrap=self.t_wrap, t_pull=self.t_pull,
            u=[round(float(v), 4) for v in u_plan],
        )
        return u_plan, grids

    # --- predicted pit under the plan = setpoint reference --------------------
    def predict_segment(self, u_plan, grids, dt_seg):
        n = max(3, round(dt_seg / self.cfg.dt_status) + 1)
        tt = np.linspace(0.0, dt_seg, n)
        u_seg = np.interp(tt, grids.u_tknots, u_plan)
        tw = (
            (self.t_wrap_exec - self.state.t) if self.wrapped
            else 1.0e9
        )
        _, seg = mpc.advance_plant(
            self.state, u_seg, dt_seg, self.theta, self.phi,
            self.ambient(self.state.t), t_wrap_seg=tw, rest=self.rest,
        )
        tpit = np.asarray(seg["tpit"])
        t_ref = np.linspace(0.0, dt_seg, len(tpit))
        return tt, np.interp(tt, t_ref, tpit)

    # --- one executed segment -------------------------------------------------
    def execute_segment(self, dt_seg, sp_t, sp_K):
        """Tick the pit for dt_seg: push setpoints, collect measurements."""
        t0 = self.state.t
        meas = {"t": [], "tpit": [], "tcore": [], "u": []}
        last_sp = None
        t_local = 0.0
        while t_local < dt_seg - 1e-6:
            sp = float(np.clip(
                np.interp(t_local, sp_t, sp_K),
                self.cfg.sp_min_K, self.cfg.sp_max_K,
            ))
            if last_sp is None or abs(sp - last_sp) > self.cfg.sp_push_eps_K:
                self.pit.set_setpoint(sp)
                last_sp = sp
            s: PitStatus = self.pit.read()
            t_local = s.t - t0
            meas["t"].append(s.t)
            meas["tpit"].append(s.tpit_K)
            meas["tcore"].append(s.tcore_K)
            meas["u"].append(s.fan_avg_pct / 100.0)
            # live advisories: wrap / pull when due
            for name, t_ev, fired_attr, exec_attr in (
                ("wrap", self.t_wrap, "wrapped", "t_wrap_exec"),
                ("pull", self.t_pull, "pulled", "t_pull_exec"),
            ):
                if (
                    t_ev is not None
                    and not getattr(self, fired_attr)
                    and s.t >= t_ev
                ):
                    self._advise(name, s.t)
            if self.pulled:
                break
            if self.cfg.wall_pace > 0:
                time.sleep(self.cfg.dt_status * self.cfg.wall_pace)
        self.j.write(
            "segment", t0=t0,
            t=[round(v, 1) for v in meas["t"]],
            tpit=[round(v, 2) for v in meas["tpit"]],
            tcore=[None if v is None else round(v, 2) for v in meas["tcore"]],
            u=[round(v, 4) for v in meas["u"]],
            sp_t=[round(t0 + v, 1) for v in sp_t],
            sp=[round(v, 2) for v in sp_K],
        )
        return meas

    def _advise(self, name: str, t_now: float):
        msg = f"ADVISORY at {t_now / HR:.2f} h: {name.upper()} the meat now"
        self.log(f"  {msg}")
        executed = False
        if self.cfg.auto_events:
            hook = getattr(self.pit, f"execute_{name}", None)
            if hook is not None:
                hook(t_now)
                executed = True
        setattr(self, "wrapped" if name == "wrap" else "pulled", True)
        setattr(self, f"t_{name}_exec", t_now)
        self.j.write("advisory", event=name, t=t_now, executed=executed)

    # --- estimator ------------------------------------------------------------
    def assimilate(self, meas, dt_seg, tpit_pred_end):
        """Advance the belief state under measured u; KF on the innovation."""
        u_meas = np.asarray(meas["u"], dtype=float)
        if u_meas.size < 2:
            u_meas = np.full(2, u_meas[0] if u_meas.size else 0.5)
        tw = 1.0e9
        if self.wrapped:
            tw = self.t_wrap_exec - self.state.t
        tp = 1.0e9
        if self.pulled:
            tp = self.t_pull_exec - self.state.t
        self.state, _ = mpc.advance_plant(
            self.state, u_meas, dt_seg, self.theta, self.phi,
            self.ambient(self.state.t), t_wrap_seg=tw, t_pull_seg=tp,
            rest=self.rest,
        )
        tpit_meas = float(meas["tpit"][-1])
        innovation = tpit_meas - float(tpit_pred_end)
        ua, c_th = float(self.phi[1]), float(self.phi[0])
        self._kf_predict_feedforward(self.state.t)
        self.kf.update(innovation, ua, c_th, dt_seg)
        # the pit sensor is truth for the pit-temperature state
        self.state = replace(self.state, tpit0=tpit_meas)
        self.j.write(
            "assimilate", t=self.state.t, tpit_meas=tpit_meas,
            tpit_pred=float(tpit_pred_end), innovation=innovation,
            t_amb_belief=self.kf.x, t_amb_var=self.kf.p,
        )

    # --- the cook -------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self.j.write("start", t_f=cfg.t_f, dt_replan=cfg.dt_replan,
                     theta=list(map(float, self.theta)),
                     phi=list(map(float, self.phi)))
        while self.state.t < cfg.t_f - 1e-6 and not self.pulled:
            t_remaining = cfg.t_f - self.state.t
            dt_seg = min(cfg.dt_replan, t_remaining)
            u_plan, grids = self.replan(t_remaining)
            sp_t, sp_K = self.predict_segment(u_plan, grids, dt_seg)
            self.log(
                f"t={self.state.t / HR:5.2f} h  amb belief "
                f"{self.kf.x - 273.15:+5.1f} C  sp "
                f"{K2F(sp_K[0]):.0f}->{K2F(sp_K[-1]):.0f} F"
                + (f"  wrap@{self.t_wrap / HR:.2f}h" if self.t_wrap and not self.wrapped else "")
                + (f"  pull@{self.t_pull / HR:.2f}h" if self.t_pull and not self.pulled else "")
            )
            meas = self.execute_segment(dt_seg, sp_t, sp_K)
            dt_exec = (meas["t"][-1] - self.state.t) if meas["t"] else dt_seg
            self.assimilate(meas, max(dt_exec, 1.0), sp_K[-1])
        self.j.write(
            "end", t=self.state.t, wrapped=self.wrapped, pulled=self.pulled,
            t_wrap_exec=self.t_wrap_exec, t_pull_exec=self.t_pull_exec,
        )
        return {
            "t_end": self.state.t,
            "wrapped": self.wrapped,
            "t_wrap_exec": self.t_wrap_exec,
            "pulled": self.pulled,
            "t_pull_exec": self.t_pull_exec,
            "t_amb_belief": self.kf.x,
        }
