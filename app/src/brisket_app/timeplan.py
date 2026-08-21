"""Cook time as a decision variable (roadmap stretch item 1).

Framing: **the perfect brisket defines its own time.** Duration is chosen
by pure cook quality — hit target doneness without burning bark, retain
maximum retained moisture, spend fuel lightly — with no external
costs bending it. The deadline never changes the cook; it positions the
*start*: light the fire (planned duration) before serving, where the
planned duration is the quality optimum padded by an uncertainty margin
from the posterior. Finishing early is nearly free (the resting hold
exists for this), so the safety padding is one-sided — exactly the
asymmetry the MPC objective already encodes.

Cheap first implementation (per roadmap): an outer scan over candidate
durations with the inner Optax schedule optimization per duration
(warm-started along the scan), then a chance layer that re-simulates each
duration's optimized schedule over posterior draws to find the shortest
planned duration with P(serveably done) above target. The differentiable-
duration variant (s = t/t_f time-scaling) is the documented next step.
"""

from __future__ import annotations

import numpy as np

from .control import ControlConfig, optimize_schedule
from .coupled import HR, CookGrids, coupled_cook


def _grids_for(t_f: float) -> CookGrids:
    """Grids for a candidate duration: ~45-min damper knots, 10-min exchange."""
    n_u = max(8, round(t_f / (0.75 * HR)))
    return CookGrids(t_end=t_f, dt_exchange=600.0, n_u=n_u)


def plan_duration(
    theta,
    phi,
    meat_tx,
    pit_tx,
    durations_h,
    cc: ControlConfig | None = None,
    ics: dict | None = None,
    inner_iters: int = 60,
    log=print,
    seed: dict | None = None,
):
    """Outer scan over candidate cook durations, inner schedule optimum.

    Pure quality objective (doneness band + bark + juiciness + light fuel
    term). When cc.opt_wrap is set, the WRAP TIME is optimized jointly with
    the schedule inside each duration — the full composition duration x
    wrap x schedule.
    Warm-starts each duration from the previous optimum: the schedule by
    fractional cook progress, the wrap time by horizon fraction (z_w
    carries over directly since t_wrap = t_end * sigmoid(z_w)).
    """
    cc = cc or ControlConfig(n_iter=inner_iters)
    records = []
    u_prev, grids_prev, zw_prev = None, None, 0.0
    if seed is not None:  # warm-start the FIRST duration from a prior scan
        u_prev = np.asarray(seed["u"])
        grids_prev = _grids_for(seed["t_f_h"] * HR)
        f = min(max(seed.get("t_wrap_frac", 0.5), 1e-4), 1 - 1e-4)
        zw_prev = float(np.log(f / (1 - f)))
    for th_f in durations_h:
        t_f = th_f * HR
        grids = _grids_for(t_f)
        if u_prev is None:
            u_init = 0.5
        else:
            frac_prev = grids_prev.u_tknots / grids_prev.u_tknots[-1]
            frac_new = grids.u_tknots / grids.u_tknots[-1]
            u_init = np.clip(np.interp(frac_new, frac_prev, u_prev), 1e-3, 1 - 1e-3)
        u_opt, hist = optimize_schedule(
            theta,
            phi,
            meat_tx,
            pit_tx,
            grids=grids,
            cc=cc,
            u_init=u_init,
            ics=ics,
            z_w_init=zw_prev,
        )
        u_prev, grids_prev = u_opt, grids
        t_wrap = hist.get("t_wrap", 1.0e9)
        if cc.opt_wrap:
            frac = min(max(t_wrap / t_f, 1e-4), 1 - 1e-4)
            zw_prev = float(np.log(frac / (1 - frac)))
        sim = coupled_cook(
            np.asarray(theta),
            np.asarray(phi),
            np.asarray(u_opt),
            meat_tx,
            pit_tx,
            grids,
            K=2,
            ics=ics,
            t_wrap=t_wrap if cc.opt_wrap else 1.0e9,
        )
        c_end = float(np.asarray(sim["c_field_final"])[0])
        w_ret = float(np.asarray(sim["w_retained"]))
        juicy = float(np.asarray(sim["juiciness"]))
        l_end = float(np.asarray(sim["l_star"])[-1])
        records.append(
            {
                "t_f_h": float(th_f),
                "u_opt": u_opt,
                "grids": grids,
                "j_quality": float(hist["losses"][-1]),
                "doneness": c_end,
                "w_retained": w_ret,
                "juiciness": juicy,
                "t_wrap_h": (t_wrap / HR) if cc.opt_wrap else None,
                "l_star_end": l_end,
            }
        )
        log(
            f"  t_f={th_f:5.1f} h  J={records[-1]['j_quality']:7.3f}  "
            f"doneness={c_end:.3f}  juiciness={juicy:.3f}  "
            f"(water {w_ret:.3f})  L*={l_end:.1f}"
            + (f"  wrap@{t_wrap / HR:.2f} h" if cc.opt_wrap else "")
        )
    return records


def optimal_duration(records, plateau_frac: float = 0.02) -> dict:
    """Pick the quality optimum from a duration scan, with a plateau rule.

    The *shortest* duration whose objective is within `plateau_frac` of the
    best: if quality saturates ("anything >= X h is perfect"), the optimum
    is the plateau's left edge, not an arbitrary interior point.
    """
    js = np.array([r["j_quality"] for r in records])
    j_best = float(js.min())
    tol = abs(j_best) * plateau_frac + 1e-9
    for r, j in zip(records, js, strict=True):
        if j <= j_best + tol:
            return r
    return records[int(np.argmin(js))]


def chance_of_done(
    record,
    thetas,
    phis,
    meat_tx,
    pit_tx,
    c_serveable: float = 0.85,
    ics: dict | None = None,
    log=print,
):
    """P(serveably done at t_f) for one duration's optimized schedule.

    Evaluated on posterior draws — the uncertainty side of the start-time
    decision (start = serving deadline - planned duration).
    """
    grids, u_opt = record["grids"], record["u_opt"]
    done = []
    for i in range(len(thetas)):
        sim = coupled_cook(
            np.asarray(thetas[i]),
            np.asarray(phis[i]),
            np.asarray(u_opt),
            meat_tx,
            pit_tx,
            grids,
            K=2,
            ics=ics,
        )
        done.append(float(np.asarray(sim["c_field_final"])[0]))
    done = np.asarray(done)
    p = float(np.mean(done >= c_serveable))
    log(
        f"  t_f={record['t_f_h']:5.1f} h  P(done)={p:.2f}  "
        f"doneness draws: {np.percentile(done, 5):.3f}..."
        f"{np.percentile(done, 95):.3f}"
    )
    return p, done
