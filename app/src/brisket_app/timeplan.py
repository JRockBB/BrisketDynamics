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
duration variant (s = t/t_f time-scaling) is `run_duration_opt.py` /
`coupled_cook_tf` (docs/results.md, differentiable duration); this scan
remains the reference that shows the plateau it climbs.
"""

from __future__ import annotations

import numpy as np

from .control import ControlConfig, optimize_schedule, safety_report
from .coupled import HR, CookGrids, coupled_cook


def _grids_for(t_f: float) -> CookGrids:
    """Grids for a candidate cook length: ~45-min damper knots, 10-min exchange."""
    n_u = max(8, round(t_f / (0.75 * HR)))
    return CookGrids(t_end=t_f, dt_exchange=600.0, n_u=n_u)


def plan_duration(
    theta,
    phi,
    tx,
    durations_h,
    cc: ControlConfig | None = None,
    ics: dict | None = None,
    inner_iters: int = 60,
    log=print,
    seed: dict | None = None,
    rest: dict | None = None,
    t_salt: float = 0.0,
):
    """Outer scan over candidate cook durations, inner schedule optimum.

    Pure quality objective (doneness band + bark + juiciness + light fuel
    term). When cc.opt_wrap is set, the WRAP TIME is optimized jointly with
    the schedule inside each duration — the full composition duration x
    wrap x schedule.
    Warm-starts each duration from the previous optimum: the schedule by
    fractional cook progress, the wrap time by horizon fraction (z_w
    carries over directly since t_wrap = t_end * sigmoid(z_w)).

    Each duration is the cook length, lighting to serve (the pit preheats
    empty and the meat goes on at the optimized load temperature, warm-started
    duration to duration): with cc.opt_pull the
    pull is optimized inside it and the meat rests under the ``rest``
    environment (None = faux-cambro defaults) to serve, where the quality
    terms are scored.
    """
    cc = cc or ControlConfig(n_iter=inner_iters)
    records = []
    u_prev, grids_prev, zw_prev, zp_prev = None, None, 0.0, None
    za_prev = zs_prev = zd_prev = zm_prev = zl_prev = None
    if seed is not None:  # warm-start the FIRST duration from a prior scan
        u_prev = np.asarray(seed["u"])
        grids_prev = _grids_for(seed["t_f_h"] * HR)
        f = min(max(seed.get("t_wrap_frac", 0.5), 1e-4), 1 - 1e-4)
        zw_prev = float(np.log(f / (1 - f)))
        if "t_pull_frac" in seed:
            fp = min(max(seed["t_pull_frac"], 1e-4), 1 - 1e-4)
            zp_prev = float(np.log(fp / (1 - fp)))
        for key, var in (
            ("t_add_frac", "za"),
            ("t_spritz_frac", "zs"),
            ("dt_spritz_frac", "zd"),
        ):
            if key in seed:
                fr = min(max(seed[key], 1e-4), 1 - 1e-4)
                val = float(np.log(fr / (1 - fr)))
                za_prev, zs_prev, zd_prev = (
                    (val, zs_prev, zd_prev)
                    if var == "za"
                    else (za_prev, val, zd_prev)
                    if var == "zs"
                    else (za_prev, zs_prev, val)
                )
    for th_f in durations_h:
        t_f = th_f * HR
        grids = _grids_for(t_f)
        t_pull = 1.0e9
        if u_prev is None:
            u_init = 0.5
        else:
            frac_prev = grids_prev.u_tknots / grids_prev.u_tknots[-1]
            frac_new = grids.u_tknots / grids.u_tknots[-1]
            u_init = np.clip(np.interp(frac_new, frac_prev, u_prev), 1e-3, 1 - 1e-3)
        u_opt, hist = optimize_schedule(
            theta,
            phi,
            tx,
            grids=grids,
            cc=cc,
            u_init=u_init,
            ics=ics,
            z_w_init=zw_prev,
            t_pull=t_pull,
            rest=rest,
            z_p_init=zp_prev,
            z_a_init=za_prev,
            z_s_init=zs_prev,
            z_d_init=zd_prev,
            z_m_init=zm_prev,
            t_salt=t_salt,
            z_l_init=zl_prev,
        )
        u_prev, grids_prev = u_opt, grids
        T_load = float(hist["T_load"]) if cc.opt_load else 0.0
        if cc.opt_load:
            fl = (T_load - cc.T_load_min) / (cc.T_load_max - cc.T_load_min)
            fl = min(max(fl, 1e-4), 1 - 1e-4)
            zl_prev = float(np.log(fl / (1 - fl)))
        t_wrap = hist.get("t_wrap", 1.0e9)
        if cc.opt_wrap:
            frac = min(max(t_wrap / t_f, 1e-4), 1 - 1e-4)
            zw_prev = float(np.log(frac / (1 - frac)))
        t_pull = hist.get("t_pull", t_pull)
        if cc.opt_pull:
            fp = min(max(t_pull / t_f, 1e-4), 1 - 1e-4)
            zp_prev = float(np.log(fp / (1 - fp)))
        t_add = hist.get("t_add", 1.0e9)
        t_spritz = hist.get("t_spritz", 1.0e9)
        dt_spritz = hist.get("dt_spritz", 3600.0)
        spritz_dose = hist.get("spritz_dose", 0.05)
        if cc.opt_add:
            fa = min(max(t_add / t_f, 1e-4), 1 - 1e-4)
            za_prev = float(np.log(fa / (1 - fa)))
        if cc.opt_spritz:
            fs = min(max(t_spritz / t_f, 1e-4), 1 - 1e-4)
            zs_prev = float(np.log(fs / (1 - fs)))
            fd = (dt_spritz - cc.dt_spritz_min) / (cc.dt_spritz_max - cc.dt_spritz_min)
            fd = min(max(fd, 1e-4), 1 - 1e-4)
            zd_prev = float(np.log(fd / (1 - fd)))
            fm = (spritz_dose - cc.m_spritz_min) / (cc.m_spritz_max - cc.m_spritz_min)
            fm = min(max(fm, 1e-4), 1 - 1e-4)
            zm_prev = float(np.log(fm / (1 - fm)))
        sim = coupled_cook(
            np.asarray(theta),
            np.asarray(phi),
            np.asarray(u_opt),
            tx,
            grids,
            ics=ics,
            t_wrap=t_wrap if cc.opt_wrap else 1.0e9,
            t_pull=t_pull,
            rest=rest,
            t_add=t_add,
            t_spritz=t_spritz,
            dt_spritz=dt_spritz,
            spritz_dose=spritz_dose,
            t_salt=t_salt,
            T_load=T_load,
        )
        t_load = float(np.asarray(sim["t_load_used"]))
        load_ok = bool(float(np.asarray(sim["load_reached"])) > 0.5)
        c_end = float(np.asarray(sim["c_field_final"])[0])
        w_ret = float(np.asarray(sim["w_retained"]))
        juicy = float(np.asarray(sim["juiciness"]))
        l_end = float(np.asarray(sim["l_star"])[-1])
        safety = safety_report(
            sim["tsurf"], sim["tcore"], grids.tknots, cc, t_load=t_load
        )
        # smoke ring in physical millimetres: material depth × final stretch
        ring_mm = 1e3 * float(sim["ring"]) * float(sim["l_thick"][-1]) / float(theta[3])
        # the optimizer parks an unused event beyond the horizon: a chimney
        # after the pull, or a spritz train starting after the wrap or the
        # pull, never fires and is recorded as none
        adds = t_add < t_pull
        spritzes = t_spritz < min(t_wrap if cc.opt_wrap else 1.0e9, t_pull)
        records.append(
            {
                "t_f_h": float(th_f),
                "u_opt": u_opt,
                "grids": grids,
                "j_quality": float(hist.get("best_loss", hist["losses"][-1])),
                "doneness": c_end,
                "w_retained": w_ret,
                "juiciness": juicy,
                "t_wrap": float(t_wrap) if cc.opt_wrap else 1.0e9,
                "t_wrap_h": (t_wrap / HR) if cc.opt_wrap else None,
                "l_star_end": l_end,
                "t_pull": float(t_pull),
                "t_pull_h": float(t_pull / HR) if t_pull < 1.0e8 else None,
                "rest": rest,
                "t_add": float(t_add) if adds else 1.0e9,
                "t_add_h": float(t_add / HR) if adds else None,
                "t_spritz": float(t_spritz) if spritzes else 1.0e9,
                "dt_spritz": float(dt_spritz),
                "t_spritz_h": float(t_spritz / HR) if spritzes else None,
                "dt_spritz_min": float(dt_spritz / 60.0) if spritzes else None,
                "spritz_dose": float(spritz_dose),
                "spritz_dose_gm2": float(1e3 * spritz_dose) if spritzes else None,
                "ring_mm": ring_mm,
                "safety_ok": safety["ok"],
                "F_core": safety["F_core"],
                "cut_h_core": safety["cut_h_core"],
                "T_load": T_load,
                "t_load": t_load,
                "t_load_h": t_load / HR,
                "load_reached": load_ok,
                "cook_from_load_h": float(th_f) - t_load / HR,
            }
        )
        log(
            f"  lit to serve {th_f:4.1f} h  on at {t_load / HR:.2f} h "
            f"(pit {T_load - 273.15:.0f} C{'' if load_ok else ', NOT REACHED'})  "
            f"J={records[-1]['j_quality']:7.3f}  "
            f"doneness={c_end:.3f}  juiciness={juicy:.3f}  "
            f"(water {w_ret:.3f})  L*={l_end:.1f}  ring={ring_mm:.1f}mm"
            + (f"  wrap@{t_wrap / HR:.2f} h" if cc.opt_wrap else "")
            + (f"  pull@{t_pull / HR:.2f} h" if t_pull < 1.0e8 else "")
            + (f"  chimney@{t_add / HR:.2f} h" if adds else "  no chimney")
            + (
                f"  spritz@{t_spritz / HR:.2f} h/{dt_spritz / 60:.0f} min/"
                f"{1e3 * spritz_dose:.0f} g/m²"
                if spritzes
                else "  no spritz"
            )
            + f"  iters={hist.get('n_iter_run', len(hist['losses']))}"
            + ("" if hist.get("converged", True) else " (cap)")
            + ("" if safety["ok"] else "  UNSAFE (raw safety residuals fail)")
        )
    return records


def optimal_duration(records, plateau_frac: float = 0.02) -> dict:
    """Pick the quality optimum from a duration scan, with a plateau rule.

    The *shortest* duration whose objective is within `plateau_frac` of the
    best: if quality saturates ("anything >= X h is perfect"), the optimum
    is the plateau's left edge, not an arbitrary interior point. Durations
    whose raw safety residuals fail (``safety_ok``) are rejected; if none
    passes, the pick is over all of them and carries ``safety_ok=False``.
    """
    safe = [r for r in records if r.get("safety_ok", True)]
    pool = safe or records
    js = np.array([r["j_quality"] for r in pool])
    j_best = float(js.min())
    tol = abs(j_best) * plateau_frac + 1e-9
    for r, j in zip(pool, js, strict=True):
        if j <= j_best + tol:
            return r
    return pool[int(np.argmin(js))]


def chance_of_done(
    record,
    thetas,
    phis,
    tx,
    c_serveable: float = 0.85,
    ics: dict | None = None,
    log=print,
    t_salt: float = 0.0,
):
    """Empirical fraction of posterior draws that are serveably done at t_f.

    This is a Monte Carlo frequency over N draws, not a 95% certification.
    Replays the record's optimized wrap, pull and schedule, so "done"
    means done at serve.
    """
    grids, u_opt = record["grids"], record["u_opt"]
    t_wrap = record.get("t_wrap", 1.0e9)
    if t_wrap is None:
        t_wrap = 1.0e9
    t_pull = record.get("t_pull", 1.0e9)
    done = []
    for i in range(len(thetas)):
        sim = coupled_cook(
            np.asarray(thetas[i]),
            np.asarray(phis[i]),
            np.asarray(u_opt),
            tx,
            grids,
            ics=ics,
            t_wrap=t_wrap,
            t_pull=t_pull,
            rest=record.get("rest"),
            t_salt=t_salt,
        )
        done.append(float(np.asarray(sim["c_field_final"])[0]))
    done = np.asarray(done)
    p = float(np.mean(done >= c_serveable))
    log(
        f"  t_f={record['t_f_h']:5.1f} h  empirical P(done)={p:.2f}  "
        f"(n={len(done)} draws; not a 95% certification)  "
        f"doneness draws: {np.percentile(done, 5):.3f}..."
        f"{np.percentile(done, 95):.3f}"
    )
    return p, done
