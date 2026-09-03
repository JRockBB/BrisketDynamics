"""Wrap constants from a documented butcher-paper wrap (roadmap item 6).

erickson_2013 (tools/loghunt/logs/erickson_2013): ~12 lb brisket on a
Char-Griller Kamado, ~15 h; wrapped in BUTCHER PAPER at a documented
moment (the chart's off/on lid-band pair), pulled at 15.2 h. The
documentation makes the wrap a known-input step-response experiment for
eta_evap (evaporation suppression) and eps_wrap (wrapped emissivity),
literature-scale constants until now.

Forward model — three chained segments (the observed core DROPS ~7 C
across the wrap; the brisket spent ~24 min off the pit being wrapped and
the probe was re-seated, so the naive single-segment model is wrong in
ways no constant can absorb):
  1. meat-on -> off-pit (523 min): measured pit trace as the boundary
     forcing (gaps interpolated), unwrapped;
  2. off-pit -> back-on (24 min): room environment (292.5 K) — the wet
     surface keeps evaporating into room air;
  3. back-on -> pull: measured pit forcing, wrapped from the start.
Segments chain through the full state (T/W/C/F fields + bark, smoke,
ring scalars). The probe is NOT assumed to sit at the center: its
material depth is a parameter, separately before (xi_pre) and after
(xi_post) the wrap — re-seating the probe in a different spot is the
physical explanation for the step down, and the fit measures it.

Fit: 8 parameters to the masked core trace. [D_eff, k_m, h_c, k_e]
(log-space, priors at repo defaults, sd 0.5), [eta_evap, eps_wrap]
(logit, priors at butcher-paper literature 0.5 / 0.3, sd 1.0),
[xi_pre, xi_post] (logit, prior 0.2 — near-center probe, sd 1.0).
Fixed: L = 0.033 m, k_mult = 1, a_myosin = 0.2, m_conv = 0.6,
eta_h = 0.2, slab. Levenberg-Marquardt with an FD Jacobian in z
(MeatConfig fields are Float64 — the wrap constants are not
ForwardDiff-reachable without a parametric-struct change; FD on an
8-parameter single-log fit is the right-sized tool). Observation noise:
iid sigma_core = 2.5 K (JPEG digitization ~+/-1.7 K plus model form).

Deliverables: the fit, Laplace sds, and counterfactuals at the fitted
parameters (never wrapped / foil constants), all through the same
three-segment structure.

Run from app/:  uv run python -u ../scripts/run_wrap_fit.py
Writes data/calibration/wrap_fit_erickson.json and
figures/wrap_fit_erickson.png.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

from brisket_app import cooklog

SIGMA_CORE = 2.5  # [K] iid observation noise (see docstring)
T_ROOM = 292.5  # [K] room during the wrap interval
T_OFF_MIN = 523.0  # off the pit (center of the first wrap lid band)
T_ON_MIN = 547.0  # back on, wrapped (center of the second band)
NEVER = 1.0e12
FIXED = {"L": 0.033, "k_mult": 1.0, "a_myosin": 0.2, "m_conv": 0.6}
SITES = [
    ("D_eff", "log", 3.8e-9, 0.5),
    ("k_m", "log", 1.8e-3, 0.5),
    ("h_c", "log", 10.0, 0.5),
    ("k_e", "log", 5.6e-4, 0.5),
    ("eta_evap", "logit", 0.5, 1.0),  # butcher paper (foil ~0.9)
    ("eps_wrap", "logit", 0.3, 1.0),  # butcher paper (foil ~0.1)
    ("xi_pre", "logit", 0.2, 1.0),  # probe material depth, before wrap
    ("xi_post", "logit", 0.2, 1.0),  # after re-seating at the wrap
]

logit = lambda p: np.log(p / (1 - p))
inv = {"log": np.exp, "logit": lambda z: 1 / (1 + np.exp(-z))}
fwd = {"log": np.log, "logit": logit}


def unpack(z):
    return {
        name: float(inv[tf](zi))
        for (name, tf, _, _), zi in zip(SITES, z, strict=True)
    }


class Forward:
    """Three-segment chained solve; returns the probe reading on the grid."""

    def __init__(self, jl, log):
        self.jl = jl
        self.t = log["t"]
        self.tcore0 = float(log["meta"]["tcore0_K"])
        mp = log["mask_pit"] & np.isfinite(log["obs_pit"])
        self.tpit = np.interp(self.t, self.t[mp], np.asarray(log["obs_pit"])[mp])
        s1 = self.t <= T_OFF_MIN * 60.0
        s3 = self.t >= T_ON_MIN * 60.0
        self.i3 = np.flatnonzero(s3)
        # segment-local time bases (each solve starts at 0)
        self.t1 = np.append(self.t[s1], T_OFF_MIN * 60.0)
        self.t2 = np.array([0.0, (T_ON_MIN - T_OFF_MIN) * 60.0])
        self.t3 = self.t[s3] - T_ON_MIN * 60.0
        self.p1 = np.append(self.tpit[s1], np.interp(T_OFF_MIN * 60, self.t, self.tpit))
        self.p3 = self.tpit[s3]
        self.n1 = int(s1.sum())

    def _cfg(self, x, **kw):
        return self.jl.MeatConfig(
            geom_m=0.0, η_evap=x["eta_evap"], ε_wrap=x["eps_wrap"], **kw
        )

    def _solve(self, x, tpit, tk, ts, cfg, t_wrap, state=None):
        jl = self.jl
        kw = {}
        if state is not None:
            T, W, C, F, B, S, R = state
            kw = dict(
                T0f=jl.Vector(T), W0f=jl.Vector(W), C0f=jl.Vector(C),
                F0f=jl.Vector(F), B0=float(B), S0=float(S), R0=float(R),
            )
        theta = [
            x["D_eff"], x["k_m"], x["h_c"], FIXED["L"], FIXED["k_mult"],
            x["k_e"], FIXED["a_myosin"], FIXED["m_conv"],
        ]
        out = jl.meat_solve(
            jl.Vector([float(v) for v in theta]),
            jl.Vector(np.asarray(tpit, dtype=float)),
            jl.Vector(np.asarray(tk, dtype=float)),
            jl.Vector(np.asarray(ts, dtype=float)),
            cfg=cfg,
            t_wrap=float(t_wrap),
            **kw,
        )
        Tf = np.asarray(out.T_field).copy()
        end = (
            Tf[:, -1],
            np.asarray(out.W_field)[:, -1].copy(),
            np.asarray(out.C_field)[:, -1].copy(),
            np.asarray(out.F_field)[:, -1].copy(),
            # PythonCall maps a trailing _b to Julia's !, so go via Symbol
            float(np.asarray(jl.getproperty(out, jl.Symbol("bark_b")))[-1]),
            float(np.asarray(jl.getproperty(out, jl.Symbol("bark_s")))[-1]),
            float(np.asarray(out.ring)[-1]),
        )
        return Tf, end

    def probe(self, Tf, xi):
        nx = Tf.shape[0]
        g = np.linspace(0.0, 1.0, nx)
        # xi measured from the CENTER (0 = center, 1 = surface)
        i = np.clip(np.searchsorted(g, xi), 1, nx - 1)
        f = (xi - g[i - 1]) / (g[i] - g[i - 1])
        return (1 - f) * Tf[i - 1] + f * Tf[i]

    def run(self, x, wrapped=True, eta=None, eps=None):
        cfg1 = self._cfg(x, T0=self.tcore0)
        Tf1, st = self._solve(x, self.p1, self.t1, self.t1, cfg1, NEVER)
        cfg2 = self._cfg(x)
        Tf2, st = self._solve(
            x, [T_ROOM, T_ROOM], self.t2, self.t2, cfg2, NEVER, state=st
        )
        x3 = dict(x)
        if eta is not None:
            x3["eta_evap"], x3["eps_wrap"] = eta, eps
        cfg3 = self._cfg(x3)
        Tf3, _ = self._solve(
            x, self.p3, self.t3, self.t3, cfg3,
            -36000.0 if wrapped else NEVER, state=st,
        )
        pre = self.probe(Tf1, x["xi_pre"])[: self.n1]
        post = self.probe(Tf3, x["xi_post"])
        # grid points between off and on (excluded from the fit by masks)
        mid = np.full(len(self.t) - self.n1 - len(self.i3), np.nan)
        return np.concatenate([pre, mid, post])


def main():
    import argparse

    from juliacall import Main as jl

    ap = argparse.ArgumentParser()
    # default L: the probe sits in the thick (point-end) section — this is
    # the configuration whose nuisance posteriors all land physical (h_c
    # ~12, k_m/k_e at priors); --L 0.033 --tag _L33 is the thin-flat
    # sensitivity (fits equally well but pushes h_c to ~1.3, trading the
    # radiative/convective split — see roadmap item 6)
    ap.add_argument("--L", type=float, default=0.045,
                    help="half-thickness [m] (0.045 point-end primary; 0.033 variant)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    a = ap.parse_args()
    FIXED["L"] = a.L

    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics: meat_solve, MeatConfig")
    print("Julia ready", flush=True)

    log = cooklog.load(dt_min=10.0, log="erickson_2013")
    f = Forward(jl, log)
    mk = (
        log["mask_core"]
        & np.isfinite(log["obs_core"])
        & ((log["t"] <= T_OFF_MIN * 60.0) | (log["t"] >= T_ON_MIN * 60.0))
    )
    obs = np.asarray(log["obs_core"])[mk]
    print(
        f"erickson_2013: {log['meta']['t_end_h']:.1f} h, {mk.sum()} core obs; "
        f"off-pit {T_OFF_MIN:.0f}-{T_ON_MIN:.0f} min",
        flush=True,
    )

    zp = np.array([fwd[tf](c) for _, tf, c, _ in SITES])
    sd = np.array([s for _, _, _, s in SITES])

    def resid(z):
        tc = f.run(unpack(z))
        return np.concatenate([(tc[mk] - obs) / SIGMA_CORE, (z - zp) / sd])

    z, n = zp.copy(), len(zp)
    r = resid(z)
    F = 0.5 * float(r @ r)
    lam = 1e-3
    t0 = time.perf_counter()
    print(f"  it  0  F={F:9.3f}", flush=True)
    for it in range(1, 26):
        J = np.empty((r.size, n))
        for j in range(n):  # FD Jacobian, see docstring
            dz = np.zeros(n)
            dz[j] = 1e-4
            J[:, j] = (resid(z + dz) - r) / 1e-4
        g = J.T @ r
        H = J.T @ J
        ok = False
        for _ in range(8):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H)), -g)
            r2 = resid(z + step)
            F2 = 0.5 * float(r2 @ r2)
            if F2 < F:
                z, r, F = z + step, r2, F2
                lam = max(lam / 3, 1e-7)
                ok = True
                break
            lam *= 10
        print(
            f"  it {it:2d}  F={F:9.3f}  |g|={np.abs(g).max():8.2g}  "
            f"|step|={np.abs(step).max():7.3g}  lam={lam:7.1e}",
            flush=True,
        )
        if not ok or np.abs(step).max() < 1e-5:
            break
    wall = time.perf_counter() - t0

    Hz = J.T @ J
    cov = np.linalg.inv(0.5 * (Hz + Hz.T) + 1e-9 * np.eye(n))
    x = unpack(z)
    zsd = np.sqrt(np.diag(cov))
    tc_fit = f.run(x)
    rms = float(np.sqrt(np.mean((tc_fit[mk] - obs) ** 2)))
    print(f"\nfit ({wall:.0f} s): rms core {rms:.2f} K")
    for (name, tf, c, _), zi, si in zip(SITES, z, zsd, strict=True):
        lo, hi = inv[tf](zi - si), inv[tf](zi + si)
        print(f"  {name:9s} {x[name]:10.4g}  [{lo:.4g}, {hi:.4g}]  (prior {c:g})")

    tc_nowrap = f.run(x, wrapped=False)
    tc_foil = f.run(x, eta=0.9, eps=0.1)
    end_C = {
        nm: float(tc[-1] - 273.15)
        for nm, tc in (("fit", tc_fit), ("no_wrap", tc_nowrap), ("foil", tc_foil))
    }
    print(f"probe at pull [C]: {({k: round(v, 1) for k, v in end_C.items()})}")

    out = {
        "log": "erickson_2013",
        "segments_min": [0.0, T_OFF_MIN, T_ON_MIN],
        "sigma_core_K": SIGMA_CORE,
        "fixed": FIXED,
        "fit": {
            name: {
                "value": x[name],
                "z_sd": float(si),
                "lo1sd": float(inv[tf](zi - si)),
                "hi1sd": float(inv[tf](zi + si)),
                "prior_center": c,
            }
            for (name, tf, c, _), zi, si in zip(SITES, z, zsd, strict=True)
        },
        "rms_core_K": rms,
        "probe_at_pull_C": end_C,
        "z_cov": cov.tolist(),
        "wall_s": wall,
    }
    out["L_m"] = a.L
    (REPO / "data" / "calibration" / f"wrap_fit_erickson{a.tag}.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"wrote data/calibration/wrap_fit_erickson{a.tag}.json")
    _figure(log, mk, obs, f, tc_fit, tc_nowrap, tc_foil, x, a.tag)


def _figure(log, mk, obs, f, tc_fit, tc_nowrap, tc_foil, x, tag=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = log["t"] / 3600.0
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(th, f.tpit - 273.15, color="tab:orange", alpha=0.4, lw=1,
            label="pit (measured, drives the model)")
    ax.plot(th[mk], obs - 273.15, ".", ms=3.5, color="k", alpha=0.6,
            label="core observed")
    ax.plot(th, tc_fit - 273.15, color="tab:green", lw=1.8,
            label=(f"fit (paper: η_evap {x['eta_evap']:.2f}, "
                   f"ε_wrap {x['eps_wrap']:.2f}; probe ξ "
                   f"{x['xi_pre']:.2f}→{x['xi_post']:.2f})"))
    ax.plot(th, tc_nowrap - 273.15, "--", color="tab:red", lw=1.4,
            label="counterfactual: never wrapped")
    ax.plot(th, tc_foil - 273.15, ":", color="tab:blue", lw=1.4,
            label="counterfactual: foil (0.9 / 0.1)")
    ax.axvspan(T_OFF_MIN / 60, T_ON_MIN / 60, color="gray", alpha=0.25)
    ax.annotate("off pit, wrapped in paper,\nprobe re-seated",
                (T_OFF_MIN / 60, 103), textcoords="offset points",
                xytext=(-2, 0), fontsize=8, color="gray", ha="right")
    ax.set(
        xlabel="time after meat-on [h]", ylabel="temperature [C]",
        title="Wrap constants from a documented wrap: erickson_2013 step response",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / f"wrap_fit_erickson{tag}.png", dpi=140)
    plt.close(fig)
    print(f"wrote figures/wrap_fit_erickson{tag}.png")


if __name__ == "__main__":
    main()
