"""The next cook's prior from the calibration's chain, site by site in the
estimator's own families and by the pool's roles.

The pilot's first hours are flown on its prior, and the cook is decided
there. This writes the prior a second cook on the same cooker deserves:

- shared (cooker) sites, `PHI_SITES` in the app: the calibration chain's
  marginal, a lognormal from the log moments, a beta by moments for the
  bounded ones, or a logbeta (a Beta on the fraction of a bounded log
  range) where the chain's log marginal is skewed (r_max), so the
  chain's shape is carried, not only its bounds;
- partial (meat) sites, `THETA_SITES`: the chain's marginal widened by
  the pool's between-cook variance `tau2` in the unconstrained
  coordinate (in quadrature on a log sd; a beta's or logbeta's
  concentration scaled down to the same effect), because the next
  brisket is another animal;
- independent (ignition) sites, `PHI_SITES_IGNITION`: the literature
  prior, every fire lights its own way;
- measured sites (`--measured L=0.06`): what a pitmaster reads off the
  flat with a ruler before the cook, a lognormal around the reading with
  the reading's own precision (`--measured-sd`, log).

Writes data/calibration/<out> with the same "prior" list the gate file
carries, and prints where the mock's truth sits under the old and the new
prior. `next_cook.py` pools pilot records into the chain and calls
`fit_families` here. Run from app/:

  uv run python ../estimator/export_chain_prior.py --measured L=<m> --out prior_next_cook.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

SKEW_LOGBETA = 0.5      # a lognormal site whose chain is this skewed in the log carries a logbeta instead
LOGBETA_MARGIN = 0.25   # the bounds of the log range: the chain's range extended by this fraction each side


def beta_moments(x):
    m, v = float(np.mean(x)), float(np.var(x))
    c = m * (1 - m) / v - 1
    return m * c, (1 - m) * c


def logit_var_beta(a, b):
    """Var of logit(X), X ~ Beta(a, b): trigamma(a) + trigamma(b)."""
    from scipy.special import polygamma
    return float(polygamma(1, a) + polygamma(1, b))


def z_of(x, p):
    """The unconstrained coordinate of x under a prior site (as the estimator's `to_z`)."""
    x = np.asarray(x, dtype=float)
    fam = p["family"]
    if fam == "beta":
        return np.log(x / (1 - x))
    if fam == "logbeta":
        f = np.clip((np.log(x) - p["p3"]) / (p["p4"] - p["p3"]), 1e-9, 1 - 1e-9)
        return np.log(f / (1 - f))
    return np.log(x)


def z_score(x, p):
    """(z(x) - prior location) / prior scale in the unconstrained coordinate; a beta's by its logit moments."""
    if p["family"] in ("beta", "logbeta"):
        from scipy.special import digamma
        a, b = p["p1"], p["p2"]
        loc = float(digamma(a) - digamma(b))
        return float((z_of(x, p) - loc) / np.sqrt(logit_var_beta(a, b)))
    return float((z_of(x, p) - p["p1"]) / p["p2"])


def skewness(v):
    v = np.asarray(v, dtype=float)
    s = np.std(v)
    return float(np.mean((v - np.mean(v)) ** 3) / s ** 3) if s > 0 else 0.0


def fit_site(name, x, fam, role, tau2, skew_logbeta=SKEW_LOGBETA):
    """One site's prior in the graph's families from samples `x` of its value: a beta by moments for a bounded
    site, a logbeta where a positive site's log samples are skewed, else a lognormal from the log moments; a
    partial site widened by tau2 in the unconstrained coordinate."""
    x = np.asarray(x, dtype=float)
    if fam == "beta":
        a, b = beta_moments(x)
        if role == "partial":
            v = logit_var_beta(a, b)
            c = v / (v + tau2)                       # scale the concentration: the logit variance grows by tau2
            a, b = c * a, c * b
        return {"name": name, "family": "beta", "p1": float(a), "p2": float(b)}
    lx = np.log(x)
    if abs(skewness(lx)) > skew_logbeta:
        lo, hi = float(lx.min()), float(lx.max())
        span = hi - lo
        lo, hi = lo - LOGBETA_MARGIN * span, hi + LOGBETA_MARGIN * span
        f = np.clip((lx - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        a, b = beta_moments(f)
        if role == "partial":
            v = logit_var_beta(a, b)
            c = v / (v + tau2)
            a, b = c * a, c * b
        return {"name": name, "family": "logbeta", "p1": float(a), "p2": float(b), "p3": lo, "p4": hi}
    mu, sd = float(np.mean(lx)), float(np.std(lx))
    if role == "partial":
        sd = float(np.sqrt(sd ** 2 + tau2))
    return {"name": name, "family": "lognormal", "p1": mu, "p2": sd}


def fit_families(chain, names, lit, tau2, measured, measured_sd, ignition="partial", roles=None):
    """The prior list from per-site samples `chain[name]` (the calibration chain, or a pooled pseudo-chain): each
    site by its role (`roles[name]` when given, else the pool's defaults by the app's site lists), in the graph's
    families. Returns (prior list, rows for the report)."""
    from brisket_app.approx import IGNITION_IID, MEAT_PARTIAL, PIT_SHARED
    out, rows = [], []
    for n in names:
        old = lit[n]
        fam = old["family"]
        x = np.asarray(chain[n], dtype=float)
        if n in measured:
            role, p = "measured", {"name": n, "family": "lognormal", "p1": float(np.log(measured[n])), "p2": measured_sd}
        elif n in IGNITION_IID and ignition == "independent":
            role, p = "independent", dict(old)
        else:
            # the ignition sites are per-cook in the pool (no population to pool them with); on the same cooker lit
            # the same way the next cook starts from this cooker's chain widened like the meat, unless told otherwise
            role = roles[n] if roles is not None and n in roles else \
                "shared" if n in PIT_SHARED else "partial" if n in MEAT_PARTIAL or n in IGNITION_IID else "independent"
            p = fit_site(n, x, fam, role, tau2)
        out.append(p)
        rows.append((n, role, float(np.median(x)), float(np.std(np.log(x))) if fam != "beta" else float(np.std(x)), p))
    return out, rows


def describe(p):
    if p["family"] == "lognormal":
        return f"lognormal(median {np.exp(p['p1']):.4g}, log-sd {p['p2']:.3f})"
    if p["family"] == "logbeta":
        return f"logbeta({p['p1']:.2f}, {p['p2']:.2f} on log [{p['p3']:.2f}, {p['p4']:.2f}])"
    return f"beta({p['p1']:.2f}, {p['p2']:.2f})"


def report(rows, truth, lit):
    print(f"{'site':10s} {'role':12s} {'chain median':>13s} {'chain sd':>9s} {'truth z, old':>13s} {'truth z, new':>13s}  new prior")
    zo, zn = [], []
    for n, role, med, sd, p in rows:
        o = z_score(truth[n], lit[n]) if n in truth else float("nan")
        w = z_score(truth[n], p) if n in truth else float("nan")
        zo.append(o); zn.append(w)
        print(f"{n:10s} {role:12s} {med:13.4g} {sd:9.3f} {o:13.2f} {w:13.2f}  {describe(p)}")
    zo, zn = np.array(zo), np.array(zn)
    print(f"the mock's truth under the prior, |z| median: old {np.nanmedian(np.abs(zo)):.2f} new {np.nanmedian(np.abs(zn)):.2f}; "
          f"max: old {np.nanmax(np.abs(zo)):.2f} new {np.nanmax(np.abs(zn)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="nuts_mono_samples_real.npz")
    ap.add_argument("--gate", default="estimator_gate_real.json")
    ap.add_argument("--tau2", type=float, default=0.04, help="between-cook variance on the meat sites, in z (the pool's default)")
    ap.add_argument("--measured", nargs="*", default=[], help="site=value read before the cook, e.g. L=0.062")
    ap.add_argument("--measured-sd", type=float, default=0.1, help="log-sd of a measured site")
    ap.add_argument("--ignition", choices=("partial", "independent"), default="partial",
                    help="the ignition sites: this cooker's chain widened like the meat (partial), or the literature prior (independent)")
    ap.add_argument("--mock", default="mock_1d.json", help="the mock whose truth is reported against the priors")
    ap.add_argument("--out", default="prior_next_cook.json")
    args = ap.parse_args()

    cal = REPO / "data" / "calibration"
    chain = np.load(cal / args.chain)
    gate = json.loads((cal / args.gate).read_text())
    names = list(gate["names"])
    lit = {p["name"]: p for p in gate["prior"]}
    measured = {k: float(v) for k, v in (s.split("=") for s in args.measured)}
    mock = json.loads((REPO / "data" / "mock" / args.mock).read_text())
    mnames = mock.get("names") or mock["truth"]["names"]
    truth = {n: float(v) for n, v in zip(mnames, mock["truth"]["x"])}

    out, rows = fit_families({n: chain[n] for n in names}, names, lit, args.tau2, measured, args.measured_sd, args.ignition)
    report(rows, truth, lit)
    dst = cal / args.out
    dst.write_text(json.dumps({"names": names, "prior": out, "source": args.chain, "tau2": args.tau2, "measured": measured,
                               "measured_sd": args.measured_sd, "roles": {r[0]: r[1] for r in rows}}, indent=1))
    print("wrote", dst.relative_to(REPO))


if __name__ == "__main__":
    main()
