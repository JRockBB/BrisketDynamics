"""The next cook's prior from the calibration's chain, site by site in the
estimator's own families and by the pool's roles.

The pilot's first hours are flown on its prior, and the cook is decided
there. Until now that prior was the calibration's literature prior. This
writes the prior a second cook on the same cooker deserves:

- shared (cooker) sites, `PHI_SITES` in the app: the calibration chain's
  marginal, a lognormal from the log moments or a beta by moments (the
  chain's marginals are not Gaussian: u_leak is bounded, r_max skewed);
- partial (meat) sites, `THETA_SITES`: the chain's marginal widened by
  the pool's between-cook variance `tau2` in the unconstrained
  coordinate (in quadrature on a log sd; a beta's concentration scaled
  down to the same effect), because the next brisket is another animal;
- independent (ignition) sites, `PHI_SITES_IGNITION`: the literature
  prior, every fire lights its own way;
- measured sites (`--measured L=0.06`): what a pitmaster reads off the
  flat with a ruler before the cook, a lognormal around the reading with
  the reading's own precision (`--measured-sd`, log).

Writes data/calibration/<out> with the same "prior" list the gate file
carries, and prints where the mock's truth sits under the old and the new
prior. Run from app/:

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


def beta_moments(x):
    m, v = float(np.mean(x)), float(np.var(x))
    c = m * (1 - m) / v - 1
    return m * c, (1 - m) * c


def logit_var_beta(a, b):
    """Var of logit(X), X ~ Beta(a, b): trigamma(a) + trigamma(b)."""
    from scipy.special import polygamma
    return float(polygamma(1, a) + polygamma(1, b))


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

    from brisket_app.approx import IGNITION_IID, MEAT_PARTIAL, PIT_SHARED

    cal = REPO / "data" / "calibration"
    chain = np.load(cal / args.chain)
    gate = json.loads((cal / args.gate).read_text())
    names = list(gate["names"])
    lit = {p["name"]: p for p in gate["prior"]}
    measured = {k: float(v) for k, v in (s.split("=") for s in args.measured)}
    mock = json.loads((REPO / "data" / "mock" / args.mock).read_text())
    mnames = mock.get("names") or mock["truth"]["names"]
    truth = {n: float(v) for n, v in zip(mnames, mock["truth"]["x"])}

    def z_of(x, fam):
        return np.log(x / (1 - x)) if fam == "beta" else np.log(x)

    def z_score(x, p):
        """(z(x) - prior location) / prior scale in the unconstrained coordinate; a beta's by its logit moments."""
        if p["family"] == "beta":
            a, b = p["p1"], p["p2"]
            from scipy.special import digamma
            loc = float(digamma(a) - digamma(b))
            return (z_of(x, "beta") - loc) / np.sqrt(logit_var_beta(a, b))
        return (z_of(x, "lognormal") - p["p1"]) / p["p2"]

    out, rows = [], []
    for n in names:
        old = lit[n]
        fam = old["family"]
        x = np.asarray(chain[n], dtype=float)
        if n in measured:
            role, p = "measured", {"name": n, "family": "lognormal", "p1": float(np.log(measured[n])), "p2": args.measured_sd}
        elif n in IGNITION_IID and args.ignition == "independent":
            role, p = "independent", dict(old)
        else:
            # the ignition sites are per-cook in the pool (no population to pool them with); on the same cooker lit
            # the same way the next cook starts from this cooker's chain widened like the meat, unless told otherwise
            role = "shared" if n in PIT_SHARED else "partial" if n in MEAT_PARTIAL or n in IGNITION_IID else "independent"
            if fam == "beta":
                a, b = beta_moments(x)
                if role == "partial":
                    v = logit_var_beta(a, b)
                    c = v / (v + args.tau2)                       # scale the concentration: the logit variance grows by tau2
                    a, b = c * a, c * b
                p = {"name": n, "family": "beta", "p1": float(a), "p2": float(b)}
            else:
                mu, sd = float(np.mean(np.log(x))), float(np.std(np.log(x)))
                if role == "partial":
                    sd = float(np.sqrt(sd ** 2 + args.tau2))
                p = {"name": n, "family": "lognormal", "p1": mu, "p2": sd}
        out.append(p)
        med = float(np.median(x))
        rows.append((n, role, med, float(np.std(np.log(x))) if fam != "beta" else float(np.std(x)),
                     z_score(truth[n], old) if n in truth else float("nan"), z_score(truth[n], p) if n in truth else float("nan"), p))

    print(f"{'site':10s} {'role':12s} {'chain median':>13s} {'chain sd':>9s} {'truth z, old':>13s} {'truth z, new':>13s}  new prior")
    for n, role, med, sd, zo, zn, p in rows:
        desc = (f"lognormal(median {np.exp(p['p1']):.4g}, log-sd {p['p2']:.3f})" if p["family"] == "lognormal"
                else f"beta({p['p1']:.2f}, {p['p2']:.2f})")
        print(f"{n:10s} {role:12s} {med:13.4g} {sd:9.3f} {zo:13.2f} {zn:13.2f}  {desc}")
    zo = np.array([r[4] for r in rows]); zn = np.array([r[5] for r in rows])
    print(f"the mock's truth under the prior, |z| median: old {np.nanmedian(np.abs(zo)):.2f} new {np.nanmedian(np.abs(zn)):.2f}; "
          f"max: old {np.nanmax(np.abs(zo)):.2f} new {np.nanmax(np.abs(zn)):.2f}")
    dst = cal / args.out
    dst.write_text(json.dumps({"names": names, "prior": out, "source": args.chain, "tau2": args.tau2, "measured": measured,
                               "measured_sd": args.measured_sd, "roles": {r[0]: r[1] for r in rows}}, indent=1))
    print("wrote", dst.relative_to(REPO))


if __name__ == "__main__":
    main()
