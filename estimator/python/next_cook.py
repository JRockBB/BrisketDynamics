"""Learning across cooks: the next cook's prior, noise prior and drift rates from the cooks flown so far.

A pilot record is a cook: the filter's final belief over the latents is this cook's posterior (a Gaussian in
z), its noise posteriors per channel say how the instruments behaved, and the belief's walk on the moving sites
says how fast the plant drifted. This script pools them with the calibration's chain (the cooker's own real
cook) through the app's two-stage pool (`brisket_app.approx.hierarchical_pool`: the cooker's sites shared, the
meat's partially pooled with a between-cook variance, the ignition per cook) and writes:

- `<out>` (data/calibration): the next cook's prior in the graph's families (`export_chain_prior.fit_families`
  on a pseudo-chain drawn from the pooled Gaussians), the between-cook variance estimated from the cooks once
  three or more share a role (`tau2 = "auto"` in the pool), the placeholder until then; with a `drift` field:
  the moving sites' random-walk rates identified from the beliefs' hourly increments (method of moments net of
  the estimation noise, floored at half the modelling choice), which the filter's callers read in place of the
  constant;
- `<noise-out>`: the noise prior per channel, the cooks' posteriors pooled with a between-cook spread measured
  across them (in place of the fixed floor once two cooks exist), on the calibration's ten-minute grid.

A record's belief is a posterior under the prior it flew on; its likelihood is recovered by dividing that prior
out (the precision at the posterior mean), so the chain's prior is counted once across the pool. Run from app/:

  uv run python ../estimator/python/next_cook.py --records closed_loop_realcook_drift_filter.json [more...] --out prior_next_cook_2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_chain_prior import describe, fit_families, logit_var_beta, report, z_of  # noqa: E402

MOVING = ("u_leak", "k_e", "h_c")
JUMP_LOG_SD = {"r_max": 0.15, "dH_eff": 0.10}      # drift.jl's modelling choice: one cook's add cannot identify it
DRIFT_CHOICE = {"u_leak": 0.05, "k_e": 0.05, "h_c": 0.03}
NOISE_GRID_S = 600.0
CHANNELS = ("pit", "core", "humidity", "mass")
NOISE_FLOORS = {"pit": None, "core": None, "humidity": 60.0, "mass": 0.015}


def prior_precision(p, z):
    """The prior's curvature at z in the unconstrained coordinate (the graph's local quadratic): the Gaussian's
    precision, or the logit-Beta's (a+b) s (1-s)."""
    if p["family"] in ("beta", "logbeta"):
        s = 1 / (1 + np.exp(-z))
        return (p["p1"] + p["p2"]) * s * (1 - s)
    return 1 / p["p2"] ** 2


def prior_center(p):
    if p["family"] in ("beta", "logbeta"):
        return float(np.log(p["p1"] / p["p2"]))
    return float(p["p1"])


def record_gaussian(rec, names15):
    """A pilot record as a cook's Gaussian in z over the 15 physics sites: the filter's final belief, its prior
    divided out; the moving sites at the cook-mean of the belief with the trajectory's spread."""
    rows = rec["checks"]
    rnames = list(rec["names"])
    b = rec["belief"]
    z = np.asarray(b["z"], dtype=float)
    cov = np.asarray(b["cov"], dtype=float)
    prior = json.loads((REPO / "data" / "calibration" / rec["prior_file"]).read_text())
    psites = {p["name"]: p for p in prior["prior"]}
    idx = [rnames.index(n) for n in names15]
    z15, cov15 = z[idx], cov[np.ix_(idx, idx)]
    # the moving sites: the belief over the cook, not its last value
    zs = np.array([r["posterior_z"] for r in rows if r["updated"]])
    sds = np.array([r["posterior_sd"] for r in rows if r["updated"]])
    for n in MOVING:
        i, j = names15.index(n), rnames.index(n)
        z15[i] = float(np.mean(zs[:, j]))
        v = float(np.mean(sds[:, j] ** 2) + np.var(zs[:, j]))
        cov15[i, :] = 0.0; cov15[:, i] = 0.0; cov15[i, i] = v
    # divide the prior out: the likelihood's Gaussian
    lam_post = np.linalg.inv(0.5 * (cov15 + cov15.T) + 1e-10 * np.eye(len(idx)))
    lam_prior = np.diag([prior_precision(psites[n], z15[i]) for i, n in enumerate(names15)])
    mu_prior = np.array([prior_center(psites[n]) for n in names15])
    lam_lik = lam_post - lam_prior
    w, V = np.linalg.eigh(0.5 * (lam_lik + lam_lik.T))
    floor = 1e-3 * max(np.max(w), 1.0)
    lam_lik = V @ np.diag(np.maximum(w, floor)) @ V.T                 # a site the cook did not inform keeps a wide likelihood
    eta = lam_post @ z15 - lam_prior @ mu_prior
    cov_lik = np.linalg.inv(lam_lik)
    return cov_lik @ eta, 0.5 * (cov_lik + cov_lik.T)


def chain_gaussian(chain, names15, lit):
    z = np.stack([z_of(chain[n], lit[n]) for n in names15], axis=1)
    return z.mean(axis=0), np.cov(z, rowvar=False)


def embed(mean15, cov15, names15, all_names):
    d = len(all_names)
    mean, cov = np.zeros(d), np.eye(d) * 0.05
    for i, ni in enumerate(names15):
        mean[all_names.index(ni)] = mean15[i]
        for k, nk in enumerate(names15):
            cov[all_names.index(ni), all_names.index(nk)] = cov15[i, k]
    return mean, cov


def pseudo_chain(pool, names15, lit, rng, n=4000):
    """Samples of each site's value from the pooled Gaussians: the shared block's, the partial block's hypermean
    (the family fit widens it by tau2), the ignition sites from the chain's own marginal (the pool leaves them per
    cook), in the site's constrained coordinate."""
    from scipy.stats import multivariate_normal
    out = {}
    def draw(mean, cov):
        cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(len(mean))
        return multivariate_normal(mean, cov, allow_singular=True).rvs(n, random_state=rng).reshape(n, -1)
    zs = draw(pool.shared_mean, pool.shared_cov) if pool.shared_idx.size else None
    zp = draw(pool.partial_hyper_mean, pool.partial_hyper_cov) if pool.partial_idx.size else None
    for j, gi in enumerate(pool.shared_idx):
        nm = pool.names[gi]
        out[nm] = zs[:, j]
    for j, gi in enumerate(pool.partial_idx):
        nm = pool.names[gi]
        out[nm] = zp[:, j]
    def x_of(z, p):
        if p["family"] == "beta":
            return 1 / (1 + np.exp(-z))
        if p["family"] == "logbeta":
            return np.exp(p["p3"] + (p["p4"] - p["p3"]) / (1 + np.exp(-z)))
        return np.exp(z)
    return {n: x_of(out[n], lit[n]) for n in out}


def pooled_noise(chain_noise, records, dt_record):
    """Per channel the cooks' noise posteriors (σ, ρ on the ten-minute grid) pooled: the mean of the means, the
    within-cook width in quadrature with the spread between cooks."""
    a = dt_record / NOISE_GRID_S                     # ρ per ten minutes from ρ per record step: ρ_600 = ρ_dt^(600/dt)
    out = {}
    for c in CHANNELS:
        entries = []
        if c in chain_noise:
            v = chain_noise[c]
            entries.append((v["sigma_mean"], v["sigma_sd"], v["rho_mean"], v["rho_sd"]))
        for rec in records:
            last = next(r for r in reversed(rec["checks"]) if r["updated"])
            nz = last["noise"][c]
            rho = float(nz["rho"]) ** (1 / a)
            rho_sd = float(nz["rho_sd"]) * (1 / a) * float(nz["rho"]) ** (1 / a - 1)
            entries.append((float(nz["sigma"]), float(nz["sigma_sd"]), rho, rho_sd))
        if not entries:
            continue
        e = np.array(entries)
        K = len(entries)
        between = lambda col, sdcol: max(float(np.var(e[:, col], ddof=1)) - float(np.mean(e[:, sdcol] ** 2)), 0.0) if K > 1 else 0.0
        out[c] = {"sigma_mean": float(np.mean(e[:, 0])), "sigma_sd": float(np.sqrt(np.mean(e[:, 1] ** 2) + between(0, 1))),
                  "rho_mean": float(np.mean(e[:, 2])), "rho_sd": float(np.sqrt(np.mean(e[:, 3] ** 2) + between(2, 3))),
                  "n_cooks": K, "between_sigma_sd": float(np.sqrt(between(0, 1))), "between_rho_sd": float(np.sqrt(between(2, 3)))}
    return out


def drift_rates(records):
    """The moving sites' random-walk rates from the beliefs' hourly increments across the records: the increments'
    variance net of the estimation noise (method of moments), floored at half the modelling choice."""
    rates = {}
    for n in MOVING:
        vs = []
        for rec in records:
            j = list(rec["names"]).index(n)
            rows = [r for r in rec["checks"] if r["updated"]]
            hours = {}
            for r in rows:
                h = int(np.floor(r["t_h"] + 1e-9))
                hours[h] = r                                   # the last update of each hour
            hs = sorted(hours)
            if len(hs) < 4:
                continue
            z = np.array([hours[h]["posterior_z"][j] for h in hs]); sd = np.array([hours[h]["posterior_sd"][j] for h in hs])
            inc = np.diff(z) / np.diff(np.array(hs, dtype=float))
            vs.append(max(float(np.var(inc)) - float(np.mean(sd ** 2)), 0.0))
        rates[n] = float(np.sqrt(max(np.mean(vs) if vs else 0.0, (0.5 * DRIFT_CHOICE[n]) ** 2)))
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="pilot records in data/mock (mock_closed_loop_realcook.jl)")
    ap.add_argument("--chain", default="nuts_mono_samples_real.npz")
    ap.add_argument("--gate", default="estimator_gate_real.json")
    ap.add_argument("--noise-in", default="noise_prior_real.json")
    ap.add_argument("--tau2", default="auto", help="'auto' (from the cooks once three share a role) or a number")
    ap.add_argument("--tau2-placeholder", type=float, default=0.04)
    ap.add_argument("--measured", nargs="*", default=[], help="site=value read before the next cook, e.g. L=0.062")
    ap.add_argument("--measured-sd", type=float, default=0.1)
    ap.add_argument("--mock", default="mock_1d.json")
    ap.add_argument("--out", default="prior_next_cook_2.json")
    ap.add_argument("--noise-out", default="noise_prior_next.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from brisket_app.approx import IGNITION_IID, MEAT_PARTIAL, PIT_SHARED, hierarchical_pool
    from brisket_app.monolith import ALL_NAMES_SORTED

    cal = REPO / "data" / "calibration"
    chain = np.load(cal / args.chain)
    gate = json.loads((cal / args.gate).read_text())
    names15 = list(gate["names"])
    lit = {p["name"]: p for p in gate["prior"]}
    records = [json.loads((REPO / "data" / "mock" / r).read_text(encoding="utf-8")) for r in args.records]
    for r, name in zip(records, args.records):
        if "belief" not in r or "actual" not in r:
            raise SystemExit(f"{name}: not a finished pilot record (no final belief)")
    all_names = list(ALL_NAMES_SORTED)
    means, covs = [], []
    m0, c0 = chain_gaussian(chain, names15, lit)
    means.append(embed(m0, c0, names15, all_names)[0]); covs.append(embed(m0, c0, names15, all_names)[1])
    for rec in records:
        m1, c1 = record_gaussian(rec, names15)
        e = embed(m1, c1, names15, all_names)
        means.append(e[0]); covs.append(e[1])
    K = len(means)
    tau2 = "auto" if args.tau2 == "auto" and K >= 3 else (args.tau2_placeholder if args.tau2 == "auto" else float(args.tau2))
    pool = hierarchical_pool(means, covs, names=all_names, shared=PIT_SHARED, partial=MEAT_PARTIAL, independent=IGNITION_IID,
                             tau2=tau2, subtract_shared_prior=False)
    tau2_used = float(np.mean(np.asarray(pool.partial_hyper_cov).diagonal() * 0 + (tau2 if not isinstance(tau2, str) else 0.0)))
    if isinstance(tau2, str):
        from brisket_app.approx import estimate_tau2
        t2 = estimate_tau2(means, covs, pool.partial_idx)
        tau2_used = float(np.mean(t2))
        print("between-cook variance from the cooks (per meat site):", {all_names[i]: round(float(v), 4) for i, v in zip(pool.partial_idx, t2)})
    rng = np.random.default_rng(args.seed)
    pc = pseudo_chain(pool, names15, lit, rng)
    for n in IGNITION_IID:
        pc[n] = np.asarray(chain[n], dtype=float)              # the ignition sites: this cooker's chain, widened like the meat by the family fit
    measured = {k: float(v) for k, v in (s.split("=") for s in args.measured)}
    out, rows = fit_families(pc, names15, lit, tau2_used, measured, args.measured_sd, "partial")
    mock = json.loads((REPO / "data" / "mock" / args.mock).read_text())
    mnames = mock.get("names") or mock["truth"]["names"]
    truth = {n: float(v) for n, v in zip(mnames, mock["truth"]["x"])}
    print(f"{K} cooks pooled: the calibration's chain and {len(records)} pilot record(s); tau2 {tau2_used:.4f}"
          + (" (estimated)" if isinstance(tau2, str) else " (the placeholder: fewer than three cooks share a role)"))
    report(rows, truth, lit)
    rates = drift_rates(records)
    print("drift rates per hour from the beliefs' walks:", {k: round(v, 4) for k, v in rates.items()}, "(the modelling choice was", DRIFT_CHOICE, ")")
    dst = cal / args.out
    dst.write_text(json.dumps({"names": names15, "prior": out, "source": f"{args.chain} + {', '.join(args.records)}", "tau2": tau2_used,
                               "tau2_estimated": isinstance(tau2, str), "n_cooks": K, "measured": measured, "measured_sd": args.measured_sd,
                               "roles": {r[0]: r[1] for r in rows},
                               "drift": {"sd_per_hour": rates, "jump_log_sd": JUMP_LOG_SD, "from_records": args.records}}, indent=1))
    print("wrote", dst.relative_to(REPO))
    noise_in = json.loads((cal / args.noise_in).read_text())
    dt_rec = float(records[0].get("grid_dt_s", 120.0))
    pooled = pooled_noise(noise_in["channels"], records, dt_rec)
    for c, v in pooled.items():
        fl = NOISE_FLOORS[c] if NOISE_FLOORS[c] is not None else noise_in["sigma_floor_K"]
        print(f"  noise {c:9s}: sigma {v['sigma_mean']:.3f} ± {v['sigma_sd']:.3f} (between cooks {v['between_sigma_sd']:.3f}), "
              f"rho/10 min {v['rho_mean']:.3f} ± {v['rho_sd']:.3f} (between {v['between_rho_sd']:.3f}); {v['n_cooks']} cooks; floor {fl}")
    ndst = cal / args.noise_out
    ndst.write_text(json.dumps({"source": f"{args.noise_in} + {', '.join(args.records)}: the cooks' noise posteriors pooled, the spread between cooks measured",
                                "sigma_floor_K": noise_in["sigma_floor_K"], "grid_s": NOISE_GRID_S,
                                "floors": {c: (NOISE_FLOORS[c] if NOISE_FLOORS[c] is not None else noise_in["sigma_floor_K"]) for c in pooled},
                                "channels": pooled}, indent=1))
    print("wrote", ndst.relative_to(REPO))


if __name__ == "__main__":
    main()
