"""The state's process-noise rates identified from the innovations' autocovariances: the bench's step 2.

The filter's worker (estimator/scripts/coverage_filter.jl) runs each cook with an accumulator (estimator/src/als.jl)
that carries, along the run, the sensitivity of every innovation autocovariance to each state group's variance
rate (the square of its rate) under the gains the filter actually used: the autocovariance least-squares method
(Odelson, Rajamani, Rawlings 2006) in its fixed-gain, time-varying form (Åkesson et al. 2008; Zhang et al. 2020,
section IV). Under the filter's own model its innovations are white with the predicted covariance; the products
of the standardized innovations at every pair of updates inside the lag window, less that model, regress on the
change of the twelve variance rates, and the accumulator's normal equations pooled over the cooks give the least
squares fit. This module solves it:

  * the rows are weighted by their own residual variance per channel pair and kind (the squares at one update,
    the products across updates), the method's weighting made feasible from the rows themselves: the weights
    start at one, each block's residual variance at the fit re-weights it, and the fit is repeated until the
    weights hold (a probe whose innovations are tiny against its predicted spread has products a hundredth as
    scattered as another's, and would otherwise be drowned); the weighted rows' noise is one when the weights
    are right;
  * a group enters the fit when the innovations see it: when its own column pins its variance rate to a
    relative standard error under `REL_SE_MAX`, or when the unbounded fit moves it by more than `MOVE_SIGMAS`
    standard errors (a rate that is small today but that the innovations want larger); the rows' noise is
    the residual of that unbounded fit over every group with a column, so a large change is not mistaken for
    noise. A group the innovations cannot see keeps its numbers and is reported as unidentified, exactly as a
    group without a finite NEES did under the covariance-matching step;
  * the fit is bounded: no variance rate leaves [θ/cap², θ·cap²] (the same cap on the rate as before), and no
    variance rate is negative;
  * after the bounded fit a group whose joint standard error exceeds its fitted variance rate without a
    significant move (the fit cannot tell its sign) is held at its number and the fit is solved again;
  * the initial widths are not identified here (their transient is one; the rates recur) and are kept;
  * the standard errors are quoted twice: from the model's rows, which treats the products as independent, and
    from a leave-one-cook-out jackknife over the cooks' own statistics, which does not; a cook's products share
    its plant and its belief, so the jackknife is the one to read when the two disagree.

The fit needs no truth, so the same step applies to a real cook's innovations, where the bench's covariance
matching against the plant's state could not.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

WEIGHT_ROUNDS = 6       # the feasible weighting: the blocks re-weighted by their residual variance at the fit, until the weights hold
REL_SE_MAX = 0.5        # a group enters the fit when its own column pins its variance rate to better than ±50 % ...
MOVE_SIGMAS = 2.0       # ... or when the unbounded fit moves it by more than this many standard errors
REL_SE_HOLD = 1.0       # after the fit, a group whose joint standard error exceeds its variance rate, and which the unbounded fit did not move, is held


def _finite(v) -> float | None:
    """A number the rates file can carry: JSON has no infinity and the Julia side parses the file strictly."""
    v = float(v)
    return v if np.isfinite(v) else None


def pool(summaries: list[dict]) -> dict:
    """The cooks' accumulators pooled: the same groups, variance rates, window and bin on every cook."""
    if not summaries:
        raise ValueError("no accumulator summaries to pool")
    first = summaries[0]
    groups = list(first["groups"]); theta = np.array(first["theta"], dtype=float); n_g = len(groups)
    G = np.zeros((n_g, n_g)); h = np.zeros(n_g); rss = 0.0; n_rows = 0; n_updates = 0
    pairs: dict[tuple[int, int, int], np.ndarray] = {}
    blocks: dict[tuple[int, int, bool], dict] = {}
    for s in summaries:
        if list(s["groups"]) != groups or not np.allclose(s["theta"], theta) or s["lag_s"] != first["lag_s"] or s["bin_s"] != first["bin_s"]:
            raise ValueError("the cooks' accumulators disagree on the groups, the rates, the window or the bin")
        G += np.array(s["G"], dtype=float); h += np.array(s["h"], dtype=float); rss += float(s["rss"])
        n_rows += int(s["n_rows"]); n_updates += int(s["n_updates"])
        for p in s["pairs"]:
            key = (int(p["a"]), int(p["b"]), int(p["bin"]))
            v = pairs.setdefault(key, np.zeros(4 + n_g))
            v[0] += p["n"]; v[1] += p["sum_product"]; v[2] += p["sum_product2"]; v[3] += p["sum_base"]; v[4:] += np.array(p["sum_b"], dtype=float)
        for b in s["blocks"]:
            key = (int(b["a"]), int(b["b"]), bool(b["same"]))
            blk = blocks.setdefault(key, {"G": np.zeros((n_g, n_g)), "h": np.zeros(n_g), "rss": 0.0, "n": 0, "s_sum": 0.0, "s_sq": 0.0})
            blk["G"] += np.array(b["G"], dtype=float); blk["h"] += np.array(b["h"], dtype=float); blk["rss"] += float(b["rss"])
            blk["n"] += int(b["n"]); blk["s_sum"] += float(b["s_sum"]); blk["s_sq"] += float(b["s_sq"])
    return {"groups": groups, "theta": theta, "G": G, "h": h, "rss": rss, "n_rows": n_rows, "n_updates": n_updates,
            "n_cooks": len(summaries), "lag_s": float(first["lag_s"]), "bin_s": float(first["bin_s"]), "pairs": pairs, "blocks": blocks}


def _solve(G: np.ndarray, h: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The bounded least squares min δᵀGδ − 2hᵀδ over lo ≤ δ ≤ hi, through the normal equations' factor."""
    n = len(h)
    if n == 0:
        return np.zeros(0)
    Gs = 0.5 * (G + G.T)
    ridge = 1e-12 * max(np.trace(Gs) / n, 1e-300)
    L = np.linalg.cholesky(Gs + ridge * np.eye(n))
    A = L.T
    b = np.linalg.solve(L, h)
    res = lsq_linear(A, b, bounds=(lo, hi), method="bvls", tol=1e-12, max_iter=10_000)
    return np.clip(res.x, lo, hi)


def _weighted(blocks: dict, weights: dict, n_g: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The normal equations and the cost at δθ = 0 with each block at its weight."""
    G = np.zeros((n_g, n_g)); h = np.zeros(n_g); rss = 0.0
    for k, b in blocks.items():
        w = weights[k]
        G += w * b["G"]; h += w * b["h"]; rss += w * b["rss"]
    return G, h, rss


def _block_variances(blocks: dict, delta: np.ndarray) -> dict:
    """Each block's residual variance at the fit δ: its cost there over its rows."""
    out = {}
    for k, b in blocks.items():
        rss_k = b["rss"] - 2 * b["h"] @ delta + delta @ b["G"] @ delta
        out[k] = max(rss_k, 0.0) / max(b["n"] - 1, 1) if b["n"] >= 2 else float("nan")
    return out


def als_fit(pooled: dict, cap: float = 4.0, summaries: list[dict] | None = None) -> dict:
    """The rates from the pooled statistics: per group the variance rate before and after, the factor on the rate,
    the standard error of the rate, whether it was identified and whether the cap bound; the fit's cost before
    and after and the rows' noise; the marginal relative standard error that judged each group; per block its
    weight (the inverse of its residual variance); with the cooks' `summaries` (three or more) the leave-one-cook-out
    jackknife of the whole procedure: each group's rate's jackknife standard error and in how many of the
    leave-one-out fits it was identified and at the cap."""
    groups = list(pooled["groups"]); theta = np.asarray(pooled["theta"], dtype=float); n_g = len(groups)
    N = int(pooled["n_rows"])
    if N < n_g + 2:
        raise ValueError(f"{N} rows cannot identify {n_g} rates")
    blocks = pooled["blocks"]
    if not blocks:
        raise ValueError("the pooled statistics carry no blocks")
    # the feasible weighting: the unbounded fit over every group with a column, each block re-weighted by its
    # residual variance at that fit, until the weights hold
    weights = {k: 1.0 for k in blocks}
    G, h, rss0 = _weighted(blocks, weights, n_g)
    seen = [g for g in range(n_g) if np.diag(G)[g] > 0]
    for _ in range(WEIGHT_ROUNDS):
        du_full = np.zeros(n_g)
        if seen:
            Gs = G[np.ix_(seen, seen)]; Gs = Gs + 1e-12 * np.trace(Gs) / len(seen) * np.eye(len(seen))
            du_full[seen] = np.linalg.solve(Gs, h[seen])
        var = _block_variances(blocks, du_full)
        new = {k: (1.0 / v if np.isfinite(v) and v > 0 else weights[k]) for k, v in var.items()}
        changed = max(abs(new[k] / weights[k] - 1) for k in blocks)
        weights = new
        G, h, rss0 = _weighted(blocks, weights, n_g)
        if changed < 1e-3:
            break
    diag = np.diag(G)
    # the unbounded fit over every group with a column: the rows' noise from its residual, each group's own
    # precision (its column alone) and its move in standard errors (the joint fit)
    seen = [g for g in range(n_g) if diag[g] > 0]
    sigma2 = rss0 / (N - 1)
    moved_sigmas = np.zeros(n_g)
    if seen:
        Gs = G[np.ix_(seen, seen)]; Gs = Gs + 1e-12 * np.trace(Gs) / len(seen) * np.eye(len(seen))
        du = np.linalg.solve(Gs, h[seen])
        rss_u = rss0 - 2 * h[seen] @ du + du @ Gs @ du
        sigma2 = max(rss_u, 0.0) / max(N - len(seen), 1)
        se_u = np.sqrt(np.maximum(np.diag(np.linalg.inv(Gs)) * sigma2, 0.0))
        moved_sigmas[seen] = np.abs(du) / np.maximum(se_u, 1e-300)
    rel_se_marginal = np.array([np.sqrt(sigma2 / diag[g]) / theta[g] if diag[g] > 0 and theta[g] > 0 else np.inf for g in range(n_g)])
    active = [g for g in seen if rel_se_marginal[g] < REL_SE_MAX or moved_sigmas[g] > MOVE_SIGMAS]
    held: list[tuple[int, str]] = [(g, "unidentified") for g in range(n_g) if g not in active]
    delta = np.zeros(n_g); se_theta = np.full(n_g, np.nan); rss1 = rss0
    while True:
        a = np.array(active, dtype=int)
        lo = theta[a] / cap ** 2 - theta[a]; hi = theta[a] * cap ** 2 - theta[a]
        d = _solve(G[np.ix_(a, a)], h[a], lo, hi) if len(a) else np.zeros(0)
        delta = np.zeros(n_g); delta[a] = d
        rss1 = rss0 - 2 * h @ delta + delta @ G @ delta
        sigma2 = rss1 / max(N - len(a), 1)
        se_theta = np.full(n_g, np.nan)
        if len(a):
            Ga = G[np.ix_(a, a)]
            cov = sigma2 * np.linalg.inv(Ga + 1e-12 * np.trace(Ga) / len(a) * np.eye(len(a)))
            se_theta[a] = np.sqrt(np.maximum(np.diag(cov), 0.0))
        worst = None
        for g in active:
            rel = se_theta[g] / max(theta[g] + delta[g], 1e-300)
            if rel > REL_SE_HOLD and moved_sigmas[g] < MOVE_SIGMAS and (worst is None or rel > worst[1]):
                worst = (g, rel)
        if worst is None:
            break
        active.remove(worst[0]); held.append((worst[0], "held: the fit could not tell its sign"))
    theta_new = theta + delta
    rates_new = np.sqrt(np.maximum(theta_new, 0.0)); rates = np.sqrt(theta)
    block_var = _block_variances(blocks, delta)
    at_cap = [bool(g in active and (abs(theta_new[g] - theta[g] / cap ** 2) <= 1e-9 * theta[g] or abs(theta_new[g] - theta[g] * cap ** 2) <= 1e-9 * theta[g]))
              for g in range(n_g)]
    jk = _jackknife(summaries, cap) if summaries is not None and len(summaries) >= 3 else None
    return {"jackknife": jk, "groups": groups, "theta": theta.tolist(), "theta_new": theta_new.tolist(), "rates": rates.tolist(), "rates_new": rates_new.tolist(),
            "factors": [float(rates_new[g] / rates[g]) if rates[g] > 0 else 1.0 for g in range(n_g)],
            "se_rate": [_finite(se_theta[g] / (2 * rates_new[g])) if rates_new[g] > 0 else None for g in range(n_g)],
            "identified": [bool(g in active) for g in range(n_g)],
            "held": {groups[g]: why for g, why in held},
            "cap_bound": at_cap,
            "rel_se_marginal": [_finite(v) for v in rel_se_marginal], "moved_sigmas": [_finite(v) for v in moved_sigmas],
            "rss_before": rss0, "rss_after": float(rss1), "n_rows": N, "sigma2": float(sigma2),
            "blocks": [{"a": k[0], "b": k[1], "same": k[2], "n": int(blocks[k]["n"]), "weight": _finite(weights[k]), "variance_at_fit": _finite(block_var[k])}
                       for k in sorted(blocks)],
            "n_cooks": pooled.get("n_cooks"), "n_updates": pooled.get("n_updates"), "lag_s": pooled.get("lag_s"), "bin_s": pooled.get("bin_s"), "cap": cap}


def _jackknife(summaries: list[dict], cap: float) -> dict:
    """The procedure repeated with each cook left out: the jackknife standard error of every group's rate and its
    variance rate, and the counts of leave-one-out fits in which the group was identified and at the cap."""
    n = len(summaries)
    fits = [als_fit(pool(summaries[:i] + summaries[i + 1:]), cap=cap) for i in range(n)]
    theta = np.array([f["theta_new"] for f in fits], dtype=float)
    rates = np.array([f["rates_new"] for f in fits], dtype=float)
    se_theta = np.sqrt((n - 1) / n * np.sum((theta - theta.mean(axis=0)) ** 2, axis=0))
    se_rate = np.sqrt((n - 1) / n * np.sum((rates - rates.mean(axis=0)) ** 2, axis=0))
    return {"n_cooks": n, "se_theta": [_finite(v) for v in se_theta], "se_rate": [_finite(v) for v in se_rate],
            "identified_in": [int(sum(f["identified"][g] for f in fits)) for g in range(theta.shape[1])],
            "at_cap_in": [int(sum(f["cap_bound"][g] for f in fits)) for g in range(theta.shape[1])],
            "rates_min": rates.min(axis=0).tolist(), "rates_max": rates.max(axis=0).tolist()}


def lag_table(pooled: dict, fit: dict, channels: dict[int, str], bins: list[int]) -> list[dict]:
    """Per channel (its own products) and lag bin: the sample mean product, the model at the rates the round ran
    with, and the model at the fitted rates; the standard error of the sample mean."""
    delta = np.asarray(fit["theta_new"], dtype=float) - np.asarray(fit["theta"], dtype=float)
    rows = []
    for c, lab in channels.items():
        for bn in bins:
            v = pooled["pairs"].get((c, c, bn))
            if v is None or v[0] < 2:
                continue
            n = v[0]; sample = v[1] / n; base = v[3] / n; model_new = base + (v[4:] @ delta) / n
            var_p = max(v[2] / n - sample ** 2, 0.0) * n / (n - 1)            # the products' own spread
            se = np.sqrt(var_p / n) if n >= 3 else np.sqrt((2.0 if bn == 0 else 1.0) / n)
            rows.append({"channel": lab, "bin": int(bn), "lag_s": float(bn * pooled["bin_s"]), "n": int(n), "sample": float(sample), "se": float(se),
                         "model_before": float(base), "model_after": float(model_new)})
    return rows


def format_fit(fit: dict, units: dict[str, str], channels: dict[int, str] | None = None) -> str:
    """The fit as the driver prints it."""
    ch = channels or {}
    lines = [(f"  the autocovariance fit: {fit['n_rows']} products over {fit['n_cooks']} cooks ({fit['n_updates']} updates), lags to {fit['lag_s'] / 3600:.1f} h; "
              f"weighted cost {fit['rss_before']:.0f} before, {fit['rss_after']:.0f} after (the weighted rows' noise {fit['sigma2']:.2f}, one when the weights are right)"),
             "  the blocks' residual variance at the fit (their weight is its inverse): "
             + "; ".join(f"{ch.get(b['a'], b['a'])}×{ch.get(b['b'], b['b'])} {'at one update' if b['same'] else 'across updates'} {b['variance_at_fit']:.3g} (n {b['n']})"
                         for b in fit["blocks"] if b["variance_at_fit"] is not None and b["a"] == b["b"]),
             f"  {'group':9s} {'rate':>10s} {'-> rate':>10s} {'factor':>7s} {'se (rate)':>10s} {'jackknife se':>13s} {'in k of n':>10s}  {'own rel se':>10s} {'move/se':>8s}  status"]
    jk = fit.get("jackknife")
    for g, name in enumerate(fit["groups"]):
        status = "identified" if fit["identified"][g] else fit["held"].get(name, "")
        if fit["cap_bound"][g]:
            status += ", at the cap"
        se = fit["se_rate"][g]; rel = fit["rel_se_marginal"][g]; mv = fit["moved_sigmas"][g]
        se_s = f"{se:10.3g}" if se is not None else f"{'-':>10s}"
        rel_s = f"{rel:10.2f}" if rel is not None else f"{'-':>10s}"
        mv_s = f"{mv:8.1f}" if mv is not None else f"{'-':>8s}"
        if jk is not None:
            jse = jk["se_rate"][g]
            jk_s = f"{jse:13.3g}" if jse is not None else f"{'-':>13s}"
            in_s = f"{jk['identified_in'][g]:3d} of {jk['n_cooks']:<3d}"
        else:
            jk_s = f"{'-':>13s}"; in_s = f"{'-':>10s}"
        lines.append(f"  {name:9s} {fit['rates'][g]:10.4g} {fit['rates_new'][g]:10.4g} {fit['factors'][g]:7.2f} {se_s} {jk_s} {in_s:>10s}  {rel_s} {mv_s}  {status} [{units.get(name, '')}]")
    return "\n".join(lines)


def format_lags(rows: list[dict]) -> str:
    """The lag table as the driver prints it: per channel, the sample product by lag against the model before and after."""
    if not rows:
        return "  (no lag pairs)"
    out = [f"  {'channel':10s} {'lag':>7s} {'n':>6s} {'sample':>8s} {'+-se':>6s} {'model before':>13s} {'model after':>12s}"]
    for r in rows:
        out.append(f"  {r['channel']:10s} {r['lag_s'] / 60:5.0f} m {r['n']:6d} {r['sample']:8.3f} {r['se']:6.3f} {r['model_before']:13.3f} {r['model_after']:12.3f}")
    return "\n".join(out)
