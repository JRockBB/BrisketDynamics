"""Cheap posterior approximations: PSIS + moment matching, nested Laplace, pooling.

Gauss-Newton Laplace is a Gaussian in numpyro's unconstrained space. NUTS
disagrees on ignition (``k_spread``, ``a0``): heavier tails and skew. The
TMB/INLA nested grid (`nested_tmb_grid`) is the family that can represent
that; `psis_smooth` / `khat_threshold` / `moment_match_psis` are the
diagnostics and the repair that say whether a proposal is close enough to
resample. `hierarchical_pool` is the two-stage multi-cook pool.

Nothing here calls the Tesseract. Scripts pass a potential callable when the
true ODE log-density is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES
from .monolith import ALL_NAMES_SORTED, NAMES, to_z

PIT_SHARED = [n for n, _ in PHI_SITES]
MEAT_PARTIAL = [n for n, _ in THETA_SITES]
IGNITION_IID = [n for n, _ in PHI_SITES_IGNITION]
IGNITION_NAMES = ("a0", "k_spread")  # ALL_NAMES_SORTED order


# --- PSIS (Vehtari, Gelman, Gabry 2017; Zhang & Stephens 2009 GPD) ----------


@dataclass(frozen=True)
class PSISResult:
    """Pareto-smoothed importance sampling output."""

    log_weights: np.ndarray
    khat: float
    ess: float
    n: int


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Zhang–Stephens (2009) empirical-Bayes GPD fit on positive exceedances.

    Profile-likelihood posterior mean over a grid of the rate ``b``, with the
    weak prior regularization used by loo/ArviZ (k̂ shrunk toward 0.5 by 10
    pseudo-observations). A method-of-moments fit was tried first and is
    biased low ~2x for k ≥ 0.7 (sample variance of a heavy tail
    underestimates), which defeats the k̂ gate exactly where it matters.

    Parameters
    ----------
    x : ndarray
        Positive exceedances.

    Returns
    -------
    k, sigma : float
        Shape (positive = heavy tail) and scale.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    if n < 5:
        return np.nan, np.nan
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.floor(np.sqrt(n)))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    b = b / (prior_bs * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    k_b = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    log_lik = n * (np.log(-b / k_b) - k_b - 1.0)
    w = 1.0 / np.sum(np.exp(log_lik[None, :] - log_lik[:, None]), axis=1)
    w = np.where(w < 10 * np.finfo(float).eps, 0.0, w)
    w = w / np.sum(w)
    b_post = float(np.sum(b * w))
    k = float(np.mean(np.log1p(-b_post * x)))
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    sigma = -k / b_post if b_post != 0 else np.nan
    return float(k), float(max(sigma, 1e-12))


def khat_threshold(n: int) -> float:
    """Sample-size-dependent PSIS reliability threshold (Vehtari et al. 2024).

    ``min(1 − 1/log10(n), 0.7)``: with 400 draws the gate is ~0.62, not 0.7.
    """
    n = max(int(n), 10)
    return float(min(1.0 - 1.0 / np.log10(n), 0.7))


def psis_smooth(log_weights: np.ndarray) -> PSISResult:
    """Pareto-smooth log importance weights.

    Parameters
    ----------
    log_weights : array-like
        ``log p(z) - log q(z)`` (or ``-U(z) - log q(z)``), any shift.

    Returns
    -------
    PSISResult
        Smoothed log-weights (normalized to max 0), Pareto-k, and the
        importance ESS of the smoothed weights.
    """
    lw = np.asarray(log_weights, dtype=float).reshape(-1)
    n = int(lw.size)
    if n == 0:
        raise ValueError("psis_smooth: empty weights")
    lw = lw - np.max(lw)
    w_raw = np.exp(lw)
    # degenerate: all weights equal (or numerically so)
    if float(np.std(w_raw)) <= 1e-12 * (float(np.mean(w_raw)) + 1e-15):
        return PSISResult(log_weights=lw, khat=float("-inf"), ess=float(n), n=n)

    n_tail = int(min(np.floor(0.2 * n), np.ceil(3.0 * np.sqrt(n))))
    n_tail = max(n_tail, 5)
    if n_tail >= n:
        n_tail = n - 1
    order = np.argsort(lw)
    cutoff = lw[order[-n_tail - 1]]
    tail = lw[order[-n_tail:]]
    exp_cut = np.exp(cutoff)
    exceed = np.sort(np.exp(tail) - exp_cut)
    exceed = exceed[exceed > 0]
    if exceed.size < 5:
        w = w_raw / np.sum(w_raw)
        return PSISResult(
            log_weights=lw, khat=float("nan"), ess=float(1.0 / np.sum(w * w)), n=n
        )
    k, sigma = _gpdfit(exceed)
    smoothed = lw.copy()
    if np.isfinite(k) and np.isfinite(sigma) and sigma > 0:
        probs = (np.arange(1, n_tail + 1, dtype=float) - 0.5) / n_tail
        if abs(k) < 1e-8:
            qq = -sigma * np.log1p(-probs)  # exponential limit
        else:
            qq = (sigma / k) * np.expm1(-k * np.log1p(-probs))
        new_tail = np.log(np.maximum(exp_cut + np.maximum(qq, 0.0), 1e-300))
        # truncate at the max raw weight (lw is shifted so max(lw) = 0)
        new_tail = np.minimum(new_tail, 0.0)
        smoothed[order[-n_tail:][np.argsort(tail)]] = new_tail
    w = np.exp(smoothed - np.max(smoothed))
    w = w / np.sum(w)
    ess = float(1.0 / np.sum(w * w))
    return PSISResult(log_weights=smoothed, khat=float(k), ess=ess, n=n)


def resample_weights(
    log_weights: np.ndarray,
    n: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Multinomial resample of indices from (possibly PSIS-smoothed) log-weights."""
    rng = rng or np.random.default_rng()
    lw = np.asarray(log_weights, dtype=float).reshape(-1)
    w = np.exp(lw - np.max(lw))
    w = w / np.sum(w)
    n = int(lw.size if n is None else n)
    return rng.choice(lw.size, size=n, replace=True, p=w)


@dataclass(frozen=True)
class MomentMatchResult:
    """Implicitly adaptive IS output (moment-matched proposal draws)."""

    Z: np.ndarray
    u: np.ndarray
    log_weights: np.ndarray
    khat: float
    ess: float
    transforms: tuple[str, ...]
    n_evals: int


def moment_match_psis(
    Z: np.ndarray,
    u: np.ndarray,
    log_q: np.ndarray,
    potential_fn,
    max_evals: int = 6,
    verbose=None,
) -> MomentMatchResult:
    """Implicitly adaptive importance sampling (Paananen et al. 2021).

    Affine-transform the proposal draws toward their importance-weighted
    moments — shift (T1), shift+scale (T2), shift+covariance (T3) — keeping
    a transform when it lowers Pareto-k̂. Each attempt costs one potential
    evaluation per draw. ``log_q`` is the proposal log-density of the
    ORIGINAL draws; an affine map changes it by a constant log-det that
    cancels in self-normalized weights, so it is reused unchanged.

    Parameters
    ----------
    Z, u, log_q
        Proposal draws ``(n, d)``, their potentials ``-log p`` and proposal
        log-densities (all length n).
    potential_fn
        ``z -> U(z)`` on a single unconstrained vector (the expensive call).
    max_evals
        Cap on transform attempts (each is ``n`` potential calls).
    verbose
        Optional ``print``-like progress callback.

    Returns
    -------
    MomentMatchResult
        Final draws, potentials, smoothed log-weights, k̂, importance ESS,
        accepted transform tags, and the number of attempts spent.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float)).copy()
    u = np.asarray(u, dtype=float).reshape(-1).copy()
    log_q = np.asarray(log_q, dtype=float).reshape(-1)
    _n, d = Z.shape

    def _score(u_arr):
        return psis_smooth(-u_arr - log_q)

    psis = _score(u)
    accepted: list[str] = []
    n_evals = 0

    def _weights(p):
        w = np.exp(p.log_weights - np.max(p.log_weights))
        return w / np.sum(w)

    def _candidate(tag, w):
        mu_w = w @ Z
        z_bar = Z.mean(axis=0)
        c = Z - z_bar
        if tag == "shift":
            return c + mu_w
        cw = Z - mu_w
        if tag == "scale":
            s_w = np.sqrt(np.sum(w[:, None] * cw * cw, axis=0))
            s0 = Z.std(axis=0)
            s0 = np.where(s0 > 0, s0, 1.0)
            return c * (s_w / s0) + mu_w
        # full covariance
        C_w = cw.T @ (w[:, None] * cw)
        C0 = np.cov(Z, rowvar=False)
        L_w = np.linalg.cholesky(0.5 * (C_w + C_w.T) + 1e-12 * np.eye(d))
        L0 = np.linalg.cholesky(0.5 * (C0 + C0.T) + 1e-12 * np.eye(d))
        A = L_w @ np.linalg.inv(L0)
        return c @ A.T + mu_w

    improved = True
    while improved and n_evals < max_evals:
        improved = False
        w = _weights(psis)
        for tag in ("shift", "scale", "cov"):
            if n_evals >= max_evals:
                break
            Z_try = _candidate(tag, w)
            u_try = np.array([float(potential_fn(z)) for z in Z_try])
            n_evals += 1
            psis_try = _score(u_try)
            if verbose:
                verbose(
                    f"  moment-match {tag}: khat {psis.khat:.3f} -> "
                    f"{psis_try.khat:.3f}  ess {psis.ess:.0f} -> {psis_try.ess:.0f}"
                )
            if np.isfinite(psis_try.khat) and psis_try.khat < psis.khat:
                Z, u, psis = Z_try, u_try, psis_try
                accepted.append(tag)
                improved = True
                break  # restart the T1→T3 ladder from the new draws
        # no early exit on the khat threshold: scale/cov often lift the
        # importance ESS well after khat has dipped under the gate
    return MomentMatchResult(
        Z=Z,
        u=u,
        log_weights=psis.log_weights,
        khat=psis.khat,
        ess=psis.ess,
        transforms=tuple(accepted),
        n_evals=n_evals,
    )


# --- Gaussians, t, conditionals ----------------------------------------------


def mvn_logpdf(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log-density of N(mean, cov). ``z`` is (n, d) or (d,)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    d = mean.size
    L = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(d))
    dev = z - mean
    y = np.linalg.solve(L, dev.T)
    quad = np.sum(y * y, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def gaussian_conditional(
    mean: np.ndarray, cov: np.ndarray, idx_a: np.ndarray, z_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional N(z_b | z_a) of a joint Gaussian.

    Parameters
    ----------
    mean, cov : ndarray
        Joint mean and covariance.
    idx_a : array-like of int
        Indices of the conditioned block.
    z_a : ndarray
        Value of that block, shape (n_a,) or (n, n_a).

    Returns
    -------
    mean_b, cov_b : ndarray
        Conditional mean (matching the leading axis of ``z_a``) and the
        conditional covariance (independent of ``z_a``).
    """
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    idx_a = np.asarray(idx_a, dtype=int).reshape(-1)
    d = mean.size
    idx_b = np.array([i for i in range(d) if i not in set(idx_a.tolist())], dtype=int)
    Saa = cov[np.ix_(idx_a, idx_a)]
    Sbb = cov[np.ix_(idx_b, idx_b)]
    Sba = cov[np.ix_(idx_b, idx_a)]
    Saa_inv = np.linalg.inv(0.5 * (Saa + Saa.T) + 1e-12 * np.eye(idx_a.size))
    cov_b = Sbb - Sba @ Saa_inv @ Sba.T
    cov_b = 0.5 * (cov_b + cov_b.T)
    z_a = np.asarray(z_a, dtype=float)
    single = z_a.ndim == 1
    z_a = np.atleast_2d(z_a)
    mean_b = mean[idx_b] + (Sba @ Saa_inv @ (z_a - mean[idx_a]).T).T
    if single:
        mean_b = mean_b[0]
    return mean_b, cov_b


def sample_mvn(
    mean: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` rows from N(mean, cov)."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(mean.size))
    return mean + rng.standard_normal((n, mean.size)) @ L.T


# --- Structured q(z_ign) q(z_rest | z_ign) -----------------------------------


def ignition_indices(names: list[str] | None = None) -> np.ndarray:
    """Integer indices of ``(a0, k_spread)`` in a name list (sorted-site default)."""
    names = list(names or ALL_NAMES_SORTED)
    return np.array([names.index(n) for n in IGNITION_NAMES], dtype=int)


def rest_indices(names: list[str] | None = None) -> np.ndarray:
    """Complement of :func:`ignition_indices`."""
    names = list(names or ALL_NAMES_SORTED)
    ign = set(ignition_indices(names).tolist())
    return np.array([i for i in range(len(names)) if i not in ign], dtype=int)


def ignition_moments_from_grid(
    z0: np.ndarray, z1: np.ndarray, log_marg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted 2-D Gaussian moments of a nested-Laplace ``log_marg`` grid.

    ``z0`` is ``a0``, ``z1`` is ``k_spread`` (``IGNITION_NAMES``).

    Parameters
    ----------
    z0, z1, log_marg
        Axes and unnormalized log-marginal from :func:`nested_tmb_grid`.

    Returns
    -------
    mean, cov : ndarray
        Length-2 mean and 2×2 covariance.
    """
    z0 = np.asarray(z0, dtype=float).reshape(-1)
    z1 = np.asarray(z1, dtype=float).reshape(-1)
    log_m = np.asarray(log_marg, dtype=float)
    p = np.exp(log_m - np.max(log_m))
    p = p / np.sum(p)
    pa, pk = p.sum(axis=1), p.sum(axis=0)
    mu0, mu1 = float(np.sum(pa * z0)), float(np.sum(pk * z1))
    A, K = np.meshgrid(z0, z1, indexing="ij")
    v00 = float(np.sum(p * (A - mu0) ** 2))
    v11 = float(np.sum(p * (K - mu1) ** 2))
    v01 = float(np.sum(p * (A - mu0) * (K - mu1)))
    cov = np.array([[v00, v01], [v01, v11]])
    cov = 0.5 * (cov + cov.T)
    lam = np.linalg.eigvalsh(cov)
    if lam.min() <= 0:
        cov = cov + (1e-8 - lam.min()) * np.eye(2)
    return np.array([mu0, mu1]), cov


# --- Nested Laplace on ignition ----------------------------------------------


@dataclass(frozen=True)
class NestedLaplace:
    """2-D nested-Laplace marginal on ignition plus the Laplace rest."""

    names: tuple[str, str]
    z0: np.ndarray
    z1: np.ndarray
    log_marg: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    ign: np.ndarray


def nested_tmb_grid(
    mean: np.ndarray,
    cov: np.ndarray,
    cell_fn,
    n_grid: int = 11,
    n_sd: float | np.ndarray = 7.0,
    names: list[str] | None = None,
) -> NestedLaplace:
    """TMB / INLA nested Laplace: ``log π(θ) ≈ -U(θ, û) - ½ log det H_rest``.

    ``cell_fn(z)`` is called at the Laplace-conditional rest and must
    return ``(U, logdet_Hrr)`` after any inner Newton on rest. The
    ``½ d log(2π)`` term is dropped (constant on a fixed rest dimension).

    Parameters
    ----------
    mean, cov
        Joint Laplace mean and covariance (rest start).
    cell_fn
        ``z -> (U, logdet_Hrr)``.
    n_grid, n_sd, names
        Grid size, half-widths in Laplace sd, and site order.
    """
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    names = list(names or ALL_NAMES_SORTED)
    ign = ignition_indices(names)
    rest = rest_indices(names)
    sd = np.sqrt(np.maximum(np.diag(cov)[ign], 1e-12))
    n_sd = np.broadcast_to(np.asarray(n_sd, dtype=float).reshape(-1), (2,))
    z0 = np.linspace(mean[ign[0]] - n_sd[0] * sd[0], mean[ign[0]] + n_sd[0] * sd[0], n_grid)
    z1 = np.linspace(mean[ign[1]] - n_sd[1] * sd[1], mean[ign[1]] + n_sd[1] * sd[1], n_grid)
    log_marg = np.empty((n_grid, n_grid))
    for i, a in enumerate(z0):
        for j, b in enumerate(z1):
            z = mean.copy()
            z[ign] = np.array([a, b])
            z[rest] = gaussian_conditional(mean, cov, ign, z[ign])[0]
            u, logdet = cell_fn(z)
            log_marg[i, j] = -float(u) - 0.5 * float(logdet)
    return NestedLaplace(
        names=IGNITION_NAMES,
        z0=z0,
        z1=z1,
        log_marg=log_marg,
        mean=mean,
        cov=cov,
        ign=ign,
    )


def sample_nested_laplace(
    nested: NestedLaplace, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw from the discrete 2-D nested marginal times the Gaussian rest."""
    log_m = nested.log_marg - np.max(nested.log_marg)
    p = np.exp(log_m)
    p = p / np.sum(p)
    flat = rng.choice(p.size, size=n, replace=True, p=p.ravel())
    i0, i1 = np.unravel_index(flat, p.shape)
    # jitter inside the cell so we don't sit on the grid
    dz0 = (nested.z0[1] - nested.z0[0]) if nested.z0.size > 1 else 0.0
    dz1 = (nested.z1[1] - nested.z1[0]) if nested.z1.size > 1 else 0.0
    z_i = np.column_stack(
        [
            nested.z0[i0] + dz0 * rng.uniform(-0.5, 0.5, n),
            nested.z1[i1] + dz1 * rng.uniform(-0.5, 0.5, n),
        ]
    )
    names = list(ALL_NAMES_SORTED)
    rest = rest_indices(names)
    mu_r, S_rr = gaussian_conditional(nested.mean, nested.cov, nested.ign, z_i)
    Lr = np.linalg.cholesky(0.5 * (S_rr + S_rr.T) + 1e-12 * np.eye(S_rr.shape[0]))
    z_r = mu_r + rng.standard_normal((n, rest.size)) @ Lr.T
    z = np.empty((n, nested.mean.size))
    z[:, nested.ign] = z_i
    z[:, rest] = z_r
    return z


def nested_log_q(z: np.ndarray, nested: NestedLaplace) -> np.ndarray:
    """Log density of :func:`sample_nested_laplace` (cell jitter x Gaussian rest)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    ign = np.asarray(nested.ign, dtype=int)
    rest = rest_indices()
    log_m = nested.log_marg - np.max(nested.log_marg)
    p = np.exp(log_m)
    p = p / np.sum(p)
    dz0 = float(nested.z0[1] - nested.z0[0]) if nested.z0.size > 1 else 1.0
    dz1 = float(nested.z1[1] - nested.z1[0]) if nested.z1.size > 1 else 1.0
    za = z[:, ign[0]]
    zk = z[:, ign[1]]
    i0 = np.clip(np.round((za - nested.z0[0]) / dz0).astype(int), 0, nested.z0.size - 1)
    i1 = np.clip(np.round((zk - nested.z1[0]) / dz1).astype(int), 0, nested.z1.size - 1)
    log_ign = np.log(np.maximum(p[i0, i1], 1e-300)) - np.log(dz0) - np.log(dz1)
    mu_r, s_rr = gaussian_conditional(nested.mean, nested.cov, ign, z[:, ign])
    log_rest = mvn_logpdf(z[:, rest] - mu_r, np.zeros(rest.size), s_rr)
    return log_ign + log_rest


# --- Hierarchical pooling of Laplace Gaussians --------------------------------


@dataclass(frozen=True)
class HierPool:
    """Second-stage pool of per-cook Laplace Gaussians."""

    names: list[str]
    shared_mean: np.ndarray
    shared_cov: np.ndarray
    partial_hyper_mean: np.ndarray
    partial_hyper_cov: np.ndarray
    cook_partial_mean: list[np.ndarray]
    cook_independent_mean: list[np.ndarray]
    shared_idx: np.ndarray
    partial_idx: np.ndarray
    independent_idx: np.ndarray


def _idx(names: list[str], block: list[str]) -> np.ndarray:
    return np.array([names.index(n) for n in block], dtype=int)


def _prior_quadratic(block: list[str], z_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal Gaussian (precision, precision·mean) matching −log prior at ``z_ref``.

    Exact for LogNormal sites (Gaussian in z); local quadratic at ``z_ref``
    for Beta-through-logit sites.
    """
    from .calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES

    dists = dict(THETA_SITES + PHI_SITES + PHI_SITES_IGNITION)
    h = np.empty(len(block))
    eta = np.empty(len(block))
    for j, (nm, zr) in enumerate(zip(block, np.asarray(z_ref, float), strict=True)):
        d = dists[nm]
        if hasattr(d, "concentration1"):  # Beta through logit
            a, b = float(d.concentration1), float(d.concentration0)
            s = 1.0 / (1.0 + np.exp(-zr))
            g = -a * (1.0 - s) + b * s
            hj = (a + b) * s * (1.0 - s)
            mu = zr - g / hj
        else:  # LogNormal: Gaussian in z
            hj = 1.0 / float(d.scale) ** 2
            mu = float(d.loc)
        h[j], eta[j] = hj, hj * mu
    return h, eta


def pool_shared(
    means: list[np.ndarray],
    covs: list[np.ndarray],
    idx: np.ndarray,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance pool of a shared block (hard pooling).

    Treats each cook's Laplace marginal on ``idx`` as a Gaussian observation
    of a common value. Each cook's Laplace already folds in one copy of the
    prior, so naively summing K precisions counts the prior K times; when
    ``names`` is given the duplicated (K−1) copies are subtracted via a
    diagonal-Gaussian match of the prior at the uncorrected pooled mean
    (exact for LogNormal sites, local for Beta). Without ``names`` the
    uncorrected pool is returned (negligible for K=2 with wide priors).
    """
    prec = np.zeros((idx.size, idx.size))
    rhs = np.zeros(idx.size)
    for m, c in zip(means, covs, strict=True):
        s = c[np.ix_(idx, idx)]
        lam = np.linalg.inv(0.5 * (s + s.T) + 1e-10 * np.eye(idx.size))
        prec = prec + lam
        rhs = rhs + lam @ np.asarray(m, dtype=float).reshape(-1)[idx]
    n_cooks = len(means)
    if names is not None and n_cooks > 1:
        mean0 = np.linalg.solve(prec, rhs)
        block = [names[i] for i in np.asarray(idx, int)]
        h, eta = _prior_quadratic(block, mean0)
        prec_c = prec - (n_cooks - 1) * np.diag(h)
        rhs_c = rhs - (n_cooks - 1) * eta
        if np.linalg.eigvalsh(0.5 * (prec_c + prec_c.T)).min() > 0:
            prec, rhs = prec_c, rhs_c
    cov = np.linalg.inv(prec)
    return cov @ rhs, 0.5 * (cov + cov.T)


def estimate_tau2(
    means: list[np.ndarray], covs: list[np.ndarray], idx: np.ndarray
) -> np.ndarray:
    """Method-of-moments between-cook variance per coordinate.

    ``τ²_j = max(s²_between,j − mean_i C_i,jj, floor)`` — the DerSimonian–
    Laird-style estimate; very noisy for 2 cooks, but data-driven.
    """
    idx = np.asarray(idx, dtype=int).reshape(-1)
    M = np.stack([np.asarray(m, dtype=float).reshape(-1)[idx] for m in means])
    Cd = np.stack([np.asarray(c, dtype=float)[np.ix_(idx, idx)].diagonal() for c in covs])
    vb = M.var(axis=0, ddof=1)
    return np.maximum(vb - Cd.mean(axis=0), 1e-4)


def pool_partial(
    means: list[np.ndarray],
    covs: list[np.ndarray],
    idx: np.ndarray,
    tau2: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Random-effects pool: ``θ_i | μ ~ N(μ, C_i + diag(τ²))``, Gaussian hypermean.

    ``tau2`` may be a scalar or a per-coordinate vector (see
    :func:`estimate_tau2`).

    Returns
    -------
    hyper_mean, hyper_cov, cook_means
        The hypermean of μ, its covariance, and the cook-level posterior
        means of θ_i (shrinkage toward μ).
    """
    k = idx.size
    tau2 = np.maximum(np.broadcast_to(np.asarray(tau2, dtype=float), (k,)), 1e-12)
    tau2_eye = np.diag(tau2)
    # hypermean precision: Σ_i (C_i + diag(τ²))^{-1}
    prec = np.zeros((k, k))
    rhs = np.zeros(k)
    cook_marg_prec = []
    cook_m = []
    for m, c in zip(means, covs, strict=True):
        s = c[np.ix_(idx, idx)] + tau2_eye
        lam = np.linalg.inv(0.5 * (s + s.T) + 1e-10 * np.eye(k))
        cook_marg_prec.append(lam)
        cook_m.append(np.asarray(m, dtype=float).reshape(-1)[idx])
        prec = prec + lam
        rhs = rhs + lam @ cook_m[-1]
    hyper_cov = np.linalg.inv(prec)
    hyper_mean = hyper_cov @ rhs
    # cook-level: precision C_i^{-1} + diag(τ²)^{-1}, mean mix of m_i and μ
    out = []
    for m_i, c, _lam in zip(cook_m, covs, cook_marg_prec, strict=True):
        ci = c[np.ix_(idx, idx)]
        lam_i = np.linalg.inv(0.5 * (ci + ci.T) + 1e-10 * np.eye(k))
        lam_tau = np.diag(1.0 / tau2)
        post_cov = np.linalg.inv(lam_i + lam_tau)
        out.append(post_cov @ (lam_i @ m_i + lam_tau @ hyper_mean))
    return hyper_mean, 0.5 * (hyper_cov + hyper_cov.T), out


def hierarchical_pool(
    means: list[np.ndarray],
    covs: list[np.ndarray],
    names: list[str] | None = None,
    shared: list[str] | None = None,
    partial: list[str] | None = None,
    independent: list[str] | None = None,
    tau2: float | str = 0.04,
    subtract_shared_prior: bool = True,
) -> HierPool:
    """Two-stage hierarchical pool of per-cook Laplace Gaussians.

    Parameters
    ----------
    means, covs : list
        Unconstrained Laplace mean and covariance per cook, same ``names``
        order.
    names : list of str, optional
        Default ``ALL_NAMES_SORTED``.
    shared, partial, independent : list of str, optional
        Site names in each pooling role. Defaults: pit / meat / ignition.
        Sites in ``names`` not listed stay independent (noise).
    tau2 : float or "auto"
        Between-cook variance on the partial-pool (meat) block, in z.
        ``"auto"`` estimates a per-coordinate value from the spread of the
        cook means (:func:`estimate_tau2`; noisy for few cooks).
    subtract_shared_prior : bool
        Remove the (K−1) duplicated prior copies from the shared pool
        (see :func:`pool_shared`). Requires the cook Laplaces to be
        posteriors under this module's priors.

    Returns
    -------
    HierPool
    """
    names = list(names or ALL_NAMES_SORTED)
    shared = list(shared or PIT_SHARED)
    partial = list(partial or MEAT_PARTIAL)
    independent = list(independent or IGNITION_IID)
    sidx = _idx(names, shared)
    pidx = _idx(names, partial)
    iidx = _idx(names, independent)
    sm, sc = pool_shared(
        means, covs, sidx, names=names if subtract_shared_prior else None
    )
    if isinstance(tau2, str):
        if tau2 != "auto":
            raise ValueError(f"tau2 must be a float or 'auto', got {tau2!r}")
        tau2 = estimate_tau2(means, covs, pidx)
    pm, pc, cook_p = pool_partial(means, covs, pidx, tau2)
    cook_i = [np.asarray(m, dtype=float).reshape(-1)[iidx] for m in means]
    return HierPool(
        names=names,
        shared_mean=sm,
        shared_cov=sc,
        partial_hyper_mean=pm,
        partial_hyper_cov=pc,
        cook_partial_mean=cook_p,
        cook_independent_mean=cook_i,
        shared_idx=sidx,
        partial_idx=pidx,
        independent_idx=iidx,
    )


# --- Widths and distances in z ------------------------------------------------


def z_matrix_physics(samples: dict) -> np.ndarray:
    """Constrained physics draws → ``(n, N_PHYS)`` unconstrained matrix (NAMES order)."""
    n = len(np.asarray(samples[NAMES[0]]).reshape(-1))
    return np.stack(
        [
            to_z([float(np.asarray(samples[k]).reshape(-1)[i]) for k in NAMES])
            for i in range(n)
        ]
    )


def sd_z_physics(samples: dict) -> dict[str, float]:
    """Posterior sd in unconstrained physics coordinates, per site in NAMES."""
    Z = z_matrix_physics(samples)
    return dict(zip(NAMES, np.std(Z, axis=0), strict=True))


def w1_z_columns(Za: np.ndarray, Zb: np.ndarray, n_q: int = 199) -> np.ndarray:
    """Per-column 1-Wasserstein distance between two draw matrices.

    Quantile-function form: ``W1 = ∫ |F_a⁻¹(p) − F_b⁻¹(p)| dp`` on a
    trimmed grid (p ∈ [0.005, 0.995] — the extreme order statistics of a
    few hundred draws are noise). Unlike an sd comparison this sees mean
    shifts and asymmetric tails.
    """
    qs = np.linspace(0.005, 0.995, int(n_q))
    qa = np.quantile(np.atleast_2d(Za), qs, axis=0)
    qb = np.quantile(np.atleast_2d(Zb), qs, axis=0)
    return np.mean(np.abs(qa - qb), axis=0)


def w1_z_physics(samples: dict, ref: dict) -> dict[str, float]:
    """Per-site W1 in z between constrained draw dicts (NAMES order)."""
    d = w1_z_columns(z_matrix_physics(samples), z_matrix_physics(ref))
    return dict(zip(NAMES, map(float, d), strict=True))


def sd_z_from_draws(Z: np.ndarray, names: list[str] | None = None) -> dict[str, float]:
    """Column-wise sd of unconstrained draws, keyed by ``names``."""
    names = list(names or ALL_NAMES_SORTED)
    Z = np.asarray(Z, dtype=float)
    return {n: float(np.std(Z[:, j])) for j, n in enumerate(names)}


def load_laplace_hess(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``laplace_hess_real.npz`` → ``(z0, H, cov)``."""
    h = np.load(path)
    z0 = np.asarray(h["z0"], dtype=float).reshape(-1)
    H = np.asarray(h["H"], dtype=float)
    H = 0.5 * (H + H.T)
    lam = np.linalg.eigvalsh(H)
    if lam.min() <= 0:
        H = H + (1e-6 * max(lam.max(), 1.0) - lam.min()) * np.eye(z0.size)
    cov = np.linalg.inv(H)
    return z0, H, cov
