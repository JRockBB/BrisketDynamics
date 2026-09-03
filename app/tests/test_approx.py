"""PSIS + moment matching, TMB nested Laplace, hierarchical pooling — no Tesseract."""

import numpy as np
from brisket_app.approx import (
    IGNITION_IID,
    MEAT_PARTIAL,
    PIT_SHARED,
    gaussian_conditional,
    hierarchical_pool,
    ignition_indices,
    ignition_moments_from_grid,
    mvn_logpdf,
    nested_log_q,
    nested_tmb_grid,
    pool_shared,
    psis_smooth,
    rest_indices,
    sample_nested_laplace,
    sd_z_from_draws,
)
from brisket_app.monolith import ALL_NAMES_SORTED


def test_psis_equal_weights_high_ess():
    n = 200
    r = psis_smooth(np.zeros(n))
    assert r.ess > 0.8 * n
    assert r.khat < 0.5 or not np.isfinite(r.khat)


def test_psis_one_outlier_low_ess():
    lw = np.zeros(100)
    lw[-1] = 20.0
    r = psis_smooth(lw)
    assert r.ess < 20


def test_psis_khat_cauchy_vs_normal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    log_p = -np.log(np.pi * (1.0 + x * x))
    log_q = -0.5 * np.log(2.0 * np.pi) - 0.5 * x * x
    r = psis_smooth(log_p - log_q)
    assert r.khat > 0.3
    assert r.ess < 0.6 * r.n


def test_psis_gaussian_self_normalized():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(400)
    # target = proposal = N(0,1) → constant weights
    r = psis_smooth(np.zeros_like(z) + 0.01 * rng.standard_normal(400))
    assert r.khat < 0.5
    assert r.ess > 200


def test_gaussian_conditional_matches_joint():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    cov = A @ A.T + np.eye(6)
    mean = rng.normal(size=6)
    idx_a = np.array([0, 3])
    z_a = mean[idx_a] + 0.2
    mu_b, cov_b = gaussian_conditional(mean, cov, idx_a, z_a)
    # draw from joint, compare empirical conditional mean via precision formula
    Saa = cov[np.ix_(idx_a, idx_a)]
    Sba = cov[np.ix_([i for i in range(6) if i not in idx_a], idx_a)]
    idx_b = np.array([i for i in range(6) if i not in set(idx_a.tolist())])
    want = mean[idx_b] + Sba @ np.linalg.inv(Saa) @ (z_a - mean[idx_a])
    np.testing.assert_allclose(mu_b, want, rtol=1e-10)
    assert cov_b.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(cov_b) > 0)


def test_ignition_moments_from_grid_recovers_gaussian():
    z0 = np.linspace(-2.0, 2.0, 21)
    z1 = np.linspace(-3.0, 3.0, 21)
    A, K = np.meshgrid(z0, z1, indexing="ij")
    mu = np.array([0.4, -0.8])
    # correlated Gaussian log-density
    S = np.array([[0.5, -0.2], [-0.2, 1.1]])
    prec = np.linalg.inv(S)
    dA, dK = A - mu[0], K - mu[1]
    quad = prec[0, 0] * dA**2 + 2 * prec[0, 1] * dA * dK + prec[1, 1] * dK**2
    log_m = -0.5 * quad
    got_mu, got_S = ignition_moments_from_grid(z0, z1, log_m)
    np.testing.assert_allclose(got_mu, mu, atol=0.08)
    np.testing.assert_allclose(np.sqrt(np.diag(got_S)), np.sqrt(np.diag(S)), rtol=0.15)
    want_c = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
    got_c = got_S[0, 1] / np.sqrt(got_S[0, 0] * got_S[1, 1])
    np.testing.assert_allclose(got_c, want_c, atol=0.12)


def test_pool_shared_inverse_variance():
    # two 1-D Gaussians N(0, 1) and N(2, 1) → pooled at 1 with var 0.5
    means = [np.array([0.0, 9.0]), np.array([2.0, 9.0])]
    covs = [np.diag([1.0, 4.0]), np.diag([1.0, 4.0])]
    mu, cov = pool_shared(means, covs, np.array([0]))
    np.testing.assert_allclose(mu, [1.0], atol=1e-10)
    np.testing.assert_allclose(cov, [[0.5]], atol=1e-10)


def test_hierarchical_pool_shared_tightens_pit_not_meat():
    names = list(ALL_NAMES_SORTED)
    d = len(names)
    true_pit = {n: 0.5 for n in PIT_SHARED}
    meat_a = {n: 1.0 for n in MEAT_PARTIAL}
    meat_b = {n: -1.0 for n in MEAT_PARTIAL}
    cov = np.eye(d) * 0.09
    m1 = np.zeros(d)
    m2 = np.zeros(d)
    for n in PIT_SHARED:
        m1[names.index(n)] = true_pit[n] + 0.15
        m2[names.index(n)] = true_pit[n] - 0.15
    for n in MEAT_PARTIAL:
        m1[names.index(n)] = meat_a[n]
        m2[names.index(n)] = meat_b[n]
    for n in IGNITION_IID:
        m1[names.index(n)] = 0.4
        m2[names.index(n)] = -0.3
    # fictitious z values, inconsistent with the real priors: keep the
    # prior-dedup off and test only the pooling mechanics here
    pool = hierarchical_pool(
        [m1, m2], [cov, cov], names=names, tau2=0.25, subtract_shared_prior=False
    )
    # shared pit closer to 0.5 than either cook
    for i, _n in enumerate(PIT_SHARED):
        assert abs(pool.shared_mean[i] - 0.5) < 0.05
        assert pool.shared_cov[i, i] < 0.09
    # meat hypermean between the two cooks; cook-level still distinct
    assert pool.cook_partial_mean[0][0] > 0
    assert pool.cook_partial_mean[1][0] < 0
    # ignition stays cook-specific
    np.testing.assert_allclose(pool.cook_independent_mean[0][0], 0.4, atol=1e-12)
    np.testing.assert_allclose(pool.cook_independent_mean[1][0], -0.3, atol=1e-12)


def test_hierarchical_pool_explicit_empty_roles():
    # an explicit [] means "no sites in this role" (it must NOT fall back
    # to the default lists — regression for the `or`-default bug)
    names = list(ALL_NAMES_SORTED)
    d = len(names)
    rng = np.random.default_rng(3)
    means = [rng.normal(size=d), rng.normal(size=d)]
    covs = [np.eye(d) * 0.1] * 2
    pool = hierarchical_pool(
        means, covs, names=names, shared=[], partial=MEAT_PARTIAL,
        independent=PIT_SHARED + IGNITION_IID, tau2=0.25,
        subtract_shared_prior=False,
    )
    assert pool.shared_mean.size == 0 and pool.shared_cov.shape == (0, 0)
    assert len(pool.cook_independent_mean[0]) == len(PIT_SHARED + IGNITION_IID)


def test_sd_z_from_draws_keys():
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(20, len(ALL_NAMES_SORTED)))
    sd = sd_z_from_draws(Z)
    assert set(sd) == set(ALL_NAMES_SORTED)
    assert all(v > 0 for v in sd.values())


def test_sorted_phys_roundtrip():
    from brisket_app.monolith import NAMES, apply_phys, sorted_to_phys

    rng = np.random.default_rng(0)
    z = rng.normal(size=len(ALL_NAMES_SORTED))
    zp = sorted_to_phys(z)
    assert zp.shape == (len(NAMES),)
    back = apply_phys(np.zeros_like(z), zp)
    for n in NAMES:
        assert back[ALL_NAMES_SORTED.index(n)] == zp[list(NAMES).index(n)]


def test_nested_tmb_matches_profile_when_logdet_constant():
    rng = np.random.default_rng(8)
    d = len(ALL_NAMES_SORTED)
    a = rng.normal(size=(d, d))
    cov = 0.05 * (a @ a.T) + 0.25 * np.eye(d)
    mean = rng.normal(size=d) * 0.05
    prec = np.linalg.inv(cov)

    def pot(z):
        r = np.asarray(z) - mean
        return 0.5 * float(r @ prec @ r)

    profile = nested_tmb_grid(mean, cov, lambda z: (pot(z), 0.0), n_grid=7, n_sd=2.0)

    def cell(z):
        return pot(z), 10.0

    tmb = nested_tmb_grid(mean, cov, cell, n_grid=7, n_sd=2.0)
    np.testing.assert_allclose(tmb.log_marg - tmb.log_marg.max(), profile.log_marg - profile.log_marg.max(), atol=1e-8)


def test_nested_tmb_logdet_fattens_ignition():
    """Rest precision that drops with k_spread^2 fattens the TMB marginal."""
    d = len(ALL_NAMES_SORTED)
    mean = np.zeros(d)
    cov = np.eye(d) * 0.25
    ign = ignition_indices()
    rest = rest_indices()
    ik = ign[1]
    n_rest = rest.size

    def pot(z):
        z = np.asarray(z, dtype=float)
        kappa = np.exp(-0.8 * z[ik] ** 2)
        return 0.5 * float(z[ign] @ z[ign]) / 0.25 + 0.5 * kappa * float(
            z[rest] @ z[rest]
        )

    def cell_profile(z):
        return pot(z), 0.0

    def cell_tmb(z):
        kappa = np.exp(-0.8 * z[ik] ** 2)
        return pot(z), n_rest * np.log(kappa)

    prof = nested_tmb_grid(mean, cov, cell_profile, n_grid=9, n_sd=2.5)
    tmb = nested_tmb_grid(mean, cov, cell_tmb, n_grid=9, n_sd=2.5)
    _mp, sp = ignition_moments_from_grid(prof.z0, prof.z1, prof.log_marg)
    _mt, st = ignition_moments_from_grid(tmb.z0, tmb.z1, tmb.log_marg)
    assert np.sqrt(st[1, 1]) > 1.15 * np.sqrt(sp[1, 1])


def test_khat_threshold_sample_size():
    from brisket_app.approx import khat_threshold

    assert abs(khat_threshold(400) - (1.0 - 1.0 / np.log10(400))) < 1e-12
    assert khat_threshold(10**7) == 0.7


def test_gpdfit_recovers_tail_shape():
    """ZS khat tracks the true GPD shape; the moment fit saturated near 0.4."""
    rng = np.random.default_rng(0)
    for k_true, lo, hi in ((0.7, 0.45, 0.95), (1.1, 0.75, 1.45)):
        khats = []
        for _ in range(5):
            u = rng.random(4000)
            w = (u ** (-k_true) - 1.0) / k_true  # GPD(k_true, sigma=1)
            khats.append(psis_smooth(np.log1p(w)).khat)
        k_med = float(np.median(khats))
        assert lo < k_med < hi, (k_true, k_med)


def test_psis_smoothed_tail_truncated_at_max():
    rng = np.random.default_rng(1)
    u = rng.random(400)
    w = (u ** (-1.0) - 1.0)  # very heavy tail
    r = psis_smooth(np.log1p(w))
    assert np.max(r.log_weights) <= 1e-12  # normalized to max 0, never above


def test_moment_match_psis_fixes_gaussian_mismatch():
    """Shift+scale moment matching repairs a too-narrow shifted proposal."""
    from brisket_app.approx import khat_threshold, moment_match_psis

    rng = np.random.default_rng(2)
    n, d = 400, 3
    mu_t, sd_t = 2.0, 1.6
    Z = rng.standard_normal((n, d))
    log_q = mvn_logpdf(Z, np.zeros(d), np.eye(d))

    def pot(z):
        z = np.asarray(z, dtype=float)
        return 0.5 * float(np.sum(((z - mu_t) / sd_t) ** 2))

    u0 = np.array([pot(z) for z in Z])
    k0 = psis_smooth(-u0 - log_q).khat
    mm = moment_match_psis(Z, u0, log_q, pot, max_evals=6)
    assert mm.transforms, "no transform accepted"
    assert mm.khat < k0
    assert mm.khat < khat_threshold(n)
    # matched draws should sit near the target moments
    np.testing.assert_allclose(mm.Z.mean(), mu_t, atol=0.4)
    np.testing.assert_allclose(mm.Z.std(), sd_t, rtol=0.3)


def test_pool_shared_prior_dedup_matches_joint():
    """Corrected 2-cook pool equals prior x two likelihoods (LogNormal sites)."""
    from brisket_app.calibrate import PHI_SITES

    names = list(ALL_NAMES_SORTED)
    d = len(names)
    f = 25.0  # likelihood precision per cook
    offsets = (0.3, -0.2)
    means, covs = [], []
    prior = {}
    for n, dist_ in PHI_SITES:
        if hasattr(dist_, "concentration1"):
            prior[n] = None
        else:
            prior[n] = (float(dist_.loc), 1.0 / float(dist_.scale) ** 2)
    for off in offsets:
        m = np.zeros(d)
        c = np.eye(d) * 0.04
        for n, _dist in PHI_SITES:
            j = names.index(n)
            if prior[n] is None:  # u_leak: plausible logit values
                m[j] = -1.3 + off
                c[j, j] = 1.0 / (2.3 + f)
            else:
                m_p, p = prior[n]
                y = m_p + off
                m[j] = (p * m_p + f * y) / (p + f)
                c[j, j] = 1.0 / (p + f)
        means.append(m)
        covs.append(c)
    sidx = np.array([names.index(n) for n in PIT_SHARED])
    _mu_naive, cov_naive = pool_shared(means, covs, sidx)
    mu_c, cov_c = pool_shared(means, covs, sidx, names=names)
    for i, n in enumerate(PIT_SHARED):
        if prior[n] is None:
            # subtraction must widen, not tighten
            assert cov_c[i, i] > cov_naive[i, i]
            continue
        m_p, p = prior[n]
        y1, y2 = m_p + offsets[0], m_p + offsets[1]
        want_prec = p + 2 * f
        want_mean = (p * m_p + f * (y1 + y2)) / want_prec
        np.testing.assert_allclose(cov_c[i, i], 1.0 / want_prec, rtol=1e-6)
        np.testing.assert_allclose(mu_c[i], want_mean, rtol=1e-6)


def test_w1_z_columns_sees_shift_and_scale():
    from brisket_app.approx import w1_z_columns

    rng = np.random.default_rng(3)
    A = rng.standard_normal((4000, 2))
    B = A.copy()
    B[:, 0] += 0.5
    d = w1_z_columns(A, B)
    np.testing.assert_allclose(d[0], 0.5, atol=0.03)
    assert d[1] < 0.03
    # same draws → ~0
    assert np.all(w1_z_columns(A, A) < 1e-12)


def test_estimate_tau2_between_cook_spread():
    from brisket_app.approx import estimate_tau2

    idx = np.array([0, 1])
    covs = [np.diag([0.01, 0.01, 9.0])] * 3
    means = [np.array([0.0, 1.0, 0.0]), np.array([0.6, 1.0, 0.0]), np.array([-0.6, 1.0, 0.0])]
    t2 = estimate_tau2(means, covs, idx)
    assert t2[0] > 0.2  # real spread on coord 0
    assert t2[1] <= 1e-3 + 1e-12  # no spread beyond the floor


def test_nested_log_q_finite_on_samples():
    rng = np.random.default_rng(21)
    d = len(ALL_NAMES_SORTED)
    cov = 0.25 * np.eye(d)
    mean = np.zeros(d)
    prec = np.linalg.inv(cov)

    def pot(z):
        r = np.asarray(z) - mean
        return 0.5 * float(r @ prec @ r)

    nested = nested_tmb_grid(mean, cov, lambda z: (pot(z), 0.0), n_grid=7, n_sd=2.0)
    z = sample_nested_laplace(nested, 40, rng)
    lq = nested_log_q(z, nested)
    assert np.all(np.isfinite(lq))
    assert lq.shape == (40,)


def test_nested_grid_sees_t_tails():
    """Potential that is t on ignition, Gaussian on rest → grid wider than Laplace."""
    d = len(ALL_NAMES_SORTED)
    mean = np.zeros(d)
    cov = np.eye(d) * 0.25
    ign = ignition_indices()
    rest = rest_indices()
    df = 4.0

    def pot(z):
        z = np.asarray(z, dtype=float)
        zi = z[ign]
        zr = z[rest]
        # t_4 on ignition (scale = I), N(0, 0.25 I) on rest
        quad_i = float(zi @ zi)
        lp_i = -0.5 * (df + 2) * np.log1p(quad_i / df)
        lp_r = -0.5 * float(zr @ zr) / 0.25
        return -(lp_i + lp_r)

    nested = nested_tmb_grid(mean, cov, lambda z: (pot(z), 0.0), n_grid=11, n_sd=4.0)
    log_m = nested.log_marg - np.max(nested.log_marg)
    p = np.exp(log_m)
    p = p / np.sum(p)
    mu1 = np.sum(p.sum(axis=0) * nested.z1)
    m2 = np.sum(p.sum(axis=0) * (nested.z1 - mu1) ** 2)
    sd_grid = np.sqrt(m2)
    sd_laplace = np.sqrt(cov[ign[1], ign[1]])
    assert sd_grid > 1.2 * sd_laplace
