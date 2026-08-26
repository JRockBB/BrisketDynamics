"""Real-cook AR(1) likelihood: segments, ρ=0 ≡ iid, constant bias is cheaper at high ρ."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from brisket_app.calibrate import ar1_log_prob, ar1_observe, contiguous_runs
from brisket_app.cooklog import load
from numpyro.infer.util import log_density


def test_contiguous_runs_splits_on_holes():
    mask = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    assert contiguous_runs(mask) == ((2, 3, 4), (6, 7))
    assert contiguous_runs(np.zeros(4, dtype=bool)) == ()
    assert contiguous_runs(np.ones(3, dtype=bool)) == ((0, 1, 2),)
    assert contiguous_runs(np.array([1, 0, 1], dtype=bool)) == ((0,), (2,))


def test_cooklog_lid_breaks_the_chain():
    log = load(dt_min=10.0)
    pit = contiguous_runs(log["mask_pit"])
    core = contiguous_runs(log["mask_core"])
    assert len(pit) >= 2
    assert len(core) >= 2
    assert sum(len(r) for r in pit) == int(log["mask_pit"].sum())
    assert sum(len(r) for r in core) == int(log["mask_core"].sum())
    # a gap between first two pit runs is a masked (lid) sample
    assert pit[0][-1] + 1 < pit[1][0]


def test_ar1_rho0_matches_iid():
    y = jnp.array([1.0, 1.4, 0.7, 1.1])
    mu = jnp.array([1.0, 1.0, 1.0, 1.0])
    sigma = 0.5
    got = ar1_log_prob(y, mu, sigma, 0.0)
    want = dist.Normal(mu, sigma).log_prob(y).sum()
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10)


def test_ar1_constant_offset_cheaper_at_high_rho():
    mu = jnp.zeros(20)
    y = mu + 2.0
    sigma = 1.0
    iid = ar1_log_prob(y, mu, sigma, 0.0)
    ar = ar1_log_prob(y, mu, sigma, 0.95)
    assert np.asarray(ar) > np.asarray(iid)


def test_ar1_observe_matches_log_prob():
    y = jnp.array([0.2, 0.5, 0.1])
    mu = jnp.array([0.0, 0.1, 0.0])
    sigma, rho = 0.4, 0.7

    def model():
        ar1_observe("seg", y, mu, sigma, rho)

    lp_model, _ = log_density(model, (), {}, {})
    lp_direct = ar1_log_prob(y, mu, sigma, rho)
    np.testing.assert_allclose(np.asarray(lp_model), np.asarray(lp_direct), rtol=1e-10)


def test_vi_guide_draws_without_tesseract():
    import jax
    import numpyro
    from brisket_app.calibrate import draw_vi_posterior, make_vi_guide
    from numpyro.infer import SVI, Trace_ELBO

    def model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(1.0, 1.0))

    guide = make_vi_guide(model, kind="full")
    svi = SVI(model, guide, numpyro.optim.Adam(0.2), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(0), 40, progress_bar=False)
    draws = draw_vi_posterior(guide, result.params, 50, seed=1)
    assert draws["x"].shape == (50,)
    assert draws["y"].shape == (50,)
    assert np.std(draws["x"]) > 0.05
