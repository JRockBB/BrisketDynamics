"""calibrate.model_real on the monolith: its gradient equals the whitened LM gradient."""

import json
from pathlib import Path

import jax
import numpy as np
from brisket_app import calibrate, cooklog
from brisket_app.clients import monolith_client
from brisket_app.monolith import (
    ALL_NAMES_SORTED,
    NAMES,
    NOISE_NAMES,
    SITES,
    Whitened,
    _is_beta,
    effective_sigma,
    model_mean,
    prior_terms,
)
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[2]


def test_model_mono_gradient_matches_lm_objective():
    """d(potential)/dz for the physics sites == the LM objective's gradient.

    Same Jacobian, two code paths (numpyro's AR(1) sites through the
    Tesseract VJP vs the explicit whitening). The comparison must be made
    at numpyro's exact constrained bit pattern: a ONE-ULP input difference
    (JAX vs numpy exp of the same z) re-routes the adaptive stepper, and
    near-optimum cancellation amplifies that into ~13% on pit-ridge
    gradient components at tol 1e-6 (measured 2026-09-02; see
    docs/verification.md). At the shared bit pattern the two paths agree
    to ~1e-13.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    log = cooklog.load(dt_min=10.0)
    x = np.array([m[n] for n in NAMES])
    init = dict(zip(NAMES, x, strict=True))
    init.update({n: float(m[n]) for n in NOISE_NAMES})
    model = calibrate.bound_model_real(log, monolith_client())
    mi = initialize_model(
        jax.random.PRNGKey(0),
        model,
        init_strategy=init_to_value(values=init),
    )
    # the exact constrained values numpyro evaluates at (may differ from x
    # by one ULP: JAX's exp vs numpy's on the same unconstrained z)
    from numpyro.infer.util import constrain_fn

    x_used = constrain_fn(model, (), {}, mi.param_info.z)
    x_np = np.array([float(x_used[n]) for n in NAMES])
    z_np = np.array([float(mi.param_info.z[n]) for n in NAMES])
    assert np.max(np.abs(x_np - x) / np.abs(x)) < 1e-14  # one-ULP territory

    g_np = jax.grad(mi.potential_fn)(mi.param_info.z)
    g_model = np.array([float(g_np[n]) for n in NAMES])
    assert set(g_np) == set(ALL_NAMES_SORTED)

    # LM-side gradient assembled at numpyro's exact (z, x) pair: solver
    # quantities at x_np, the analytic chain factor and prior at z_np.
    # (to_z/from_z round-trips are not bit-exact on Beta sites, and a
    # one-ULP input change re-routes the adaptive stepper.)
    wh = Whitened.build(log, *effective_sigma(m))
    mu, J = model_mean(x_np, log)
    ct = -(wh.L.T @ wh.residual(mu))
    chain = np.array(
        [
            xv * (1.0 - xv) if _is_beta(d) else xv
            for xv, (_, d) in zip(x_np, SITES, strict=True)
        ]
    )
    _fp, gp, _hp = prior_terms(z_np)
    g_lm = (J.T @ ct) * chain + gp
    scale = np.max(np.abs(g_lm)) + 1e-12
    assert np.max(np.abs(g_model - g_lm)) / scale < 1e-6
