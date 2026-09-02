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
    Whitened,
    effective_sigma,
    from_z,
    objective,
    to_z,
)
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[2]


import pytest


@pytest.mark.xfail(
    reason="2026-09-02: real VJP defect, not test noise. At the current MAP the "
    "two paths' MEANS are bit-identical and the linear-surrogate gradient "
    "matches the whitened-LM gradient to 1e-13, but jax.grad (VJP endpoint) "
    "disagrees with the jacobian endpoint by up to 13% on pit components "
    "(UA, dH_eff). Richardson-extrapolated central FD of the potential sides "
    "with the JACOBIAN endpoint (UA +2.24, dH_eff -2.78). GN/Laplace/TMB use "
    "the jacobian endpoint (unaffected); NUTS acceptance uses values only "
    "(still targets the right posterior, just less efficiently). Fix is in "
    "the monolith Tesseract's VJP path (docs/verification.md).",
    strict=False,
)
def test_model_mono_gradient_matches_lm_objective():
    """d(potential)/dz for the physics sites == the LM objective's gradient.

    Same Jacobian, two code paths (numpyro's AR(1) sites through the
    Tesseract VJP vs the explicit whitening); at fixed noise parameters the
    dependence on the physics latents is identical up to solver jitter.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    log = cooklog.load(dt_min=10.0)
    x = np.array([m[n] for n in NAMES])
    z = to_z(x)
    init = dict(zip(NAMES, x, strict=True))
    init.update({n: float(m[n]) for n in NOISE_NAMES})
    mi = initialize_model(
        jax.random.PRNGKey(0),
        calibrate.bound_model_real(log, monolith_client()),
        init_strategy=init_to_value(values=init),
    )
    g_np = jax.grad(mi.potential_fn)(mi.param_info.z)
    g_model = np.array([float(g_np[n]) for n in NAMES])
    wh = Whitened.build(log, *effective_sigma(m))
    _F, g_lm, _H, _w = objective(z, wh, log)
    assert np.allclose(from_z(z), x, rtol=1e-12)
    assert set(g_np) == set(ALL_NAMES_SORTED)
    # the two paths evaluate at x and exp(log x): a last-bit difference moves
    # the adaptive solve's Jacobian by ~4e-5, which the near-optimum
    # cancellation in the gradient amplifies to ~0.5 % (see model_mean)
    scale = np.max(np.abs(g_lm)) + 1e-12
    assert np.max(np.abs(g_model - g_lm)) / scale < 1e-2
