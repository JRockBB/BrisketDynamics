"""The monolith bridge: the numpyro model's gradient equals the whitened LM gradient."""

import json
from pathlib import Path

import jax
import numpy as np
from brisket_app import cooklog
from brisket_app.monolith import (
    ALL_NAMES_SORTED,
    NAMES,
    NOISE_NAMES,
    Whitened,
    bound_model_mono,
    effective_sigma,
    from_z,
    objective,
    to_z,
)
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

REPO = Path(__file__).resolve().parents[2]


def test_model_mono_gradient_matches_lm_objective():
    """d(potential)/dz for the physics sites == the LM objective's gradient.

    Same Jacobian, two code paths (numpyro's AR(1) sites through the
    custom_vjp vs the explicit whitening); at fixed noise parameters the
    dependence on the physics latents is identical up to float error.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    log = cooklog.load(dt_min=10.0)
    x = np.array([m[n] for n in NAMES])
    z = to_z(x)
    init = dict(zip(NAMES, x, strict=True))
    init.update({n: float(m[n]) for n in NOISE_NAMES})
    mi = initialize_model(
        jax.random.PRNGKey(0),
        bound_model_mono(log),
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
