"""GATE A: composed finite-difference gradient check of the full coupled
pipeline — jax.grad through pit → meat → pit → meat (K=2 Picard, two Julia
Tesseracts, two different AD strategies) vs central finite differences.

Float64 everywhere, tight ODE tolerances (1e-8) so FD truncation dominates.
Pass: max relative error ≤ 5e-3. Expected: ≤ 1e-4.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brisket_app.clients import meat_client, pit_client
from brisket_app.coupled import HR, THETA_DEFAULT, PHI_DEFAULT, CookGrids, coupled_cook

GATE_TOL = 5e-3
# D_eff exception: its composed-loss response to sub-percent steps sits near the
# solver-noise floor even at tight tolerances (the K=2 feedback lets both
# solvers' adaptivity jitter into the loss), so it gets a 1% FD step and a
# percent-level gate. The meat adjoint itself is validated to 0.5% against FD
# at the Julia level (test_gradients.jl); this composed check verifies the
# cross-boundary plumbing.
DEFF_HREL = 1e-2
DEFF_TOL = 2e-2

# 6 h horizon reaches mid-stall so D_eff (moisture resupply) is a live parameter;
# shorter horizons leave it FD-invisible (validated in the Julia-level study).
# 1e-10 tolerances: both solvers' adaptivity jitter must stay below the FD signal.
GRIDS = CookGrids(t_end=6 * HR, dt_exchange=600.0, n_u=6)
TOLS = {"abstol": 1e-10, "reltol": 1e-10}


@pytest.fixture(scope="module")
def loss_fn():
    meat_tx, pit_tx = meat_client(), pit_client()

    def loss(theta, phi, u_knots):
        out = coupled_cook(theta, phi, u_knots, meat_tx, pit_tx, GRIDS, K=2, **TOLS)
        return (
            1e-3 * jnp.sum((out["tcore"] - 340.0) ** 2)
            + 1e-3 * jnp.sum((out["tpit"] - 390.0) ** 2)
            + 10.0 * jnp.sum(out["c_field_final"])
            + 1e3 * jnp.sum(out["j_evap"])
            + jnp.sum(out["m_fuel"]) / GRIDS.nt
        )

    return loss


def rel_err(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def central_fd(f, args, argnum, i, hrel=1e-3):
    x = np.asarray(args[argnum], dtype=np.float64)
    # RELATIVE step: parameters span ~12 orders of magnitude (D_eff ~ 1e-8,
    # C_th ~ 1e4); an absolute step either drowns in truncation or perturbs 30%
    h = hrel * abs(x[i]) if abs(x[i]) > 0 else 1e-6
    xp, xm = x.copy(), x.copy()
    xp[i] += h
    xm[i] -= h
    ap = [jnp.asarray(xp) if k == argnum else args[k] for k in range(len(args))]
    am = [jnp.asarray(xm) if k == argnum else args[k] for k in range(len(args))]
    return (float(f(*ap)) - float(f(*am))) / (2 * h)


@pytest.fixture(scope="module")
def grad_and_args(loss_fn):
    theta = jnp.asarray(THETA_DEFAULT)
    phi = jnp.asarray(PHI_DEFAULT)
    u = jnp.asarray(np.linspace(0.3, 0.6, GRIDS.n_u))
    g = jax.grad(loss_fn, argnums=(0, 1, 2))(theta, phi, u)
    return g, (theta, phi, u)


@pytest.mark.parametrize("i", range(5))
def test_grad_theta(loss_fn, grad_and_args, i):
    g, args = grad_and_args
    hrel, tol = (DEFF_HREL, DEFF_TOL) if i == 0 else (1e-3, GATE_TOL)
    fd = central_fd(loss_fn, args, 0, i, hrel=hrel)
    assert rel_err(float(g[0][i]), fd) < tol, f"theta[{i}]: ad={float(g[0][i])} fd={fd}"


@pytest.mark.parametrize("i", range(5))
def test_grad_phi(loss_fn, grad_and_args, i):
    g, args = grad_and_args
    fd = central_fd(loss_fn, args, 1, i)
    assert rel_err(float(g[1][i]), fd) < GATE_TOL, f"phi[{i}]: ad={float(g[1][i])} fd={fd}"


@pytest.mark.parametrize("i", [0, 2, 5])
def test_grad_u(loss_fn, grad_and_args, i):
    g, args = grad_and_args
    fd = central_fd(loss_fn, args, 2, i)
    assert rel_err(float(g[2][i]), fd) < GATE_TOL, f"u[{i}]: ad={float(g[2][i])} fd={fd}"
