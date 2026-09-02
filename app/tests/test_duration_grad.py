"""GATE: differentiable duration — d/dt_f through the monolith vs finite differences.

Fractional-time mode (s = t/t_f): the schedule shape is fixed on [0, 1] and
the cook length stretches it, so ``jax.grad`` w.r.t. ``t_f`` is one more
forward-mode direction. Float64, tight tolerances (1e-8) so FD truncation
dominates over solver jitter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brisket_app.clients import monolith_client
from brisket_app.coupled import (
    HR,
    PHI_DEFAULT,
    THETA_DEFAULT,
    CookGrids,
    coupled_cook,
    coupled_cook_tf,
)

T_F = 6 * HR
N_SAVE = 37
S_SAVE = np.linspace(0.0, 1.0, N_SAVE)
S_U = np.linspace(0.0, 1.0, 6)
U = np.full(6, 0.5)
FW, FP = 0.5, 0.8
TOLS = {"abstol": 1e-8, "reltol": 1e-8}


@pytest.fixture(scope="module")
def tx():
    return monolith_client()


def test_tf_mode_matches_seconds_mode(tx):
    """At fixed t_f the fractional-time solve is the seconds solve."""
    grids = CookGrids(t_end=T_F, dt_exchange=600.0, n_u=6)
    assert grids.nt == N_SAVE
    a = coupled_cook(
        jnp.asarray(THETA_DEFAULT),
        jnp.asarray(PHI_DEFAULT),
        jnp.asarray(U),
        tx,
        grids,
        t_wrap=FW * T_F,
        t_pull=FP * T_F,
        **TOLS,
    )
    b = coupled_cook_tf(
        jnp.asarray(THETA_DEFAULT),
        jnp.asarray(PHI_DEFAULT),
        jnp.asarray(U),
        T_F,
        tx,
        S_U,
        S_SAVE,
        frac_wrap=FW,
        frac_pull=FP,
        **TOLS,
    )
    assert float(jnp.max(jnp.abs(a["tpit"] - b["tpit"]))) < 0.05
    assert float(jnp.max(jnp.abs(a["tcore"] - b["tcore"]))) < 0.05
    assert abs(float(a["w_retained"]) - float(b["w_retained"])) < 1e-4


def test_grad_wrt_duration_matches_fd(tx):
    """jax.grad d/dt_f of a quality-shaped scalar vs central FD."""

    def loss(t_f):
        out = coupled_cook_tf(
            jnp.asarray(THETA_DEFAULT),
            jnp.asarray(PHI_DEFAULT),
            jnp.asarray(U),
            t_f,
            tx,
            S_U,
            S_SAVE,
            frac_wrap=FW,
            frac_pull=FP,
            **TOLS,
        )
        return (
            1e-2 * jnp.sum((out["tcore"] - 340.0) ** 2)
            + 10.0 * jnp.sum(out["c_field_final"])
            + 100.0 * out["w_retained"]
        )

    g = float(jax.grad(loss)(jnp.asarray(T_F)))
    h = 60.0
    fd = (float(loss(T_F + h)) - float(loss(T_F - h))) / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-12) < 5e-3, (g, fd)


def test_grad_wrt_pull_fraction_matches_fd(tx):
    """jax.grad d/d(frac_pull): the event-fraction path through the same solve."""

    def loss(fp):
        out = coupled_cook_tf(
            jnp.asarray(THETA_DEFAULT),
            jnp.asarray(PHI_DEFAULT),
            jnp.asarray(U),
            T_F,
            tx,
            S_U,
            S_SAVE,
            frac_wrap=FW,
            frac_pull=fp,
            **TOLS,
        )
        return 1e-2 * jnp.sum((out["tcore"] - 340.0) ** 2) + 100.0 * out["w_retained"]

    g = float(jax.grad(loss)(jnp.asarray(FP)))
    h = 2e-3
    fd = (float(loss(FP + h)) - float(loss(FP - h))) / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-12) < 5e-3, (g, fd)
