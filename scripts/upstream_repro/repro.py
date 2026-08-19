"""Repro: jax.grad through apply_tesseract w.r.t. a SUBSET of the
differentiable inputs misbehaves; differentiating w.r.t. ALL of them is
fine.

Model: y = sum(a**2) + sum(3*b)  →  dy/da = 2a, dy/db = 3 (exact).

Run:  uv run python repro.py   (from this directory)
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

tx = Tesseract.from_tesseract_api(Path(__file__).parent / "tesseract_api.py")

a0 = jnp.asarray(np.array([1.0, 2.0, 3.0]))
b0 = jnp.asarray(np.array([10.0, 20.0]))
k0 = jnp.asarray(np.array([0.5, 0.5, 0.5, 0.5]))

# uses BOTH outputs; non-diff input interleaved between the diff ones
# analytic: L = y + sum(z) = sum(a^2)+3*sum(b)+sum(k) + sum(a)*sum(b)
#   dL/da = 2a + sum(b)          = [32 34 36]
#   dL/db = 3 + sum(a)           = [9 9]


def f(a, b):
    out = apply_tesseract(tx, {"a": a, "knots": k0, "b": b})
    return out["y"] + jnp.sum(out["z"])


print("forward:", float(f(a0, b0)), " (expect 104+2+180 = 286)")

for name, fn in [("eager", lambda g: g), ("jit", jax.jit)]:
    g_all = fn(jax.grad(f, argnums=(0, 1)))(a0, b0)
    g_a = fn(jax.grad(f, argnums=0))(a0, b0)
    g_b = fn(jax.grad(f, argnums=1))(a0, b0)
    print(f"[{name}] all: da={np.asarray(g_all[0])} db={np.asarray(g_all[1])}")
    print(f"[{name}] subset a: da={np.asarray(g_a)}  (expect [32 34 36])")
    print(f"[{name}] subset b: db={np.asarray(g_b)}  (expect [9 9])")
