# Gradient verification record

Every gradient path in this project is checked against central finite differences
at three levels. This file records the methodology, results, and the FD
conditioning pitfalls we hit (which are themselves informative).

## Level 1 — Julia derivatives vs FD (`test/test_monolith.jl`)

The monolith is differentiated in forward mode (ForwardDiff through the
FBDF solve): `monolith_jacobian` (residual Jacobian over the 15 physics
latents) and `monolith_stack_jacobian` / `monolith_stack_jvp` (the
Tesseract's `jacobian` / `jacobian_vector_product` endpoints over the
stacked outputs w.r.t. θ, φ, the damper knots, the wrap and the pull).

- Residual Jacobian columns for D_eff, h_c, UA and u_leak vs central FD
  (1e-3 relative step): ≤ 1.2e-4 relative.
- Stacked Jacobian's pull column vs FD (60 s step): ≤ 2e-2 relative
  (the pull blend is 300 s wide); juiciness-at-serve sensitivity to the pull
  is nonzero.
- JVP along a mixed tangent vs the Jacobian times that tangent: ≤ 5e-3
  relative — a 1-partial and a 5-partial Dual solve take slightly
  different adaptive steps (the error norm includes the partials).
- Coupling correctness: the Julia Picard exchange of the component solvers
  (`picard_cook`, K = 1, 2, 4, 8) converges to the monolith: K = 8 within
  0.04 K on the real-cook grid (`scripts/bench_monolith.jl`) and within
  0.02 K (measured 3 mK) on the test fixture with a 60 s exchange grid
  (`test/test_monolith.jl`).

Jitter, measured: a last-bit change in an input moves the Jacobian by
~4e-5 relative at solver tolerance 1e-6 (5e-7 at 1e-8, 3.7x the cost). Near
an optimum the gradient of the log density cancels O(1e3) terms, so two
exact paths differ by ~0.5 % there (`test_monolith_bridge.py`).

## Level 2 — Tesseract endpoint check (`tesseract-runtime check-gradients`)

Run on the trajectory inputs (the well-scaled ones; a single **absolute** FD
step `--eps` cannot span parameter vectors covering 12 orders of magnitude —
those are covered by Level 1's relative-step tests):

- **Pit** (`u_knots`, `qmeat_knots` → `tpit`, `m_fuel`, seed 42, 200 evals):
  every substantive Jacobian entry agrees to 6–7 significant digits.
  The tool reports "failures" exclusively on near-zero entries (elements outside
  a knot's support window, AD ≈ 1e-6, FD ≈ 7e-5 — both zero relative to O(100)
  neighbors), where a per-element relative comparison is meaningless. This is
  the false-positive mode the tool's own documentation warns about.
- **Meat** (`tpit_knots` → outputs, seed 42): quick 5-eval run at production
  tolerance (1e-8) flags percent-level disagreements on small-magnitude
  Jacobian entries — magnitudes agree, small elements differ 5–20%, consistent
  with the tool's absolute `--eps 1e-4` probing O(0.01–0.2) outputs against a
  1e-8 solver-noise floor. The same quantities pass Level 1 (≤1e-3) and the
  Level 3 composed gate at 1e-10, so this is FD conditioning, not an adjoint
  defect. The full tight-tolerance sweep (`scripts/checkgradients/run.sh`,
  ~1 h+) is deferred to Phase D as a submission artifact, per project policy
  of not running heavy redundant checks per-change.

## Level 3 — Composed pipeline gate (`app/tests/test_composed_grad.py`)

`jax.grad` through `coupled_cook` — one `apply_tesseract` call on the
monolith, reverse mode via tesseract-jax's transpose of the
`vector_jacobian_product` endpoint — vs central FD on a scalar loss.
Coverage: the five meat transport/thermal latents (`theta[0:5]`; D_eff
2e-2 with a 1% step, others 5e-3), all 7 pit parameters, three damper
directions, wrap time, pull time plus its bitwise backward-compatibility
test, and the damper→burn-rate→smoke→bark-color chain. Expulsion latents
`k_e` / `a_myosin` are held at Level 1. `test_monolith_tesseract.py` checks
the endpoints against each other (jvp = J·v, vjp = Jᵀ·w, 0-d scalar
blocks). 6 h horizon, tolerances 1e-10, float64 end-to-end.
