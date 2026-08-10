# Gradient verification record

Every gradient path in this project is checked against central finite differences
at three levels. This file records the methodology, results, and the FD
conditioning pitfalls we hit (which are themselves informative).

## Level 1 — Julia solver adjoints vs FD (`test/test_gradients.jl`)

- **Meat** (`meat_vjp`, SciMLSensitivity `GaussAdjoint(autojacvec = EnzymeVJP())` —
  switched from `InterpolatingAdjoint(ReverseDiffVJP(false))` on Aug 10 after
  profiling showed 27x speedup with gradients agreeing to ~2e-5; the full
  suite below re-passed 32/32 after the switch):
  8 h horizon, tol 1e-9, mixed loss over all five outputs. All five physical
  parameters and four forcing-trajectory knots agree with central FD:
  - `D_eff` ≤ 2e-2 (1% FD step; see "FD conditioning" below)
  - all other parameters and knots ≤ 1e-3 (typically 1e-5 to 1e-6)
- **Pit** (`pit_vjp`, exact dense ForwardDiff Jacobian, transposed):
  all five parameters, damper knots, and heat-load knots ≤ 1e-3
  (heat-load knots ≤ 1e-3 rather than 1e-4 purely because the per-knot FD signal
  is tiny; the Jacobian itself is exact to machine precision).

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

`jax.grad` through the full K=2 Picard composition
(pit → meat → pit → meat, gradients crossing the Tesseract VJP boundary
4 times) vs central FD on a scalar loss: **13/13 pass** —
5 meat parameters, 5 pit parameters (5e-3 gate; D_eff 2e-2 with a 1% step),
and damper-knot directions. 6 h horizon, tolerances 1e-10, float64 end-to-end.

## The NaN-gradient hunt (Phase C) — symptoms vs root cause

A single root cause produced four distinct-looking symptoms over one evening:
`GaussAdjoint`'s quadrature emits NaN gradients when the meat backward pass
crosses a hot fuel-collapse pit trajectory carrying a coupling-feedback
(`q_meat`) cotangent. The NaN reached the optimizer state, and downstream
appeared as: (1) BoundsError from a 1-point pit solve (NaN damper → NaN dt →
solver exits at t0), (2) "all-NaN u_knots" at the pit endpoint, (3) flaky
iteration-dependent onset (the trajectory only collapses once the optimizer
opens the damper), (4) apparent thread races. Fixed by switching to
`InterpolatingAdjoint(EnzymeVJP())` (agrees with ReverseDiff to 6 digits on
the captured failing call, 4x faster than Gauss there). The failing call was
captured via an endpoint hook that dumps inputs+cotangents to disk on any NaN
gradient and raises FloatingPointError — that guard stays in production.

Separately-real threading rules learned en route (all load-bearing, all in
the tesseract_api.py files): all juliacall work on one dedicated thread;
stringify Julia exceptions on that thread; never let zero-copy views of
Julia or XLA memory cross threads (copy on entry AND exit); solvers fail
loudly on bad retcodes rather than returning truncated arrays.

## FD conditioning lessons (worth knowing before touching these tests)

1. **`ReverseDiffVJP(true)` (compiled tape) is silently wrong here** — the
   time-dependent interpolation weights of the forcing get frozen into the
   compiled tape as constants, zeroing the adjoint w.r.t. all forcing knots
   and corrupting parameters entering the surface BC. `compile=false` is
   correct (and faster, since the RHS closure is rebuilt per call anyway).
   This is why `MEAT_SENSEALG` is what it is. Do not "optimize" it back.
2. **`D_eff` is FD-invisible on short horizons.** Early in the cook the surface
   is demand-limited, so the loss response to sub-percent `D_eff` steps sits
   below solver-adaptivity noise, and central FD returns noise scaled by 1/h
   (sign flips included). Gradient tests for `D_eff` must run ≥ 6–8 h horizons
   (stall active) with a ~1% relative step.
3. **Both solvers' tolerances bound the FD noise floor of the composed loss.**
   The pit Tesseract originally had fixed 1e-8 tolerances; through the Picard
   feedback (θ → q_meat → pit → forcing) its adaptivity jitter entered every
   FD evaluation at ~1e-5, drowning small-parameter signals. Tolerances are
   now explicit inputs of both Tesseracts, and the composed gate runs both at
   1e-10.
