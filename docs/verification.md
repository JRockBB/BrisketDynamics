# Gradient verification record

Every gradient path in this project is checked against central finite differences
at three levels. This file records the methodology, results, and the FD
conditioning pitfalls we hit (which are themselves informative).

## Level 1 — Julia solver adjoints vs FD (`test/test_gradients.jl`)

- **Meat** (`meat_vjp`, SciMLSensitivity
  `InterpolatingAdjoint(autojacvec = EnzymeVJP())` — the survivor of two
  documented sensealg incidents: the Aug-10 Gauss NaN hunt and the Aug-20
  silent-zero hunt, both below): 8 h horizon, tol 1e-9, mixed loss over
  the output heads. All seven θ parameters (incl. the expulsion latents
  k_e/a_myosin via juiciness AND coupled-Tcore paths), forcing-trajectory
  knots, the wrap time, smoke knots, bark/moisture output
  cotangents, and a **mid-cook (nonuniform IC) VJP** agree with central FD:
  - `D_eff` ≤ 2e-2 (1% FD step; see "FD conditioning" below)
  - all other parameters and knots ≤ 1e-3–5e-3 (typically 1e-5 to 1e-6)
- **Pit** (`pit_vjp`, exact dense ForwardDiff Jacobian, transposed):
  all seven φ parameters (incl. the ignition dynamics k_spread/a0),
  damper knots, and heat-load knots ≤ 1e-3
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
4 times) vs central FD on a scalar loss. Coverage: the five meat
transport/thermal latents (`theta[0:5]`; D_eff 2e-2 with a 1% step, others
5e-3), all 7 pit parameters, three damper directions, wrap time, and the
damper→burn-rate→smoke→bark-color chain. Expulsion latents `k_e` /
`a_myosin` are not in this Python parametrize; they are held at Level 1
(moisture adjoint + juiciness/Tcore paths). The full app suite is 27
collected tests (`test_composed_grad` 18, `test_mpc` 6, `test_local_apply`
3). CI runs `test_local_apply` only. 6 h horizon, tolerances 1e-10,
float64 end-to-end.

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

## The silent-zero hunt (moisture era, 2026-08-20)

The moisture build's enlarged output set exposed a second sensealg failure
mode, nastier than NaN because nothing raises: with large cotangent seeds
(||cot|| ~ 1e3), the backward adjoint solve can abort its FIRST step
(dt < machine epsilon) and SciMLSensitivity returns **identically zero
gradients**. Both Enzyme and ReverseDiff backends returned the same wrong
value — the tell that the defect sat above the VJP backend. Bisection
localized it to seed magnitude: the identical cotangent at unit norm was
exact to 2e-6. Fix, both permanent: `meat_vjp` normalizes cotangents to
unit magnitude on entry and rescales the gradient on exit (exact by
linearity of the VJP), and a loud guard throws if the adjoint ever
returns all-zero gradients against nonzero seeds. The lesson generalizes:
an adjoint that agrees with an independent backend can still be wrong —
backends share the backward *solve*, so cross-backend agreement only
exonerates the VJP layer.

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
   Tolerances are explicit inputs of both Tesseracts, and the composed gate runs both at
   1e-10.

## In-container check-gradients sweep (submission artifact)

The full trajectory-input sweep also runs **inside the shipped
containers** (`scripts/checkgradients/run_container.sh`; the tesseract CLI's
`run` command does not forward check-gradients flags, so the script invokes
the containers' runtime entrypoint via `docker run` directly — launched
foreground from a live WSL session, since WSL tears down idle VMs and
SIGTERMs detached containers mid-Julia-load). Results at seed 42
(`data/verification/cg_{meat,pit}_full.log`):

| Endpoint | Checks | Tool-flagged | After analysis |
|---|---|---|---|
| meat `tpit_knots` | 125 | 59 | **0 real** |
| pit `u_knots`+`qmeat_knots` | 398 | 205 | **0 real** |

The tool compares elementwise with `--rtol` only (no absolute floor), so
every flag is one of two benign classes, verified mechanically by
`scripts/checkgradients/analyze_failures.py` over all 264 detailed failure
blocks:

1. **Near-zero kink elements** (213 blocks): both VJP and FD are ≤1e-3 of
   the row scale at the interpolation-kink element; two-sided FD straddles
   the piecewise-linear kink the adjoint integrates one side of. Relative
   error is meaningless at these magnitudes; every informative element in
   the same rows agrees to ≤1e-5 relative.
2. **Causality-onset FD artifacts** (51 blocks): at the first save time a
   perturbed knot can influence, the adjoint reports exactly zero — correct
   by causality, since the forcing perturbation ramps from zero at that
   instant — while FD reports up to ~1e-2 of row scale. The eps-dependence
   experiment (`data/verification/cg_meat_eps3.log`) is decisive: growing
   `--eps` 10× (1e-4 → 1e-3) shrinks the FD onset value ~265×
   (2.16e-3 → 8.2e-6). A genuine slope is eps-independent; artifact divided
   by eps is not. The FD side is perturbed-solve dense-output noise / eps;
   the adjoint is right.

Conclusion: **zero real gradient mismatches in the shipped containers**;
all informative elements agree within the tool's tolerance at both eps
values. Parameter vectors remain covered by `test/test_gradients.jl` (a
single absolute `--eps` cannot span 12 decades; see above).

## Bark-era container sweep (rebuilt images, 2026-08-20)

Both images rebuilt with the wrap + bark model (122 meat states, pit
`mdot` output) and re-swept in-container
(`data/verification/sweep_bark_containers.log`). The payload activates the
wrap (t_wrap = 4 h) and a realistic smoke flux, so the new code paths
execute inside the shipped containers; the swept FD *directions* are the
trajectory forcings only. `t_wrap` and `smoke_knots` are **excluded as FD
directions by the same absolute-eps doctrine as the parameter vectors**
(one --eps cannot span 1.4e4 s and 1.2e-4 kg/s; a first sweep that included
them produced exactly the predicted artifacts: an 83% relative kick on
smoke measuring solver noise against the adjoint's correct zero, and a
7e-9 relative step on t_wrap measuring nothing). Both directions are held
at the Julia level with per-parameter relative steps (agreement 1e-5–2e-3)
and in the composed gate (5 meat + 7 pit + damper + wrap + bark chain).

Classified by `analyze_failures.py --exclude-inputs smoke_knots,t_wrap`:
pit 126 tool-flags and meat 39 tool-flags → **0 real mismatches**; classes
are the two established benign patterns plus a third documented here:
*FD-unresolvable weak rows* (entire row response < 1e-5 — e.g.
`tpit_knots → bark_s`, where a single forcing knot moves smoke deposition
at the 1e-6 level and VJP/FD agree in sign and order inside solver noise).

The in-container sweep logs live in `data/verification/`. Rebuild the
images and re-sweep after changing the meat payload (162 states, θ = 7,
fat/bark handoff). `meat_payload.json` already carries θ = 7.

Known issue: the in-container meat runtime intermittently hangs at
initialization (worker frozen at 0.29 s CPU; observed 2 of 3 launches on
the rebuilt image; the successful run supplied this artifact).
`probe_container.sh` detects the hang in one 45 s sample; sweep runs now
carry an 8-minute liveness gate. Suspected juliacall-init timing race —
tracked for the HeaterMeter/service era, where container startup
reliability actually matters.
