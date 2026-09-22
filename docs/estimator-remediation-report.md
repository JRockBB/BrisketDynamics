# BrisketDynamics Remediation Report

Started: 2026-09-21  
Source audit: `docs/repo-estimator-audit.md`

This report records implementation changes in audit-issue order. An issue is
removed from the source audit only after its focused regression test passes.
Broader calibration or validation work made stale by a code correction is
called out separately rather than treated as proof of correctness.

## Issue 1 — Process noise was applied after the endpoint observation

Status: **fixed and focused tests passing**

### Changes

- Reordered `filter_update!` into a true predict-then-update cycle:
  parameter process noise enters before interval propagation, physical and
  event process noise enters the predicted endpoint covariance, and the
  endpoint RxInfer graph then assimilates the reading.
- Added a dedicated endpoint observation context. It reconstructs
  instantaneous channel observables from the predicted endpoint state while
  retaining the existing channel registry and RxInfer graph.
- Removed event-specific pit-probe measurement inflation. Event uncertainty
  is now state uncertainty and can be conditioned by the endpoint reading.
- Initial-time readings are assimilated before the first transition instead
  of sharing one discrepancy state with the first endpoint.
- Added a square-root/whitened RxInfer update. A nearly deterministic
  transition exposed a correlation eigenvalue near `4e-14`; directly
  inverting that covariance produced a posterior wider than its prior.
  The graph now infers a standardized variable with an identity Gaussian
  prior under `v = m + Sξ`, where `P = SS'`, and transforms the posterior
  back with triangular products.
- Retained only a roundoff-scale correlation-eigenvalue repair before
  Cholesky factorization; it is no longer used as a substitute for a stable
  precision update.
- Added a regression case comparing an event-crossing prediction with and
  without a precise endpoint pit reading. The reading must remove most of
  the event-induced pit-state variance.

### Verification

- Module load: passed.
- Square-root covariance diagnostic: passed; latent standard deviations
  remained finite and prior-scale instead of expanding by factors of
  roughly 20–100.
- `estimator/test/test_sequential.jl`: **141/141 passed**, including the new
  endpoint-event conditioning regression.
- `estimator/test/test_segments.jl`: **16/16 passed**, including compatibility
  of the original full-precision `FilterPrior` path.
- The initial-time/first-endpoint discrepancy defect formerly listed as
  audit issue 11 was resolved by assimilating initial readings before the
  first transition.
- State-noise calibration: must be rerun after this change because the filter
  fingerprint and process-noise semantics changed.

## Issue 3 — Scale model double-counted fat

Status: **fixed and focused tests passing**

### Changes

- Added one composition-conserving mass function shared by estimator stacks,
  mock instruments, generators, and tests:

  `non-fat solids + unrendered fat + retained rendered fat + water`.

- Corrected both 1-D and 2-D scale outputs. At the raw state (`F = F0`) the
  mass factor is now exactly `1 + W`, rather than `1 + W + F0`.
- Threaded `MeatConfig` through real-cook mock readings so nondefault
  `F0` and `eta_drain` values cannot silently use default mass bookkeeping.
- Updated all tracked mock and coverage generators to use the shared
  function.

### Verification

- `estimator/test/test_channels.jl`: **26/26 passed**, including four new
  raw-mass, rendered-fat-loss, and derivative checks.
- Regenerated `mock_1d.json`, `mock_2d.json`, and `mock_realcook.json` from
  the corrected observation model.
- `estimator/test/test_physics2d.jl`: **17/17 passed**; corrected 1-D and
  2-D stack masses reproduce their independently generated clean readings.
- `estimator/test/test_realcook.jl`: **16/16 passed**.
- `estimator/test/test_sequential.jl`: **141/141 passed** with the corrected
  scale readings.
- `test_smoother.jl` reached a pre-existing unstable monolith solve at the
  prior-median physics call, before the scale output is evaluated. The mass
  path itself is covered by the passing stack and sequential suites above;
  the serialized-prior cause remains tracked under issue 8 and was not
  treated as an issue-3 regression.

## Issue 4 — Monolith observables disagreed with the dynamics

Status: **fixed and focused tests passing**

### Changes

- Centralized the surface mass-flux result used by both the RHS and output
  reconstruction. One call now returns total meat flux, the pull-gated flux
  entering the pit, and the resulting pit vapor pressure.
- Applied skin resistance, salt activity, wrap suppression, pull blending,
  and vapor feedback once in that shared path.
- Reconstructed monolith outputs with the effective wrap state combining the
  scheduled wrap and irreversible temperature-trigger latch.
- Added `wrap_latch` and `wrap_effective` trajectories to monolith outputs
  for direct parity checks and downstream state handoff.
- Updated the standalone meat output path to use the same flux result.

### Verification

- Added unit checks for shared flux/observable equality, skin resistance,
  and zero meat contribution to pit humidity after pull.
- Added a triggered-wrap monolith check that independently reconstructs
  `j_evap`, `pv_pit`, and `q_meat` from the integrated state.
- Root physics suite: **376/376 passed**, including the new parity checks.
- Regenerated `mock_realcook.json` so post-pull humidity observations follow
  the corrected pit-share flux.
- Regenerated `plan_experiment.json` because planner outcomes include the
  corrected monolith observables.
- `estimator/test/test_realcook.jl`: **16/16 passed**.
- `estimator/test/test_physics2d.jl`: **17/17 passed**.

## Issue 6 — Sigma-point update did not optimize one coherent objective

Status: **fixed and focused tests passing**

### Changes

- Made the statistical-linearization covariance an explicit outer
  majorization quantity. Each graph proposal and every line-search trial are
  now compared under one frozen set of nonlinear-residual weights.
- After an accepted move, the sigma-point affine model and weights are
  rebuilt and the baseline objective is immediately reevaluated before the
  next proposal. Objectives from two different weight sets are no longer
  compared.
- Extended iteration history with `F_before` and `F_trial`, while `F` records
  the relinearized surrogate baseline. The accepted-step decrease is always
  `F_before - F_trial`.
- Documented that the reported local Gaussian covariance is conditional on
  the final statistical-linearization surrogate; it is not labeled as a
  covariance from a globally normalized nonlinear posterior.
- Added a moving-weight regression whose relinearized objective deliberately
  differs from the accepted trial objective.

### Verification

- Regenerated `plan_experiment.json` after the issue-4 observable correction.
- `estimator/test/test_iterated.jl`: **13/13 passed**, including the
  moving-surrogate regression and RxInfer planner/estimator gates.
- `estimator/test/test_sequential.jl`: **141/141 passed** with actual
  sigma-point state propagation and the RxInfer graph update.

## Issue 7 — Nonconverged online updates were committed

Status: **fixed and focused tests passing**

### Changes

- A short nonconverged online fit is continued once from its last accepted
  center with a larger iteration budget.
- Endpoint fits commit their mean and covariance only when the initial or
  retry solve reports convergence.
- A failed fit advances time with the process-predicted endpoint belief,
  emits a warning, and reports `attempted=true`, `updated=false`.
- Rejected readings do not enter discrepancy residuals, volatility levels,
  AR-noise re-identification, or fault adaptation.
- A nonconverged initial-time assimilation similarly retains the original
  initial belief.

### Verification

- Added a forced-zero-iteration regression comparing a rejected reading with
  the identical no-reading prediction. Means and covariances must agree and
  all adaptive histories must remain empty.
- `estimator/test/test_sequential.jl`: **146/146 passed**. Routine updates
  converge on continuation; the deliberately forced failure retained the
  no-reading prediction exactly.

## Issue 8 — Serialized priors were silently misread

Status: **fixed and focused tests passing**

### Changes

- Replaced every manual three-field `PriorSite` reconstruction in tracked
  estimator drivers and tests with `read_sites`, preserving `p3`, `p4`, and
  `coord`.
- Updated tracked analysis scripts as well; no tracked manual serialized-site
  constructor remains.
- Replaced manual log/logit conversions in those paths with the shared
  `to_z`, `from_z`, and `draw_truth` family-aware transforms.
- Made `read_sites` reject serialized `logbeta` entries that omit either
  bound instead of silently substituting `[0, 1]`.
- Added round-trip checks covering family, shape parameters, bounds,
  score coordinate, and constrained values.
- Updated the smoother's drift-effect threshold to the correctly loaded
  score-coordinate prior; the old, larger effect was produced by the
  erroneous logit interpretation.

### Verification

- Search gate: no manual three-field serialized-prior constructor remains in
  tracked files.
- `estimator/test/test_smoother.jl`: **43/43 passed**; the formerly unstable
  prior-median monolith solve is now stable.
- `estimator/test/test_horizon.jl`: **278/278 passed**.

## Issue 9 — Censored humidity used a squared-hinge surrogate

Status: **fixed and focused tests passing**

### Changes

- Removed the soft clamp from `CensoredHumidity`; its prediction is now the
  latent, uncapped sensor value.
- A capped sample now contributes the normal survival loss
  `-log Φ((prediction - cap) / σ)`. The lower tail and inverse Mills ratio
  use `erfcx`, so severe cap violations remain finite.
- Kept inference on the existing RxInfer graph by replacing each capped row
  at a linearization center with a Gaussian site whose score and curvature
  exactly match the survival loss there. Line search and convergence use the
  survival loss itself, not that local quadratic.
- Applied statistical-linearization variance and levelled sensor noise to
  the censoring scale before constructing the site.
- In the sequential estimator, the likelihood acts on
  `physics prediction + AR discrepancy state` before clipping. A capped
  observation therefore conditions the discrepancy, whose transition
  carries the dependence to later arrivals.
- Kept capped batch samples out of AR residual identification; they are
  marginal survival terms rather than artificial zero residuals. The
  compatibility field `s` remains readable but no longer controls a
  soft-hinge approximation.
- Changed ridge-posterior scoring and all nonlinear graph objectives to use
  the same survival likelihood.
- Excluded local survival sites from Gaussian residual-power, NIS, and
  chi-squared diagnostics.

### Verification

- Finite-difference checks verify that the RxInfer site's gradient and
  curvature match the censored-normal loss.
- Severe-tail regression verifies finite likelihood and site parameters.
- `estimator/test/test_realcook.jl`: **18/18 passed**, including a direct
  censored RxInfer update of the sequential discrepancy state.
- `estimator/test/test_iterated.jl`: **13/13 passed**, including the
  real-cook graph estimator.
- `estimator/test/test_sequential.jl`: **149/149 passed** with capped
  readings connected through the discrepancy-state graph path.
- `estimator/test/test_misfit.jl`: **1208/1208 passed**, including exclusion
  of survival sites from Gaussian diagnostics.
- `estimator/test/test_ridge.jl`: **7/7 passed** with exact survival scoring.

## Issue 12 — Probe-depth prior and coverage truth disagreed

Status: **fixed and focused tests passing**

### Changes

- Replaced the independent LogNormal insertion distance with
  `core_depth_fraction ~ Beta(8, 2)` in the normal-score coordinate.
  Its physical insertion depth is now formed jointly with thickness as
  `depth = L * fraction`.
- Updated `MeatProbe` to interpolate at material coordinate
  `s = 1 - fraction` and differentiate through the bounded prior transform.
  The old clamp and its zero-sensitivity region are unreachable.
- Made a latent probe channel reject an old positive/lognormal depth site,
  so stale drivers fail explicitly instead of silently changing units.
- Removed truth-only clipping from `draw_truth`. Coverage truth now comes
  from exactly the same bounded site used for inference; use of the former
  `depth=` clipping keyword is rejected.
- Updated estimator drivers, coverage workers, mock generators, planner
  fixtures, tests, and tracked analysis scripts to store the fraction and
  convert to a physical depth only at instrument generation.
- Renamed the latent and mock metadata to `core_depth_fraction`.
- Added `ProbeDepthReading` for a ruler measurement in physical metres; its
  Jacobian couples thickness and fraction instead of treating 3 mm as a
  dimensionless fraction error.
- Regenerated the 1-D, 2-D, real-cook, and drifting mock records with a
  fraction truth of `0.8`; their physical probe location and readings are
  unchanged.
- Marked prior coverage and sensor-value results that used `core_depth` as
  stale. They cannot establish coverage for the new parameterization.

### Verification

- `estimator/test/test_channels.jl`: **34/34 passed**, including the bounded
  transform, physical-depth relation, analytic Jacobian, and rejection of a
  stale LogNormal site.
- `estimator/test/test_physics2d.jl`: **17/17 passed**; both physics stacks
  reproduce their clean probe readings at the fraction truth.
- `estimator/test/test_segments.jl`: **16/16 passed** with the bounded site in
  the full-Gaussian graph prior.
- `estimator/test/test_sequential.jl`: **151/151 passed** through the
  sequential RxInfer path after the prior migration.
- `estimator/scripts/sensor_value.jl` completed and regenerated
  `sensor_value_real.json` with the fraction latent and joint physical-depth
  ruler.
- `estimator/scripts/plan_experiment.jl` completed and regenerated the planner
  reference after explicitly translating its historical six-hour MAP into the
  fraction coordinate and rebuilding covariance there.
- Truth-draw checks require all fractions to lie strictly inside `(0, 1)`,
  recover the declared mean, produce ordinary draws above `0.9`, and reject
  the removed clipping API.
- Full held-out coverage must be rerun because the latent coordinate, prior,
  mocks, and physics fingerprint changed.

## Issue 13 — Sequential Gaussian state could leave the physical domain

Status: **fixed and focused tests passing**

### Changes

- Moved the sequential plant state to unconstrained filter coordinates:
  positive quantities use a stable shifted-softplus map and unit-interval quantities
  (collagen, smoke ring, and ignited fraction) use a scaled logit map.
  Every Gaussian mean, sigma point, and posterior draw therefore decodes to
  positive temperatures, moisture, fat, salt, bark, fuel, pan, and firebox
  states and to strictly bounded fractions.
- Kept the RxInfer factor graph as the update path. `PhysicsState` decodes a
  graph point before the monolith solve and re-encodes the handed-off endpoint;
  ForwardDiff consequently supplies Jacobians in the new coordinates.
- Mapped initial, continuous-process, and event covariances from their documented
  physical units into local state coordinates. A per-interval coordinate-width
  ceiling prevents a physical additive-noise approximation from becoming
  singular at a boundary.
- Removed ignition-mean clamps. Bounds are now structural and apply to the whole
  Gaussian belief after decoding, not only to one reported mean.
- Changed covariance prediction to propagate and combine square-root factors by
  QR. Statistical prediction now obtains conditional regression with a
  Cholesky solve, repairs/checks the conditional covariance to PSD, and never
  forms an explicit inverse or covariance division.
- Stabilized the logistic map in both tails.
- Converted reported pit/core/surface moments back to physical units while
  retaining coordinate covariance for state-aware planning and NEES checks.
- The state-noise fingerprint is intentionally invalidated; consistency rates
  must be re-earned in the new coordinates.

### Verification

- `estimator/test/test_sequential.jl`: **213/213 passed**.
- The suite covers state round trips, extreme-coordinate support, physical
  reporting, actual posterior draws, event uncertainty, covariance PSD,
  square-root propagation against the covariance identity, and the
  state-aware planning restart contract.
- The updated files have no linter errors.

## Issue 15 — Hardware pilot safety and correctness defects

Status: **fixed and focused tests passing**

### Changes

- Normalized every pit timestamp to elapsed cook time from the first status
  sample. Non-finite timestamps and backward clock motion now stop the
  pilot instead of creating a multi-billion-second segment.
- Added status validation for pit temperature and average fan duty, converted
  non-finite optional food readings to missing values, and prevented a
  zero-pace hardware caller from spinning HTTP requests while the pit clock
  is unchanged.
- Events no longer become model truth merely because an advisory was emitted.
  Wrap, pull, fuel-add, and spritz remain pending until a backend hook or an
  explicit human callback acknowledges execution; actual execution time and
  acknowledgement source are journaled.
- Forwarded all four event families through segment prediction and
  assimilation. Executed chimney and recurring-spritz times retain their
  original phase on later replans instead of being reset at each horizon.
- Added fuel-add and spritz hooks to the in-process and HTTP mock pits,
  including spritz interval and dose.
- Added a true `--shadow` mode that never sends a setpoint or invokes a
  backend event hook. Real hardware now defaults to real-time polling and
  interactive event acknowledgement; simulation and mock backends retain
  free-running execution.
- Made the Python estimator boundary explicit: the launch food reading seeds
  the uniform initial meat temperature, later food readings are journal-only,
  and the separate RxInfer estimator remains the intended estimation loop.
- Completed receding-horizon state handoff for pan mass, pan temperature,
  firebox temperature, and the irreversible trigger-wrap latch. The
  Tesseract now exposes those trajectories and accepts their initial values;
  `CookState` carries them and the firebox configuration through every
  segment.
- Updated the mock HeaterMeter, pilot CLI help, README example, and roadmap
  to describe acknowledgement and non-actuating shadow operation.

### Verification

- `app/tests/test_pilot.py`: **6/6 passed**, covering UNIX-clock
  normalization, clock/fan validation, pending acknowledgement, all backend
  events, shadow non-actuation, add/spritz forwarding, and complete Python
  handoff wiring.
- A focused Julia split-versus-whole solve passed for a nondefault water pan,
  two-zone firebox, and nonzero wrap latch. Terminal core/pit temperatures,
  pan mass/temperature, firebox temperature, and latch agreed across the
  restart tolerances.
- Julia package load and Python compilation passed; modified Python files
  pass Ruff's import/name checks.
- The local Python Tesseract endpoint test remains blocked at setup by the
  already-audited Windows `PythonCall` `sys.dll` initialization failure.
  This occurs before endpoint code runs; direct Julia handoff and mocked
  Python boundary tests cover the changed contracts pending that environment
  repair.
