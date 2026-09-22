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

## Issue 2 (part) — Stale state-noise rates were loaded with a warning

Status: **fixed and focused tests passing**

### Changes

- `load_state_noise` is strict by default: a missing rates file or a
  fingerprint that is not the tree's is an error naming both fingerprints
  and the bench command. `strict=false` keeps the old warning and exists for
  one caller, the bench's tuning round, which starts from the file's rates
  by design; `coverage_filter.jl` takes that as an explicit `stale_ok`
  argument and records it, and `bench_state_noise.py` passes it for the
  tuning round only.
- The bench's fresh round now refuses to run on rates earned on another
  tree (`SystemExit` before the rates file is written; previously a printed
  note), and both rounds record their own fingerprint in the file.
- `physics_fingerprint` now also covers `stream.jl` (the graph fit the
  filter calls) and `misfit.jl`.
- The family-test branch of the bench (`--truth-gaussian`) was removed: with
  every bounded site carried in its normal score, the "Gaussian truths" draw
  was the family's own draw.
- The bench's default is two workers with two BLAS threads each; the
  three-worker run of 2026-09-21 exhausted this laptop's memory.
- Every flight and coverage driver that loads the rates is strict; the
  committed rates file (`ba928b214b27`) is refused on the current tree
  until the bench re-earns it.

### Verification

- `estimator/test/test_sequential.jl`, the state-block testset: **37/37 passed**,
  including the new checks: a missing file throws, a file whose fingerprint
  is `000000000000` throws and loads with `strict=false`, `stream.jl` and
  `misfit.jl` are in the hashed set, and on a synthetic tree a byte in either
  moves the fingerprint.
- The tree's fingerprint has moved from `fb25d6a57234` (the audited
  remediation) by these changes; `data/calibration/state_noise.json` still
  records `ba928b214b27` and is now refused by every driver but the bench's
  tuning round.

## Roadmap item (ii) — No trust region on the online update

Status: **fixed and focused tests passing**

### Changes

- `IteratedProblem` carries an optional `step_cap(c, step) -> α_max`; the
  graph's step is cut to the trust region before the line search, the line
  search runs from there (its floor relative to the cap), the
  Levenberg-Marquardt relaxation fires on a first-trial acceptance capped or
  not, and convergence is never declared on a capped step. The history
  records `alpha_max`.
- The whitened filter solve caps the step so that no coordinate of the block
  (latents, state, discrepancies) moves beyond its own predicted width
  `sqrt(P_ii)` in one iteration; the raw solve caps by each site's local
  width at the centre. Both solves take `trust=true` by default.
- The filter's update record counts the capped iterations (`capped`), so
  the bench records show how often the region binds.

### Verification

- `estimator/test/test_iterated.jl`: **20/20 passed**, the new testset 7/7 (a site
  thirty widths away is reached in 29 capped moves of one width and one
  free step, convergence declared only on the free step; the whitened cap
  brings a three-width move to a third); the real-cook gate still
  reproduces the Gauss-Newton posterior with the cap on by default.
- `estimator/test/test_sequential.jl`: **219/219 passed**; the filter's own first
  update (the initial readings assimilated, the prediction, the update at
  the first arrival) commits with the cap on, 4 of 14 iterations capped
  against 4 iterations uncapped (the capped initial assimilation needs the
  continuation), and the `capped` count in the record includes the initial
  assimilation.
- What the cap does on routine updates (the filter's first ninety minutes on
  `mock_realcook.json`, 45 updates, cap on against cap off): every update
  commits either way, 64 of 240 iterations are capped, the belief at ninety
  minutes differs by at most 0.15 of a width on one state entry and by less
  than 0.1 sd on every latent, the latent widths by less than 2.5 %; the
  capped run takes 600 s against 357 s, because a capped iteration is
  accepted and rebuilds the surrogate where the uncapped line search would
  have rejected the proposal cheaply.
- Where it binds, iteration by iteration: at the cook's first grid point the
  exposed area (1.2 widths; the initial readings pull hard from the prior
  centre); from the second update on, one direction pairing the probe's
  depth fraction with the temperature-field nodes bracketing the probe and
  the core probe's discrepancy, at 2 to 9 widths on the first updates and
  10 to 60 widths by the sixth (the proposal moves the probe deeper and
  heats the nodes there, which leaves the reading unchanged). The exact
  objective rejects those proposals with or without the cap; they are the
  update surrogate's ridge between the probe's position and the field around
  it, the same pairing the per-node field noise once fed at the noise level
  (fixed 2026-09-16), now visible in the affine model. The cap is kept on by
  default as decided; the bench's records count the capped iterations per
  update, and the seed-12 lock the cap was built for is the bench's to show.

## Issue 10 (part) — The AR coefficient's posterior was clamped after inference

Status: **fixed and focused tests passing**

### Changes

- RxInfer's AR node keeps its Gaussian on the coefficient (its rules emit and
  consume Gaussian messages; a delta node in front of it would not bound the
  posterior the node uses under mean-field, only its read-out). The
  post-hoc point clamp is replaced by the posterior's truncated-normal
  moments on the stationary interval (−1, 1), in `erfcx` forms that stay
  finite when the Gaussian sits tens of sd beyond the bound. The summary is
  the Gaussian's own moments when its mass lies inside; the raw posterior
  stays in the result for diagnostics. An interval the Gaussian has no
  numerically representable mass in is an error, not a clamp.
- The remaining clamps downstream (the grid-step power, the level process)
  are no-ops on the summary and stay for the negative side.
- Not done here: carrying the noise posterior's width into the discrepancy
  transition; the bench's low innovation power argues against widening it
  before the bench is rerun.

### Verification

- `estimator/test/test_noise.jl`: **19/19 passed**, including the new testset (11
  checks): the truncated moments in the tails (a posterior at 1.03 with sd
  0.001 gives a mean in (0.9999, 1) and a finite width), an explosive
  residual whose least-squares slope exceeds one gives a node posterior
  mean above 1 and a summary strictly inside the bound narrower than the
  Gaussian, and a stationary AR(0.8) series keeps the node's own moments to
  1 % of its width.

## Issue 16a — The spritz mask ran past the wrap

Status: **fixed and focused tests passing**

### Changes

- `lid_windows` and `lid_open_at` stop the spritz train at the wrap or the
  pull, whichever is first, as the monolith gates the lid lift.
- Regenerated `mock_realcook.json` and `mock_drift.json` (their masks and
  the readings drawn under them change).

### Verification

- `estimator/test/test_realcook.jl`: **all 21 checks passed** on the regenerated
  mock, including the new ones: no lid window between the wrap's own window
  and the pull on the grid or at any time, a spritz before the wrap still
  opens the lid, and an unwrapped cook's train still runs to the pull.
- `mock_realcook.json` regenerated: 886 pit and core samples, 177 humidity
  (34 at the cap), 58 scale, 15 lid-open grid points; `mock_drift.json`
  regenerated from it (seed 23).

## Issue 16b — The water pan's mass could overshoot below zero and freeze

Status: **fixed and focused tests passing**

### Changes

- Both monolith solves reject an adaptive step that takes the pan's water
  mass materially below zero (`isoutofdomain`), so the state converges onto
  the dry pan instead of freezing beneath it. The bound sits a millionth of
  the taper scale below zero: as the pan dries the state converges onto zero
  itself, and a trial a rounding error below it is not an overshoot (a
  strict bound at zero made the step size collapse there).

### Verification

- `test/test_monolith.jl` run alone: **107/107 passed** (7 min 27 s), including the new
  dry-out regression (a 0.25 kg pan dries to within a nanogram of zero, the
  humidity and pan temperature finite). With a strict bound at zero the same
  solve went `Unstable` at the dry-out (17 of 49 points saved); the tolerant
  bound is what made it pass.
- The full root suite (`Pkg.test()`): **386/386 passed** (9 min 53 s) with the
  domain guard on every monolith solve.

## Issue 16c — Interpolated pit gaps were marked usable

Status: **fixed and focused tests passing**

### Changes

- `cooklog.load` masks a span of missing pit samples out of the likelihood
  as it already did for the meat probe; the span is still interpolated for
  the grid. The tracked calibration log has no pit gaps, so no tracked
  calibration artifact changes.

### Verification

- `app/tests/test_calibrate.py` and `test_objective_parity.py`: **9/9 passed**
  (no Julia involved), including the new gap test: a 50-minute pit gap in a
  synthetic log is interpolated on the grid, masked out of the pit
  likelihood, and leaves the meat mask untouched.
- The tracked calibration log (`mache_brisket_cook.csv`) has no missing pit
  samples (4 missing meat samples, 1 missing fan sample), so no tracked
  calibration changes.

## Issue 16d — No term-by-term parity gate on the objective

Status: **fixed and focused tests passing**

### Changes

- `estimator/scripts/objective_parity.jl` writes
  `data/mock/objective_parity.json`: one cook's surface, core and fuel
  trajectories, its executed damper as hourly steps, the scalars the
  objective reads at serve, the goal parameters, and the terms Julia's port
  gives them. Both suites score that stored cook without running physics:
  the Julia test rebuilds the cook output from the trajectories and must
  reproduce the terms; the app test scores the same cook with
  `mpc.executed_objective` and must match tenderness, appearance, taste,
  fuel and smoothness to 1e-9, and checks the goal parameters field by
  field. The bark-burn mean and the lethality integrals are held to a 5 %
  tolerance because their windows differ by design (Julia blends the pull
  and the serve smoothly; the executed score cuts at the pull).
- The 0.9 L* gap between the two engines' bark remains a roadmap item; the
  crosscheck script's disagreement came from the plants, not the formula.

### Verification

- `estimator/test/test_horizon.jl`: **283/283 passed**, the parity testset 5/5: the
  goals equal the fixture's field by field, the meat configuration's water
  and fat constants equal the fixture's, and the cook output rebuilt from
  the stored trajectories reproduces juiciness, the burn term, the fuel
  fraction, both lethality integrals and every term to 1e-9.
- `app/tests/test_objective_parity.py`: **2/2 passed** (no Julia): the goal
  parameters match `ControlConfig` field by field, and on the stored cook
  `executed_objective` matches tenderness, appearance, taste, fuel and
  smoothness to 1e-9; the burn mean is zero on both sides, the lethality
  integrals agree within the serve-horizon tolerance, and the safety terms
  are zero on both sides.
- Found on the way and fixed in the same file: the replan test's "never worse
  than the warm start alone" compared the objective with the plan prior while
  the settled pick rule judges by the goals alone; it failed at the last
  commit with the same numbers (goals 0.537 against 0.572 for the scan, the
  full objective 0.889 against 0.820) and now compares the goals.

## Issue 16f — The module header described logit-only sites and an outer loop

Status: **fixed**

### Changes

- `BrisketEstimator.jl`'s header now describes the four families, the logit
  and score coordinates, the state block's coordinates, the local-quadratic
  prior including a filter's whitened full Gaussian, the callback-hosted
  iteration inside one `infer` call with its trust region, and the files.




## Leftover of issues 3 and 12 — The streaming record behind two gates was not regenerated

Status: **fixed and focused tests passing**

### Changes

- `data/mock/mock_1d_stream.json` (the hourly refits of the 1-D mock, made on
  2026-09-07) was left as it was when issues 3 and 12 regenerated
  `mock_1d.json` (the scale's readings and the probe-depth latent's units
  changed). `estimator/test/test_engine.jl` gates the graph engine against
  that record's posterior and failed by 13 widths on the first batch; the
  planner experiment (`plan_experiment_setup.jl`) reads the same record's
  six-hour posterior, translated from the old depth coordinate.
- The record is regenerated by `estimator/scripts/mock_stream.jl` on the
  current mock, then the planner reference `plan_experiment.json` by
  `plan_experiment.jl`, and both gates rerun. The record now carries the
  fraction latent directly, so the setup's translation branch is no longer
  taken.

### Verification

- `estimator/test/test_engine.jl` on the regenerated record: **9/9 passed**, the
  three batches at zero difference in means and widths and the objectives
  equal (33.8386, 42.3760, 55.1139); the record is now the graph engine's own
  run, so the gate is the regression that the solve, the channels and the
  mock have not moved, and its header says so. The streaming figure
  `stream_mock_1d.png` is redrawn from it (the prose of the 2026-09-08
  results entry quotes the old run and is left for the documentation pass
  after the bench).
- `estimator/test/test_iterated.jl` on the regenerated planner reference: **20/20
  passed** (the planner reproduces the expected-loss reference, the real-cook
  gate holds). With that, every one of the fifteen estimator test files
  passes on the committed tree.

## The bench on the remediated filter: the smoke round, and the bench itself pending a go

Status: **the driver, the worker, the summary and the figures verified end to end; the bench not run**

### What ran

- `bench_state_noise.py --smoke` on the committed tree (physics `e6d030f47a57`):
  one two-hour cook per round, one worker. The tuning round started from the
  stale rates file with `stale_ok` (the worker printed the warning and ran),
  the one covariance-matching step was applied, the fresh cook ran on the
  smoke's own rates without a warning, the summary JSON and both figures
  were written (`*_smoke` names, nothing quoted from them). 60 updates per
  cook, 600 s per cook: about 10 s per update at the start of a cook.
- Estimate for the bench proper (nine cooks of fourteen hours per round,
  about 400 updates each, two workers): 60 to 80 minutes per cook, five cooks
  per worker per round, about six hours per round, **about twelve hours for
  both rounds**; the trust region's extra rebuilds are part of that (about
  70 % more wall time per update than the uncapped solve on the first ninety
  minutes of a cook).
- The Sep 21 partial records of the score-coordinate run were deleted; the
  smoke's own records and figures are untracked scratch.

### Verification

- The tree is frozen for the bench: nothing in `src/` or the filter's hashed
  files changes until the bench has run. The command, from `app/`:
  `uv run python ../estimator/python/bench_state_noise.py --workers 2`.

## Immediate items 1 and 4 — Estimator suite in CI; PythonCall on Windows

Status: **done locally; the CI job is verified on the next push**

### Changes

- `estimator/test/runtests.jl` takes a list of files and prints each file's
  wall time; `.github/workflows/CI.yml` gains an `estimator-tests` job that
  runs the suite in four shards on every push and pull request.
- `app/tests/conftest.py`: when the Julia runtime behind the app cannot
  start, the first Julia-backed test fails with the cause and the repair and
  the rest are skipped with a one-line reason, instead of one setup error
  per test.

### Verification

- `estimator/test/runtests.jl test_noise.jl` runs that one file and reports its wall
  time; an unknown file name is refused with the list of files. The full runner
  on the committed tree: **14 of 15 files passed** (the fifteenth, `test_engine.jl`,
  is the leftover below), about 80 minutes serial on a loaded laptop, the
  longest file 14 minutes; the four CI shards are balanced on those times to
  about twenty minutes each (sequential with the five short files; segments
  with the smoother; the planner with the iterator and the noise; the engine
  with the ridge, the real cook and the 2-D physics). The job itself is verified
  on the next push.
- PythonCall on Windows, the cause: juliacall keeps a Julia project of its own
  (`app/.venv/julia_env`) and picked juliaup's `release` channel (1.12) while
  the package, the estimator and CI run 1.11; with the executable pinned to
  the `julia` on PATH its project still loaded `Preferences` at a version
  other than the repository manifest's, JSON then could not precompile inside
  the same session, and the run waited on precompile lock files left by the
  earlier attempts (the two runs that looked hung: a Julia child with seconds
  of CPU).
- The fix: on Windows both Tesseract APIs set `PYTHON_JULIAPKG_EXE` to the
  juliaup-resolved `julia.exe` and `PYTHON_JULIAPKG_PROJECT` to the repository
  project (PythonCall is a direct dependency of the package). juliapkg then
  manages that project: it added `OpenSSL_jll` (compat `~3.0`), pinned
  `PythonCall` to `=0.9.35`, appended an empty `[preferences]` section, and
  re-resolved `Manifest.toml`. The committed manifest could not be resolved
  as pinned (its `OrdinaryDiffEqDifferentiation` 3.7.0 and
  `OrdinaryDiffEqCore` 4.13.0 are no longer allowed together by the
  registry), so the re-resolution bumped about a dozen packages
  (`OrdinaryDiffEqCore` 7.28 to 7.30 among them) and added
  `LHLFactorization`; the estimator's own manifest is separate and unchanged.
  `lock.pid` and `pyjuliapkg/` at the root are juliapkg's and ignored.
- On that manifest: the bridge test passes (96 s), the root suite `Pkg.test()`
  **386/386 passed** (8 min 40 s), and the whole app suite on Windows
  **86 passed** in 6 min 21 s, against 32 passed, 2 failed and 42 setup errors
  at the audit. With Julia unable to start (the manifest out of step with the
  project for one run), the suite reported it as designed: 37 passed, 48
  skipped, one failure carrying the cause.
