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

## Roadmap item (ii) — The update's convergence: a trust region built, then retired for the posterior-linearization form

Status: **replaced; focused tests passing** (this session, 2026-09-22)

### What was built first and why it was retired

- A trust region on the graph's step (no coordinate beyond its own predicted
  width per iteration) was built as the named fix for the seed-12 lock. On
  the filter's own updates it bound on 27 of 45 routine updates: the graph
  proposed steps of 10 to 60 widths along one direction (the probe's depth
  fraction with the field nodes bracketing the probe), the exact objective
  rejected them with or without the cap, and the cap cost about 70 % more
  wall time for a belief within 0.15 sd of the uncapped one.
- The literature explains the proposals rather than the cap: the update
  placed its sigma points at the prior for the whole update, mixed the
  latents' regression slopes with a state tangent computed once and left
  stale, and took the graph's direction from the regression's expected
  reading while the line search judged the physics at the centre. That is
  the construction the posterior-linearisation paper (García-Fernández et
  al. 2015, Section IV-C) names as ad hoc, and the reason the damped IPLF
  (Raitoharju et al. 2018) descends one cost with the points fixed and
  re-places them between rounds. The region was removed in full.

### Changes

- The whitened filter solve is now the damped posterior-linearization
  update: the line search descends the regression's own cost
  (`slr_objective`: each channel's expected reading over the sigma points
  at the trial centre, the rows inflated by the last linearization's
  residual, plus the whitened prior), so the direction and the acceptance
  judge one cost; between rounds the points are placed again at the round's
  posterior covariance and the state's tangent is taken at its centre
  (`_place_points!`); the rounds stop when the round's posterior mean has
  moved less than `round_kl` nats from the belief the points were placed by
  (`_gauss_shift`: half the whitened squared shift of the mean, the mean term
  of the IPLF's KL criterion, the widths' change left out since the
  regression carries it exactly) or at `rounds` (defaults 5 nats and 3
  rounds). An accepted trial's points serve its linearization,
  so a first-trial acceptance costs no extra solves. The update record
  carries `iters` and `rounds`, and the bench worker writes them with
  `converged`.
- Fixed on the way: a solve restarted at a centre whose linearization was
  cached could find the points at a rejected trial's centre.

### Verification

- `estimator/test/test_sequential.jl`: **221/221 passed** on the new update; the
  filter's own first update (the initial readings assimilated, the prediction,
  the update at the first arrival) commits in 1 round and 11 iterations, and
  the shift rule is checked on two known Gaussians. The update testset ran in
  9 min 18 s on the placement rule (11 min 26 s under the KL criterion, 12 min
  57 s with the trust region).
- `estimator/test/test_iterated.jl`: **13/13 passed** (the planner and the
  real-cook gates, which use the batch solve, are untouched by the change).
- The rounds on the filter's first ninety minutes (45 updates, one round against
  up to three, before the placement rule): every update commits either way,
  the second round runs on nearly every update (mean 2.02), the belief at
  ninety minutes differs by at most 0.23 sd on a latent (k_m, k_e, m_conv), the
  widths by under 4 % except the probe's depth fraction, 19 % narrower with the
  points at the posterior; the iterations rise by an eighth (224 against
  199; the wall times, 596 s against 287 s, are confounded by the compilation
  the first run in the process carries, measured below). The
  innovations do not move: NIS/dof 0.35 against 0.36, the pit's lag-one −0.30
  against −0.31, the core's +0.47 against +0.52 with its mean −0.17 against
  −0.15. So the placement rule: the points are placed again only when the
  round's posterior has moved more than `round_kl` nats from the belief they
  were placed by, which leaves a routine update at one round.
- The placement rule on the same ninety minutes (`round_kl` 5 nats on the
  mean's whitened shift): with `rounds` at 3 every update stays at one round,
  the iterations are the one-round run's 199, and the belief at ninety
  minutes is the one-round belief to the last digit (largest difference 0.0
  widths, every sd ratio 1.0), so a routine update costs what it did before
  the rounds existed. The same pair of runs measures the compilation: for
  identical work the first run in the process took 483 s and the second
  244 s, so the second round's true cost in the run above was about an
  eighth more, matching its iterations, not double.
- What the innovations now say, and the update cannot change: the pit's
  negative lag-one is over-correction (its earned rate, 0.42 K per step, is
  as large as the probe's own innovation), the core's positive lag-one with a
  persistent negative mean is under-correction on the core's side (its offset
  re-explained through the field, which relaxes by diffusion between updates,
  rather than through the discrepancy state). Both are the rates' structure,
  invisible to a covariance-matching step against the truth and exactly what
  the innovation autocovariances identify (Odelson et al. 2006; Zhang et al.
  2020, Section IV). The bench's step 2 is therefore the next decision, and
  it is put to the user rather than built.

### The noise check that the update's restructure was ordered behind

The smoke cook (seed 10, two hours, the smoke's rates) rerun with the filter
told the mock's own noise instead of the calibration's prior, and once more
on a constant plant:

| configuration | NIS/dof | core mean square | core lag-one (band 0.26) | core mean |
|---|---|---|---|---|
| A. calibration prior (the bench's) | 0.40 | 0.29 | +0.38 | −0.31 |
| B. the mock's own noise, no floors | 0.47 | 0.42 | +0.31 | −0.29 |
| C. as B, constant plant | 0.47 | 0.42 | +0.30 | −0.28 |

The prior's sd is a minor part of the low NIS and none of the core's
autocorrelation; the drift is none of it either. By arithmetic the rest of
the pit's overstated innovation variance is its earned process-noise rate
(2.28 K per root-hour, 0.42 K per two-minute step, as large as the probe's
own innovation): the covariance-matching step turned a persistent,
latent-driven pit error into white per-step noise, the failure mode Zhang et
al. (2020) record for covariance matching. The core's biased, correlated
innovations are the residual the update left unabsorbed. The bench's noise
configuration therefore stays; what the tuning round must be read for is
whether the posterior-placed update absorbs that residual and how the
matching step treats the pit.

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

## The bench's step 2 — The rates identified from the innovations' autocovariances, in place of covariance matching against the truth

Status: **built, tested and smoke-run; the bench itself pending a go** (this session, 2026-09-23)

### Why the step was replaced

- The bench earned the state's process-noise rates by one covariance-matching
  step against the plant's state: each group's rate scaled by the square root
  of its NEES per degree of freedom. On the remediated filter the smoke cook
  showed what that step cannot see (the entry above): the pit's earned rate,
  0.42 K per two-minute step, was as large as the probe's own innovation, the
  innovations were smaller than the filter expected (NIS/dof 0.4) and the
  pit's were negatively autocorrelated (over-correction) while the core's
  stayed positively autocorrelated with a persistent negative mean
  (under-correction). NEES against the truth measures the belief's width;
  it does not say whether a width came from a persistent, latent-driven error
  or from per-step noise, and the step turned the former into the latter.
  The innovations' autocovariances at several lags tell the two apart: that
  is what the autocovariance least-squares method identifies from (Odelson,
  Rajamani, Rawlings 2006; in the fixed-gain, time-varying form, Åkesson et
  al. 2008; Zhang et al. 2020, section IV). The step needs no truth, so it is
  the same step on a real cook's innovations.

### The mechanism, in the filter's own pieces

- The filter's update at an arrival is, to first order, a Kalman update with
  the affine rows H at the predicted mean (one per reading, whitened by its
  noise; the same rows the diagnostics whiten the innovations with) and the
  gain K = P⁻Hᵀ S⁻¹ on the predicted innovation covariance S = H P⁻ Hᵀ + I.
  The prediction between arrivals is a transition F with the process noise
  Q_known + Σ_g θ_g Q̇_g, θ_g the square of group g's rate and Q̇_g the group's
  covariance per unit of it in the state's coordinate at the interval's end.
  The sigma-point prediction is exactly such a transition: the identity on
  the latents, the points' regression slope less the conditional coupling on
  the latents' columns of the state and the discrepancies, the tangent on the
  state's own columns; the predicted covariance is F P Fᵀ plus the
  regression's residual, a covariance (`_sigma_prediction` now returns F).
- Under its own model the filter's innovations are white with covariance S:
  at lag zero the predicted covariance, beyond it nothing (the Kalman-gain
  identity F[(I − KH)P⁻Hᵀ − K R] = 0). Under the rates θ + δθ, with the gains
  the filter actually used, the prediction error carries an extra covariance
  Π_g per unit δθ_g, propagated as Π_g ← F Π_g Fᵀ + Q̇_g at a prediction and
  Π_g ← (I − KH) Π_g (I − KH)ᵀ at an update, and the innovation autocovariance
  between the update at t and an earlier one at t′ is [t = t′] S + Σ_g δθ_g
  H(t) Φ(t, t′) Π_g(t′) H(t′)ᵀ with Φ the product of the closed-loop
  transitions between them. In units standardized by the filter's predicted
  innovation sds, the products of the innovations at every pair of updates
  inside the lag window, less the base, are a linear regression on the twelve
  δθ_g. The accumulator (`estimator/src/als.jl`) carries its sufficient
  statistics along the run: the normal equations, and per channel pair and
  time lag the sample products, the base and each group's coefficient. The
  driver pools them over the cooks and solves for δθ ≥ −θ.

### Changes

- `estimator/src/als.jl`: `ALSAccumulator`, `als_update!` (the twelve
  sensitivity recursions, the pending updates inside the window with their
  propagated Π Hᵀ, the rows' statistics; innovation replicates as columns
  for the tests' simulation), `als_summary` (what the worker writes per cook).
- `estimator/src/sequential.jl`: `_coord_scales` shared by the coordinate map
  and the sensitivity; `_sigma_prediction` returns the equivalent transition;
  `_state_process_sens` (per group the derivative of the interval's mapped
  process covariance with respect to its variance rate, nothing where the
  coordinate cap holds an entry, half where the cap holds one side of a
  covariance); the `als` slot on `FilterState` (`filter_init(...; als)`), fed
  at the end of every update with the interval's transition (the
  discrepancies' AR factors preceding the physics), the sensitivities, the
  standardized rows, S, the gain on the standardized innovations (zero when
  the predicted belief was kept) and the innovations. The belief is the same
  with or without it.
- `estimator/scripts/coverage_filter.jl`: a tenth argument, the accumulator's
  lag window in hours (2 by default, 0 without); the accumulator per cook and
  its summary under `als` in the record.
- `estimator/python/als_fit.py`: `pool`, `als_fit`, `lag_table`, the
  printing. The rows are weighted by their own residual variance per channel
  pair and kind (the squares at one update, the products across updates; the
  accumulator keeps each block's normal equations for it), the method's
  weighting made feasible from the rows themselves: the weights start at one,
  each block's residual variance at the unbounded fit re-weights it, and the
  fit is repeated until the weights hold. The first smoke showed why: the
  core probe's innovations are tiny against their predicted spread, and its
  products were a twelfth as scattered as the pit's; at unit weight the pooled
  rows' noise read 0.10 and the core was drowned. The rule: the rows' noise is
  the residual of the unbounded fit over every group with a column (so a large
  change is not mistaken for noise); a group enters the fit when its own
  column pins its variance rate
  to a relative standard error under a half, or when the unbounded fit moves
  it by more than two standard errors (a rate small today that the
  innovations want larger); the fit is bounded to [θ/cap², θ·cap²] on the
  variance rate (the same cap of four on the rate as before) and to θ ≥ 0;
  after it, a group whose joint standard error exceeds its variance rate and
  which the unbounded fit did not move is held at its number and the fit is
  solved again. A group the innovations cannot see keeps its numbers and is
  reported, as a group without a finite NEES did before. The initial widths
  are kept: their transient is one, the rates recur. Non-finite numbers are
  written as null, since the Julia side parses the rates file strictly.
- `estimator/python/bench_state_noise.py`: `identify` in place of `match`;
  `--lag-h`; the fit's whole record under `bench.tune.als` in the rates file;
  the fit table and the lag table printed; the figure rendered after the fit.
- `estimator/python/plot_als_fit.py`: per channel and lag the sample
  autocovariance (± its standard error) against the model at the round's
  rates and at the identified rates (`figures/estimator/als_autocovariance.png`;
  `_smoke` from the smoke). Its palette is the repository's blue and orange
  with dark-ink markers; the sample is markers with error bars and the models
  are lines, so identity is not colour alone (the palette validator could not
  run here: no node on this machine).
- `estimator/test/test_als.jl` (the planner CI shard); `app/tests/test_bench_als.py`;
  `docs/results.md` and `figures/README.md` say what the step is now.
- After the review (2026-09-24): `estimator/python/bench_manifest.py` and the
  manifests, the driver's whole-block cook count, held-out fresh seeds and
  validated reuse, the jackknife in `als_fit.py`, the cook-cluster interval
  in the driver's printout, the renderers on the manifest;
  `app/tests/test_bench_manifest.py`.

### Verification

- `estimator/test/test_als.jl`: **21/21 passed** (9 min). The transition
  reproduces the sigma prediction's cross block to 1e-8 and leaves a
  positive semidefinite residual on the state block; the per-group
  sensitivities sum to the interval's mapped process covariance on every
  entry the cap does not hold, and have no latent entries. The accumulator
  against a simulation of its own error dynamics on the filter's first thirty
  minutes (the run's transitions, rows and gains; 400 replicates at once): a
  planted change of the pit's variance rate of **+8.0 was fitted as +7.73 ±
  0.21**, the temperature field (relative standard error 0.30) was identified
  and fitted within three standard errors of its planted zero, the fit's
  cost fell; the belief with the accumulator is the belief without it to
  1e-10; the summary survives JSON, its pairs count the rows and its blocks'
  normal equations sum to the whole.
- `app/tests/test_bench_als.py`: **3 passed**. A planted change of two
  identified groups over three pooled cooks, with one block's rows three
  times as scattered as the other's, is recovered within its standard error
  while the nine unseen groups keep their numbers, the weighting finds the
  variance ratio of nine and the error bars come from the products' own
  spread; the cap binds; a rate small today that the innovations want larger
  enters and moves to the cap while a sample uncorrelated with its column is
  held; cooks that ran on different rates, and no cooks, are refused.
- The driver's smoke (one two-hour cook per round: seed 1 tuned from the
  stale rates file, seed 10 fresh; nothing quoted from it), run twice, at unit
  weight and with the blocks' weighting. The tuning cook's innovations against
  the filter's model: at lag zero the pit's mean square **0.43 ± 0.08 against
  1.00**, the core's **0.12 ± 0.02 against 1.00**, and the pit's positive
  autocorrelation out to twenty minutes (0.17 ± 0.06 at one step). At unit
  weight the pooled rows' noise read 0.10, because the core's products were a
  twelfth as scattered as the pit's, and the fit sent both the pit's rate and
  the temperature field's to the cap. With the weighting the rows' noise is
  1.00 and the fit identifies one group: the pit's rate, to the cap (2.54 to
  0.64 K per root-hour, eight standard errors), which brings the pit's
  lag-zero model to 0.54; the field is held (its column alone pins its rate to
  ±55 % and the fit moves it one standard error): the unweighted move had
  been the pit's rows speaking for the core's. The other ten groups are
  unidentified on a two-hour cook. On the fresh cook, with the one identified
  rate, the innovations came in whiter: NIS/dof 0.31 to 0.49, the pit's
  lag-one +0.30 to +0.10 and the core's +0.40 to +0.20 (both inside the band
  of 0.26), the temperatures' NEES/3 0.48 to 0.45 and the pit's NEES/dof 0.64
  to 0.53 (one cook each, on different seeds: a direction, not a number).
- What no rate can explain is in the figure
  (`figures/estimator/als_autocovariance_smoke.png`): the core's innovation
  variance is an eighth of the model's at lag zero, before and after (the
  field's rate at its cap would move the model by under a percent), and the
  humidity's about six tenths. That gap is not the rates'. It is the channel's noise model:
  the meat probe's white row floor is 0.1 K (`white_floor`), and the 1.2 K
  is the floor on the AR discrepancy's sigma, whose calibration prior is
  1.65 K against the mock's 1.2 K with a similar correlation time; the
  discrepancy's transition variance is the largest term of the core's
  predicted innovation variance, and the belief's own predicted spread of
  the core temperature is the rest (the noise check above: telling the
  filter the mock's own sigma and rho, without floors, moved the core's mean
  square from 0.29 to 0.42 on seed 10). A first draft of this entry called
  the 1.2 K a white floor; it is not. The bench proper measures the gap over
  the cooks; it is a finding to read, not to tune away, and the noise model
  is the user's decision.

### The bench (2026-09-23, on the go; the driver's own numbers, ten cooks per round)

- The tuning round (seeds 1 to 10, 4080 updates, 4 h 35 min on two workers)
  from the stale rates file: NIS/dof 0.50, the latents' NEES/16 1.42, the
  temperatures' NEES/3 1.33; the temperature field (1.39) and the pit (1.56)
  too narrow against the truth, the water (0.39) too wide; the core's lag-one
  +0.19 against a band of 0.10, the pit's +0.05.
- The fit, on 1.12 million products with the weighted rows' noise at 1.00:
  three groups identified and all three sent to the cap, the temperature
  field, the water and the pit each to a quarter of their rate (63, 5 and 39
  standard errors); the other nine unidentified. At lag zero the pit's sample
  0.50 ± 0.01 against the model 1.00 before and **0.52 after**: the pit's gap
  is the rate's and closes. The core's 0.34 ± 0.01 against 1.00 before and
  **0.90 after**: the field's rate at its cap moves the model a tenth, the
  gap is not a rate's. The humidity's 1.25 ± 0.08 against 1.00, which no rate
  addresses, and the scale's 0.78 ± 0.07.
  (`figures/estimator/als_autocovariance.png`.)
- The fresh round (seeds 10 to 19, 4080 updates, 4 h 9 min) on the
  identified rates: **NIS/dof 0.74** (from 0.50), the pit's lag-one +0.02,
  the humidity's +0.06 (from +0.18), the core's +0.18 (from +0.19, still out
  of band: the core's gap is not a rate's); the temperatures' NEES/3 1.40;
  the temperature field 1.26 and the pit 1.50 still too narrow, the water
  0.54 still too wide, the ignited fraction 2.00 with one cook at 10. The
  latents' NEES/16 **1.78 is undercoverage**, not the band's edge: the
  chi-squared band for the mean of ten cooks is about 0.79 to 1.23 (0.43 to
  1.80 is one cook's), and the cook-cluster 95 % interval of the mean from
  the ten per-cook values (2.65, 2.40, 2.72, 2.04, 0.94, 2.64, 1.02, 0.83,
  1.16, 1.35) is 1.22 to 2.34, excluding one. The rates fell by a factor
  four and the state's widths against the truth did not worsen (different
  seeds; a direction, not a pair). The innovations are whiter and closer to
  their predicted size; what remains is on the channels' side and on the
  latents', not the rates'.
- Three qualifications from the review of this run (2026-09-24), each
  verified in the code and the records. (1) The fresh round was not fully
  held out: the driver rounded nine cooks up to two blocks of five, ran seeds
  1 to 10 in the tuning round and started the fresh seeds at 10, so seed 10
  is in both while the file said nine cooks per round. (2) All three
  identified rates sit on the cap's lower bound: they are cap-limited
  candidates, not interior estimates, and the model's standard errors treat
  1.1 million overlapping products as independent when a cook's products
  share its plant and its belief. (3) The noise terminology: the white row
  floor is 0.1 K and the 1.2 K is the AR discrepancy's sigma floor (above);
  the core's lag-one stays +0.18, so fitting a white measurement variance
  alone would not resolve the channel's mismatch: the AR sigma and rho are
  what disagrees with the mock, and a joint fit of the channel noise (white
  and AR, prior-regularized) with the state noise, validated on held-out
  cooks, is the design item, not dropping physically justified floors.
- The protocol flaw, and its fix as the review asked (a manifest, not
  deletion plus a glob): the consistency and coverage figures (and their
  summary JSONs) were rendered from a glob on the round's stem, and the
  record names carry the seed block, which changed with the worker count.
  The old fresh records `coverage_filter_13.json` and `coverage_filter_16.json`
  (2026-09-17, physics `ba928b214b27`, the old rates) matched the glob, so
  those two figures pooled sixteen cooks of two physics and two rate sets and
  are wrong; the driver's printed numbers, the rates file's record and the
  fit's figure read the round's files and are right. Now each round's records
  are listed in a manifest (`data/mock/bench_manifest_<round>.json`:
  `estimator/python/bench_manifest.py`) with their seed blocks, the cook
  count, the physics and the state noise they ran with; every renderer
  (`plot_filter_consistency.py`, `plot_coverage.py`, `plot_als_fit.py`)
  takes the manifest and refuses a record whose seeds, physics or rates are
  not the manifest's; `--reuse` accepts existing records only when they
  validate the same way. The cook count is whole blocks, the fresh seeds
  start where the tuning seeds end unless given, and overlapping seed sets
  are refused. The fit quotes a leave-one-cook-out jackknife standard error
  beside the model's, with the count of leave-one-out fits in which each
  group was identified and at the cap; the driver prints the cook-cluster
  interval of the latents' and the temperatures' NEES beside the band.
  Tests: `test_bench_manifest.py` (the round's records validate, a record
  of another bench is refused with the reason, the manifest round-trips and
  a changed record is refused at load) and the jackknife in
  `test_bench_als.py` (a borderline group identified on the pool but not by
  every two of three cooks is what it flags).

### What remains

- The corrected check: the fresh round on seeds 11 to 20, held out from the
  tuning round's 1 to 10, rendered from its manifest. The quarter-rate result
  stands as a strong directional finding; the calibration (the rates file,
  the records, the figures) is committed only once that check has run.
- The channel noise in the identification: white and AR jointly with the
  state noise, prior-regularized, validated on held-out cooks; a design item
  for the user.
- The latents' undercoverage (1.78 over ten cooks), to be read first on the
  corrected coverage figure: a latent inconsistency the rates do not govern.

## The bench on the remediated filter: the smoke round, and the bench itself pending a go

Status: **the driver, the worker, the summary and the figures verified end to end; the bench not run**
(its step 2 has since been replaced: the entry above)

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
  both rounds** on the update as it was in the smoke round; the
  posterior-linearization form's cost is measured below.
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
