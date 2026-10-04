# Estimator consistency after merging the from-lighting model

Work started 2026-10-04 on `estimator-consistency` at `96553a3`, merging
`Photo-segementation` at `7aace2c`. The user authorized estimator changes
and overnight validation. Model and observation-graph defects were found
and corrected, but the merged estimator has **not established statistical
consistency**. Full-cook validation is still blocked by numerical convergence
and runtime failures. Tuning runs are not held-out evidence.

The requested source exists locally at `7aace2c`. A remote fetch on
2026-10-04 returned `Repository not found` for the configured GitHub
remote; this integration therefore uses that local source commit.

| Gate | Recorded outcome |
|---|---|
| Root physics | 703/703 pass |
| Factor solver | 259/259 pass |
| Probability-model consistency | 104/104 pass |
| Preheat / slice observations / capped-slice regressions | 25/25, 18/18, 17/17 pass |
| Online estimator | 165/165 pass |
| Multi-piece assimilation and event handling | 126/126 pass |
| Multi-piece offline check, with explicit integration accuracy | 3/3 pass |
| Original offline hierarchy, with explicit integration accuracy | 22/22 pass |
| Heterogeneous offline hierarchy, with explicit integration accuracy | 18/18 pass; expensive accuracy setting |
| Planner | Single piece 19/19 pass; multi-piece convergence failure |
| Streaming scorer | 21/21 pass |
| Held-out complete-cook statistical consistency | Not established; no held-out round launched |

These are separate targeted runs, with their timing and source-change order
recorded below. They are not a claim that the entire repository test suite passed.

## Integration and diagnostics

- The propagation conflict combines observation-time saves from the
  streaming branch with latent-dependent ROM exchange corrections from
  the photo branch. Streaming observation corrections now also evaluate
  at the current latents and carry their derivatives.
- The estimator manifest lacked the physics package's new dependencies.
  Resolving it adds SparseArrays and OpenSSL_jll to that package's entry
  and aligns OpenSSL_jll with the root project's 3.0 compatibility bound.
- The legacy streaming fixture starts the pit at 362.43 K with meat already
  loaded and uses the old ignition prior. A separate `STREAM_START=lighting`
  design starts at ambient, loads at the refit's documented 90 minutes,
  and shifts the subsequent controls and events consistently.
- The lighting validation prior uses independent marginal distributions
  centered on the from-lighting MAP, with its Fisher widths. Bounded
  fractions use beta distributions matching the local delta-method width.
  Initial ignition has a declared beta prior with mean 0.15 and sd 0.05;
  this spread was **not inferred** by the refit, which fixed a0. The old
  production prior and calibration chain are preserved.
- `STREAM_NOISE=matched` matches the core interpolation and stationary
  OU plus white instrument noise. The default legacy stress case keeps
  the linear probe interpolation and the post-add pit-noise bump. Neither
  deterministic plant is a draw from the estimator's stochastic process
  model; the process-rate scale is recorded explicitly.
- Segmented handoff now preserves physical ignition endpoints instead of
  pushing a fully lit bed back to `1 - 1e-6`. System solves use the same
  load-transition integration stops as the monolith.
- The monolith's vapor closure included the entire meat source during
  partial loading, while the system included only the loaded fraction.
  The monolith now gates the source inside the closure as well as the
  final flux. Six RHS comparisons across the load cover this regression.
- Asynchronous slices resampled the damper onto a new grid, smoothing
  over original control corners. `damper_knots` now returns the original
  covering knots; integration still starts and ends at the requested
  times. This removes a 0.085 K pit discrepancy at the loading boundary
  in the full-resolution seed-41002 diagnostic. The final 4 h diagnostic
  has every state group's whitened mean-square residual below 0.5 at
  default process rates, and matched sensor predictions agree to rounding
  (`scratchpad/estimator_preheat/truth_exact_controls.log`).
- Batched innovations now retain every reading, whitened in chronological
  order against the joint predicted covariance. Keeping only each
  channel's last marginally standardized residual could not test
  innovation whiteness. Their squared norm remains the same NIS.
- Capped readings on the slice path inserted a state node without closing
  the preceding observation slice. Delayed capped readings also left
  later readings attached to the wrong anchor. Node-backed readings now
  split/rebuild the process and OU bridges, preserving the shared end
  state and the dynamics/slice solve cache. A focused regression failed
  9 of 12 checks before this fix and passes all 12 afterward.
- A capped observation in a mixed batch was included in the covariance
  used to score that batch's uncapped readings. The slice prediction now
  excludes all newly arriving observation factors, matching the node
  path. Five regression checks compare with explicitly removing the
  capped factor, verify restoration of factor order, and demonstrate that
  including it would change the forecast uncertainty.
- The stopping test divided each score coordinate by its own curvature,
  ignoring correlations. For `J = [1e6 -1e6; 1 1]` and data `[0, 2]`, it
  certified `[0, 0]` although the exact mean is `[1, 1]`, two posterior
  standard deviations away. The check now solves for the full undamped
  Gauss--Newton displacement. Its public diagnostic field is still named
  `gradient`; it measures a coupled coordinate displacement. Regression
  checks cover the solution, residual scaling and sparse QR.
- A failed line search could keep using a stale Jacobian for the entire
  iteration budget, only refreshing it at the final certificate. An
  analytic nonlinear regression reached its optimum in one step, then
  spent nine iterations rejecting the same stale direction. Failed
  searches now invalidate stale Jacobians immediately. The same example
  finishes in three iterations with unchanged tolerances. Trial residual
  evaluations also preserve a Jacobian already made at the restored
  accepted point; the primal and derivative cache points are distinct.
- The online solver can request its configured restart early after three
  failed searches if a freshly computed joint step already lies inside
  the existing posterior-width resolution envelope. That request returns
  `converged=false`; only the final attempt may accept under the existing
  relaxed resolution rule. This avoids spending the first attempt's
  entire budget damping an unchanged, numerically stalled point.
- `SystemModel` now separates actual integration accuracy from its
  process-covariance error allowance through `floor_abstol` and
  `floor_reltol`. Defaults preserve the previous relationship. Holding
  those fields fixed allows tighter solves without silently reducing
  state uncertainty. Execution updates preserve both settings; benchmark
  records and pooling checks include them.
- The bench saves its starting source/backend fingerprints, complete
  declared priors, failure details and actual scoring times. Short runs
  stop the plant at the requested horizon. Both plant and estimator use
  1e-10 absolute / 1e-8 relative integration tolerances.

The legacy full-resolution truth diagnostic over 14 h showed a maximum
listed temperature-field dynamics residual of 1.6 times its process
variance per coordinate at 8.5 h, and sub-0.01 K physical pit discrepancies.
Its core-observation discrepancy is about 0.0045 K early in the cook
because the two probe interpolations differ. This is diagnostic evidence,
not statistical consistency. The log is
`scratchpad/estimator_preheat/truth_merged_14h.log`.

## Validation record

Commands use Julia 1.11.9 and one OpenBLAS thread. Direct benchmark runs
use four Julia threads unless their command overrides it. The test shard
driver originally dropped the parent's `--threads=4`, leaving its children
at Julia's default; it now forwards the requested pool explicitly. The
already running shards continue on their original pool. Their timings
also include competition with tuning jobs on this 16 GB machine.
`scratchpad/estimator_preheat/run.py` captures Julia's actual exit
code and combined log; PowerShell's native-stderr formatting otherwise
reports an error merely when `Pkg.activate` writes its normal message.

- Initial merge checks: factors 248/248, consistency 90/90 and slices
  11/11 passed. ROM passed 60/60 after the physics fixes were stable.
  New controls and innovation diagnostics require another affected-suite
  run, which is in progress.
- The root physics suite passed 703/703 after the vapor-closure fix.
- Fixture/preheat/handoff checks passed 19/19. Six new checks for
  preservation of control corners also passed (25 total). Updated
  consistency checks passed 94/94, including chronological whitening.
- The first preheat assertion sampled only 2.5 logistic widths before
  loading, where the model intentionally blends the load. It now samples
  20 widths before loading, consistent with the existing physics tests.
- Seed 41001's initial 3 h smoke used fixed initial conditions and long
  slices (four updates). It is historical tuning evidence only. Subsequent
  lighting runs draw initial coordinates from the estimator's declared
  initial-state prior and truth latents directly in their prior coordinates.
- With process rates scaled by 0.001, the node threshold is scaled by
  its square: `node_ratio=1.5e-7` corresponds to 0.15 at default rates.
  This retains roughly five-minute slices instead of allowing many hours
  of observations to accumulate in one dense factor. Two-hour scoring
  checkpoints also stop batches before the edge. Neither choice changes
  the production defaults or the frozen comparison backends.
- A full 18 h, 40-node truth diagnostic with sampled initial conditions
  passed through all events. At process-rate scale 0.001, the largest
  per-group whitened mean-square residual was 0.0229 (pit temperature).
  Maximum physical errors were 0.000240 K in the pit and 0.0000474 K in
  the meat field; sensor prediction errors were at floating-point rounding.
  The complete record is `data/mock/truth_lighting_41002.json`.
- The optional `max_slice_s` age cap passed four placement checks. Its
  default is `Inf`; a benchmark can use `STREAM_MAX_SLICE_S` to bound
  dense factors while keeping longer nearly deterministic propagation
  slices. This option's statistical and runtime assessment is pending.
- The planning regression's single-slab checks passed 19/19. The two-piece
  replan failed to reach stationarity within its test's 30-iteration budget.
  This remains under investigation; it is not counted as a passing suite.
- Six analytic checks of the scorer's chi-square and Student-t bands and
  independent-cook handling passed. Tuning records are not validation.
- The coupled stopping test passed all 253 factor checks, including the
  Kalman/RTS oracle and marginalization tests (`factors_coupled.log`).
- With the cache-retry fix, factors passed 255/255 and probability-model
  consistency checks passed 96/96 (`cache_factor_consistency.log`).
  The cache tests check that restoring an accepted mean reuses its exact
  Jacobian even when a trial evaluation moved the primal cache.
- Early-restart safeguards passed with all 259 factor checks
  (`restart_factor_tests.log`), including refusal to label a requested
  restart as convergence and refusal outside its posterior-step bound.
- The updated slice-observation suite passed all 18 checks on four Julia
  threads in 496.5 s (`cache_slice_tests.log`), including age-cap placement,
  cached derivatives, node/slice equivalence and conditioned innovations.
- Online regressions passed 165/165 in 3379 s on two Julia threads,
  including transaction rollback, delayed observations, full-history
  comparison and the frozen RxInfer tangent oracle. This run preceded
  the capped-slice fix; the latter's 17 targeted checks passed separately
  in 204.8 s (`censored_slice_after.log`).
- After adding independent covariance-floor controls, probability-model
  consistency passed 104/104 (including eight new covariance/constructor/
  execution checks), preheat passed 25/25, and scoring passed 21/21.
  The original offline hierarchy passed 22/22 at its explicitly tighter
  integration accuracy. The heterogeneous hierarchy initially passed 17
  assertions but failed convergence at 0.0000483 against `tol=2e-5`, so
  that combined run failed (`floor_accuracy_regressions.log`). A subsequent
  run at 1e-15/1e-13 accuracy passed all 18 assertions in eight iterations,
  with final displacement 0.00000771. It cost 12 min 43.6 s within the
  testset, 808.9 s including startup (`semantic_accuracy_probe.log`). The
  final fixture uses that accuracy only for the heterogeneous models;
  production defaults remain unchanged. The two offline testsets passed
  in separate runs, not one final combined invocation.
- The production benchmark exercised the new controls successfully on a
  three-minute lighting prefix: one committed update, four iterations,
  40.2 s for the update, 139.6 s including startup and plant construction.
  `stream_lighting_floor_api_smoke_41002.json` records both sets of
  tolerances. This is an API smoke test on a tuning seed, not a cook-length
  or statistical-consistency result.
- Multi-piece online checks passed: 12 layout, 73 assimilation, 36 piece
  permutation and 5 complete-event-cook checks. Maximum permutation
  differences were 0.000224 K and 0.000792 latent-coordinate units.
  Its separate sparse offline check failed convergence at 15 iterations;
  the two unobserved-group assertions passed. That failure is being
  diagnosed, not counted as a passing file.
- Four concurrent short-slice tuning workers loaded the old stopping
  test. They were stopped after preserving their logs when memory
  contention prevented the corrected-solver diagnostic from starting.
  They are **incomplete, superseded tuning runs**, not completed cooks
  or statistical evidence. The process commands and reason are recorded
  in `scratchpad/estimator_preheat/superseded_tuning.json`. Seed 41001's
  full-horizon worker had already rejected its two-hour arrival.
  Subsequent heavy diagnostics run serially alongside the regression
  shard, with memory headroom checked before starting.
- New lighting runs stop and save diagnostics at the first uncommitted
  arrival (`STREAM_FAIL_FAST=true`). Such a run already fails the
  complete-cook criterion; continuing it cannot recover that criterion.
  The scorer records even a round with zero completed cooks as a failure.
- The first bounded-slice tuning run (seed 41002, 3 h requested,
  `max_slice_s=3600`, `node_ratio=0.00015`, lag 2, rate scale 0.001)
  used the coupled certificate before the cache/restart fixes. It
  committed through 2.39 h, then rejected the 2.43 h update: remaining
  coordinate step 0.0206 and posterior step 0.0782 sd. Its two-hour
  update took 595 s and 53 iterations; this configuration had not met
  the runtime gate. `data/mock/stream_lighting_tune_capped_41002.json`
  retains the failed trajectory; it contains zero completed cooks.

The source branch's fingerprinted ROM and process-noise artifacts are
not re-earned by these full-order checks. The physics changes invalidate
their old fingerprints; strict artifact checks remain enabled. The
lighting fixture's declared validation prior is not a newly calibrated
production prior.

## Remaining work

The updated cache/restart path still needed 50 iterations and 610 s at
the two-hour loading checkpoint of seed 41002 (the earlier path needed
53 iterations and 595 s on this contended machine). It did not remove
that bottleneck. It subsequently rejected the 2.39472 h update, with
coordinate step 0.002263 and posterior step 0.013518 sd, in
`data/mock/stream_lighting_tune_cache_41002.json` (zero completed cooks).
A same-seed diagnostic used 1e-12 / 1e-10 integration tolerances: the
1.496 h update took 21 iterations and 158.4 s, against 5 iterations and
54.1 s at baseline tolerances. It was stopped after missing the declared
120-second runtime gate; only three loading iterations ran, so this is
not a completed cook. These tolerances also enter the existing integrator
covariance floor, so this is a numerical-model sensitivity check, not a
comparison with an identical likelihood. A separate scratch experiment tests a
nonlinear dynamics retraction and labels its backend as experimental;
it is not production code or held-out validation.

With the capped-slice correction, the experimental retraction/accuracy run
committed seven updates through 2.4975 h, including three capped humidity
readings. The formerly failing 2.43056 h update committed with a remaining
joint step of 0.000411 posterior sd. It still took 241.9 s; the 2.39472 h
and 2.4975 h updates took 340.2 s and 222.1 s. The run was stopped after
1326.1 s because it repeatedly missed the 120-second update gate.
`stream_lighting_censor_fixed_41002.json` remains an **incomplete active
cook**, with zero completed cooks; the scorer refuses to certify it.
The retraction remains a scratch experiment. Production solver settings
and acceptance thresholds were not relaxed to adopt it.

The first retraction experiment reduced the two-hour update from 50 to
16 iterations (174.2 s on two Julia threads, versus 610.1 s on four in
the cache baseline; these runs shared the machine with regressions).
It committed the 2.39472 h update in 19 iterations / 241.6 s, but rejected
2.43056 h after 37 iterations / 397.1 s: its posterior step was 0.013493 sd,
outside the existing 0.01 bound. It therefore remains experimental.
`stream_lighting_retraction_probe_41002.json` contains zero completed cooks.
The next scratch diagnostic holds the original integration-error covariance
allowance fixed while tightening the actual solves, to separate numerical
accuracy from changing the factor weights.

The multi-piece planning regression also failed a separate diagnostic
using its production budget of 50 iterations per start. Neither start
reached the new coupled-displacement certificate. The failed result is
preserved in `scratchpad/estimator_preheat/plan_default_budget.log`;
the budget has not been raised and the stopping gate has not been relaxed.
The 30-iteration failure occurred before the coupled convergence check was
added (the first solver probe was at 03:59, after that planning run ended).
An isolated no-adaptive-damping trial also failed at the production budget
of 50 iterations per start (`plan_no_adaptive_probe.log`, 485.9 s).
A Powell-damped BFGS trial also failed both starts at that budget
(`plan_bfgs_probe.log`, 403.1 s): final coupled displacements were 0.842
and 0.830. Neither experimental planning algorithm is in production.
Tightening the planner's integration to 1e-12/1e-10 also failed both
starts at the original test budget of 30 (displacements 0.142 and 0.0849;
`plan_accuracy_probe.log`, 385.2 s). That change was not adopted either.

The offline regressions also expose unresolved stopping failures. The
two-cook hierarchy and the standalone cook failed their convergence
assertions; the other 20 assertions in that testset passed. The two-cook
solve stopped with coupled displacement 0.000683 against `tol=1e-5`.
The multi-piece offline test passed its two state-group assertions but
failed convergence at 0.001601 against `tol=1e-4`. Normalizing humidity
discrepancy coordinates by their sensor sd left that result unchanged:
the largest remaining step belongs to a physical state coordinate.
Tightening integration from 1e-10/1e-8 to 1e-12/1e-10 while keeping
the original covariance allowance reduced the two-cook displacement to
0.0000918, but both convergence assertions still failed. This is an
accuracy diagnostic, not a production tolerance change or a relaxed gate.
The same accuracy change passed all three multi-piece offline assertions
in its original 15-iteration budget (final displacement 0.0000984,
`multi_precision_probe.log`, 191.5 s including startup). This isolates a
numerical-accuracy contribution to that failure; it does not establish
calibration or justify changing every model's default tolerances.
The actual API subsequently reproduced the multi-piece offline pass,
3/3 checks in 116.9 s including startup (`multi_floor_accuracy.log`).
At 1e-14/1e-12 integration accuracy, still holding the allowance at
1e-10/1e-8, the hierarchy converged in nine iterations at 0.00000982;
all 22 legacy offline assertions passed (`offline_precision_tight_probe.log`,
216.2 s including startup). The offline fixtures now explicitly request
the accuracy their existing convergence thresholds need; neither their
iteration budgets nor their thresholds nor covariance allowances changed.
The production API for this choice is:

```julia
SystemModel(sys, log, blk; abstol=1e-14, reltol=1e-12,
            floor_abstol=1e-10, floor_reltol=1e-8)
```

For streaming diagnostics the corresponding controls are
`STREAM_ABSTOL`, `STREAM_RELTOL`, `STREAM_FLOOR_ABSTOL`, and
`STREAM_FLOOR_RELTOL`. This does not make the remaining streaming
convergence or calibration gates pass by construction.

The scorer now accepts the benchmark's nonfinite diagnostic encoding and
writes an explicit failure when there are zero completed cooks. Twenty-one
regression checks verify preserved NaN/Inf failure details, refusal to
score an active worker, analytic chi-square/Student-t values, and
per-site/state-group failures despite passing aggregate NEES. The scorer
now gates every latent's independent-cook bias, variance and coverage,
and every reported state group's normalized NEES. These are pointwise
95% tests; with many comparisons, some failures are expected by chance.
The first rejected capped-reading update had 11 readings but only 10
uncensored innovation degrees of freedom. Filtering out capped readings
selects a truncated subset, so unit-Gaussian innovation tests no longer
have their advertised null distribution. The scorer reports this and
cannot certify all gates when censoring occurred; dedicated censored
predictive-calibration tests remain necessary.
Per-update records now retain latent means, widths and truths even when
a later update fails, so tuning can distinguish a moving mean from a
changing standardized error. The obsolete, duplicate `test_online.jl` worker
was stopped after verifying its process ancestry; the replacement runs
the current solver and forwards the requested Julia thread count.

Remaining acceptance work is realistic streaming latency, the unresolved
numerical regression failures, and censored predictive calibration. Freeze
a working configuration before running disjoint held-out seeds through
loading, wrapping, fuel addition, pulling and resting. Statistical claims
must distinguish independent cooks from correlated arrivals, count every
failed/uncommitted update, and state the configuration and time interval
actually assessed. The 18-hour truth-trajectory result does not substitute
for posterior coverage or forecast calibration. Existing calibrated-noise
and ROM artifacts still require their normal fingerprint checks and fresh
validation; their fingerprints were not rewritten to bypass those checks.
