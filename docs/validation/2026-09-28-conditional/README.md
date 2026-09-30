# Conditional estimator validation — 2026-09-28

Status: experimental; statistical consistency and live-controller qualification
are not established. Failed approximation/solve checks are retained, not excluded.

## Source and execution

- Backend fingerprint: `d31877a10f94`.
- Physics/filter fingerprint: `c2c7bbf8c0ec`. This hash includes the estimator module
  include list; its change is not a change to the cooking physics or calibration.
- [SHA-256 manifest](source-manifest.json) records the complete physics and estimator
  source, calibration prior, environment files, validation harness and tested files.
- Julia 1.11.9, Windows, four Julia threads, one OpenBLAS thread, offline package
  environment, `--startup-file=no --compiled-modules=existing --project=estimator`.
- Final checks loaded a fresh estimator module inside an isolated parent module in
  a process with already-loaded dependencies. Earlier prototype modules were not
  reused for the final regression assertions. The two density/online test files were subsequently rechecked in a fresh OS process, as noted below.
- Timings include warm compilation effects and some concurrent prototype work. They
  are development timings, not a controller-hardware benchmark.

## Exact and structural checks

[Regression log](regression.log) plus the [fresh-process test log](standalone.log): **619 distinct assertions passed** across six files. The warm run passed 616; the fresh run passed 163 overlapping assertions, including three additional offline/online agreement checks.

| File | Assertions |
|---|---:|
| `test_conditional.jl` | 60 |
| `test_conditional_online.jl` | 58 |
| `test_conditional_density.jl` | 116 |
| `test_deterministic_online.jl` | 47 |
| existing `test_factors.jl` | 248 |
| existing `test_consistency.jl` | 90 |

The first fresh-process attempt exposed test imports placed after a test set, masked by the preloaded helper in the warm runner. The test initialization order was corrected without changing estimator code. The two affected files then passed independently in a fresh OS process: 47 online/offline assertions and 116 density assertions, using the same backend fingerprint. The [initial failure](offline-import-failure.log) is retained.

These check normalized densities, conditional determinant weighting, covariance
and evidence algebra, small contrasts, continuous CDFs, the actual cooking model's
unobserved nonlinear prior, sensor-information invariance and transactional updates.
The entire repository test suite and the expensive MC suites were not rerun.

[Regression driver](regression-driver.jl) preserves the executed warm-dependency
harness. It used the original `scratchpad/online_work/deterministic_final` location;
its relative includes must be adjusted if replayed from this archive. The six test
files are also runnable directly through `estimator/test/runtests.jl`.

## Cook validation protocol

The original four development cases are the already exposed 22001–22004 comparison
cases: rich sensors, pit/core dropout, one probe, two asynchronous probes; mesh nx=4,
last observation at 1800 seconds. Defaults are automatic directions, orders 5/9,
maximum order 17, maximum 729 nodes per bank and three integrated axes. Full nonlinear
history is retained. Acceptance tolerances were not relaxed to make cases pass.

After the implementation freeze, two new sparse-sensor cases were declared and run
once with nx=5: seed 23011 at 0/80/360 seconds with one probe, and seed 23012 at
0/110/390 seconds with two probes and a missing channel at the middle arrival.
These are limited unseen implementation checks, not a coverage study. The reserved
21001–21064 calibration seeds remain untouched. No truth is passed to inference. A worker log saying `PASS ...job.jl` means the driver completed without an uncaught exception; the JSON `passed`/`completed` fields determine whether inference passed. The standalone validation command exits with an error if any cook is unresolved.

## Cook results

The final implementation did **not** pass the four longer development cooks.
[Complete development-case results](final_cooks.json):

| Suite | Last accepted time | Rejected arrival | Reason | Update time |
|---|---:|---:|---|---:|
| Rich suite | 120 s | 600 s | Conditional shape; density checks also fail | 437 s |
| Pit/core dropout | 420 s | 1800 s | Conditional shape; density checks also fail | 262 s |
| One probe | 75 s | 510 s | Density refinement/normalizer | 30.6 s |
| Two probes | 150 s | 630 s | Density refinement | 14.4 s |

In the rich case the conditional shape defect was 0.580 (limit 0.25) and the
estimated density discrepancy was 0.123 (limit 0.025). The pit/core case remained
unresolved even with three integration directions: shape defect 2.54 and estimated
discrepancy 0.671. The reported reason is the first rejection, not necessarily the
only failed diagnostic. Rejected arrivals were rolled back.

[Unseen sparse-sensor results](unseen_sparse_cooks.json), unchanged implementation:

| Suite / seed / nx | Result |
|---|---|
| One probe / 23011 / 5 | All arrivals at 0, 80, 360 s accepted; last update 1.27 s |
| Two probes / 23012 / 5 | Accepted 0 s; rejected 110 s on density resolution in 14.9 s |

For the unseen two-probe arrival the estimated discrepancy was 0.0318 and the
normalizer change 0.0123, both above their declared limits. A small conditional
shape defect did not conceal those integration failures. No subsequent tuning was
performed on these unseen cases. The completed one-probe case is only a short
implementation check, not evidence of long-cook or real-cook coverage.

The [final worker log](final-worker.log) records both runs. Its executor later hit a
file-rename bookkeeping error while the completed job was being moved; the inference
and JSON outputs for both runs had already completed. This is recorded in
[the worker stderr](final-worker.err), and is not an inference pass/fail result.
## Development evidence and rejected alternatives

The [development archive](development/) includes drivers, logs and JSON for failed
experiments as well as improvements. The drivers depend on earlier worker state;
they are an audit trail, not independent fresh-process benchmarks. Jobs before the
final validation used evolving methods and must not be attributed to the final hash.
The duplicate job 21 was interrupted once the final-module equivalent was running.

- Fixed state-Gaussian conditional banks failed joint covariance refinement on the
  first dynamic arrival in all four cases. Enlarging the grid did not address the
  non-Gaussian shape of the remaining state distribution.
- A continuous Gaussian-component experiment did not repair that issue.
- At the first dynamic arrivals, weighted one-SD parameter-direction log-density
  defects were approximately 1,826–5,306 in state coordinates, versus 0.0016–0.123
  with nonlinear innovation transport. These are local diagnostic improvements,
  not coverage probabilities or a global approximation-error bound.
- Later probe-only errors exposed probe-depth directions; humidity exposed a
  different direction. Automatic axis selection therefore uses graph shape
  diagnostics, rather than a sensor-name-specific ignition block.
- Continuous reconstruction was needed to represent density between fixed nodes.
  Its nonlinear remainders and density-comparison integrals require their own
  refinement. Increasing node count cannot correct a coarse comparison integral.
- A constant conditioned prior term perturbed the numerical optimizer despite
  containing no information about the other coordinates. Removing constant rows
  from optimization while retaining their original evidence restored the exact
  unobserved-depth invariant.

## Interpretation

Small finite shape defects and numerical agreement between approximations do not
establish posterior calibration. The conditional covariance is still Gauss–Newton;
remote modes and tail errors can be missed. The adaptive budget is finite. Default
full-history cost grows, and optional Gaussian fixed-lag compression can discard
nonlinear history. Generated-state tail/decision integration is not yet qualified.

The external calibration and prior export were not modified. The earlier MC work
provided reference and diagnostic evidence, not an improved calibration. Nothing in
this record promotes the experimental deterministic backend to the live controller.
## What the failures do and do not identify

The conditional shape check rejects the particular local quadratic being used. It
cannot, by itself, distinguish an inaccurate curvature matrix from skewness or
another mode. For least squares, the observed Hessian contains both `J'J` and
residual-weighted second derivatives; this implementation retains the Gauss–Newton
part. A proper Laplace approximation uses the observed log-posterior Hessian
([Helin and Kretschmann, error estimates for Laplace approximation](https://link.springer.com/article/10.1007/s00211-021-01266-9)).
Auditing that difference is necessary before attributing every conditional failure
to a need for a more expressive distribution. No observed-Hessian correction was
implemented or validated in this change.

If conditional Gaussian errors persist after that audit, the unresolved work is a
likelihood-adapted nonlinear conditional map and more efficient adaptive integration,
not simply more moments of the old state Gaussian. One relevant research direction
is [Sharp, van Bloemen Waanders and Marzouk's iterated transport/quadrature framework](https://arxiv.org/abs/2603.12448),
which learns non-Gaussian surrogates while reusing expensive model evaluations. Its
reported experiments concern low-dimensional inverse problems; it is not evidence
that this cooking estimator will meet accuracy or online latency requirements.