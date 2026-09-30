# Conditional posterior curvature audit

The estimator is **not yet statistically consistent**. The audit identifies a real
Gauss–Newton curvature error, but correcting the full Hessian is insufficient for
the rich-sensor and pit/core failures. Higher-order structure remains in conditional
parameters, including `k_spread` and, in the rich case, uncertain probe depth.
No inference backend, prior, calibration file, live controller or acceptance tolerance
was changed by this audit.

The table reports weighted one-standard-deviation log-density defects over the three
largest-weight components. Smaller is better; the existing diagnostic threshold is
0.25. The corrected column uses each full observed-Hessian covariance to define the
directions and their standard deviations. Original component weights are retained.

| Suite / failed arrival | Checked mass | Original GN defect | Corrected-curvature defect |
| --- | ---: | ---: | ---: |
| One probe / 510 s | 81.2% | 0.0535 | 0.00765 |
| Two probes / 630 s | 80.9% | 0.0159 | 0.0112 |
| Rich sensors / 600 s | 53.2% | 0.580 | 2.128 |
| Pit/core dropout / 1800 s | 20.0% | 2.542 | 2.393 |

These are counterfactual local-approximation checks, **not** posterior coverage results
or an updated mixture posterior. In particular, 20% checked mass in the pit/core case
is insufficient to characterize the complete posterior. Raw component and directional
results are retained in the JSON files; `summary.json` and `summarize.ps1` reproduce
the aggregation.

This audit investigates the exposed 22001–22004 failures of backend `d31877a10f94`
and physics/filter fingerprint `c2c7bbf8c0ec`.
It does not retune noise, priors, acceptance thresholds or the test cases. Reserved
21001–21064 seeds and the previously exposed 23011–23012 cases are not used here.

The negative log posterior is evaluated in parameter, initial-state and process-
innovation coordinates. Process whiteners remain fixed, so the decoder has a
constant Jacobian determinant. Process losses are evaluated directly as half the
squared innovation norm, avoiding cancellation from subtracting nearby states.
Other factors retain their original exact losses. Their gradients are differentiated
directly; AD dynamics sensitivities are propagated backwards through the decoder.
Numerical ODE accuracy remains part of this calculation and is checked separately.

Central differences of these gradients at steps 0.04, 0.02 and 0.01 estimate
observed Hessian columns in conditional Gauss–Newton covariance units. Parameter-
span audits also report curvature coupling to the omitted innovation directions;
they are **projected audits**, not full posterior Hessians. Full audits are labeled
separately. No indefinite eigenvalues are clipped.

Each report retains step-size changes, Hessian asymmetry, stationarity, covariance
whitening error and numerical round-trip loss error. Directional probes at 0.5, 1
and 2 old standard deviations separate local quadratic changes from odd and even
higher-order departures. These are local diagnostics; they cannot rule out distant
modes or establish interval coverage.

`estimator/test/test_curvature_audit.jl` has independent analytic targets: an exactly
Gaussian target with an incorrect Gauss–Newton metric, a quartic target, and a
projected audit with nonzero coupling outside its subspace.

Reproduce snapshots with:

```powershell
julia --threads=4 --project=estimator estimator/scripts/freeze_deterministic_failures.jl scratchpad/curvature_audit/replay
julia --threads=4 --project=estimator estimator/scripts/curvature_audit.jl scratchpad/curvature_audit/replay/core_22003.jls core_audit.json full 0.0001
julia --threads=4 --project=estimator estimator/test/test_curvature_audit.jl
```

Snapshots include the rejected density, all banks still present at rejection,
their original graphs, the accepted prefix, observations, case definition and
source fingerprints. They do not retain every discarded intermediate grid from
earlier axis changes. Julia serialization requires a compatible Julia version and
the defining modules to be loaded before deserialization; snapshots are local
artifacts, not a portable public interchange format.

Research basis: Laplace covariance uses the observed negative-log-posterior Hessian,
whereas Gauss–Newton omits residual-weighted second derivatives. A large defect
against Gauss–Newton alone therefore does not disprove a Gaussian conditional.
See [Helin and Kretschmann (2022)](https://link.springer.com/article/10.1007/s00211-021-01266-9).

## Numerical accuracy

At the original ODE tolerances, independent scalar finite differences failed 6 of
8 gradient checks (absolute tolerance 0.0002, relative tolerance 0.0001). The largest
discrepancy was 0.0130. These failures are preserved in `worker.log`; no threshold was
relaxed. For the same eight directions, reducing both ODE tolerances by 10,000 brought
the largest discrepancy below 1e-6. The original process whiteners, noise anchors,
parameter values, priors, measurements and innovation-coordinate chart were retained.

This is a **numerical convergence study**, not a revised stochastic model or a new
calibration. The first one-probe component's loss changed by only 3.6e-6, but the final
parameter-span Hessian step change dropped from 0.00994 to 3.86e-6. Finite differencing
for curvature requires more accurate solves than evaluating the forward likelihood.
The large rich-sensor shape defects are much larger than these numerical effects.

Files ending in `_tight_full_curvature.json` use ODE tolerances 1e-14/1e-12 and Hessian
steps 0.02, 0.01 and 0.005. Original-tolerance full and projected results are retained
separately. Projected results are not substituted for full covariance/evidence corrections.
The counterfactual correction uses the full positive Hessian, changes the covariance
root by its inverse Cholesky factor, and reports the corresponding log-volume change.
It does not refit modes, renormalize all conditional component weights or replace the
estimator's posterior. Aggregate comparisons keep the original component weights.

The archive uses strict JSON: an unavailable first-step difference is `null`. Earlier
raw scratch reports used the repository's extended JSON writer with `Infinity`; those
values are converted to `null` in the archive without discarding numerical results.

## What the evidence supports

All four original failures reproduced at the same arrivals, with their prior rejection
diagnostics unchanged. The full observed Hessians at the checked modes were positive;
this audit did not find saddles mistaken for minima. The supplied covariance roots
whitened the Gauss–Newton metric to approximately 1e-11 at the original solve accuracy.
Numerical reconstruction of process residuals was not responsible for the large shape
defects: direct innovation losses and decoded graph losses agreed to numerical precision.

The sparse cases were rejected for disagreement between depth-integration rules.
Their checked Gaussian conditionals become more accurate with observed curvature,
especially for one probe. This is reason to reconstruct and compare **all** nodes
with corrected covariance and evidence before judging whether additional depth-grid
refinement is sufficient. It is not a claim that their posterior intervals are calibrated.

The rich and pit/core cases still have large odd and even departures from the corrected
quadratic. The rich case's broader observed-Hessian covariance exposes more non-Gaussian
structure. An unconditional switch from Gauss–Newton covariance to a Laplace covariance
would therefore not resolve the frozen suite.

The next inference implementation should preserve these remaining conditional shapes
with adaptive deterministic integration or a checked nonlinear transport, selecting
directions from the target rather than sensor names or simulation seeds. The current
tensor-grid budget and fixed Gaussian remainder did not resolve these cases. Validate that method
offline against the stored full-history targets, then compare its incremental online
updates at matching prefixes. External calibration, predictive checks, held-out coverage,
real cooks and bounded-memory controller latency remain separate requirements.

## Verification and artifacts

- Final analytic regression: **25/25** assertions passed. This includes Gaussian
  and quartic oracles, a rotated full basis, projected cross-curvature and strict JSON.
- Cooking derivative/noise-law verification: **28/28** assertions passed at the
  refined solver accuracy, including the same eight directional gradient checks
  that originally had six failures. Process whiteners and their anchors were preserved.
- An earlier independent fresh process passed the initial **22/22** analytic checks.
  The three later assertions were verified in the final isolated test module.
- All four exposed estimator failures reproduced. No coverage study or full repository
  regression suite was run; the estimator source and calibration were unchanged.
- `source-manifest.json`: 74 files verified with no hash mismatches, including the
  frozen estimator/physics sources and the new diagnostic scripts/tests.
- `snapshot-manifest.json`: paths, sizes and SHA-256 hashes of the four local `.jls`
  snapshots under `scratchpad/curvature_audit/`. These large binary working artifacts
  are ignored by Git; the report, strict JSON results and reproduction scripts remain
  in the reviewable workspace.

`worker.log` retains successes and failures. A driver completion is not an estimator
acceptance. The `07/08/10/11` driver failures were JSON parsing/indexing problems in
auxiliary reporting; final aggregation uses `summarize.ps1` over verified strict JSON.
`drivers/` records the executed development scripts, whose relative paths assume their
original scratch location. Prefer the repository commands above for replay. Both Julia
processes used for this audit exited, and no background validation remains running.
