# Sampled online/offline implementation validation

The baseline runs below used backend `411c33d5521d`, physics `4e4a6800e4a2`. The [source hashes](source_hashes.json) identify
the implementation, comparison script, control generator and prior. Julia 1.11.9, eight Julia threads,
one BLAS thread, Windows; `--startup-file=no --compiled-modules=existing --project=estimator`.
The earlier archive in `../2026-09-27` is a separate historical version and has not been overwritten.

The subsequent log-weight correction changes the backend to `84201766a16f`. Its separate
[validation record](logweight-fix/README.md) distinguishes the current code from these baseline cook
comparisons. It does not resolve the full-history refresh cost or turn the failed comparison into a pass.

## Fresh-process correctness checks

The [regression log](regression.log) contains **702 passing assertions**, with no observed failed
assertions. Ten files completed, followed by four completed sets in `test_online.jl`:

| Test | Passed assertions |
| --- | ---: |
| Nonlinear posterior sampling | 32 |
| SMC analytic and measurement oracles | 38 |
| Full-refit sampled transactions | 26 |
| Incremental likelihood, updates, checkpoints and batching | 48 |
| Offline sampled entry point | 10 |
| Sampled histories, planning and forecasts | 63 |
| Existing single-piece planner parity | 19 |
| Existing factor graph | 248 |
| Consistency and coordinate regressions | 90 |
| Calibration-harness bookkeeping | 10 |
| Existing online layout, observables, transactions and arrivals | 118 |

The optional legacy full-history/delayed-reading comparison was interrupted after a prolonged run;
the remainder of `test_online.jl` is **incomplete**, not passed. See [the run status](regression_status.txt).
The multi-piece planner set and the full repository suite were not run in this validation.

An additional fresh standard-library process passed all **six** statistical comparator tests in
`test_online_comparison.jl` ([log](comparison_oracle.log)). A constructed dependence error is rejected
even when the marginal checks pass. This brings the completed repository assertions to **708**;
the comparison oracle uses its own known distribution and does not fit or inspect cook truths.

The [development log](development.log) is retained separately. Initial failures exposed missing-addition
NaNs and a non-invertible plan encoding. Another finite-difference comparison straddled an interpolation
knot and was moved inside a cell without loosening its tolerance. An unreliable assumption that an
outlier must exhaust the sampler was replaced by an explicit insufficient-refresh-budget test. The
fresh checks above include the fixes; development reruns are not added to the assertion count.

## Interpretation

These are implementation checks. They do not establish real-cook calibration, population transfer,
hours-long runtime, or robust closed-loop control. The public interfaces and remaining model/controller
limits are described in [sampled estimation](../../sampled-estimation.md).

## Cook comparison status

**The frozen four-suite comparison (`5d69152caee3`) failed overall.** All four cases were attempted;
three passed their online/offline comparison and one rejected its final online arrival for insufficient
sampling precision. The [combined record](comparison.json) retains the failure. The full-suite case, seed 22001,
completed all four online arrivals and its independent offline comparison. The maximum standardized
differences were 2.00 for means, 2.71 for CDF values, and 2.69 for joint second moments, all below the
predeclared limit of 6. The largest mean difference was 0.067 posterior standard deviations. The
independent offline run passed estimated precision at 1,024 particles per system; the online path
had retained a 4,096-particle budget following its initial refresh.

Its online update times were 122.25, 337.29, 14.67 and **2,243.23 seconds** at cook times 0, 120, 600
and 1,800 seconds. The last update is a clear live-use latency problem at this observation cadence.
These are observed elapsed times, including compilation and some concurrent development computation,
not isolated hardware benchmarks. Passing numerical precision and implementation agreement does not
make this online path suitable for deployment.

The full-suite seed was subsequently exposed during proposal development. It remains a predeclared
comparison of the frozen baseline, but cannot serve as held-out validation of those new proposals.
The single-probe (22003) and two-probe (22004) cases also passed their independent offline comparisons.
Their largest mean differences were 0.068 and 0.048 posterior standard deviations. Maximum standardized
mean/CDF/joint-moment discrepancies were 2.99/3.02/3.04 and 2.72/3.03/3.10, respectively.
Their [raw records](sparse_comparison.json) and [log](sparse_comparison.log) are retained.
They used a four-thread warm worker after restoring the two experimental method overrides from the
exact frozen production files; the [driver](sparse_driver.jl) records that restoration. Extra candidate
and oracle methods remained loaded but were not called by the production inference path. These were
not fresh-process runs. The driver files retain their original workspace-relative paths.

The one-probe update times were 1.62, 6.14, 150.16 and 9.99 seconds; the two-probe times were 0.45,
4.70, 354.51 and 519.26 seconds. These three completed comparisons each contain only one cook at
four spatial nodes.

The pit/core case, seed 22002, accepted arrivals at 0, 90 and 420 seconds. At 1,800 seconds it exhausted
the 4,096-particle-per-system budget, with minimum mean ESS 258.60 and tail ESS 355.37, and returned
`committed=false`. That arrival took **5,632.78 seconds (93.88 minutes)**. The previous accepted graph
and posterior were retained. No final online/offline comparison was possible for this case, and the
offline comparison was not run after the rejection. Failure to deliver a qualified update at this
cadence is a blocking live-use limitation, not a passing calibration result.

The [first two cases](first_two_comparison.json), [complete log](first_two_comparison.log),
[driver](first_two_driver.jl) and [segment status](comparison_segment_status.txt) are retained. The
first process was deliberately stopped at the third-case boundary after writing the pit/core failure;
the already-completed sparse segment supplies cases three and four. Its exit status 1 reflects this
boundary stop. Source hashes were verified unchanged before promotion of the later log-weight fix.
The combined report is false because of the recorded pit/core failure, not because of that stop.

Separate [runtime diagnostics](runtime-diagnostics/README.md) reached the full likelihood with a small
unqualified particle system and reproduced the first three original arrivals in an instrumented replay.
The last incremental propagation completed and triggered a full-history refresh. Case 22002 is now
development-exposed as well; subsequent changes require independent validation.

Three [proposal experiments](proposal-experiments/README.md) were evaluated separately and were not
promoted. Their exact-weight/kernel oracles passed, but their cook results did not establish a reliable
performance improvement. Source, failures, interrupted attempts and diagnostics are retained there.

The [reference profile](transport_profile.txt) sampled 512 serial full-history evaluations on the
development full-suite graph in 4.49 seconds. About 85% of sampled stacks were inside propagation and
73% inside the ODE solve (inclusive counts). This indicates that repeated physics replay dominates
this workload; observation bookkeeping alone cannot remove the refresh-cost problem. This is a
profile of the declared four-node model, not an hours-long production performance qualification.
