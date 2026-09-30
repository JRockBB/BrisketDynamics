# Deterministic nonlinear coordinates: measured comparison

The nonlinear coordinate map lowers the late two-probe refinement indicator,
but it does not resolve it within the matched budget. Neither this experiment nor
the earlier adaptive work establishes estimator consistency or online readiness.

## What changed

`estimator/src/conditional_charts.jl` adds an invertible triangular map. In
standardized selected-parameter coordinates, each later coordinate follows a
quadratic conditional center and a positive, exponentiated quadratic log width.
The pilot design is fixed and symmetric. It uses conditional fits of the original
graph, with no truth, sensor identities, or validation-case branching. Two, three,
and four integrated dimensions require 5, 18, and 43 pilots respectively.

The map changes integration coordinates; it does not declare the posterior
Gaussian. Every downstream node evaluates the original conditional target and
includes the map Jacobian. The remaining conditional Gaussian approximation and
its shape checks remain explicit. The density in the original graph coordinates
divides out that same Jacobian. The original serialized candidate type is preserved; mapped candidates
use a separate type.

The map is available to offline full-graph fitting and the existing full-history
online candidate adapter. Selected-variable CDFs and comparisons between maps
have separate numerical integration checks. A resolved density does not imply a
resolved nonlinear CDF. Earlier process-noise anchors, likelihoods, physics,
externally calibrated priors, and live defaults are unchanged.

CDF thresholds are in the original graph's selected coordinates, usually the
unconstrained parameter coordinates, rather than reference coordinates `q`.
Callers must convert physical parameter thresholds through the site's transform.
These queries do not yet provide predictive temperature or doneness CDFs.

## Comparison protocol

Both methods receive the same graph, selected axes, base center/covariance,
tolerance, integration orders, total conditional-fit budget, and 600-second soft
wall budget. Map pilots are deducted before posterior integration. Shape fits
also consume the same budget. Failed fits and exhausted budgets are reported,
not discarded or converted into successful beliefs. The wall budget is checked
between fits and batches of geometry evaluations. A counted fit is a completed
conditional evaluation; it may require more than one optimizer attempt, whose
cost remains included in wall time. The deadline
cannot interrupt a single
in-flight optimizer. Timings are single warm-process measurements, not a repeated
latency benchmark. Shared axis-selection setup is reported separately.

The finite mapped and affine boxes generally cover different physical domains.
These comparisons measure integration efficiency within each declared domain;
they are not equal-domain posterior accuracy comparisons. The analytic example
has independently computed domain masses. Cook domain coverage is unqualified.

Only previously exposed development cases 22001–22004 are used. Reserved
21001–21064 and 23011/23012 cases are untouched. No Monte Carlo inference or
calibration fitting is involved.

## Results

The late two-probe target at 1,800 seconds uses both probe-depth coordinates.
With 257 completed fits allowed, affine coordinates take 404.2 seconds and the
map takes 349.4 seconds, including 17.3 seconds for five pilots. The last complete
joint refinement indicators are 0.193718 and 0.136934, against the unchanged 0.002
criterion. Both end with an incomplete refinement and incomplete shape checks.
The final mapped marginal quadrature indicator is only 0.0000221, which must not
be mistaken for a resolved joint approximation.

Six additional off-grid fits, outside the matched budget, inspect the three most
massive leaves of each candidate. The mapped approximation has conditional KL
errors of approximately 0.000041, 0.000023, and 0.00227 at those points, with
absolute marginal log-density interpolation errors up to 0.0236. The worst of
these mapped conditional discrepancies is mostly a mean error. These are local
diagnostics against corrected conditional Gaussian fits, not true-posterior
error bounds or a substitute for complete refinement and shape checks.

A separate check uses the already fitted cell-center nodes, with no additional
graph fits, to inspect the three most massive leaves. All six finite shape checks
pass the 0.25 criterion: maximum defects are 0.133715 for affine coordinates and
0.126314 for the map. Repeated-solve differences at previously checked centers
are recorded in `late_two_cached_shape.json`. These cached checks do not change
the original budgeted outcome or establish tail accuracy; the integration result
remains unresolved.

The analytic curved, heteroscedastic example resolves in 142 total evaluations
including five map pilots, with relative normalizer error 8.62e-10 against an
independent domain integral. Affine interpolation exhausts 257 evaluations and
remains unresolved. The mapped box contains 0.999873 of this known target mass;
the affine box contains 0.999937. This deliberately curved example verifies the
mechanism; it is not evidence of general cook performance.

One-probe and early two-probe cases each integrate one axis, so the triangular
map reduces to the affine base and requires no pilots. Both methods resolve on
their declared boxes with the same fit counts and shape results.

The rich-sensor map uses all 729 allowed fits (18 pilots and 711 integration/shape
fits), taking 587.6 seconds in the repaired run. Its last complete refinement
indicator is 0.938563 against 0.01. It also finds a conditional-shape defect of
0.362053 against 0.25, despite completing only two of three shape checks. Thus
improving numerical point selection alone would not qualify this candidate;
the remaining conditional approximation also needs attention.

Paired results below use affine/map ordering. Indicators refer to the last
complete refinement, not an unfinished final refinement or a true-posterior
error bound. The four-direction runs stop on the time limit; the unresolved
two- and three-direction runs stop because their next work exceeds the node
budget. A batch of required nodes may leave a few evaluations unused.

| Development target | Integrated directions | Completed fits | Seconds | Last complete indicators | Result |
|---|---:|---:|---:|---:|---|
| One probe, 510 s | 1 | 56 / 56 | 45.6 / 48.4 | 0.001763 / 0.001763 | Box resolution and checked shape pass |
| Two probes, 630 s | 1 | 64 / 64 | 52.6 / 51.1 | 0.001987 / 0.001987 | Box resolution and checked shape pass |
| Two probes, 1,800 s | 2 | 257 / 257 | 404.2 / 349.4 | 0.193718 / 0.136934 | Integration unresolved |
| Rich suite, 600 s | 3 | 727 / 729 | 571.4 / 587.6 | 0.934368 / 0.938563 | Integration unresolved; checked shape fails |
| Pit/core dropout, 1,800 s | 4 | 403 / 453 | 601.1 / 600.3 | 7.435302 / 6.591399 | Time budget; integration unresolved |

The numerical criterion is 0.002 for the first three rows and 0.01 for the last
two. The affine rich-suite shape check is complete and also fails, at 0.361356.
Neither method demonstrates a consistent cook posterior or validated domain
coverage. The map is not a generally sufficient improvement over affine
coordinates on these development targets.

All 172 final focused assertions pass, including regression checks of the
adaptive density, target adapter, covariance/evidence correction, nonlinear
map, original-coordinate CDFs, cancellation-safe CDF slopes, query budgets,
map-width numerical range, caller ownership, and unchanged recorded cook-map
values. Analytic updates reuse the same nonlinear map through two additions of
new data while preserving the earlier candidate. The earlier nine mapped online
cook checks also pass: replay from 75 to 510 seconds agrees with the offline
candidate to 2.70e-12 on the estimated TV-bound diagnostic, preserves noise
anchors and the caller, and takes 45.5 seconds. That cook replay has only one
integrated direction, so it does not qualify nonlinear multi-axis chart reuse,
long-history cost, or controller latency.

## Limits and next decision

A conditional center/width map can help follow a curved ridge, but cannot by
itself flatten a multimodal conditional. The adversarial test deliberately fits
one of two modes: a narrow reference box resolves numerically while retaining
less than 60% of the target mass. Expanding the box recovers both modes. Cook
tail and missing-mode checks therefore remain mandatory.

The present local tensor interpolation still has substantial work left after
mapping. Its first complete refinement check needs
`3^d + 2d*3^(d-1)` conditional evaluations: 21 in two dimensions, 81 in three,
and 297 in four, before map pilots. The next efficiency experiment is dimension-adaptive sparse
interpolation with explicit joint-density and conditional-geometry checks,
allowing more directions to be integrated when conditional shape requires it.
That method has not been implemented or validated in this change. Increasing
moment order, loosening thresholds, or merely running the candidate online does
not close the demonstrated accuracy gap. Reserved-cook coverage, predictive
checks, real-cook discrepancy, and online latency qualification remain pending.

## Research basis

The use of invertible coordinates and explicit Jacobians follows the general
measure-transport framework in [Marzouk et al.](https://arxiv.org/abs/1602.05023).
The broader representation of monotone triangular maps is discussed by
[Baptista, Marzouk, and Zahm](https://arxiv.org/abs/2009.10303). This implementation
is a conditional-pilot center/width map; it does not implement their full
variational or adaptive KR-map learning algorithms, and no convergence theorem
from those papers is claimed here.

[Farcas et al.](https://arxiv.org/abs/1904.12204) provide a relevant deterministic
sparse-interpolation approach for expensive Bayesian inverse problems. Adapting
that idea to this estimator's conditional means, covariance, shape, and online
history is proposed work, not an established result for cooking.

## Reproduction and provenance

The production backend fingerprint is `d31877a10f94`; physics is `c2c7bbf8c0ec`.
The comparison uses Julia 1.11.9, four Julia threads, and one OpenBLAS thread.
The seven exact inference sources used for the matched comparison are preserved
under `sources/benchmark/`, with hashes in `benchmark-source-manifest.json`.
The rich-sensor rerun uses the same inference with a repaired ownership/reporting
wrapper, preserved under `sources/reporting-fix/` and identified by
`reporting-fix-source-manifest.json`. The original wrapper incorrectly treated
deep-copied observation closures as evidence of a mutated graph; it fails even
before fitting. The corrected wrapper compares serialized caller state before
and after fitting and preserves reports on failure. `ownership-audit.json` and
the retained worker log document that reporting defect and the discarded run.
Frozen graph snapshots are identified by SHA-256 in each case report and remain
in the existing scratch snapshot directory because they are large.

The reusable driver is `estimator/scripts/validate_transport.jl`. Driver jobs,
strict JSON reports, tests, and logs are archived beside this report. The worker
log retains failed development tests as well as their successful corrected runs;
those earlier failures are not qualification results.

Final numerical guards and CDF query changes have their own source hashes in
`source-manifest.json`; the exact comparison versions remain preserved separately.
The final tests verify identical forward-map values and Jacobians at every
recorded one-/two-probe mapped cook node. CDF reports have an explicit evaluation
budget and report unresolved queries rather than silently accepting a partial
integration check. Width underflow/overflow is rejected without clipping.

`reproduce.jl` replays the matched fits using archived inference and the repaired
reporting wrapper. It verifies historical source and archive hashes first. The
snapshot directory and output directory can be supplied as its two arguments.

`summary.json` and `summarize.ps1` provide the paired results. `verify.ps1` checks
all ten reports for matched settings, fit accounting, preserved source hashes,
unchanged caller state, and unchanged production/calibration sources. The final
worker also passes 23 artifact/syntax checks over 22 strict JSON documents and
the replay script. `local-candidate-manifest.json` identifies the larger Julia
candidate files retained in scratch storage. `artifact-manifest.json` hashes the
archived files other than itself.

New map tests run with:

```powershell
julia --threads=4 --project=estimator estimator/test/runtests.jl test_conditional_charts.jl test_mapped_conditional.jl test_transport_budget.jl test_target_ownership.jl
```

Regression tests for the shared adapter, curvature, and adaptive integration are
`test_adaptive_conditional.jl`, `test_adaptive_curvature.jl`, and
`test_adaptive_density.jl`. The graph tests can now bootstrap the estimator when
run alone as well as in a warmed validation worker.
