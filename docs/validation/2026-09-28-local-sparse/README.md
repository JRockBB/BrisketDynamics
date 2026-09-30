# Local sparse refinement across coupled parameters

The optional spatial sparse candidate is implemented for offline and full-history online
fitting. It is **not a qualified replacement for the existing method**. The late
two-probe cases still exhaust their fit budgets, and the analytic comparisons
show that spatial refinement alone does not reliably handle oblique curvature.
Production defaults, priors, process noise, physics and calibration are unchanged.

## Implementation

`estimator/src/local_sparse_density.jl` represents each rectangular region with
a sparse polynomial of marginal log density, conditional means and log-Cholesky
geometry. Positive density and positive conditional covariance are preserved.
Positive quadrature normalizes the density; Bernstein bounds guard against peaks
of the interpolant missed by both quadrature orders. No Monte Carlo is used.

Each region begins with a complete total-degree-two polynomial. Its complete
next-degree frontier and two off-grid panels are evaluated before it can pass.
Higher-order mixed interactions therefore receive checks even when axis probes
are blind. Order can increase to eight. After two order increments, a region is
bisected if its unsaturated joint discrepancy fails to decrease by a factor of
two. The frontier's directional contributions choose the split axis; relative
width breaks ties. Children start at degree two.

All interpolation, unused frontier, off-grid and discarded child evaluations
share one exact oracle cache and one fit budget. A parent is retained until both
child polynomials exist, so budget exhaustion cannot leave a hole in the domain.
Every retained leaf must pass the same **uniform finite-point** joint discrepancy
criterion used by the global sparse path. Small posterior mass does not excuse
a failing leaf. This is deliberately distinct from the older tensor method's
integrated refinement indicator. Normalization has a separate two-million-point
evaluation limit. Failures and incomplete checks remain unresolved.

The unsaturated score is used for refinement decisions only. The bounded score
used for stopping is unchanged. Finite checks are not true-posterior error bounds,
tail or missing-mode guarantees, conditional Gaussian validation, or empirical
cook coverage. Panels are excluded from fitting but reused for adaptation and
stopping; they are not held-out validation. Adjacent patches may have boundary
jumps. Continuity is not asserted.

The spatial-refinement rationale is supported by
[Wilka and Lang's local hp sparse interpolation work](https://arxiv.org/html/2404.02556v2),
while the polynomial hierarchy uses
[nested Leja interpolation](https://arxiv.org/html/1911.08312v2).
This implementation uses rectangular polynomial patches; it does not implement
the former paper's continuous localized basis, kink detector, or convergence
guarantees.

The shared adapter selects this candidate with `local_refinement=true`:

```julia
offline = expanding_sparse_fit(graph;
    axes, shape_key, local_refinement=true, max_nodes=257)
arrival = sparse_arrival_candidate(anchor, records;
    axes, shape_key, local_refinement=true, max_nodes=257)
```

The existing static-latent and full-original-history restrictions remain. The
online path returns a candidate, never a live commit. Automatic parameter
promotion, original-factor reconstruction, exact chart Jacobians, actual-node
shape checks and caller ownership are shared with the prior adapter. There are
no sensor names, cook IDs, or simulation-truth branches in the integration code.

## Refinement-policy correction

The initial policy used the bounded stopping score to decide whether polynomial
order was helping. Large conditional errors saturate that score, hiding genuine
improvement and causing premature subdivision. A generic analytic counterexample
uses a Gaussian marginal, conditional mean `x^4 + 0.3*x*y^2`, and standard
deviation `1e-7`. The initial policy exhausts 65 fits without resolving this exact
polynomial; the final policy resolves it in **29 fits in one region**.

The final policy uses unsaturated log-mass and conditional KL-derived distances
for allocation and permits two order increments before diagnosing stagnation.
It retains the original bounded stopping criterion. Both source versions and
their results are archived. The final regression suite passes **359 assertions
across eleven files**, including a regression for this saturation failure.

## Protocol and analytic controls

Only previously exposed development snapshots 22001–22004 and the recorded
1,800-second two-probe continuation are used. Reserved 21001–21064 and 23011/23012
cooks remain untouched. The existing finite boxes, tolerances, graph-fit budgets,
integration orders, priors, physical model and noise anchors are retained.
Each cook comparison permits 600 seconds, checked between operations; an in-flight
optimizer cannot be interrupted. One discarded warmup fit precedes a timed cook
comparison. Wall times are observations from a warm process, not latency claims.

Mapped comparisons charge five fresh chart pilots against the same 257-fit total.
Automatic rich-cook promotion charges every screening and discarded stage against
729 fits. Original and final archives are loaded into separate Julia modules,
so recorded source hashes identify the executed implementations.

Two coupled C1 analytic targets use a Gaussian marginal on `[-2,2]^2`, conditional
standard deviation 0.02, and means `max(x-0.137,0)^2*(1+0.2*y)` or
`max(x+0.45*y-0.137,0)^2`. Every method receives 257 fits and tolerance 0.002.

| Target | Global sparse | Final local sparse | Existing local tensor |
|---|---:|---:|---:|
| Aligned coupling: fits / resolved | 257 / no | 209 / yes | 177 / yes |
| Aligned coupling: averaged conditional TV estimate | 0.001707 | 0.00002738 | 0.00003222 |
| Tilted coupling: fits / resolved | 257 / no | 257 / no | 257 / no |
| Tilted coupling: averaged conditional TV estimate | 0.07896 | 0.04797 | 0.009090 |

TV estimates average the analytic equal-variance Gaussian conditional distance
under the known marginal. Positive diagnostic quadrature uses orders 19 and 31;
these are numerical estimates, not certified bounds. The global tilted estimate
changes by about 0.00037 and the final local estimate by about 0.00018. Marginal
log-density and conditional-root checks are recorded separately. Resolution
flags use different numerical diagnostics across sparse and tensor methods, so
the flags alone do not establish equal accuracy. The tilted example favors the
existing tensor method and limits any claim of general improvement.

## Cook outcomes

| Target, final policy | Fits | Seconds | Outcome |
|---|---:|---:|---|
| One probe, 510 s; fixed chart | 113 | 50.4 | Numerical and finite shape checks pass |
| Two probes, 630 s; two integrated parameters | 129 | 70.7 | Fit budget; unresolved |
| Late two probes, 1,800 s; affine chart | 257 | 236.4 | Fit budget; unresolved |
| Late two probes, 1,800 s; fresh nonlinear chart | 257 | 193.0 | Fit budget; unresolved |
| Rich cook, 600 s; automatic two to four parameters | 729 | 501.1 | Fit budget; unresolved |

The one-probe result is not an improvement over the previously recorded global
sparse result at the same fixed chart (54 fits). The early coupled two-probe
case has finite shape defect 0.01265 but recorded off-grid discrepancy 0.1301;
its conditional Gaussian shape check does not establish numerical resolution.

The final mapped late two-probe run uses all 257 fits, including five pilots.
Its recorded off-grid discrepancy is **0.9706** against the unchanged **0.002**
criterion; shape defect is 0.1194. The affine run also exhausts 257 fits, with
recorded off-grid discrepancy **1.2602** and shape defect 0.1148. Both remain
unresolved. Passing a finite Gaussian-shape check does not repair inaccurate
interpolation of the joint conditional density. Final checks can be incomplete
at budget exhaustion, so the recorded discrepancies are not uniform error bounds.

The rich run spends three fits screening/promoting parameters and 726 on the
four-dimensional integration. Its final candidate has three retained patches,
degrees 4, 4 and 5, and finite shape defect 0.1923. The last complete refinement
check reports frontier discrepancy 1.9087 and off-grid discrepancy 1.7187 against
0.01. Normalization error is also unresolved at 0.002084 against its separate
0.001 tolerance. Spent child fits remain charged when an incomplete split keeps
its parent. No approximation is accepted by loosening a threshold.

### Common conditional-oracle comparison

Nine extra conditional fits evaluate the same central affine Gauss grid used in
the previous report, outside the matched fit budgets. Every compared point lies
inside every candidate's support. Coordinates, original factors and innovation
decoders agree. Results compare each representation with the same polished
conditional Gaussian oracle, not the true posterior.

| Representation | Largest absolute marginal log-mass error | Largest conditional KL |
|---|---:|---:|
| Existing local affine | 0.07735 | 0.06106 |
| Existing local mapped | **0.01323** | **0.003924** |
| Global sparse mapped, degree 64 | 0.04235 | 0.01885 |
| Initial spatial sparse affine | 0.4619 | 30.82 |
| Initial spatial sparse mapped | 0.3435 | 17.04 |
| Final spatial sparse affine | 0.4383 | 6.002 |
| Final spatial sparse mapped | 0.3721 | 0.4233 |

The policy correction improves the failed initial spatial construction, but the
existing mapped local method remains substantially more accurate on this grid.
The new method's failure is visible at central comparison points, not only in
remote tails. This evidence rejects adopting it as the consistency fix.

The present construction resets children to degree two, which can discard useful
mixed polynomial structure when splitting one direction. Its axis-aligned
patches also struggle with oblique features, as the analytic tilted example
demonstrates. Preserving mixed structure across splits and allocating effort by
joint-density contribution are possible further experiments; neither has been
implemented or validated here. This result does not rule out other local sparse
constructions.

### Coupled online replay

The full-history 630 to 1,800-second two-probe arrival uses the same fixed affine
chart and the shared local fitter. It spends 257 fits and takes 247.0 seconds,
including assimilation and five-point comparison, excluding the separate offline
fit. At those five points, the maximum log-density difference from offline is
zero at floating-point precision and conditional KL is approximately 5.14e-28.
All seven recorded agreement/budget checks pass, and the caller is unchanged.
Process-noise `W` and `wy` anchors agree exactly.

Both candidates remain unresolved and uncommitted. Agreement verifies target
construction and the shared implementation; it does not establish accuracy.
This is a fixed-chart replay, not a qualification of learning nonlinear charts
online. No result here establishes estimator consistency, bounded-memory
filtering, predictive coverage or controller latency.

## Reproduction

`reproduce.jl CASE METHOD OUTPUT.json [initial|final]` checks production baseline,
source-archive and input hashes, then runs the selected comparison. The default
source version is final. Cases are `core`, `two_core`, `two_core_coupled`,
`late_two`, and `full`; methods are `local`, `expanding_local`, and `mapped`.
The coupled early two-probe check adds the highest-scoring remaining parameter
from the frozen shape diagnostic, retaining the 129-fit budget.

`verify.ps1` checks hashes, ownership, budgets and regression results. Executed
drivers and the full worker logs preserve the experimental sequence. The frozen
input snapshots and large serialized candidates remain in ignored local scratch
directories; reproduction requires those files and the estimator Julia
environment. Source archives distinguish the initial and final policies.
The first artifact-audit driver had a parsing typo; its corrected successor and
the original failure are retained in the worker log. No inference source or cook
result was changed by that audit correction.
The corrected artifact audit passes 70 checks, including loading both archived
implementations and parsing the replay script. All 74 production/calibration
baseline files remain unchanged: backend `d31877a10f94`, physics `c2c7bbf8c0ec`.
