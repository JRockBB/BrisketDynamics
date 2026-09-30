# Sparse conditional integration and automatic parameter promotion

The one-probe online arrival now resolves numerically within its 129-fit budget,
using a shared-budget local fallback, and agrees with offline fitting. Automatic
parameter promotion also removes a central Gaussian-shape failure in the richer
cook. **Overall estimator consistency remains unqualified: the late two-probe
and richer integration problems are still unresolved.** Live defaults are
unchanged. These are development results on previously exposed cooks.

## Representation and selection

`estimator/src/sparse_density.jl` builds a downward-closed, nested Leja polynomial
interpolant of the marginal log density, conditional mean, and log-Cholesky
geometry. It admits individual nodes and interacting polynomial terms rather
than fitting every node of a tensor cell. The coefficient hierarchy is selected
using joint conditional discrepancies, with periodic breadth steps so low-order
interactions cannot be permanently starved by another coordinate.

Exponentiating the interpolated log density and log diagonal of the Cholesky
factor preserves positive density and positive definite conditional covariance.
Normalization uses positive Gauss quadrature, adaptive subdivision, and a
Bernstein bound on peaks of the interpolant. Sparse interpolation coefficients
are never interpreted as signed probability weights. There is no Monte Carlo.

The method follows the nested hierarchical interpolation construction described
by [Loukrezis and De Gersem](https://arxiv.org/html/1911.08312v2). The broader
[adaptive sparse Leja inverse-problem work](https://arxiv.org/abs/1904.12204)
motivates reducing expensive model evaluations. This implementation does not
implement that paper's multilevel forward model or claim its results for cooks.

`expanding_sparse_fit` first checks the remaining Gaussian at an actual
conditional fit. If an eligible parameter fails the existing 0.25 shape check,
it adds the worst failing parameter to the integrated coordinates and screens
again. The main integration then checks up to three high-density actual oracle
nodes. It can promote another parameter after these checks if budget remains.
All screening, discarded stages, validation nodes, and unused frontier nodes
consume one shared conditional-fit budget. Axis and fit limits stay unresolved.

Axis selection uses the original target and eligible parameter coordinates;
there are no branches on cook IDs, truth, probe count, or sensor identities.
Promotion currently rebuilds an affine chart from the original graph covariance.
Fixed-axis sparse fitting also accepts the learned nonlinear chart, including
its exact Jacobian; the separate mapped comparison charges fresh map pilots.

The final degree cap is 64. If global interpolation stalls in one dimension,
the fitter can use the existing local quadratic integrator on the SAME finite
box and exact target, reusing every cached oracle value. Screening and both
integration attempts share one fit budget. A successful fallback returns the
existing mapped local candidate type; an unsuccessful attempt retains the
unresolved global candidate and charges the spent fits. No multidimensional
tensor fallback is enabled implicitly.

Positive cubature has a two-million-evaluation limit. A conservative bound on
local mass and geometry quadrature reserves capacity for the fallback; a binary
search limits its cell count even when a caller supplies a huge cell budget.
Initial graph linearization is cloned because it can populate factor caches.

Shape probes reuse actual fitted conditional means and covariance through the
common innovation decoder. They do not refit a node or mistake interpolated
geometry for an exact conditional solve. They test finite directions in the
specified parameter key, not every state direction or every posterior mode.

## What the diagnostics mean

The sparse refinement diagnostic differs from the older tensor-cell indicator.
It checks the maximum finite-point discrepancy on the admissible interpolation
frontier and two off-grid tensor panels. The discrepancy includes marginal
log-density error and conditional Gaussian KL, preserving sensitivity to very
narrow correlated directions. The off-grid points are excluded from polynomial
fitting but reused for adaptation and stopping; they are not held-out validation.

`resolved` only says those finite checks and numerical normalization passed on
the declared box. It is not integrated target error, a tail or missing-mode
bound, conditional Gaussian validation, or empirical cook coverage. Its number
cannot be compared directly with the old integrated refinement indicator.
Matched fit/time budgets therefore compare resource use and outcomes, not equal
accuracy proven by identical metrics. A separate common off-grid experiment
compares representations against the same conditional oracle fits.
For a successful local fallback, `diagnostic_kind` explicitly switches to the
older integrated joint-refinement diagnostic; it does not relabel a failed
global point check as a successful one.

The initial sparse quadratic fit has `(d+1)(d+2)/2` nodes. Checking its complete
cubic frontier and both off-grid panels takes `binomial(d+3,3) + 2^(d+1)` calls
before overlap: 67 in four dimensions versus the older 297-node first tensor
refinement check. This reduces an initial fit barrier; it does not guarantee
faster convergence for strongly curved conditional geometry.

## Protocol

Only development snapshots 22001–22004 and the previously recorded late
two-probe continuation are used. Reserved 21001–21064 and 23011/23012 cases remain
untouched. Priors, physics, likelihoods, process-noise anchors and external
calibration are preserved. Caller-state fingerprints are checked after runs.

Fixed-axis comparisons retain the previous graph, coordinates, finite radius-4
box, fit budget, numerical tolerance, and integration orders. The initial Leja
degree cap is 16. Capacity experiments explicitly raise that cap while keeping
the same fit budget and tolerance; their results are recorded separately.
The degree-cap increase applies generally, not by sensor or case label. The
default soft wall limit is 600 seconds, checked between fits and cubature
cells; it cannot interrupt an in-flight optimizer. Counts are completed
conditional evaluations, possibly with more than one optimizer attempt. Wall
times are single warm-process observations, not repeated latency benchmarks.

Nonlinear charts and promoted axes can change the original-coordinate finite
domain. Comparisons across these domains do not prove equal-support accuracy.
The online adapter retains the complete original factor history. Bounded-memory
filtering, predictive CDFs, controller latency, and nonlinear multi-axis cook
updates remain unqualified.

## Results

| Target and method | Fits | Seconds | Outcome |
|---|---:|---:|---|
| One probe, 510 s; old local tensor | 56 | 51.5 | Numerical and finite shape checks pass |
| Same fixed chart; sparse degree 16 | 21 | 25.6 | Degree limit; unresolved |
| Same fixed chart; sparse degree 64 | 54 | 41.2 | Numerical and finite shape checks pass |
| Two probes, 630 s; sparse degree 64 | 42 | 33.8 | Numerical and finite shape checks pass |
| Late two probes, 1,800 s; sparse affine, degree 16 | 257 | 377.1 | Fit budget; unresolved |
| Same target; sparse mapped, degree 16 | 257 | 307.5 | Fit budget; unresolved |
| Same target; sparse mapped, degree 64 | 257 | 298.1 | Fit budget; unresolved |
| Rich cook, 600 s; three axes, degree 16 | 638 | 601.0 | Time budget; unresolved |
| Rich cook; automatic two → four axes | 617 | 600.4 | Central shape screening passes; integration times out |

The late mapped degree-64 run includes five freshly fitted chart pilots and 252
posterior evaluations. Its off-grid point discrepancy is **0.08685** versus the
unchanged **0.002** criterion. Normalization error is only 0.0001152 and finite
shape defect is 0.1164; neither substitutes for unresolved joint interpolation.

Automatic rich-cook screening adds coordinates 14 and 17 to the original 15 and
11. Three conditional fits reduce the central shape defect **0.6966 → 0.3574 →
0.1912**, crossing the existing 0.25 threshold. A separate three-fit audit records
those stage values under the owned-graph implementation. The main four-dimensional
integration still fails; passing the screening point does not validate every
conditional elsewhere in its domain.

### Online/offline agreement

The automatic one-probe chart initially remains unresolved even at degree 64.
With local fallback enabled, the 75 → 510 s arrival and its offline full-target
fit both resolve in **116 fits each**, within the **129-fit** budget. The local
joint-refinement indicator is **0.001907**, and the finite shape defect is
**0.05636**. Across the five recorded comparison points the maximum log-density
difference is zero at floating-point precision; conditional KL is approximately
5.14e-28. Caller state and process-noise anchors are preserved. Nine checks pass.

The reported **146.3 seconds covers both offline and online fits together**,
assimilation, and comparison. This is not an online latency qualification.
The earlier degree-16 and degree-64 global-only replay reports are retained;
agreement alone did not qualify either unresolved result.

### Common conditional-oracle comparison

`common_offgrid_final.json` evaluates nine common physical parameter points from
a fixed central affine Gauss grid, using nine extra conditional fits outside the
matched budgets. An earlier nine-fit audit is retained separately. Every method
uses the same corrected conditional oracle and innovation decoder. No points
are outside any compared support on this small grid.

| Representation | Largest absolute marginal log-mass error | Largest conditional KL |
|---|---:|---:|
| Previous local affine | 0.07735 | 0.06106 |
| Previous local nonlinear map | **0.01323** | **0.003924** |
| Sparse affine, degree 16 | 0.16982 | 0.04953 |
| Sparse nonlinear map, degree 16 | 0.11041 | 0.02230 |
| Sparse nonlinear map, degree 64 | 0.04235 | 0.01885 |

Higher degree helps, but this grid favors the previous local mapped method.
These are conditional approximation diagnostics, not true-posterior bounds or
independent cook coverage. They do not support replacing the local method for
the coupled cases.

### Why local refinement matters

A separate smooth four-way conditional-mean interaction is recovered in 105
evaluations, below the old four-dimensional 297-node initial-check floor. The
off-grid panels detect that interaction even when axis and cubic-frontier
surpluses are zero. Smooth polynomial log densities and conditional geometry
are favorable to the sparse hierarchy.

A generic counterexample uses a standard Gaussian marginal on [-4,4] and a
Gaussian conditional with mean `max(x - 0.137, 0)^2` and standard deviation 0.02.
This mean is continuously differentiable but has a curvature change. With the
same 129-fit budget, the local method resolves in 29 fits; the global degree-64
method stops unresolved after 69. A separate positive-quadrature comparison
against the analytic joint density gives total-variation estimates **0.00003081**
and **0.004338**, respectively. Refining that diagnostic quadrature changes
them by about 9e-18 and 1.7e-6. The sparse-plus-local fallback reuses these global
evaluations and passes in 94 total fits within the original budget; see `hybrid_control.json`.

The cook observable `_slab_temperature_at` in `estimator/src/semantic.jl` is a
piecewise cubic Hermite interpolant with continuous slope, so limited smoothness
is a plausible contributor to slow global approximation. This is an inference,
not proof that it explains every cook failure. Research on
[local hp-adaptive sparse interpolation](https://arxiv.org/abs/2404.02556)
addresses this general limitation. Extending local sparse refinement to multiple
dimensions is the next numerical task; raising global polynomial degree alone
has not resolved the coupled cooks.

## Verification and artifacts

The final regression suite passes **285 assertions** across nine test files.
Separate cook audits pass nine online/offline, five promotion-screening and 12
cached-geometry/normalization checks. The numerical safeguards preserve all
checked cached means, roots and normalizers. All 74 production/calibration
baseline files remain unchanged; backend `d31877a10f94`, physics `c2c7bbf8c0ec`.

Source archives distinguish the initial benchmark, later degree-64 numerical
safeguards, successful online fallback, and final implementation. The final
online-to-final change replaces a linear search for the conservative cell limit
with a bounded binary search; the default limit remains 128, explicitly tested.
JSON reports, worker logs, executed drivers, input hashes and local candidate
hashes preserve the experiment history. The empty duplicate `15b` driver came
from an unsuccessful edit after its first run; the complete `15c` and `15d`
drivers reproduce and refine the smoothness comparison.

The original `core_degree64.json` and `two_core_degree64.json` capacity runs used
the already-loaded benchmark module with an explicit degree-64 setting. Their
`sources` fields captured the files on disk after intermediate edits, rather
than the module's loaded code. They are retained as historical reports, not as
source-hash proof. Fresh `*_verified_capacity.json` runs load the final archived
implementation explicitly and record its hashes.
They reproduce both passing outcomes at 54 and 42 fits, taking 36.5 and 26.0
seconds, respectively. A separate artifact audit passes 259 checks, including
parsing the replay script and loading all four archived implementations.

Run `verify.ps1` here to check hashes, budgets, ownership and regression counts.
`reproduce.jl CASE METHOD OUTPUT.json [VERSION]` replays a cook with archived
sources (`benchmark`, `degree64`, `online`, or `final`). Its default is the original
benchmark. The serialized input snapshots and large candidate files remain in
the ignored local scratch directories; their hashes are recorded. Reproduction
requires those local inputs and the estimator Julia environment.
