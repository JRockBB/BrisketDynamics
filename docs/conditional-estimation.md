# Deterministic conditional estimation

This is an experimental offline and online backend. It retains a continuous,
non-Gaussian joint distribution instead of reporting one Gaussian over the cooking
state. It has **not established statistical consistency on real cooks** and does
not replace the live controller. The validation record includes failed cook updates.

## Representation

The distribution has three parts: a continuous density over a small parameter block;
a joint Gaussian over the remaining parameters, initial state and process innovations
conditional on that block; and the nonlinear cooking model that maps them into states.
The conditional means and covariances vary across the integrated block. This retains
skewness, changing conditional variance and curved parameter/state dependence without
reducing them to a few higher moments.

For a transition, the change of coordinates is

```text
innovation = W * (next_state - physics(previous_state, parameters))
next_state = physics(previous_state, parameters) + W \ innovation
```

Every component uses the same anchored process whitener `W`. The transformation is
triangular with a constant Jacobian determinant. With Gaussian parameter coordinates,
initial uncertainty and process noise, the unobserved generative prior is exactly
Gaussian in these coordinates even when its state distribution is strongly curved.
Observations can still make the conditional innovation distribution non-Gaussian.

At each integration node, exact coordinate elimination fixes the selected parameters.
The remaining factor graph is optimized, and its joint covariance root and volume are
computed with column-scaled QR. Constant conditioned prior rows remain in the original
evidence calculation but are omitted from optimization. Otherwise a constant loss
can perturb floating-point line searches despite carrying no information about the
remaining coordinates. Node evidence uses Gauss–Newton curvature, not the exact
observed Hessian; this remains an approximation after nonlinear observations.

A fitted quadratic log marginal and affine conditional-mean trend represent arbitrary
linear Gaussian updating exactly, including between nodes. Local cubic interpolation
represents the remainders, and interpolated log Cholesky diagonals preserve positive
conditional covariance. The result is a continuous density, not a lattice of possible
parameter values. Remainders are bounded outside the integration grid; distant modes
are not thereby ruled out.

The conditional-integration approach is related to
[Rue, Martino and Chopin's nested Laplace method](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2008.00700.x).
This implementation adds a physics-defined nonlinear state transformation. It is not
R-INLA, nor is this nonlinear cooking graph a directly applicable latent Gaussian model.

## Offline and online interfaces

```julia
# Uses declared sensors, noise and calibrated priors. No truth or RNG is supplied.
sm = deterministic_init(model)
result = assimilate!(sm, arriving_readings)
if result.committed
    distribution = belief_at(sm)  # ConditionalDensity, not a state Gaussian
end

# Full-history offline inference uses the same distribution and update semantics.
fit = deterministic_fit(model, readings)

# An offline calibration analysis may explicitly specify integration directions.
sm = deterministic_init(model; axes=[:a0, :k_spread])
```

Automatic selection uses departures of the exact graph log density from a conditional
quadratic to choose parameter directions. It contains no branches for sensor names,
counts or validation cases. The initial axis is the first latent coordinate; directions
are reconsidered when observations expose conditional shape error. Up to three axes
are supported. An explicit `axes` declaration keeps that block fixed. The existing
ignition calibration motivates the last example but does not establish that ignition
alone suffices for probe-only cooks.

Banks of conditional graphs advance with chronological measurements. Accumulated log
weights use ratios of new and old conditional integrals, rather than repeatedly counting
old observations. Independent node solves run on separate graphs; reductions have a
fixed order. Difficult nodes can restart from nearby converged conditional modes.
Failed nodes are neither silently discarded nor accepted because their weights are small.

The default retains full nonlinear history. A caller can explicitly set a finite `lag`,
but Gaussian marginalization can lose the nonlinear dependence this representation is
intended to preserve. Rebuilding axes or recentering the integration grid requires the
original factors, so it is disabled after that compression. Full-history online storage
and computation grow with the cook; bounded-memory nonlinear compression is unresolved.

Before accepting an arrival the backend compares successive continuous densities in
common innovation coordinates, checks numerical integration of that comparison, and
checks conditional shape. It can refine and recenter the integration rule by refitting
the original full graph. The default numerical tolerances are 0.025 for the estimated
TV bound, one quarter of that for its integration change, and one tenth for the log
normalizer change. A weighted one-standard-deviation shape defect above 0.25 triggers
axis adaptation or rejection. These are approximation diagnostics, not coverage claims.
Budgets are explicit (`orders`, `max_order`, `max_nodes`, `max_axes`).

An unresolved arrival returns `committed=false` with a reason and diagnostics, restoring
the entire previous estimator, including a multi-time batch. A returned `candidate`
is unaccepted diagnostic output. No MC fallback, hidden Gaussian fallback, noise fitting,
or change to the externally calibrated prior is performed during assimilation.

## Distribution queries

`conditional_logpdf(distribution, x)` evaluates the normalized approximate joint density
in the graph's variable ordering, including the nonlinear change-of-variables determinant.
`conditional_cdf(distribution, key, index, value)` integrates a parameter or initial-state
marginal in its latent coordinate; it does not treat grid nodes as posterior samples.
A CDF for a state generated by the nonlinear dynamics needs further predictive integration
and is explicitly unsupported by that function.

`conditional_density_distance(a, b)` estimates a TV bound between two approximations
from their marginal density difference and conditional Gaussian KL/Pinsker bounds.
It requires identical nonlinear decoders, including control history and covariance
anchors. Its integration rule refines separately from the posterior's conditional-node
rule. It is **not** a bound on distance from the true posterior.

The lower-level `conditional_fit(graph; axes, ...)` and `conditional_init` APIs retain
conditional node banks. `conditional_transport` preserves the nonlinear state mapping.
Their `conditional_points`, `conditional_expectation` and `conditional_moments` use
positive third-degree spherical-radial cubature. Those nodes are integration points,
not independent samples or validated tail quantiles. In particular, do not estimate
95% intervals by taking empirical quantiles of that cubature rule. Controller prediction
and decision queries still need a separately verified nonlinear integration budget.

## Calibration and validation

Calibration stays outside live inference: the existing MAP/Laplace, nested ignition
analysis, population pooling and noise fitting are unchanged. The prior export is not
replaced. MC remains an optional independent reference/diagnostic; this change does not
attribute better calibration to the earlier MC implementation.

Analytic tests cover correlated Gaussian densities/CDFs, a normalized curved
heteroscedastic posterior, small correlated contrasts, preserved evidence through
elimination, and the cooking model's generative-prior transport identity. Online tests
cover unobserved probe-depth invariance, batching, ownership, shared decoder checks and
rollback. Development cooks include rich sensors, pit/core dropout, one probe and two
asynchronous probes. They use the previously exposed 22001–22004 cases; the reserved
21001–21064 calibration seeds remain untouched.

Commands:

```powershell
julia --threads=4 --project=estimator estimator/test/runtests.jl test_conditional.jl test_conditional_density.jl test_conditional_online.jl test_deterministic_online.jl
julia --threads=4 --project=estimator estimator/scripts/validate_deterministic.jl deterministic_comparison.json
```

See [the validation record](validation/2026-09-28-conditional/README.md) for exact source
fingerprints, results and failures. Passing implementation tests, resolving quadrature,
or small shape defects does not establish statistical consistency. Finite shape probes
check only a few high-weight components and parameter directions, and can miss other
modes or tails. Remaining qualification includes independent joint/predictive reference
comparisons, held-out coverage across sensor suites, real-cook model discrepancy,
long-history accuracy and controller latency. These limits are material to deployment.

The follow-up [curvature audit](validation/2026-09-28-curvature/README.md) separates
Gauss–Newton curvature error from higher-order conditional shape. It preserves all
four exposed failures as reusable graph snapshots, checks numerical ODE accuracy,
and compares full observed-Hessian counterfactuals. Curvature correction helps the
checked sparse-sensor conditionals but does not resolve the rich-sensor and pit/core
shape failures. It is diagnostic work; the live estimator remains unchanged.

The subsequent [adaptive integration work](validation/2026-09-28-adaptive/README.md)
adds a positive continuous density on locally refined cells and a shared
full-history graph adapter for offline and online candidates. Innovation-coordinate
mode finishing removes a conditional-solve resolution floor in the exposed sparse
sensor cases. Observed curvature updates covariance and evidence together. These
are experimental qualification tools; richer-suite shape, tails, complete-cook
coverage and online cost remain acceptance requirements.
The final adaptive source resolves the checked one-probe target through the
1,800-second synthetic horizon. The two-probe continuation still fails conditional
shape even after full curvature correction; integrating both probe depths exhausts
the current budget. These failures remain explicit. Online/offline candidate
agreement and a common accurate dynamics decoder are verified, without changing
the live defaults or claiming statistical consistency.

The [nonlinear-coordinate comparison](validation/2026-09-28-transport/README.md)
adds a deterministic triangular map learned from conditional pilots of the
original graph, including its Jacobian and construction cost. On the late
two-probe target it lowers the last complete refinement indicator from 0.194
to 0.137, against an unchanged 0.002 criterion. Rich-sensor and four-direction
dropout targets remain unresolved; the rich target also fails conditional shape.
The map therefore does not establish consistency or online readiness. All 172
final focused assertions pass, and a one-direction online replay matches the
offline candidate, but multi-axis cook reuse, domain coverage, prediction,
held-out coverage, and long-history cost still need qualification. The next
efficiency experiment is dimension-adaptive sparse integration that can expand
the integrated block when the remaining conditional shape fails. Calibration,
live defaults, and reserved cook cases remain unchanged.
