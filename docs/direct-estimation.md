The direct posterior API is experimental. **It has not fixed statistical consistency and is not an accepted online estimator.** The numerical experiments and failed checks are recorded in [the direct-integration report](validation/2026-09-28-direct/README.md).

It evaluates the original nonlinear factor-graph target in independent parameter, initial-state and process-innovation coordinates. Gaussian OU sensor histories are integrated out analytically. A local Gaussian or a fitted nonlinear map supplies integration coordinates; its exact density/Jacobian correction is included at every point. Means and covariances are summaries of this target, not its definition. Signed sparse-quadrature coefficients are not probability weights.

After loading BrisketEstimator, explicitly include the script:

```julia
const E = BrisketEstimator
include("estimator/scripts/direct_online.jl")
stream = direct_init(model; max_evaluations=100_000, max_seconds=600.)
result = E.assimilate!(stream, readings)
if result.committed
    p = E.belief_at(stream)
    query = direct_cdf(p, coordinate_index, threshold)
    # Use query.value only if query.resolved is true.
end
```

These budgets are numerical limits, not a real-time performance claim. An unsuccessful update retains the previous posterior and timestamp; the caller must preserve the rejected readings for retry or record the gap. There is no Gaussian fallback. direct_fit uses the same graph-construction path offline. direct_forecast retains the full history and leaves the live estimator untouched; it assumes the model's declared control log. Forecast uncertainty and dynamic online acceptance have not been qualified on independent cooks.

Every fit reports separate numerical status and statistically_consistent=false. Every CDF or predictive CDF has its own error estimate and resolution flag. A resolved density does not automatically resolve a new query. direct_logpdf rejects unresolved normalization unless allow_unresolved=true is explicitly supplied for diagnostics. direct_radius similarly guards unresolved moments; a radius computed from non-Gaussian moments is not a chi-square confidence region.

By default the fitter requests joint parameter/state-coordinate moments. moments=false requests only normalization; it does not supply usable moments. physical_moments=true additionally requests decoded physical mean and variance. Unrequested summaries contain NaN internally and must not be consumed; diagnostics identify the requested/resolved quantities. Report serialization converts unavailable numbers to JSON null. Physical credible intervals must come from the distribution and monotone coordinate transformation, not from a Gaussian mean plus standard deviation.

The optional transport_axes setting chooses at most three pilot axes using the original likelihood geometry. It never selects from truth, test seed or sensor-suite names, and it does not truncate the other dimensions. curvature_rotation is an optional full-dimensional rotation. Neither option passed the difficult development integration checks. They are retained for diagnosis, not recommended production settings.

Supported factors currently require full history, static normal or normal-score parameter priors, uncensored Gaussian sensor observations and known executed controls. Drift, hierarchy, censoring and Gaussian marginal-history factors are rejected. A single probe or two probes use the same declared SensorSpec path as larger suites; statistical performance across those suites remains unproven.

Focused checks can be reproduced from the repository root:

```powershell
julia --startup-file=no --threads=4 --project=estimator estimator/test/test_direct_quadrature.jl
julia --startup-file=no --threads=4 --project=estimator estimator/test/test_direct_transport.jl
julia --startup-file=no --threads=4 --project=estimator estimator/test/test_direct_posterior.jl
julia --startup-file=no --threads=4 --project=estimator estimator/test/test_consistency_direct.jl
```

The marginal-rank harness is an incomplete development tool. It cannot declare an overall consistency pass. Qualification still requires joint and data-dependent ranks, predictive tests, retained failures, independent cook replication, held-out sensor suites, online performance and independent real-cook references.