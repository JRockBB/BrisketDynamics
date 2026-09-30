# Joint sampled estimation during and after a cook

`sampled_init(model)` now selects incremental non-Gaussian inference. `sampled_fit(model, readings)`
fits the full cook offline. Both retain joint parameter/state trajectories and use the same nonlinear
factor likelihood. `online_init` still selects the existing Gaussian approximation; callers must opt
into `sampled_init`. The existing Gaussian mock pilot has not been silently migrated, particularly
because its capped humidity readings require a censoring model the sampled backend does not yet support.

These are implemented inference paths, not a claim of established real-cook calibration or bounded
real-time cost. The evidence and remaining validation requirements are recorded below and in
[the generalization specification](estimator-generalization-validation.md).

**The first four-suite implementation comparison failed overall.** Full-suite, one-probe and two-probe
examples agreed with independent offline fits, but a pit/core example exhausted the sampling budget
and rejected its last arrival after about 94 minutes. Even accepted refreshes took minutes. This
implementation is not qualified for a live cook; see the [complete results](validation/2026-09-27-online/README.md).

## Online updates

```julia
using Random

sm = sampled_init(model; particles=1024, max_particles=4096,
                  replicates=8, moves=4, rng=MersenneTwister(42))
result = assimilate!(sm, batch)  # MeasurementRecord vector, chronological timestamps
if result.committed
    posterior = result.posterior
    intervals = posterior_quantile(posterior, [0.025, 0.5, 0.975]; physical=true)
else
    # The accepted posterior and graph remain at the previous arrival.
    # Inspect result.diagnostics; retain the batch for retry with an adequate work budget.
end

checkpoint = smoother_checkpoint(sm)
resumed = restore_checkpoint(checkpoint)  # includes coordinates, log weights, predictions and RNG
```

The common update propagates each retained trajectory through only the new intervals and evaluates
the new conditional observation likelihood. Static parameters remain static. Sensor discrepancy is
an OU process integrated out analytically, with its time correlation preserved. Repeated readings
therefore do not count as independent evidence about a persistent sensor offset.

Continuation keeps normalized log weights separately from the displayed posterior weights. A finite
log weight can exponentiate to zero in floating-point arithmetic; a later reading can still favor that
trajectory. Taking `log` of displayed weights would permanently discard it. Checkpoints and rollback
preserve the log weights, and incremental propagation retains those trajectories even when their
current displayed weight is zero.

For residual vector `r` and the lower-triangular whitening matrix `W = chol(K).L⁻¹`, the incremental
log likelihood is `-sum(abs2, (W*r)[new_rows])/2`, up to a common constant. Appending chronological
observations leaves old whitening rows unchanged. The unit test compares this increment to the full
nonlinear potential, including asynchronous readings and a repeated timestamp. Different sensor
counts and types enter through ordinary `SensorSpec` declarations, with no suite-specific settings.

If particle weights or estimated mean/tail precision become inadequate, a full-history tempered SMC
refresh regenerates the joint trajectories. The work budget doubles up to `max_particles`, measured
**per independent system**. Eight systems at 1,024 particles means 8,192 trajectories. A failed refresh
returns `committed=false`; exceptions also restore the accepted graph and posterior. Sampling consumes
randomness during a failed attempt. Restore a checkpoint if identical RNG continuation is required.
`assimilate!(sm, batch; refresh=true)` explicitly requests a full refresh.

Each independent system retains total weight `1/replicates`. Diagnostics use variation between systems
and unequal-weight variance floors, not just the final weights after resampling. They are estimated
numerical precision checks and can still miss unexplored modes. Neither a good ESS nor agreement with
another run establishes statistical calibration.

The Gaussian solver establishes deterministic process-covariance anchors and computational proposals.
Its covariance is not substituted for the sampled posterior. For a batch spanning multiple timestamps,
the graph solves each timestamp before anchoring the next interval, as the offline builder does.
This prevents transport packet grouping from changing the declared process model.

Full history, observation covariance, graph solves and refreshes still grow with elapsed cook length.
This is sequential importance updating with full-history refresh, **not SMC² or a constant-cost filter**.
The construction follows sequential importance and invariant-move principles described by
[Del Moral, Doucet and Jasra](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).
[SMC²](https://arxiv.org/abs/1101.1528) is a possible subsequent approach to sequential parameter/state
inference if full-history refresh costs prove unacceptable; it has not been implemented here.

## Offline reference

```julia
p = sampled_fit(model, all_readings; rng=MersenneTwister(43))
p.diagnostics.mc_precision_ok || error("offline posterior needs more sampling work")
summary = belief_at(p)

# Compare inference on the exact accepted online graph and its historical execution snapshots:
reference = sample_posterior_smc(sm.graph_backend; rng=MersenneTwister(44))
```

`sampled_fit` sorts a copy of the records and constructs the full graph chronologically. Returned
draws retain the independent prior/process coordinates needed to replay every trajectory. The optional
`sampled_init(model; mode=:offline)` interface performs a full posterior refit on every arrival.
`sampled_fit` is for one cook; cross-cook hierarchical learning is separate work.

Executed controls and events must be supplied. `update_execution!(sm; u=..., events=...)` replaces the
future execution record without rewriting the factors already accepted. For an exact online/offline
comparison after execution updates, use `sm.graph_backend`: rebuilding from a different final log can
change which control/event information was available when historical factors were constructed.

## Forecasts and decisions

```julia
# Actual joint draws, propagated through nonlinear dynamics, with future process disturbances.
forecast = posterior_predict(sm, [sm.posterior.t + 60, sm.posterior.t + 300];
                             rng=MersenneTwister(45))
future_intervals = posterior_quantile(forecast; physical=true)

# Every trajectory supplies its own state, parameters and accumulated cook history.
snap = belief_snapshot(sm; goals=goals_by_piece)
plan_result = replan_system(snap, schedule, goals_by_piece)
```

`posterior_predict` defaults to adding the model's future process disturbances. The optional `model`
argument supplies proposed future controls while retaining the same state layout and latent block.
Forecasts describe physical states; sensor measurement noise is not included. `process_noise=false`
is an explicit forecast conditional on deterministic future physics.

The sampled planner optimizes the weighted expected **complete goal loss** over joint scenarios.
Accumulated lethality, surface exposure and initial fuel mass are replayed separately for each history;
they are not replaced by their posterior means. Nonlinear outcome means and quantiles use the actual
scenario weights. Changing the definitions of historical integrals requires rebuilding the snapshot.

By default `belief_snapshot` retains all positive-weight trajectories. An explicit `n` uses stratified
resampling and records that approximation in provenance. This can reduce planning work substantially,
but its effect on the decision must be checked at larger budgets. A small scenario budget is not
qualified merely because inference had a high effective sample size.

**The current planner's objective and `system_plan_predictive` condition on deterministic future
physics.** They include current state, parameter and history uncertainty, but do not yet integrate
future process disturbances; prediction metadata says so explicitly. `posterior_predict` includes
those disturbances for state forecasts. Closing this gap, checking decision stability and validating
an entire adaptive cook remain necessary before claiming uncertainty-calibrated control.

## Supported inputs and evidence

The [versioned validation record](validation/2026-09-27-online/README.md) contains source hashes,
completed test counts and explicit status for the interrupted optional legacy regression.

The sampled paths currently require static latents and uncensored observations with declared noise.
The incremental path requires chronological arrivals; use offline fitting for delayed records.
One or two probes are supported and do not acquire invented measurements of unobserved quantities.
Unknown action histories, unknown sensor calibration, censoring, drifting parameters and population
transfer require explicit model support. Sampled planning currently rejects ROM discrepancy models.

Correctness checks include analytic Gaussian posteriors with correlated repeated sensor readings,
invariant sampling kernels with known nonlinear moments, preservation of unobserved priors, exact
incremental/full likelihood comparisons, chronology/batching, transaction rollback, reproducible
checkpoint continuation, joint weighted planning losses and derivatives, and future process covariance.
These checks use independent mathematical oracles rather than only replaying expected implementation
outputs. Planner tests also caught and fixed missing-addition NaNs and plan encode/decode changes to
valid decisions near time/fraction boundaries.

`estimator/scripts/validate_sampled_online.jl` declares four 30-minute implementation comparisons in
advance: full suite, pit/core with dropout, one irregularly sampled core probe, and two independently
positioned probes with alternating arrivals. These controls use a four-node spatial model; production
resolution and hours-long runtime are not established by them. Seeds 22001–22004 are dedicated to these comparisons.
Each uses identical sampling settings and compares the online posterior to independent full-history
sampling on the same graph. Mean, CDF and joint second-moment differences are reported relative to estimated Monte Carlo
error. All attempts and runtimes are retained. This is one cook per configuration, not a calibration study.

```powershell
julia --threads=8 --project=estimator estimator/scripts/validate_sampled_online.jl sampled_online_comparison.json
```

The separate reserved seeds 21001–21064 remain unused for the larger predeclared calibration study.
Posterior rank checks must use independent cooks, as motivated by
[simulation-based calibration](https://arxiv.org/abs/1804.06788). The existing calibration trace must not
be reused as held-out real-cook evidence.
