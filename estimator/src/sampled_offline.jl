export sampled_fit

"""
    sampled_fit(model, records; particles=1024, max_particles=4096, replicates=8, moves=4, kwargs...)

Fit a single cook's full nonlinear posterior offline. A copy of the readings is sorted by time and
same-time readings are grouped. The graph is constructed chronologically under the supplied execution
log; deterministic Gaussian solves supply anchors and proposals, while returned uncertainty is sampled.
For an existing online graph, use `sample_posterior_smc(sm.graph_backend)` to retain its exact historical
execution/model snapshots. Static latents and uncensored observations are currently required.

The returned `SampledPosterior` retains trajectories for prediction and planning. Check
`diagnostics.mc_precision_ok`; exhausting `max_particles` returns a failed diagnostic, never a Gaussian
substitute. `belief_at` and sampled planning refuse an unqualified posterior.
"""
function sampled_fit(model::SystemModel,records::AbstractVector{MeasurementRecord};particles::Int=1024,
                     max_particles::Int=4096,replicates::Int=8,moves::Int=4,maxstages::Int=100,
                     rng::AbstractRNG=MersenneTwister(0),verbose::Bool=false,kwargs...)
    2<=particles<=max_particles && replicates>=4 && moves>=1 && maxstages>=1 ||
        throw(ArgumentError("invalid posterior work limits"))
    (haskey(kwargs,:lag) || haskey(kwargs,:als)) && throw(ArgumentError("sampled offline inference requires full history without Gaussian ALS"))
    recs=sort(collect(records);by=r->r.t,alg=MergeSort)
    for r in recs
        isfinite(r.t) && isfinite(r.value) || throw(ArgumentError("measurements must be finite"))
        censored_at(sensor(model.log,r.sensor),r) && throw(ArgumentError("sampled fitting currently requires uncensored readings"))
    end
    sm=online_init(model;lag=typemax(Int),kwargs...)
    _append_sampled_measurements!(sm,recs;verbose)
    count=particles
    while true
        p=sample_posterior_smc(sm;particles=count,replicates,moves,maxstages,rng,verbose)
        (p.diagnostics.mc_precision_ok || count>=max_particles) && return p
        count=min(2count,max_particles)
    end
end
