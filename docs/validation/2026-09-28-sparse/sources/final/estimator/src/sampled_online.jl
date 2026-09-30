# An explicit inference backend: the graph optimizer constructs proposals, and accepted beliefs come
# from the nonlinear posterior. Full history is retained; no Gaussian marginal prior replaces it.
export SampledSmoother, sampled_init

mutable struct SampledSmoother
    graph_backend::FixedLagSmoother
    posterior::SampledPosterior
    rng::AbstractRNG
    particles::Int
    max_particles::Int
    replicates::Int
    moves::Int
    maxstages::Int
end

"""
    sampled_init(model; mode=:online, particles=1024, max_particles=4096, replicates=8, moves=4, rng=MersenneTwister(0), kwargs...)

A non-Gaussian estimator using the same sensor/factor declarations as `online_init`. The default
`mode=:online` reuses weighted trajectories between readings; `mode=:offline` refits the complete
posterior on every arrival as a reference. Both retain full history for exact-target refreshes.
`assimilate!` accepts an update only after posterior sampling meets its estimated Monte Carlo precision
checks. It doubles the particle count, up to `max_particles`, when needed; failed updates roll back.
The graph optimizer supplies proposals, not reported uncertainty. `belief_at` reads the sampled posterior,
and `sm.posterior.draws` retains the joint distribution for subsequent prediction or control.

This backend currently supports static latents and uncensored readings. It is computationally heavier
than the fixed-lag Gaussian backend and does not feed its Gaussian ALS calibration recursion. The
existing `online_init` API remains available for that approximation. Sampling precision checks are not
a substitute for validating model error on independent real cooks.
"""
function sampled_init(model::SystemModel;particles::Int=1024,max_particles::Int=4096,replicates::Int=8,
                      moves::Int=4,maxstages::Int=100,rng::AbstractRNG=MersenneTwister(0),mode::Symbol=:online,kwargs...)
    mode in (:online,:offline) || throw(ArgumentError("sampled mode must be :online or :offline"))
    mode===:online && return particle_init(model;particles,max_particles,replicates,moves,maxstages,rng,kwargs...)
    2<=particles<=max_particles && replicates>=4 && moves>=1 && maxstages>=1 ||
        throw(ArgumentError("invalid particle, replicate, move, or stage limits"))
    (haskey(kwargs,:lag) || haskey(kwargs,:als)) &&
        throw(ArgumentError("sampled inference retains full history and does not use Gaussian ALS"))
    inner=online_init(model;lag=typemax(Int),kwargs...)
    n=max(2048,particles*replicates,32*(length(model.blk)+model.bl.n))
    posterior=sample_posterior(inner;n,rng)
    posterior.diagnostics.mc_precision_ok || error("initial prior sampling failed its precision checks")
    return SampledSmoother(inner,posterior,rng,particles,max_particles,replicates,moves,maxstages)
end

belief_at(sm::SampledSmoother)=belief_at(sm.posterior)
window_nodes(sm::SampledSmoother)=window_nodes(sm.graph_backend)

# Establish each interval's deterministic covariance anchor after solving all preceding readings.
# Bundling several timestamps into one arrival must not change the graph's declared process model.
function _append_sampled_measurements!(inner,recs;verbose=false)
    first=1;proposal=nothing
    while first<=length(recs)
        last=first
        while last<length(recs) && recs[last+1].t==recs[first].t;last+=1;end
        for r in recs[first:last];add_measurement!(inner,r);end
        proposal=solve_window!(inner;verbose)
        first=last+1
    end
    return proposal
end

function assimilate!(sm::SampledSmoother,recs::AbstractVector{MeasurementRecord};verbose::Bool=false)
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    inner=sm.graph_backend
    for rec in recs
        spec=sensor(inner.model.log,rec.sensor)
        isfinite(rec.t) && isfinite(rec.value) || throw(ArgumentError("a measurement must have a finite time and value"))
        censored_at(spec,rec) && throw(ArgumentError("sampled inference currently requires uncensored readings"))
    end
    saved=_transaction_snapshot(inner);start=time();count=sm.particles;last=nothing
    try
        proposal=_append_sampled_measurements!(inner,sort(collect(recs);by=r->r.t,alg=MergeSort);verbose)
        while true
            last=sample_posterior_smc(inner;particles=count,replicates=sm.replicates,moves=sm.moves,
                                      maxstages=sm.maxstages,rng=sm.rng,verbose)
            if last.diagnostics.mc_precision_ok
                belief=belief_at(last)
                record=merge(belief,(committed=true,converged=true,proposal_converged=proposal.converged,
                    inference=:tempered_smc,diagnostics=last.diagnostics,wall=time()-start))
                _push_record!(inner,record)
                sm.posterior=last
                return merge(record,(posterior=last,))
            end
            count>=sm.max_particles && break
            count=min(2count,sm.max_particles)
        end
        _restore_transaction!(inner,saved)
        return (t=maximum(rec.t for rec in recs),committed=false,converged=false,reason=:monte_carlo_precision,
                inference=:tempered_smc,diagnostics=last.diagnostics,belief=belief_at(sm),wall=time()-start)
    catch
        _restore_transaction!(inner,saved)
        rethrow()
    end
end
assimilate!(sm::SampledSmoother,rec::MeasurementRecord;kwargs...)=assimilate!(sm,[rec];kwargs...)
