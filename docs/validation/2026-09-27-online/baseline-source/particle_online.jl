# Sequential importance updates with exact-target full-history refreshes. Static parameters remain
# static: no Liu-West shrinkage, artificial random walk, or Gaussian projection of the posterior.
export ParticleSmoother, particle_init

mutable struct ParticleSmoother
    graph_backend::FixedLagSmoother
    posterior::SampledPosterior
    predictions::Matrix{Float64}
    rng::AbstractRNG
    particles::Int
    max_particles::Int
    replicates::Int
    moves::Int
    maxstages::Int
    refreshes::Int
    increments::Int
end

"""
    particle_init(model; particles=1024, max_particles=4096, replicates=8, moves=4, kwargs...)

Incremental joint nonlinear inference. Accepted trajectories and their weights survive each arrival;
only new state intervals and conditional observation likelihoods are evaluated. Sensor OU errors are
integrated out with their time correlation intact. When sampling precision is inadequate, a full-history
SMC refresh regenerates trajectories and static parameters under the same graph, increasing work up to
`max_particles`. A failed refresh rolls the whole arrival back; it does not publish Gaussian uncertainty.

Full history is retained for refreshes and sampled planning histories. Refresh cost therefore grows
with history; this is not a constant-cost filter. Chronological, uncensored observations and static
latents are currently required. Sampling precision is not evidence of physical-model calibration.
"""
function particle_init(model::SystemModel;particles::Int=1024,max_particles::Int=4096,replicates::Int=8,
                       moves::Int=4,maxstages::Int=100,rng::AbstractRNG=MersenneTwister(0),kwargs...)
    2<=particles<=max_particles && replicates>=4 && moves>=1 && maxstages>=1 ||
        throw(ArgumentError("invalid particle, replicate, move, or stage limits"))
    (haskey(kwargs,:lag) || haskey(kwargs,:als)) &&
        throw(ArgumentError("particle inference retains full history and does not use Gaussian ALS"))
    inner=online_init(model;lag=typemax(Int),kwargs...);ref=_posterior_transport(inner)
    count=particles
    while true
        n=count*replicates;U=randn(rng,ref.n,n);X=zeros(length(ref.zm)+model.bl.n,n)
        Threads.@threads for j in 1:n;X[:,j]=_transport_draw(ref,view(U,:,j)).draw;end
        p=_replicate_posterior(ref,X,fill(1/n,n),replicates;history=PosteriorHistory(ref,U),
            extra=(proposal=:exact_prior,update=:initial,refreshed=false))
        p.diagnostics.mc_precision_ok && return ParticleSmoother(inner,p,zeros(0,n),rng,count,max_particles,
            replicates,moves,maxstages,0,0)
        count>=max_particles && throw(ArgumentError("initial particle budget is insufficient for estimated precision"))
        count=min(2count,max_particles)
    end
end

belief_at(sm::ParticleSmoother)=belief_at(sm.posterior)
window_nodes(sm::ParticleSmoother)=window_nodes(sm.graph_backend)

function _validate_particle_extension(old,new)
    length(new.ks)>=length(old.ks) && new.ks[1:length(old.ks)]==old.ks && new.zm==old.zm && new.zs==old.zs ||
        throw(ArgumentError("a sequential update must extend the same latent/state history"))
    for k in old.ks[2:end]
        old.off[k]==new.off[k] && old.roots[k]==new.roots[k] ||
            throw(ArgumentError("a sequential update changed a retained state distribution"))
    end
    m=length(old.fs)
    length(new.fs)>=m || throw(ArgumentError("observations cannot be removed from a particle update"))
    for i in 1:m
        a,b=old.fs[i],new.fs[i]
        (a.key_y,a.value,a.σ)==(b.key_y,b.value,b.σ) || throw(ArgumentError("a retained observation changed"))
    end
    isapprox(new.W[1:m,1:m],old.W;rtol=1e-11,atol=1e-12) && new.obsbias[1:m]==old.obsbias ||
        throw(ArgumentError("the retained sensor likelihood changed"))
    return nothing
end

# Cholesky whitening is causal in observation order. Its old rows are unchanged on append, so the
# new rows alone give log p(new readings | retained readings, trajectory), up to a common constant.
function _transport_extend(old,ref,u,draw,predictions)
    nz=length(ref.zm);z=copy(draw[1:nz]);y=copy(draw[nz+1:end]);m=length(old.fs)
    pred=vcat(predictions,zeros(length(ref.fs)-m))
    function observe!(k)
        for i in m+1:length(ref.fs)
            obs=ref.fs[i];obs.key_y==k || continue
            pred[i]=_obs_pred(obs,vcat(z,y,obs.key_d===nothing ? Float64[] : [0.0]))
        end
    end
    observe!(last(old.ks))
    for k in ref.ks[length(old.ks)+1:end]
        f=ref.generators[k]
        y=propagate(f.model,y,z,f.t0,f.t1;first_slice=f.first)[1]+ref.roots[k]*view(u,ref.off[k])
        observe!(k)
    end
    residuals=pred+ref.obsbias-[f.value for f in ref.fs]
    wr=ref.W[m+1:end,:]*residuals
    return (draw=vcat(z,y),predictions=pred,log_increment=-sum(abs2,wr)/2)
end

function _incremental_posterior(sm::ParticleSmoother,ref)
    old=sm.posterior;history=old.history;previous=history.reference
    _validate_particle_extension(previous,ref)
    n=length(old.weights);count=n÷sm.replicates
    U=vcat(history.coordinates,randn(sm.rng,ref.n-previous.n,n))
    X=similar(old.draws);P=zeros(length(ref.fs),n);logw=log.(old.weights)
    Threads.@threads for j in 1:n
        r=_transport_extend(previous,ref,view(U,:,j),view(old.draws,:,j),view(sm.predictions,:,j))
        all(isfinite,r.draw) && all(isfinite,r.predictions) && isfinite(r.log_increment) ||
            error("non-finite incremental posterior trajectory")
        X[:,j]=r.draw;P[:,j]=r.predictions;logw[j]+=r.log_increment
    end
    w=zeros(n)
    for replicate in 1:sm.replicates
        block=(replicate-1)*count+1:replicate*count;hi=maximum(logw[block])
        isfinite(hi) || error("all sequential particle weights vanished")
        w[block]=exp.(logw[block].-hi);w[block]./=sum(w[block])*sm.replicates
    end
    p=_replicate_posterior(ref,X,w,sm.replicates;history=PosteriorHistory(ref,U),
        extra=(proposal=:sequential_importance,update=:incremental,refreshed=false,weight_ess=inv(sum(abs2,w)),
            propagated_intervals=n*(length(ref.ks)-length(previous.ks))))
    return p,P
end

function _posterior_predictions(p)
    h=p.history;P=zeros(length(h.reference.fs),length(p.weights))
    Threads.@threads for j in axes(P,2)
        r=_transport_draw(h.reference,view(h.coordinates,:,j))
        isapprox(r.draw,p.draws[:,j];rtol=1e-10,atol=1e-10) || error("retained trajectory does not replay its endpoint")
        P[:,j]=r.predictions
    end
    return P
end

function assimilate!(sm::ParticleSmoother,recs::AbstractVector{MeasurementRecord};verbose::Bool=false,refresh::Bool=false)
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    inner=sm.graph_backend
    times=[r.t for r in recs]
    all(isfinite,times) && issorted(times) && first(times)>=sm.posterior.t ||
        throw(ArgumentError("particle arrivals must be finite and chronological; use the offline reference for delayed data"))
    for rec in recs
        spec=sensor(inner.model.log,rec.sensor)
        isfinite(rec.value) || throw(ArgumentError("measurement values must be finite"))
        censored_at(spec,rec) && throw(ArgumentError("particle inference currently requires uncensored readings"))
    end
    saved=_transaction_snapshot(inner);start=time();refreshed=false;count=sm.particles
    try
        proposal=_append_sampled_measurements!(inner,recs;verbose) # deterministic anchors/proposal, never reported uncertainty
        ref=_posterior_transport(inner)
        p,predictions=refresh ? (nothing,nothing) : _incremental_posterior(sm,ref)
        sequential_ok=!refresh && p.diagnostics.mc_precision_ok && p.diagnostics.weight_ess>=max(400,20length(p.mean))
        if !sequential_ok
            refreshed=true
            while true
                p=sample_posterior_smc(inner;particles=count,replicates=sm.replicates,moves=sm.moves,
                    maxstages=sm.maxstages,rng=sm.rng,verbose)
                p.diagnostics.mc_precision_ok && break
                count>=sm.max_particles && break
                count=min(2count,sm.max_particles)
            end
            if !p.diagnostics.mc_precision_ok
                _restore_transaction!(inner,saved)
                return (t=last(times),committed=false,converged=false,reason=:monte_carlo_precision,
                    inference=:sequential_smc,diagnostics=p.diagnostics,belief=belief_at(sm),wall=time()-start)
            end
            predictions=_posterior_predictions(p)
        end
        diagnostics=merge(p.diagnostics,(update=refreshed ? :full_history_refresh : :incremental,refreshed=refreshed,
            refreshes=sm.refreshes+Int(refreshed),increments=sm.increments+Int(!refreshed)))
        p=SampledPosterior(p.t,p.model,p.draws,p.physical,p.weights,p.mean,p.covariance,diagnostics,p.history)
        record=merge(belief_at(p),(committed=true,converged=true,proposal_converged=proposal.converged,
            inference=:sequential_smc,diagnostics=diagnostics,wall=time()-start))
        _push_record!(inner,record)
        sm.posterior=p;sm.predictions=predictions;sm.particles=count
        sm.refreshes+=Int(refreshed);sm.increments+=Int(!refreshed)
        return merge(record,(posterior=p,))
    catch
        _restore_transaction!(inner,saved)
        rethrow()
    end
end
assimilate!(sm::ParticleSmoother,rec::MeasurementRecord;kwargs...)=assimilate!(sm,[rec];kwargs...)

function update_execution!(sm::Union{ParticleSmoother,SampledSmoother};kwargs...)
    update_execution!(sm.graph_backend;kwargs...)
    p=sm.posterior
    sm.posterior=SampledPosterior(p.t,sm.graph_backend.model,p.draws,p.physical,p.weights,p.mean,p.covariance,p.diagnostics,p.history)
    return sm
end

"Versioned in-memory continuation of the graph, weighted trajectories, cached readings and RNG."
struct SampledCheckpoint{T}
    version::Int
    physics::String
    backend::String
    smoother::T
end
smoother_checkpoint(sm::Union{ParticleSmoother,SampledSmoother})=
    SampledCheckpoint(1,physics_fingerprint(),backend_fingerprint(),deepcopy(sm))
function restore_checkpoint(cp::SampledCheckpoint)
    cp.version==1 && cp.physics==physics_fingerprint() && cp.backend==backend_fingerprint() ||
        throw(ArgumentError("sampled checkpoint version, physics or backend does not match"))
    return deepcopy(cp.smoother)
end
function restore_checkpoint!(sm::T,cp::SampledCheckpoint{T}) where {T<:Union{ParticleSmoother,SampledSmoother}}
    saved=restore_checkpoint(cp)
    for name in fieldnames(T);setfield!(sm,name,getfield(saved,name));end
    return sm
end
