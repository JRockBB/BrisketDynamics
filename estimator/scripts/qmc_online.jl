isdefined(@__MODULE__,:qmc_posterior_fit) || include(joinpath(@__DIR__,"qmc_posterior.jl"))

mutable struct QMCSmoother
    anchor::E.FixedLagSmoother
    posterior::E.SampledPosterior
    predictions::Matrix{Float64}
    logweights::Vector{Float64}
    seeds::Vector{UInt64}
    uniform_dimensions::Int
    rng::AbstractRNG
    settings::NamedTuple
    refreshes::Int
    increments::Int
end

function qmc_init(model;points=256,max_points=4096,replicates=8,max_seconds=60.,
        update_seconds=10.,rng=MersenneTwister(0),kwargs...)
    (haskey(kwargs,:lag)||haskey(kwargs,:als)) && throw(ArgumentError("QMC retains nonlinear history and keeps calibration external"))
    update_seconds>0 || throw(ArgumentError("update budget must be positive"))
    anchor=E.online_init(model;lag=typemax(Int),kwargs...)
    owned=deepcopy(rng)
    result=qmc_posterior_fit(anchor;points,max_points,replicates,max_seconds,rng=owned)
    result.diagnostics.resolved || error("initial QMC posterior did not meet numerical precision")
    settings=(points=points,max_points=max_points,replicates=replicates,update_seconds=Float64(update_seconds))
    QMCSmoother(anchor,result.posterior,result.predictions,result.logweights,result.cache.seeds,
        result.cache.uniform_dimensions,owned,settings,0,0)
end
E.belief_at(sm::QMCSmoother)=E.belief_at(sm.posterior)
E.belief_snapshot(sm::QMCSmoother;kwargs...)=E.belief_snapshot(sm.posterior;kwargs...)
E.posterior_predict(sm::QMCSmoother,times;kwargs...)=E.posterior_predict(sm.posterior,times;kwargs...)

function _qmc_increment(sm::QMCSmoother,ref;deadline)
    old=sm.posterior;previous=old.history.reference
    E._validate_particle_extension(previous,ref)
    R=sm.settings.replicates;n=length(old.weights)÷R
    additional=ref.n-previous.n;nd=sm.uniform_dimensions+additional
    U=vcat(old.history.coordinates,zeros(additional,n*R))
    if additional>0
        for r in 1:R
            time()<deadline || throw(QMCTimeBudget())
            b=(r-1)*n+1:r*n
            # Continue the same net in unused independent uniform coordinates.
            net=uniform_net(n,nd;rng=MersenneTwister(sm.seeds[r]))
            U[previous.n+1:end,b]=-sqrt(2).*E.erfcinv.(2 .*net[sm.uniform_dimensions+1:end,:])
        end
    end
    X=similar(old.draws);P=zeros(length(ref.fs),n*R);lw=copy(sm.logweights)
    for firstpoint in 1:256:n*R
        time()<deadline || throw(QMCTimeBudget())
        lastpoint=min(n*R,firstpoint+255)
        Threads.@threads for j in firstpoint:lastpoint
            out=E._transport_extend(previous,ref,view(U,:,j),view(old.draws,:,j),view(sm.predictions,:,j))
            all(isfinite,out.draw)&&isfinite(out.log_increment) || error("nonfinite incremental QMC trajectory")
            X[:,j]=out.draw;P[:,j]=out.predictions;lw[j]+=out.log_increment
        end
    end
    p,logs=_qmc_summary(ref,X,lw,R;history=E.PosteriorHistory(ref,U),
        extra=(proposal=:incremental_rqmc,method=:rqmc,))
    half=n÷2;ids=vcat([collect((r-1)*n+1:(r-1)*n+half) for r in 1:R]...)
    coarse,_=_qmc_summary(ref,X[:,ids],lw[ids],R;history=E.PosteriorHistory(ref,U[:,ids]))
    refinement=_qmc_refinement(coarse,p)
    diagnostic=merge(p.diagnostics,(mc_precision_ok=p.diagnostics.precision_before_refinement&&refinement.passed,
        refinement_checked=true,refinement=refinement,propagated_intervals=n*R*(length(ref.ks)-length(previous.ks)),
        update=:incremental))
    posterior=E.SampledPosterior(p.t,p.model,p.draws,p.physical,p.weights,p.mean,p.covariance,diagnostic,p.history)
    (posterior=posterior,predictions=P,logweights=logs,uniform_dimensions=nd)
end

function E.assimilate!(sm::QMCSmoother,records::AbstractVector{E.MeasurementRecord};
        max_seconds=sm.settings.update_seconds,refresh=false,progress=nothing)
    isempty(records) && throw(ArgumentError("nothing to assimilate"))
    times=[r.t for r in records]
    all(isfinite,times)&&issorted(times)&&first(times)>=sm.posterior.t ||
        throw(ArgumentError("QMC arrivals must be finite and chronological"))
    for r in records
        spec=E.sensor(sm.anchor.model.log,r.sensor)
        isfinite(r.value)&&!E.censored_at(spec,r) || throw(ArgumentError("QMC requires finite uncensored readings"))
    end
    max_seconds>0 || throw(ArgumentError("update budget must be positive"))
    start=time();candidate=deepcopy(sm.anchor);rng=deepcopy(sm.rng)
    proposal=E.assimilate!(candidate,records)
    proposal.committed || return (committed=false,reason=:graph_solve,t=last(times),belief_t=sm.posterior.t,
        posterior=sm.posterior,wall=time()-start)
    result=nothing;refreshed=refresh
    try
        time()<start+max_seconds || throw(QMCTimeBudget())
        if !refresh
            ref=E._posterior_transport(candidate)
            result=_qmc_increment(sm,ref;deadline=start+max_seconds)
            refreshed=!result.posterior.diagnostics.mc_precision_ok
        end
        if refreshed
            remaining=start+max_seconds-time();remaining>0 || throw(QMCTimeBudget())
            fit=qmc_posterior_fit(candidate;points=sm.settings.points,max_points=sm.settings.max_points,
                replicates=sm.settings.replicates,max_seconds=remaining,rng,progress)
            result=(posterior=fit.posterior,predictions=fit.predictions,logweights=fit.logweights,
                uniform_dimensions=fit.cache.uniform_dimensions,seeds=fit.cache.seeds)
        end
    catch err
        err isa QMCTimeBudget || rethrow()
        return (committed=false,reason=:time_budget,t=last(times),belief_t=sm.posterior.t,
            posterior=sm.posterior,wall=time()-start)
    end
    p=result.posterior
    if !p.diagnostics.mc_precision_ok || time()>start+max_seconds
        return (committed=false,reason=time()>start+max_seconds ? :time_budget : :numerical_precision,
            t=last(times),belief_t=sm.posterior.t,diagnostics=p.diagnostics,posterior=sm.posterior,wall=time()-start)
    end
    diagnostic=merge(p.diagnostics,(update=refreshed ? :full_history_refresh : :incremental,
        refreshed=refreshed,proposal_converged=proposal.converged,wall=time()-start,
        refreshes=sm.refreshes+Int(refreshed),increments=sm.increments+Int(!refreshed)))
    p=E.SampledPosterior(p.t,p.model,p.draws,p.physical,p.weights,p.mean,p.covariance,diagnostic,p.history)
    sm.anchor=candidate;sm.posterior=p;sm.predictions=result.predictions;sm.logweights=result.logweights
    sm.uniform_dimensions=result.uniform_dimensions;sm.rng=rng
    if refreshed;sm.seeds=result.seeds;sm.refreshes+=1;else;sm.increments+=1;end
    (committed=true,reason=:ok,t=p.t,posterior=p,diagnostics=diagnostic,wall=time()-start)
end
E.assimilate!(sm::QMCSmoother,record::E.MeasurementRecord;kwargs...)=E.assimilate!(sm,[record];kwargs...)

function E.update_execution!(sm::QMCSmoother;kwargs...)
    E.update_execution!(sm.anchor;kwargs...)
    p=sm.posterior
    sm.posterior=E.SampledPosterior(p.t,sm.anchor.model,p.draws,p.physical,p.weights,p.mean,p.covariance,p.diagnostics,p.history)
    sm
end