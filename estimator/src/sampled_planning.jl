# A scenario owns its state, parameters AND accumulated history. Replacing any of these by their
# separate means destroys dependencies before a nonlinear forecast or decision is evaluated.
export SampledBeliefSnapshot, PlanScenario, SampledForecast, posterior_predict

struct PlanScenario
    snapshot::BeliefSnapshot
    weight::Float64
    source_index::Int
end
struct SampledBeliefSnapshot
    t::Float64
    sys::CookSystem
    scenarios::Vector{PlanScenario}
    goals::Dict{Symbol,PlanGoals}
    provenance::NamedTuple
end

function _posterior_selection(p,n,rng)
    p.diagnostics.mc_precision_ok || throw(ArgumentError("posterior precision is inadequate"))
    n===nothing || (n isa Integer && n>=2) || throw(ArgumentError("use at least two planning samples, or n=nothing for all"))
    ids=findall(>(0),p.weights)
    if n===nothing || n>=length(ids)
        return ids,p.weights[ids]./sum(p.weights[ids])
    end
    cumulative=cumsum(p.weights);cumulative[end]=1.0
    selected=[searchsortedfirst(cumulative,(j-1+rand(rng))/n) for j in 1:n]
    ids=sort(unique(selected));weights=[count(==(i),selected)/n for i in ids]
    return ids,weights
end

_history_goal_key(g::PlanGoals)=(g.leth_t_ref,g.leth_z,g.t_surf_max,g.dT_soft)
function _sampled_history(ref,u,goals,history_dt)
    z=ref.zm+ref.zs.*u[ref.off[ref.kz]];y=Float64[]
    past=Dict(id=>Past() for id in ref.model.bl.sys.pieces)
    for k in ref.ks[2:end]
        f=ref.generators[k]
        if f isa GaussianPrior
            y=f.m+ref.roots[k]*u[ref.off[k]]
            x=belief_decode(y,ref.model.bl);m0=x[first(belief_index(ref.model.bl,:pit,:m_fuel))]
            for pa in values(past);pa.m0=m0;end
        else
            ts=collect(range(f.t0,f.t1;length=max(2,ceil(Int,(f.t1-f.t0)/history_dt)+1)))
            yp,o=propagate(f.model,y,z,f.t0,f.t1;first_slice=f.first,output_times=ts)
            for pc in ref.model.sys.pieces
                ev=piece_events(f.model.log.events,pc.id;cfg=pc.cfg)
                _accumulate!(past[pc.id],o.piece[pc.id],ts,ev.t_pull,goals[pc.id])
            end
            y=yp+ref.roots[k]*u[ref.off[k]]
        end
    end
    for pa in values(past);pa.t=ref.t;end
    return past
end

"""
    belief_snapshot(p::SampledPosterior; n=nothing, goals=..., history_dt=60, rng=MersenneTwister(0))

Retain weighted joint posterior scenarios, including a replayed past for each trajectory. The default
uses every positive-weight draw. An explicit `n` reduces work by stratified resampling and is recorded
in provenance; its predictive/decision accuracy must be checked as that budget grows. No Gaussian
sigma points or mean history are introduced. `history_dt` controls past-integral output quadrature.
"""
function belief_snapshot(p::SampledPosterior;n=nothing,goals=Dict(id=>PlanGoals() for id in p.model.bl.sys.pieces),
                         history_dt::Real=60.0,rng::AbstractRNG=MersenneTwister(0))
    p.history===nothing && throw(ArgumentError("sampled planning requires retained trajectory coordinates"))
    p.model.rom===nothing || throw(ArgumentError("sampled planning does not yet model ROM discrepancy"))
    isfinite(history_dt) && history_dt>0 || throw(ArgumentError("history quadrature step must be positive"))
    gd=Dict{Symbol,PlanGoals}(goals)
    Set(keys(gd))==Set(p.model.bl.sys.pieces) || throw(ArgumentError("one goal specification per piece is required"))
    ids,w=_posterior_selection(p,n,rng);scenarios=Vector{PlanScenario}(undef,length(ids))
    nz=length(p.model.blk);ny=p.model.bl.n;C=zeros(nz+ny,nz+ny);Cx=zeros(ny,ny)
    Threads.@threads for k in eachindex(ids)
        j=ids[k];z=copy(p.draws[1:nz,j]);y=copy(p.draws[nz+1:end,j]);x=copy(p.physical[:,j])
        past=_sampled_history(p.history.reference,view(p.history.coordinates,:,j),gd,history_dt)
        context=BeliefSnapshot(p.t,p.model.sys,p.model.bl,p.model.blk,z,y,C,x,Cx,belief_handoff(x,p.model.bl),
            past,(source=:posterior_scenario,index=j),nothing)
        scenarios[k]=PlanScenario(context,w[k],j)
    end
    return SampledBeliefSnapshot(p.t,p.model.sys,scenarios,gd,(source=:joint_posterior,source_draws=length(p.weights),
        scenarios=length(ids),requested=n,subsampled=n!==nothing && n<count(>(0),p.weights),history_dt=Float64(history_dt),
        source_precision=p.diagnostics,backend=backend_fingerprint()))
end
function belief_snapshot(sm::Union{ParticleSmoother,SampledSmoother};kwargs...)
    haskey(kwargs,:goals) && return belief_snapshot(sm.posterior;kwargs...)
    return belief_snapshot(sm.posterior;goals=Dict(id=>sm.graph_backend.goals for id in sm.posterior.model.bl.sys.pieces),kwargs...)
end
belief_plan_draws(snap::SampledBeliefSnapshot)=snap.scenarios

function _check_sampled_goals(snap,goals)
    Set(keys(goals))==Set(keys(snap.goals)) || throw(ArgumentError("planning goals must cover the sampled pieces"))
    all(_history_goal_key(goals[id])==_history_goal_key(snap.goals[id]) for id in keys(goals)) ||
        throw(ArgumentError("historical-integral goals changed; rebuild the sampled snapshot with those goals"))
end
_planning_context(snap::SampledBeliefSnapshot,draw::PlanScenario)=
    (draw.snapshot,(z=draw.snapshot.z,y=draw.snapshot.y))
function _plan_residual(snap::SampledBeliefSnapshot,draw::PlanScenario,ζ,f,s,goals,control,k,N,hedge)
    _check_sampled_goals(snap,goals)
    return sqrt(draw.weight).*_plan_residual(draw.snapshot,vcat(draw.snapshot.z,draw.snapshot.y),ζ,f,s,goals,control,1,1,false)
end
function _planning_scan(snap::SampledBeliefSnapshot,zs,c,f,s,goals,control)
    return sum(.5sum(abs2,_plan_residual(snap,draw,c,f,s,goals,control,k,length(zs),true)) for (k,draw) in enumerate(zs))
end

struct WeightedPlanFactor <: Factor
    factor::PlanFactor
    root_weight::Float64
end
vars(f::WeightedPlanFactor)=vars(f.factor)
ndim(f::WeightedPlanFactor)=ndim(f.factor)
provenance(f::WeightedPlanFactor)="weighted joint posterior planning scenario"
residual(f::WeightedPlanFactor,vals)=f.root_weight.*residual(f.factor,vals)
jacobians(f::WeightedPlanFactor,vals)=[f.root_weight.*J for J in jacobians(f.factor,vals)]
function _planning_factor(snap::SampledBeliefSnapshot,draw::PlanScenario,key,f,s,goals,control,k,N,hedge,n_out)
    factor=PlanFactor(key,draw.snapshot,vcat(draw.snapshot.z,draw.snapshot.y),f,s,goals,control,1,1,false,n_out,nothing)
    return WeightedPlanFactor(factor,sqrt(draw.weight))
end

"Weighted expected full goal loss over joint scenarios; historical and current uncertainties stay coupled."
function system_plan_loss(snap::SampledBeliefSnapshot,plan,s::SystemSchedule,goals::AbstractDict{Symbol,PlanGoals};
                          control::SystemControlGoals=SystemControlGoals())
    _check_sampled_goals(snap,goals);f=system_free_set(s);ζ=encode_system_plan(plan,f,s)
    ls=[.5sum(abs2,_plan_residual(draw.snapshot,vcat(draw.snapshot.z,draw.snapshot.y),ζ,f,s,goals,control,1,1,false))
        for draw in snap.scenarios]
    w=[d.weight for d in snap.scenarios]
    return (mean=dot(w,ls),draws=ls,weights=w)
end

"Optimize weighted expected loss without projecting the scenarios onto a Gaussian."
function replan_system(snap::SampledBeliefSnapshot,s::SystemSchedule,goals::AbstractDict{Symbol,PlanGoals};kwargs...)
    _check_sampled_goals(snap,goals)
    (haskey(kwargs,:hedge) || haskey(kwargs,:n_dirs)) &&
        throw(ArgumentError("sampled planning uses weighted full loss; set scenario count when taking the snapshot"))
    return _replan_system(snap,snap.scenarios,s,goals;kwargs...)
end

function _weighted_prediction(X,w,probabilities)
    mu=X*w;delta=X.-mu
    quantiles=reduce(vcat,[permutedims(_weighted_quantiles(view(X,i,:),w,probabilities)) for i in axes(X,1)])
    return (mean=mu,sd=sqrt.(vec(sum(delta.^2 .* w';dims=2))),quantiles=quantiles,
        probabilities=collect(probabilities),draws=X,weights=copy(w))
end

"Nonlinear scenario quantiles, conditional on the future plan's deterministic physics (no future process noise)."
function system_plan_predictive(snap::SampledBeliefSnapshot,ζ,f::SystemFreeSet,s::SystemSchedule,
                                goals::AbstractDict{Symbol,PlanGoals};probabilities=[.025,.5,.975])
    _check_sampled_goals(snap,goals)
    all(q->0<=q<=1,probabilities) || throw(ArgumentError("invalid quantile probabilities"))
    outputs=Vector{Any}(undef,length(snap.scenarios))
    Threads.@threads for j in eachindex(outputs)
        context=snap.scenarios[j].snapshot
        outputs[j]=system_plan_outputs(context,context.z,ζ,f,s,goals;y=context.y)[1]
    end
    w=[d.weight for d in snap.scenarios]
    return Dict(id=>merge(_weighted_prediction(hcat([o[id] for o in outputs]...),w,probabilities),
        (uncertainty_scope=:current_state_parameters_and_history,includes_future_process_noise=false,)) for id in snap.sys.layout.pieces)
end

struct SampledForecast
    times::Vector{Float64}
    draws::Vector{Matrix{Float64}}
    physical::Vector{Matrix{Float64}}
    weights::Vector{Float64}
    model::SystemModel
    provenance::NamedTuple
end

"""
    posterior_predict(p, times; model=p.model, process_noise=true, n=nothing, rng=MersenneTwister(0))

Propagate joint draws through future physics, adding independent process disturbances by default.
The coordinate covariance of each future interval is fixed at a deterministic anchor propagated from
the graph point, as in the declared graph model. Sensor measurement noise is not added to physical-state
predictions. `model` supplies the proposed/executed future controls. This is a forecast, not a measurement
update. `n=nothing` retains all source draws and weights; finite-sample prediction error remains.
"""
function posterior_predict(p::SampledPosterior,times;model::SystemModel=p.model,process_noise::Bool=true,n=nothing,
                           rng::AbstractRNG=MersenneTwister(0))
    ts=collect(Float64,times)
    !isempty(ts) && all(isfinite,ts) && all(>(0),diff(vcat(p.t,ts))) ||
        throw(ArgumentError("forecast times must strictly increase after the posterior time"))
    p.history===nothing && throw(ArgumentError("forecast anchoring requires the retained graph"))
    model.bl.sys_index==p.model.bl.sys_index && model.bl.kinds==p.model.bl.kinds &&
        model.bl.sys.pieces==p.model.bl.sys.pieces && model.blk===p.model.blk ||
        throw(ArgumentError("forecast must preserve the state layout and use the posterior's latent block"))
    ids,w=_posterior_selection(p,n,rng);ref=p.history.reference;nz=length(ref.zm)
    current=copy(p.draws[:,ids]);zanchor=copy(ref.g.values[ref.kz]);yanchor=copy(ref.g.values[last(ref.ks)])
    draws=Matrix{Float64}[];physical=Matrix{Float64}[];t0=p.t
    for (step,t) in enumerate(ts)
        first_slice=length(ref.ks)==2 && step==1
        f=DynamicsFactor(model,ref.kz,VarKey(:state,:forecast,step),VarKey(:state,:forecast,step+1),t0,t;
            first=first_slice,events=sensor_events(model.log,t0,t))
        cache=_dyn_eval!(f,zanchor,yanchor;jac=false);root=inv(cache.W);yanchor=copy(cache.f)
        disturbances=process_noise ? root*randn(rng,model.bl.n,length(ids)) : zeros(model.bl.n,length(ids))
        next=copy(current)
        Threads.@threads for j in axes(next,2)
            y=propagate(model,view(current,nz+1:size(current,1),j),view(current,1:nz,j),t0,t;first_slice)[1]
            next[nz+1:end,j]=y+disturbances[:,j]
        end
        all(isfinite,next) || error("non-finite forecast trajectory")
        decoded=hcat([belief_decode(view(next,nz+1:size(next,1),j),model.bl) for j in axes(next,2)]...)
        push!(draws,next);push!(physical,decoded);current=next;t0=t
    end
    return SampledForecast(ts,draws,physical,w,model,(includes_process_noise=process_noise,
        source_draws=length(p.weights),forecast_draws=length(ids),subsampled=length(ids)<count(>(0),p.weights),
        source_precision=p.diagnostics,backend=backend_fingerprint()))
end
posterior_predict(sm::Union{ParticleSmoother,SampledSmoother},times;kwargs...)=posterior_predict(sm.posterior,times;kwargs...)
function posterior_quantile(f::SampledForecast,probabilities=[.025,.5,.975];physical::Bool=true)
    all(q->0<=q<=1,probabilities) || throw(ArgumentError("invalid quantile probabilities"))
    return [_weighted_prediction(X,f.weights,probabilities).quantiles for X in (physical ? f.physical : f.draws)]
end
