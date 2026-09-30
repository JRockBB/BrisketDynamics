isdefined(@__MODULE__,:DirectPosterior) || include(joinpath(@__DIR__,"direct_posterior.jl"))

mutable struct DirectSmoother
    anchor::E.FixedLagSmoother
    posterior::DirectPosterior
    settings::NamedTuple
end

"Full-history deterministic online posterior; no Gaussian history compression."
function direct_init(model;atol=.002,rtol=.002,max_evaluations=100_000,max_level=5,
                     max_seconds=600.,rule=:hermite,transport_axes=0,curvature_rotation=false,moments=true,physical_moments=false,kwargs...)
    haskey(kwargs,:lag) && throw(ArgumentError("direct inference retains the full nonlinear history"))
    anchor=E.online_init(model;lag=typemax(Int),kwargs...)
    settings=(atol=atol,rtol=rtol,max_evaluations=max_evaluations,max_level=max_level,max_seconds=max_seconds,
        rule=rule,transport_axes=transport_axes,curvature_rotation=curvature_rotation,moments=moments,physical_moments=physical_moments)
    p=direct_posterior_fit(anchor;settings...)
    p.diagnostics.resolved || error("initial direct posterior integration is unresolved: $(p.diagnostics.reason)")
    DirectSmoother(anchor,p,settings)
end

E.belief_at(sm::DirectSmoother)=sm.posterior

function E.assimilate!(sm::DirectSmoother,records::AbstractVector{E.MeasurementRecord};progress=nothing)
    isempty(records) && throw(ArgumentError("nothing to assimilate"))
    for r in records
        isfinite(r.t) && isfinite(r.value) || throw(ArgumentError("measurements must be finite"))
        E.sensor(sm.anchor.model.log,r.sensor)
        r.t>=last(sm.anchor.nodes)[1] || throw(ArgumentError("online arrivals must be chronological"))
    end
    start=time();candidate=deepcopy(sm.anchor)
    # The graph solve supplies a chart and predictable covariance anchors. All
    # original nonlinear factors remain in the target for direct integration.
    result=E.assimilate!(candidate,records)
    result.committed || return (committed=false,reason=:graph_solve,diagnostics=result,wall=time()-start)
    p=direct_posterior_fit(candidate;sm.settings...,progress)
    if !p.diagnostics.resolved
        return (committed=false,reason=:direct_integration,diagnostics=p.diagnostics,
                candidate=p,t=last(candidate.nodes)[1],wall=time()-start)
    end
    sm.anchor=candidate;sm.posterior=p
    (committed=true,reason=:ok,diagnostics=p.diagnostics,posterior=p,
     t=last(candidate.nodes)[1],wall=time()-start)
end
E.assimilate!(sm::DirectSmoother,record::E.MeasurementRecord;kwargs...)=E.assimilate!(sm,[record];kwargs...)

"Offline fit uses exactly the same graph construction and direct likelihood integrals."
function direct_fit(model,records;atol=.002,rtol=.002,max_evaluations=100_000,
                    max_level=5,max_seconds=600.,progress=nothing,rule=:hermite,transport_axes=0,curvature_rotation=false,moments=true,physical_moments=false,kwargs...)
    haskey(kwargs,:lag) && throw(ArgumentError("direct inference retains the full nonlinear history"))
    anchor=E.online_init(model;lag=typemax(Int),kwargs...)
    if !isempty(records)
        result=E.assimilate!(anchor,collect(records))
        result.committed || error("offline graph solve failed")
    end
    direct_posterior_fit(anchor;atol,rtol,max_evaluations,max_level,max_seconds,progress,rule,transport_axes,curvature_rotation,moments,physical_moments)
end

"Predict under the model's declared control log, preserving the live posterior."
function direct_forecast(sm::DirectSmoother,t::Real;progress=nothing)
    isfinite(t) && t>=last(sm.anchor.nodes)[1] || throw(ArgumentError("forecast time precedes current belief"))
    candidate=deepcopy(sm.anchor)
    E._node_at!(candidate,Float64(t))
    direct_posterior_fit(candidate;sm.settings...,progress)
end
