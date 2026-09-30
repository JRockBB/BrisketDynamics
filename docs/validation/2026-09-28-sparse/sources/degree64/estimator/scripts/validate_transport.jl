# Deterministic, source-recorded representation comparison. Never deploys a belief.
isdefined(@__MODULE__,:adaptive_strict_json) || include(joinpath(@__DIR__,"validate_adaptive.jl"))
isdefined(@__MODULE__,:fit_conditional_chart) || include(joinpath(@__DIR__,"fit_conditional_chart.jl"))
isdefined(@__MODULE__,:mapped_density_distance) || include(joinpath(@__DIR__,"mapped_diagnostics.jl"))
isdefined(@__MODULE__,:target_state_fingerprint) || include(joinpath(@__DIR__,"target_ownership.jl"))
using Serialization,SHA

if !isdefined(@__MODULE__,:TransportTimeLimit)
    struct TransportTimeLimit <: Exception end
end
Base.showerror(io::IO,::TransportTimeLimit)=print(io,"comparison time budget exhausted")

function transport_sources()
    names=["src/adaptive_density.jl","src/conditional_charts.jl","scripts/adaptive_conditional.jl",
        "scripts/adaptive_curvature.jl","scripts/fit_conditional_chart.jl","scripts/mapped_diagnostics.jl",
        "scripts/validate_transport.jl","scripts/target_ownership.jl"]
    return [(path="estimator/"*p,sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"..",p))))) for p in names]
end

"Matched fit budgets, including fresh map pilots, original graph, and unchanged thresholds."
function compare_transport_graph(graph,report_prefix;axes,center,scale,shape_key,
        max_nodes=257,tolerance=.002,orders=(5,9),radius=4.,pilot_radius=2.,max_seconds=600.,
        step_bounds=Dict(),newton_tol=0.,methods=(:affine,:triangular),metadata=nothing)
    isfinite(max_seconds) && max_seconds>0 || throw(ArgumentError("invalid time budget"))
    saved=target_state_fingerprint(graph);candidates=Dict{Symbol,Any}();reports=NamedTuple[]
    for method in methods
        method in (:affine,:triangular) || throw(ArgumentError("unknown coordinate method"))
        start=time();map_seconds=0.;map_diagnostics=nothing;candidate=nothing;training=false
        completed_pilots=Ref(0);completed_nodes=Ref(0);last_refinement=Ref{Any}(nothing)
        checkpoint()=(time()-start<=max_seconds || throw(TransportTimeLimit()))
        function progress(h)
            h.event==:chart_pilot && (completed_pilots[]=h.completed)
            h.event==:node && (completed_nodes[]=h.nodes)
            h.event==:refinement && (last_refinement[]=h)
            println(method," ",h);flush(stdout)
        end
        outcome=try
            chart=AffineChart(center,scale)
            if method==:triangular
                training=true
                fit=fit_conditional_chart(graph;axes,center,scale,radius=pilot_radius,max_fits=max_nodes,
                    step_bounds,newton_tol,progress,checkpoint)
                chart=fit.chart;map_diagnostics=fit.diagnostics;map_seconds=time()-start;training=false
            end
            remaining=max_nodes-completed_pilots[]
            candidate=adaptive_conditional_fit(graph;axes,chart,lower=fill(-radius,length(axes)),
                upper=fill(radius,length(axes)),tolerance,max_nodes=remaining,orders,
                curvature=polished_gn_conditional_curvature,shape_key,step_bounds,newton_tol,progress,checkpoint)
            candidates[method]=candidate
            serialize(report_prefix*"_$(method).jls",candidate)
            (status=:candidate,diagnostics=candidate.diagnostics)
        catch err
            err isa InterruptException && rethrow()
            training && (map_seconds=time()-start)
            timeout=err isa TransportTimeLimit || (err isa ChartFitFailure && err.cause isa TransportTimeLimit)
            (status=timeout ? :time_budget : :failed,error=sprint(showerror,err))
        end
        unchanged=target_state_fingerprint(graph)==saved
        report=(method=method,backend=E.backend_fingerprint(),physics=E.physics_fingerprint(),sources=transport_sources(),
            metadata=metadata,settings=(axes=axes,center=center,scale=scale,radius=radius,pilot_radius=pilot_radius,
                max_nodes=max_nodes,tolerance=tolerance,orders=orders,max_seconds=max_seconds),
            wall=time()-start,map_seconds=map_seconds,map=map_diagnostics,completed_pilots=completed_pilots[],
            completed_nodes=completed_nodes[],total_conditional_fits=completed_pilots[]+completed_nodes[],
            last_refinement=last_refinement[],outcome=outcome,graph_unchanged=unchanged,ownership_check=:serialized_state,
            committed=false,consistent=false,domain_validated=false)
        write(report_prefix*"_$(method).json",adaptive_strict_json(report));push!(reports,report)
        unchanged || error("comparison mutated the caller's target; report preserved")
        println("FINISH ",method," ",(wall=report.wall,fits=report.total_conditional_fits,status=outcome.status));flush(stdout)
    end
    return (reports=reports,candidates=candidates)
end

"Replay one of the four frozen failures; optional extra axis is selected from target shape."
function compare_transport_snapshot(path,report_prefix;add_axis=false,kwargs...)
    snapshot=deserialize(path)
    snapshot.backend==E.backend_fingerprint() && snapshot.physics==E.physics_fingerprint() || error("snapshot source mismatch")
    old=snapshot.result.candidate;source=old.transported.source;anchor=snapshot.failed.bank.anchor
    selected=copy(source.axes)
    if add_axis
        shape=E.conditional_shape_diagnostic(old.transported;key=anchor.key_z)
        remaining=setdiff(eachindex(shape.scores),last.(selected))
        push!(selected,(anchor.key_z,remaining[argmax(shape.scores[remaining])]))
    end
    rows=[source.offsets[key][i] for (key,i) in selected]
    center=source.mean[rows];scale=Matrix(cholesky(Symmetric(source.covariance[rows,rows])).L)
    return compare_transport_graph(anchor.graph,report_prefix;axes=selected,center,scale,shape_key=anchor.key_z,
        step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,
        metadata=(case=snapshot.case,t=last(anchor.nodes)[1],snapshot_sha256=bytes2hex(sha256(read(path)))),kwargs...)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("usage: validate_transport.jl SNAPSHOT.jls REPORT_PREFIX [add_axis=false]")
    compare_transport_snapshot(ARGS[1],ARGS[2];add_axis=length(ARGS)>=3 && parse(Bool,ARGS[3]))
end
