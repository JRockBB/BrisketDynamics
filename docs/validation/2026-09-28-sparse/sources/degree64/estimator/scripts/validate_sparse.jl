# Development comparison, not a consistency/coverage acceptance gate.
isdefined(@__MODULE__,:adaptive_strict_json) || include(joinpath(@__DIR__,"validate_adaptive.jl"))
isdefined(@__MODULE__,:SparseConditionalCandidate) || include(joinpath(@__DIR__,"sparse_conditional.jl"))
isdefined(@__MODULE__,:target_state_fingerprint) || include(joinpath(@__DIR__,"target_ownership.jl"))
using Serialization,SHA
struct SparseTimeLimit <: Exception end
Base.showerror(io::IO,::SparseTimeLimit)=print(io,"sparse comparison time budget exhausted")
function sparse_sources()
    names=["src/adaptive_density.jl","src/sparse_density.jl","src/conditional_charts.jl",
        "scripts/adaptive_conditional.jl","scripts/sparse_conditional.jl","scripts/adaptive_curvature.jl",
        "scripts/curvature_audit.jl","scripts/target_ownership.jl","scripts/validate_sparse.jl"]
    [(path="estimator/"*p,sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"..",p))))) for p in names]
end
function validate_sparse_graph(graph,report_path;axes,center,scale,shape_key,method=:sparse,
        max_nodes=257,tolerance=.002,orders=(5,9),radius=4.,max_seconds=600.,max_axes=5,
        max_degree=64,max_quadrature_evaluations=2_000_000,
        step_bounds=Dict(),newton_tol=0.,metadata=nothing)
    method in (:tensor,:sparse,:expanding) || throw(ArgumentError("unknown method"))
    isfinite(max_seconds) && max_seconds>0 || throw(ArgumentError("invalid time budget"))
    sources=sparse_sources();before=target_state_fingerprint(graph);start=time()
    candidate=nothing;completed=Ref(0);lastcheck=Ref{Any}(nothing);events=NamedTuple[]
    checkpoint()=(time()-start<=max_seconds || throw(SparseTimeLimit()))
    function progress(h)
        h.event==:node && (completed[]=get(h,:total_nodes,h.nodes))
        h.event==:refinement && (lastcheck[]=h)
        h.event in (:promote,:axis_stage) && push!(events,h)
        println(method," ",h);flush(stdout)
    end
    outcome=try
        kw=(shape_key=shape_key,tolerance=tolerance,max_nodes=max_nodes,orders=orders,
            curvature=polished_gn_conditional_curvature,step_bounds=step_bounds,
            newton_tol=newton_tol,progress=progress,checkpoint=checkpoint)
        candidate=if method==:expanding
            expanding_sparse_fit(graph;axes,radius,max_axes,max_degree,max_quadrature_evaluations,kw...)
        else
            fitter=method==:sparse ? sparse_conditional_fit : adaptive_conditional_fit
            extra=method==:sparse ? (max_degree=max_degree,max_quadrature_evaluations=max_quadrature_evaluations) : (;)
            fitter(graph;axes,chart=AffineChart(center,scale),lower=fill(-radius,length(axes)),
                upper=fill(radius,length(axes)),kw...,extra...)
        end
        (status=:candidate,diagnostics=candidate.diagnostics,axes=candidate.axes)
    catch err
        err isa InterruptException && rethrow()
        (status=err isa SparseTimeLimit ? :time_budget : :failed,error=sprint(showerror,err,catch_backtrace()))
    end
    unchanged=target_state_fingerprint(graph)==before;source_unchanged=sparse_sources()==sources
    report=(method=method,backend=E.backend_fingerprint(),physics=E.physics_fingerprint(),sources=sources,
        metadata=metadata,settings=(axes=axes,center=center,scale=scale,shape_key=shape_key,
            max_nodes=max_nodes,tolerance=tolerance,orders=orders,radius=radius,max_seconds=max_seconds,max_axes=max_axes,
            max_degree=method==:tensor ? nothing : max_degree,
            max_quadrature_evaluations=method==:tensor ? nothing : max_quadrature_evaluations),
        wall=time()-start,completed_fits=completed[],last_check=lastcheck[],promotions=events,outcome=outcome,
        graph_unchanged=unchanged,source_unchanged=source_unchanged,consistent=false,committed=false,
        domain_validated=false,diagnostics_comparable_to_tensor=false)
    write(report_path,adaptive_strict_json(report))
    unchanged && source_unchanged || error("comparison ownership/source change; report preserved")
    println("FINISH ",method," ",(wall=report.wall,fits=completed[],status=outcome.status));flush(stdout)
    (report=report,candidate=candidate)
end
