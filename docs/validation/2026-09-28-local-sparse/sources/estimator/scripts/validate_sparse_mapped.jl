isdefined(@__MODULE__,:validate_sparse_graph) || include(joinpath(@__DIR__,"validate_sparse.jl"))
isdefined(@__MODULE__,:fit_conditional_chart) || include(joinpath(@__DIR__,"fit_conditional_chart.jl"))

"Compare the sparse integrator in a freshly learned chart; charge all pilots."
function validate_sparse_mapped(graph,report_path;axes,center,scale,shape_key,max_nodes=257,
        tolerance=.002,orders=(5,9),radius=4.,max_seconds=600.,max_degree=64,
        max_quadrature_evaluations=2_000_000,local_fallback=true,local_refinement=false,local_degree=8,
        step_bounds=Dict(),newton_tol=0.)
    isfinite(max_seconds) && max_seconds>0 || throw(ArgumentError("invalid time budget"))
    sources=vcat(sparse_sources(),[(path="estimator/scripts/"*p,
        sha256=bytes2hex(sha256(read(joinpath(@__DIR__,p))))) for p in ("fit_conditional_chart.jl","validate_sparse_mapped.jl")])
    before=target_state_fingerprint(graph);start=time();pilots=Ref(0);nodes=Ref(0)
    lastcheck=Ref{Any}(nothing);candidate=nothing;mapdiag=nothing;map_seconds=nothing
    checkpoint()=(time()-start<=max_seconds || throw(SparseTimeLimit()))
    function progress(h)
        h.event==:chart_pilot && (pilots[]=h.completed)
        h.event==:node && (nodes[]=h.nodes)
        h.event==:refinement && (lastcheck[]=h)
        println("mapped_sparse ",h);flush(stdout)
    end
    outcome=try
        fitted=fit_conditional_chart(graph;axes,center,scale,radius=2.,max_fits=max_nodes,
            step_bounds,newton_tol,progress,checkpoint)
        mapdiag=fitted.diagnostics;map_seconds=time()-start
        candidate=sparse_conditional_fit(graph;axes,chart=fitted.chart,lower=fill(-radius,length(axes)),
            upper=fill(radius,length(axes)),shape_key,max_nodes=max_nodes-pilots[],tolerance,orders,
            max_degree,max_quadrature_evaluations,local_fallback,local_refinement,local_degree,
            curvature=polished_gn_conditional_curvature,step_bounds,newton_tol,progress,checkpoint)
        (status=:candidate,diagnostics=candidate.diagnostics)
    catch err
        err isa InterruptException && rethrow()
        timeout=err isa SparseTimeLimit || (err isa ChartFitFailure && err.cause isa SparseTimeLimit)
        (status=timeout ? :time_budget : :failed,error=sprint(showerror,err,catch_backtrace()))
    end
    unchanged=target_state_fingerprint(graph)==before
    source_unchanged=all(bytes2hex(sha256(read(joinpath(@__DIR__,"../..",s.path))))==s.sha256 for s in sources)
    report=(method=local_refinement ? :mapped_local_sparse : :mapped_sparse,sources=sources,backend=E.backend_fingerprint(),physics=E.physics_fingerprint(),
        settings=(axes=axes,center=center,scale=scale,max_nodes=max_nodes,tolerance=tolerance,orders=orders,
            radius=radius,max_seconds=max_seconds,max_degree=max_degree,
            max_quadrature_evaluations=max_quadrature_evaluations,local_fallback=local_fallback,
            local_refinement=local_refinement,local_degree=local_degree),
        wall=time()-start,map_seconds=map_seconds,map=mapdiag,
        completed_pilots=pilots[],completed_nodes=nodes[],completed_fits=pilots[]+nodes[],last_check=lastcheck[],
        outcome=outcome,graph_unchanged=unchanged,source_unchanged=source_unchanged,consistent=false,
        committed=false,domain_validated=false,diagnostics_comparable_to_tensor=false)
    write(report_path,adaptive_strict_json(report))
    unchanged && source_unchanged || error("source or ownership changed; report preserved")
    println("FINISH mapped_sparse ",(wall=report.wall,fits=report.completed_fits,status=outcome.status));flush(stdout)
    (report=report,candidate=candidate)
end
