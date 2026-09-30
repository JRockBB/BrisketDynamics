# Shared offline / full-history online candidate path. No live commit or
# calibration changes. Axis promotion consumes the SAME graph-fit budget.
isdefined(@__MODULE__,:adaptive_conditional_fit) || include(joinpath(@__DIR__,"adaptive_conditional.jl"))
isdefined(@__MODULE__,:SparseDensities) || include(joinpath(@__DIR__,"../src/sparse_density.jl"))
using .SparseDensities

struct SparseConditionalCandidate{C}
    density::SparseDensity
    decoder::E._ConditionalTransport
    axes::Vector{Tuple{E.VarKey,Int}}
    axisrows::Vector{Int}
    free::Vector{Int}
    chart::C
    diagnostics::NamedTuple
end
function adaptive_conditional_logpdf(p::SparseConditionalCandidate,x)
    u=E._conditional_encode(p.decoder,x);q=chart_inverse(p.chart,u[p.axisrows])
    marginal=adaptive_marginal_logpdf(p.density,q);marginal==-Inf && return -Inf
    jac=sum(logabsdet(f.cache.W)[1] for f in p.decoder.dynamics;init=0.)-chart_logjac(p.chart,q)
    adaptive_logpdf(p.density,q,u[p.free])+jac
end
function sparse_conditional_fit(graph;screen_only=false,max_degree=64,max_quadrature_evaluations=2_000_000,kwargs...)
    build(args...;kw...)=sparse_density(args...;screen_only,max_degree,max_quadrature_evaluations,kw...)
    function shape_points(values,count)
        # Stable, deterministic tie order. Select high-density ACTUAL nodes;
        # these have no leaf mass, so do not invent a cell-mass interpretation.
        points=sort(collect(keys(values));by=q->(-values[q].logmass,q))
        [(q=collect(q),mass=nothing) for q in points[1:min(count,length(points))]]
    end
    adaptive_conditional_fit(graph;_integration=(build=build,shape_points=shape_points,candidate=SparseConditionalCandidate),kwargs...)
end

"""
Screen the Gaussian remainder, promote the worst failing eligible parameter,
then integrate. Every promotion rebuilds from the original factors. Center-only
screening is followed by finite checks at high-density oracle nodes. No sensor
names, simulation truth or cook labels enter selection. Results stay candidates.
"""
function expanding_sparse_fit(graph;axes,shape_key,max_nodes=257,max_axes=5,radius=4.,
        shape_tol=.25,eligible_axes=nothing,progress=nothing,checkpoint=nothing,kwargs...)
    selected=collect(Tuple{E.VarKey,Int},axes)
    max_axes isa Integer && max_nodes isa Integer && 1<=length(selected)<=max_axes &&
        max_nodes>=1 && isfinite(radius) && radius>0 && isfinite(shape_tol) && shape_tol>0 ||
        throw(ArgumentError("invalid expansion settings"))
    # Linearization may populate factor caches. Even the chart's initial
    # covariance must be computed on an owned graph, including failure paths.
    base=deepcopy(graph)
    off,n=E._offsets(base);mu=E._stacked(base,base.keys,off,n);C=E.joint_cov(base)[1]
    eligible=eligible_axes===nothing ? [(shape_key,j) for j in eachindex(off[shape_key])] : collect(eligible_axes)
    all(k==shape_key && 1<=j<=length(off[k]) for (k,j) in eligible) || throw(ArgumentError("invalid eligible shape axes"))
    total=0;stages=NamedTuple[];candidate=nothing;screen=true;reason=:node_budget
    while total<max_nodes
        rows=[off[k][j] for (k,j) in selected]
        center=mu[rows];scale=Matrix(cholesky(Symmetric(C[rows,rows])).L)
        budget=screen ? 1 : max_nodes-total
        stage_nodes=Ref(0)
        function report(h)
            h.event==:node && (stage_nodes[]=h.nodes)
            progress===nothing || progress(merge(h,(stage=length(stages)+1,total_nodes=total+stage_nodes[])))
        end
        candidate=sparse_conditional_fit(graph;axes=selected,chart=AffineChart(center,scale),
            lower=fill(-radius,length(selected)),upper=fill(radius,length(selected)),
            shape_key,shape_tol,max_nodes=budget,screen_only=screen,progress=report,checkpoint,kwargs...)
        total+=candidate.diagnostics.graph_solves
        push!(stages,(axes=copy(selected),screen=screen,fits=candidate.diagnostics.graph_solves,
            shape_defect=candidate.diagnostics.shape_defect,shape_pass=candidate.diagnostics.shape_pass,
            resolved=candidate.diagnostics.resolved,reason=candidate.diagnostics.reason))
        progress===nothing || progress((event=:axis_stage,stage=length(stages),
            diagnostics=last(stages),total_nodes=total))
        if candidate.diagnostics.shape_pass
            if screen;screen=false;continue;end
            reason=candidate.diagnostics.resolved ? :finite_checks : candidate.diagnostics.reason;break
        end
        remaining=setdiff(eligible,selected)
        if isempty(remaining) || length(selected)>=max_axes
            reason=:axis_budget;break
        end
        scores=[maximum(s.scores[j] for s in candidate.diagnostics.shape_checks) for (_,j) in remaining]
        maximum(scores)>shape_tol || (reason=:shape_unresolved;break)
        total<max_nodes || break
        next=remaining[argmax(scores)];push!(selected,next);screen=true
        progress===nothing || progress((event=:promote,axis=next,axes=copy(selected),total_nodes=total))
    end
    diag=merge(candidate.diagnostics,(resolved=reason==:finite_checks && candidate.diagnostics.resolved,
        reason=reason,total_graph_solves=total,axis_stages=stages,automatic_axis_promotion=true))
    SparseConditionalCandidate(candidate.density,candidate.decoder,candidate.axes,candidate.axisrows,
        candidate.free,candidate.chart,diag)
end

sparse_arrival_candidate(anchor,recs;kwargs...)=adaptive_arrival_candidate(anchor,recs;fit=expanding_sparse_fit,kwargs...)
