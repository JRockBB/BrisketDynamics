# Experimental full-history adapter. Load BrisketEstimator as E first. Kept out
# of live defaults while truncation, conditional shape and cook coverage remain
# unqualified. The same function accepts an offline graph or an online anchor's
# complete graph; it never substitutes the previous posterior for original data.
isdefined(@__MODULE__,:AdaptiveDensities) || include(joinpath(@__DIR__,"../src/adaptive_density.jl"))
using .AdaptiveDensities
isdefined(@__MODULE__,:ConditionalCharts) || include(joinpath(@__DIR__,"../src/conditional_charts.jl"))
using .ConditionalCharts
using LinearAlgebra

struct AdaptiveConditionalCandidate
    density::AdaptiveDensity
    decoder::E._ConditionalTransport
    axes::Vector{Tuple{E.VarKey,Int}}
    axisrows::Vector{Int}
    free::Vector{Int}
    center::Vector{Float64}
    scale::Matrix{Float64}
    diagnostics::NamedTuple
end

struct MappedConditionalCandidate{C}
    density::AdaptiveDensity
    decoder::E._ConditionalTransport
    axes::Vector{Tuple{E.VarKey,Int}}
    axisrows::Vector{Int}
    free::Vector{Int}
    chart::C
    diagnostics::NamedTuple
end

"""
Build an adaptive candidate for a chronological online arrival using a cloned
full-history anchor. The caller's graph, factor registries and current belief are
unchanged on success AND failure. No Gaussian belief is returned as a substitute
for an unresolved density. This is a qualification path, not a live commit API.
"""
function adaptive_arrival_candidate(anchor,recs;kwargs...)
    anchor.lag==typemax(Int) || throw(ArgumentError("adaptive replay requires full history"))
    any(f->f isa E.LinearFactor,anchor.graph.factors) &&
        throw(ArgumentError("adaptive online replay cannot restore a marginalized nonlinear history"))
    anchor.split===nothing && anchor.als===nothing ||
        throw(ArgumentError("adaptive replay requires static latents and fixed calibration"))
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    all(r->r.t>=last(anchor.nodes)[1] && isfinite(r.t) && isfinite(r.value),recs) ||
        throw(ArgumentError("adaptive online arrivals must be finite and chronological"))
    work=deepcopy(anchor)
    # The same timestamp batching as offline replay keeps the process-noise
    # anchoring schedule independent of whether records arrive in one API call.
    records=sort(collect(recs);by=r->r.t,alg=MergeSort);first=1
    while first<=length(records)
        finish=first
        while finish<length(records) && records[finish+1].t==records[first].t;finish+=1;end
        result=E.assimilate!(work,records[first:finish])
        result.committed || return (committed=false,reason=:anchor_solve,candidate=nothing)
        first=finish+1
    end
    candidate=adaptive_conditional_fit(work.graph;
        step_bounds=Dict(work.key_z=>work.latent_step),newton_tol=E.ONLINE_NEWTON_TOL,kwargs...)
    return (committed=false,reason=:candidate_only,candidate=candidate,anchor=work)
end

function adaptive_conditional_logpdf(p::AdaptiveConditionalCandidate,x)
    u=E._conditional_encode(p.decoder,x)
    q=p.scale\(u[p.axisrows]-p.center)
    jac=sum(logabsdet(f.cache.W)[1] for f in p.decoder.dynamics;init=0.)-logabsdet(p.scale)[1]
    return adaptive_logpdf(p.density,q,u[p.free])+jac
end
function adaptive_conditional_logpdf(p::MappedConditionalCandidate,x)
    u=E._conditional_encode(p.decoder,x);q=chart_inverse(p.chart,u[p.axisrows])
    marginal=adaptive_marginal_logpdf(p.density,q);marginal==-Inf && return -Inf
    jac=sum(logabsdet(f.cache.W)[1] for f in p.decoder.dynamics;init=0.)-chart_logjac(p.chart,q)
    return adaptive_logpdf(p.density,q,u[p.free])+jac
end

"""
    adaptive_conditional_fit(graph; axes, center, scale, lower, upper, ...)

Refit the ORIGINAL full target at locally selected integration nodes. All factors
and process-noise anchors are cloned unchanged. `curvature` may replace a node's
innovation root and log evidence, but must return both consistently. A correction
that refines the dynamics solver must also return its `decoder`: innovation
coordinates, shape checks and decoded states must use the same numerical map.
Decoders must agree across all nodes. The correction defaults
to the existing Gauss-Newton approximation, explicitly reported in diagnostics.

This returns a candidate, never a committed live belief. `resolved` only refers
to numerical integration on the declared box. Neither that flag nor the finite
shape check qualifies consistency or justifies dropping the original factors.
"""
function adaptive_conditional_fit(graph;axes,center=nothing,scale=nothing,chart=nothing,lower,upper,
        tolerance=.01,max_nodes=257,max_cells=128,orders=(5,9),
        iters=80,tol=1e-6,ftol=1e-8,step_bounds=Dict(),newton_tol=0.,
        curvature=nothing,shape_key=nothing,max_shape_cells=3,shape_tol=.25,
        progress=nothing,checkpoint=nothing)
    g=deepcopy(graph);off,n=E._offsets(g);selected=collect(Tuple{E.VarKey,Int},axes)
    isempty(selected) && throw(ArgumentError("select at least one integration coordinate"))
    length(unique(selected))==length(selected) || throw(ArgumentError("integration axes repeat"))
    generated=Set(f.key_y1 for f in g.factors if f isa E.DynamicsFactor)
    any(k in generated for (k,_) in selected) && throw(ArgumentError("cannot integrate a generated state coordinate"))
    any(f isa E.DynamicsFactor && f.key_m!==nothing for f in g.factors) &&
        throw(ArgumentError("adaptive conditional integration currently requires static latents"))
    if chart!==nothing
        center===nothing && scale===nothing || throw(ArgumentError("supply a chart or center/scale, not both"))
        center=chart_base(chart).center;scale=chart_base(chart).scale
        chart_dimension(chart)==length(selected) || throw(ArgumentError("chart dimension differs from integration axes"))
    end
    center=collect(Float64,center);scale=Matrix{Float64}(scale);k=length(selected)
    length(center)==k && all(isfinite,center) && size(scale)==(k,k) && all(isfinite,scale) && isfinite(logabsdet(scale)[1]) ||
        throw(ArgumentError("invalid integration chart"))
    ia=[off[key][i] for (key,i) in selected];free=setdiff(1:n,ia)
    C=E.joint_cov(g)[1];mu=E._stacked(g,g.keys,off,n);slope=C[:,ia]/C[ia,ia]
    starts=Dict{Tuple,Vector{Float64}}();records=NamedTuple[]
    decoder=Ref{Union{Nothing,E._ConditionalTransport}}(nothing)
    shape_refs=Dict{Tuple,Any}()
    point_at(q)=chart===nothing ? center+scale*q : chart_forward(chart,q)

    function solve_node(q;retain=false)
        checkpoint===nothing || checkpoint()
        length(records)<max_nodes || throw(AdaptiveDensities.NodeBudget())
        point=point_at(q);jac=chart===nothing ? 0. : chart_logjac(chart,q)
        all(isfinite,point) && isfinite(jac) || error("nonfinite integration map")
        gc=deepcopy(g)
        if isempty(starts)
            start=mu+slope*(point-mu[ia])
        else
            near=argmin(Dict(t=>sum(abs2,collect(t)-q) for t in keys(starts)))
            # Full Gaussian regression provides a predictor only. Every node
            # still solves the full conditional target at its exact coordinates.
            start=starts[near]+slope*(point-point_at(collect(near)))
        end
        for key in gc.keys;gc.values[key].=start[off[key]];end
        node=E._conditional_node!(gc,selected,point;iters,tol,ftol,step_bounds,newton_tol)
        if !node.solve.converged
            # Retry from the anchor, without weakening any optimizer tolerance.
            start=mu+slope*(point-mu[ia])
            for key in gc.keys;gc.values[key].=start[off[key]];end
            node=E._conditional_node!(gc,selected,point;iters,tol,ftol,step_bounds,newton_tol)
        end
        node.solve.converged || curvature!==nothing || error("adaptive conditional solve failed at $(point)")
        ref=E._conditional_transport(node)
        corrected=curvature===nothing ? (root=ref.root,log_integral=node.log_integral,diagnostics=(method=:gauss_newton,)) : curvature(node,ref)
        node.solve.converged || get(corrected,:mode_converged,false) ||
            error("neither conditional optimizer converged at $(point)")
        size(corrected.root)==size(ref.root) && all(isfinite,corrected.root) && isfinite(corrected.log_integral) ||
            error("invalid conditional curvature correction")
        corrected_ref=get(corrected,:decoder,ref)
        corrected_ref.offsets==off || error("correction changed the decoder coordinate layout")
        if decoder[]===nothing
            decoder[]=deepcopy(corrected_ref)
        else
            E._same_conditional_decoder(decoder[],corrected_ref) ||
                error("conditional nodes do not share the same innovation decoder")
        end
        L=E._canonical_conditional_root(corrected.root[free,:])
        node_mean=get(corrected,:mean,ref.mean)
        length(node_mean)==length(ref.mean) && all(isfinite,node_mean) || error("invalid corrected conditional mean")
        result=(logmass=corrected.log_integral+jac,mean=node_mean[free],root=L)
        starts[Tuple(q)]=get(corrected,:state_mean,node.mean)
        push!(records,(q=copy(q),point=copy(point),logmass=result.logmass,chart_logjac=jac,gradient=node.solve.gradient,
            initial_solve_converged=node.solve.converged,curvature=corrected.diagnostics))
        checked_ref=E._ConditionalTransport(corrected_ref.graph,corrected_ref.offsets,
            copy(node_mean),corrected.root,corrected_ref.dynamics)
        retain && (shape_refs[Tuple(q)]=(node=node,ref=checked_ref,root=corrected.root))
        progress===nothing || progress((event=:node,nodes=length(records),q=copy(q),logmass=result.logmass))
        return result
    end

    d=adaptive_density(q->solve_node(q),lower,upper;tolerance,max_nodes,max_cells,orders,checkpoint,
        progress=progress===nothing ? nothing : h->progress(merge((event=:refinement,),h)))
    # Check actual conditional targets in the most massive leaves; no simulation
    # truth or sensor identities enter axis integration or node selection.
    shapes=NamedTuple[];shape_complete=shape_key!==nothing;shape_budget=false
    if shape_key!==nothing
        masses=[AdaptiveDensities._logmass([c],d.integration_order) for c in d.cells]
        ids=sortperm(masses;rev=true)[1:min(max_shape_cells,length(masses))]
        for id in ids
            cell=d.cells[id];q=(cell.lo+cell.hi)/2
            try
                solve_node(q;retain=true)
            catch err
                err isa AdaptiveDensities.NodeBudget || rethrow()
                shape_complete=false;shape_budget=true;break
            end
            item=shape_refs[Tuple(q)];ref=item.ref;root=item.root
            # Exact scalar loss probes against this node's actual quadratic.
            # A curvature callback can supply observed curvature; the default
            # root deliberately retains the old GN shape test's interpretation.
            function target(u)
                x=E._conditional_decode(ref,u)
                return sum(f isa E.DynamicsFactor ? sum(abs2,u[off[f.key_y1]])/2 :
                    E.loss(f,[x[off[key]] for key in E.vars(f)]) for f in ref.graph.factors)
            end
            F=target(ref.mean);scores=zeros(length(off[shape_key]))
            for (j,row) in enumerate(off[shape_key])
                sd=norm(root[row,:]);sd>0 || continue
                v=root*(root[row,:]/sd)
                # Central derivative from exact loss, separating nonzero MAP
                # residual from quadratic/higher-order approximation error.
                h=.001;linear=(target(ref.mean+h*v)-target(ref.mean-h*v))/(2h)
                scores[j]=max(abs(target(ref.mean+v)-F-linear-.5),abs(target(ref.mean-v)-F+linear-.5))
            end
            push!(shapes,(q=q,cell_mass=exp(masses[id]-d.log_normalizer),scores=scores,maximum_defect=maximum(scores)))
        end
    end
    defect=isempty(shapes) ? Inf : maximum(s.maximum_defect for s in shapes)
    diag=merge(d.diagnostics,(graph_solves=length(records),curvature=curvature===nothing ? :gauss_newton : :corrected,
        shape_complete=shape_complete,shape_budget_exhausted=shape_budget,shape_defect=defect,
        shape_pass=shape_complete && defect<=shape_tol,shape_checks=shapes,nodes=records,
        committed=false,consistent=false))
    return chart===nothing ? AdaptiveConditionalCandidate(d,decoder[],selected,ia,free,center,scale,diag) :
        MappedConditionalCandidate(d,decoder[],selected,ia,free,deepcopy(chart),diag)
end
