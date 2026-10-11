export DeterministicSmoother, deterministic_init, deterministic_fit

mutable struct DeterministicSmoother
    bank::ConditionalSmoother
    posterior::ConditionalDensity
    center::Vector{Float64}
    scale::Matrix{Float64}
    density_tol::Float64
    shape_tol::Float64
    max_order::Int
    max_nodes::Int
    adaptive_axes::Bool
    max_axes::Int
    diagnostics::NamedTuple
    # learn-then-freeze: full history while the chart adapts; after `freeze_stable` consecutive
    # committed arrivals that needed no re-centre/refine, every smoother's lag drops to `hold_lag`
    freeze_stable::Union{Nothing,Int}
    hold_lag::Int
    frozen::Bool
    stable::Int
    n_recentres::Int
    n_refines::Int
end

function _density_from_bank(sm,bank,center,scale)
    p=_conditional_current(bank,sm.axes)
    # The returned distribution must own its history when live graphs advance or
    # marginalize. The Gaussian graph is an optimizer, not the reported belief.
    p=deepcopy(p)
    return conditional_density(p;center,scale,order=bank.order,logweights=bank.logweights)
end

"""
    deterministic_init(model; axes=:auto, orders=(5,9), max_order=17, ...)

Online continuous conditional density with nonlinear physics transport. Calibrated
prior/noise values remain fixed. No particle population, random draws, resampling,
or Gaussian state ellipse is the returned posterior. `belief_at` returns the
distribution; parameter CDFs use `conditional_cdf`. Nonlinear moment/decision queries
need a separately checked deterministic integration budget.

Refinement compares continuous joint densities in their common innovation chart.
Before the first marginalization it may add a finer conditional bank from the full
graph, up to the declared budget. After compression it will reject an unresolved
arrival rather than reconstruct lost nonlinear history from an anchor Gaussian.
The default retains full history: Gaussian fixed-lag compression can erase precisely
the nonlinear dependence being retained. A finite `lag` is an explicit experimental
tradeoff. With `axes=:auto`, parameter directions are selected from posterior shape
defects, without sensor-name rules or truth; at most `max_axes` are integrated.
Finite shape probes and quadrature convergence are NOT held-out consistency proof.

LEARN-THEN-FREEZE (`freeze_stable`): the chart can only be re-centred against full
history, so the smoother keeps every factor while the integrated latents are still
being learned (the ignition pair: during lighting). After `freeze_stable` consecutive
committed arrivals that needed no re-centre or refinement, the chart freezes: every
smoother's lag drops to `hold_lag` and compaction proceeds. Each component remains a
Gaussian smoother CONDITIONED on its fixed chart point, so the nonlinear dependence on
the integrated latents is kept for the rest of the cook. After freezing an unresolved
arrival is rejected (re-centring would need the discarded history), never degraded.
`freeze_stable=nothing` keeps full history for the whole cook.
"""
function deterministic_init(model::SystemModel;axes=:auto,orders=(5,9),max_order=17,max_nodes=729,
                            density_tol=0.025,shape_tol=0.25,max_axes=3,lag=typemax(Int),
                            freeze_stable=nothing,hold_lag=3,kwargs...)
    0<density_tol<1 && shape_tol>0 || throw(ArgumentError("invalid density/shape tolerances"))
    freeze_stable===nothing || (freeze_stable>=1 && hold_lag>=0) ||
        throw(ArgumentError("freeze_stable must be positive and hold_lag non-negative"))
    max_order>=last(orders) || throw(ArgumentError("max_order precedes the initial rules"))
    1<=max_axes<=3 || throw(ArgumentError("continuous reconstruction supports one to three axes"))
    auto=axes===:auto
    selected=auto ? [1] : axes
    length(selected)<=max_axes || throw(ArgumentError("initial axes exceed max_axes"))
    bank=conditional_init(model;axes=selected,orders,max_nodes,lag,kwargs...)
    off,_=_offsets(bank.anchor.graph);ia=[off[k][i] for (k,i) in bank.axes]
    C=joint_cov(bank.anchor.graph)[1]
    center=[bank.anchor.graph.values[k][i] for (k,i) in bank.axes]
    scale=Matrix(cholesky(Symmetric(C[ia,ia])).L)
    a=_density_from_bank(bank,bank.banks[end-1],center,scale)
    b=_density_from_bank(bank,bank.banks[end],center,scale)
    r=conditional_density_distance(a,b;atol=density_tol/4)
    r.tv_bound_estimate<=density_tol && r.integration_change<=density_tol/4 &&
        r.normalization_error<=density_tol/10 || error("initial conditional density is unresolved")
    diagnostic=merge(r,(resolved=true,conditional_gaussian_validated=false,inference=:conditional_transport))
    return DeterministicSmoother(bank,b,center,scale,Float64(density_tol),Float64(shape_tol),max_order,max_nodes,auto,max_axes,diagnostic,
                                 freeze_stable===nothing ? nothing : Int(freeze_stable),Int(hold_lag),false,0,0,0)
end

# Freeze the chart: every smoother (anchor and each component) compacts to `hold_lag` from now on.
function _freeze_chart!(sm::DeterministicSmoother)
    for s in vcat([sm.bank.anchor],[inner for b in sm.bank.banks for inner in b.smoothers])
        s.lag=sm.hold_lag;s.lag_max=sm.hold_lag
    end
    sm.frozen=true
    _compact_conditional_banks!(sm.bank)
    return sm
end

# New nodes can be fitted without substituting a Gaussian history ONLY while the
# anchor still contains the full original factors. Clone the whole smoother so
# graph factors and the dynamics/discrepancy registries retain their identities.
function _refine_density_bank!(sm::DeterministicSmoother,order)
    anchor=sm.bank.anchor;g=anchor.graph;axes=sm.bank.axes
    any(f->f isa LinearFactor,g.factors) && return false
    order^big(length(axes))<=sm.max_nodes || return false
    qs,lws=_tensor_quadrature(order,length(axes);max_nodes=sm.max_nodes)
    off,n=_offsets(g);ia=[off[k][i] for (k,i) in axes]
    C=joint_cov(g)[1];mu=_stacked(g,g.keys,off,n);slope=C[:,ia]/C[ia,ia]
    inners=Vector{FixedLagSmoother}(undef,length(qs));nodes=Vector{ConditionalNode}(undef,length(qs))
    Threads.@threads for j in eachindex(qs)
      let inner=deepcopy(anchor),point=sm.center+sm.scale*qs[j],c=nothing
        start=mu+slope*(point-mu[ia])
        for k in inner.graph.keys;inner.graph.values[k].=start[off[k]];end
        for _ in 1:inner.restarts+1
            c=_conditional_node!(inner.graph,axes,point;iters=inner.iters,tol=inner.tol,ftol=inner.ftol,
                step_bounds=Dict(inner.key_z=>inner.latent_step),newton_tol=ONLINE_NEWTON_TOL)
            c.solve.converged && break
        end
        inners[j]=inner;nodes[j]=c
      end
    end
    _retry_conditional_nodes!(inners,nodes,axes)
    all(c->c.solve.converged,nodes) || return false
    logs=[lw+.5sum(abs2,q)+c.log_integral for (lw,q,c) in zip(lws,qs,nodes)]
    shift=maximum(logs);logs .-= shift+log(sum(exp.(logs.-shift)))
    push!(sm.bank.banks,_ConditionalBank(order,inners,nodes,logs))
    return true
end

function _rebuild_density_axes!(sm,indices)
    anchor=sm.bank.anchor
    any(f->f isa LinearFactor,anchor.graph.factors) && return false
    length(indices)<=sm.max_axes || return false
    orders=[b.order for b in sm.bank.banks[1:2]]
    last(orders)^big(length(indices))<=sm.max_nodes || return false
    previous=(axes=sm.bank.axes,center=sm.center,scale=sm.scale,banks=sm.bank.banks)
    sm.bank.axes=[(anchor.key_z,i) for i in indices]
    prior=[z_moments(anchor.model.blk.sites[i]) for i in indices]
    sm.center=[p[1] for p in prior]
    sm.scale=Matrix(Diagonal(sqrt.([p[2] for p in prior])))
    sm.bank.banks=_ConditionalBank[]
    ok=all(_refine_density_bank!(sm,o) for o in orders)
    if !ok
        sm.bank.axes=previous.axes;sm.center=previous.center;sm.scale=previous.scale;sm.bank.banks=previous.banks
    end
    return ok
end

function _unconditional_shape(sm)
    anchor=sm.bank.anchor
    c=_conditional_node!(deepcopy(anchor.graph),Tuple{VarKey,Int}[],Float64[];
        iters=anchor.iters,tol=anchor.tol,ftol=anchor.ftol,
        step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=ONLINE_NEWTON_TOL)
    c.solve.converged || return nothing
    p=_conditional_posterior([c],[0.0],Tuple{VarKey,Int}[])
    return conditional_shape_diagnostic(conditional_transport(p);key=anchor.key_z)
end

function _recenter_density_bank!(sm,d)
    any(f->f isa LinearFactor,sm.bank.anchor.graph.factors) && return false
    # Integrate where the current approximation places mass. This changes only
    # the numerical chart; all nodes are refitted to the original full factors.
    # It neither treats the previous approximation as new data nor fits a prior.
    movement=norm(d.base_mean)
    stretch=maximum(abs,log.(svdvals(d.base_root)))
    max(movement,stretch)>0.1 || return false
    orders=[b.order for b in sm.bank.banks[1:2]]
    previous=(center=sm.center,scale=sm.scale,banks=sm.bank.banks)
    sm.center=sm.center+sm.scale*d.base_mean
    sm.scale=sm.scale*d.base_root
    sm.bank.banks=_ConditionalBank[]
    ok=all(_refine_density_bank!(sm,o) for o in orders)
    if !ok
        sm.center=previous.center;sm.scale=previous.scale;sm.bank.banks=previous.banks
    end
    return ok
end

function _deterministic_advance!(sm::DeterministicSmoother,recs;verbose=false)
    solved=_solve_conditional_banks!(sm.bank,recs;verbose)
    solved.ok || return solved
    attempted=Set{Tuple}([Tuple(i for (_,i) in sm.bank.axes)])
    restarted_axes=false;recenters=0;refines=0;rebuilt=false
    while true
        banks=sm.bank.banks
        a=_density_from_bank(sm.bank,banks[end-1],sm.center,sm.scale)
        b=_density_from_bank(sm.bank,banks[end],sm.center,sm.scale)
        r=conditional_density_distance(a,b;atol=sm.density_tol/4)
        resolved=r.tv_bound_estimate<=sm.density_tol && r.integration_change<=sm.density_tol/4 &&
                 r.normalization_error<=sm.density_tol/10
        d=merge(r,(resolved=resolved,axes=[sm.bank.anchor.model.blk.names[i] for (_,i) in sm.bank.axes],
            orders=[x.order for x in banks],nodes=[length(x.nodes) for x in banks],
            conditional_gaussian_validated=false,inference=:conditional_transport))
        shape=conditional_shape_diagnostic(b.transported;key=sm.bank.anchor.key_z)
        d=merge(d,(shape_defect=shape.maximum_defect,shape_checked_mass=shape.checked_mass))
        if shape.maximum_defect>sm.shape_tol && sm.adaptive_axes
            selected=[i for (_,i) in sm.bank.axes]
            # Reconsider the block from the unconditioned posterior once per
            # arrival. Previously important directions need not stay important.
            if !restarted_axes
                base=_unconditional_shape(sm)
                proposed=base===nothing ? selected : [argmax(base.scores)]
                restarted_axes=true
            else
                proposed=selected
            end
            if Tuple(proposed) in attempted
                remaining=setdiff(eachindex(shape.scores),selected)
                proposed=isempty(remaining) ? selected : vcat(selected,remaining[argmax(shape.scores[remaining])])
            end
            if !(Tuple(proposed) in attempted) && _rebuild_density_axes!(sm,proposed)
                push!(attempted,Tuple(proposed));recenters=0;rebuilt=true;continue
            end
            return (ok=false,reason=:conditional_shape,diagnostics=d,candidate=b)
        end
        if !resolved
            if recenters<2 && _recenter_density_bank!(sm,b)
                recenters+=1;continue
            end
            order=min(sm.max_order,banks[end].order+4)
            if order>banks[end].order && _refine_density_bank!(sm,order)
                refines+=1;continue
            end
            return (ok=false,reason=:density_resolution,diagnostics=d,candidate=b)
        end
        shape.maximum_defect<=sm.shape_tol ||
            return (ok=false,reason=:conditional_shape,diagnostics=d,candidate=b)
        sm.posterior=b;sm.diagnostics=d
        sm.bank.posterior=b.transported.source
        _compact_conditional_banks!(sm.bank)
        return (ok=true,diagnostics=merge(d,(recentres=recenters,refines=refines)),
                adapted=recenters>0 || refines>0 || rebuilt,recenters=recenters,refines=refines)
    end
end

function assimilate!(sm::DeterministicSmoother,recs::AbstractVector{MeasurementRecord};verbose=false)
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    for r in recs
        isfinite(r.t) && isfinite(r.value) || throw(ArgumentError("measurements must be finite"))
        sensor(sm.bank.anchor.model.log,r.sensor)
        r.t>=last(sm.bank.anchor.nodes)[1] || throw(ArgumentError("online arrivals must be chronological"))
    end
    saved=deepcopy(sm);start=time()
    function restore!()
        for i in 1:fieldcount(DeterministicSmoother);setfield!(sm,i,getfield(saved,i));end
    end
    try
        anchor=sm.bank.anchor
        sorted=sort(collect(recs);by=r->r.t,alg=MergeSort)
        # Under :slice the anchor splits the batch at the nodes it will make; otherwise per timestamp.
        groups=if anchor.attach===:slice
            _slice_chunks(anchor,sorted)
        else
            gs=Vector{MeasurementRecord}[];first=1
            while first<=length(sorted)
                finish=first
                while finish<length(sorted) && sorted[finish+1].t==sorted[first].t;finish+=1;end
                push!(gs,sorted[first:finish]);first=finish+1
            end
            gs
        end
        result=nothing;recentres=0;refines=0;froze=false
        for g in groups
            result=_deterministic_advance!(sm,g;verbose)
            if !result.ok
                restore!()
                return _deterministic_record(sm,start,length(groups),false,
                    (reason=result.reason,diagnostics=result.diagnostics))
            end
            recentres+=result.recenters;refines+=result.refines
            if !sm.frozen
                sm.stable=result.adapted ? 0 : sm.stable+1
                if sm.freeze_stable!==nothing && sm.stable>=sm.freeze_stable
                    _freeze_chart!(sm);froze=true
                end
            end
        end
        sm.n_recentres+=recentres;sm.n_refines+=refines
        record=_deterministic_record(sm,start,length(groups),true,
            (diagnostics=result.diagnostics,recentres=recentres,refines=refines,froze=froze))
        _push_record!(anchor,record)
        return merge(record,(posterior=sm.posterior,))
    catch
        restore!();rethrow()
    end
end
assimilate!(sm::DeterministicSmoother,rec::MeasurementRecord;kwargs...)=assimilate!(sm,[rec];kwargs...)

# The record a consumer (bench, planner) reads: the quadrature mixture's moments at the newest node
# (on rejection the standing belief, after rollback), the phase, and the chart's adaptation counts.
# Innovation statistics are deliberately empty: a mixture predictive is not Gaussian (a PIT is the
# consistency statistic still to build).
function _deterministic_record(sm::DeterministicSmoother,start,nchunks,ok,extra)
    anchor=sm.bank.anchor;t,ky=last(anchor.nodes)
    belief=_conditional_belief(sm.bank.posterior,anchor.model,anchor.key_z,ky,t)
    merge(belief,(t_belief=belief.t,n_nodes=length(anchor.nodes),chunks=nchunks,
                  phase=sm.frozen ? :hold : :learn,stable=sm.stable,
                  recentres_total=sm.n_recentres,refines_total=sm.n_refines,
                  bank_nodes=[length(b.nodes) for b in sm.bank.banks],
                  nis=NaN,dof=0,innov_std=Dict{Symbol,Float64}(),innov_rows=NamedTuple[],
                  innovation_whitened=NamedTuple[],iters=0,attempts=1,gradient=NaN,
                  newton_sd=NaN,newton_tol=ONLINE_NEWTON_TOL,newton_joint_sd=NaN,joint_tol=NaN,
                  inference=:conditional_transport),
          extra,(committed=ok,converged=ok,wall=time()-start))
end
belief_at(sm::DeterministicSmoother)=sm.posterior
window_nodes(sm::DeterministicSmoother)=window_nodes(sm.bank)
conditional_density(sm::DeterministicSmoother)=sm.posterior
conditional_transport(sm::DeterministicSmoother)=sm.posterior.transported
update_execution!(sm::DeterministicSmoother;kwargs...)=(update_execution!(sm.bank;kwargs...);sm)

"Offline full-history deterministic inference through the same continuous-density updates."
function deterministic_fit(model::SystemModel,recs::AbstractVector{MeasurementRecord};kwargs...)
    haskey(kwargs,:lag) && throw(ArgumentError("deterministic_fit retains full history"))
    sm=deterministic_init(model;lag=typemax(Int),kwargs...)
    isempty(recs) && return (t=last(sm.bank.anchor.nodes)[1],posterior=sm.posterior,
        committed=true,converged=true,wall=0.0,diagnostics=sm.diagnostics)
    return assimilate!(sm,sort(collect(recs);by=r->r.t,alg=MergeSort))
end
