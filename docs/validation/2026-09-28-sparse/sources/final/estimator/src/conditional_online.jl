export ConditionalSmoother, conditional_init

mutable struct _ConditionalBank
    order::Int
    smoothers::Vector{FixedLagSmoother}
    nodes::Vector{ConditionalNode}
    logweights::Vector{Float64}
end

"A bounded-window bank of conditional factor graphs and their accumulated log evidence."
mutable struct ConditionalSmoother
    anchor::FixedLagSmoother
    axes::Vector{Tuple{VarKey,Int}}
    banks::Vector{_ConditionalBank}
    posterior::ConditionalPosterior
    mean_tol::Float64
    covariance_tol::Float64
end

"""
    conditional_init(model; axes, orders=(5,9), lag=3, max_nodes=729, ...)

Deterministic online conditional-Laplace estimator. `axes` names (or indices) the
static latent block selected by OFFLINE calibration/approximation analysis, e.g.
`[:a0,:k_spread]` for the existing ignition calibration. This is explicit because
there is no evidence that the same two directions suffice for every cook or suite.
The factor/sensor interfaces are unchanged; no sensor identities are hard-coded.

Each positive quadrature node retains its own fixed-lag graph, joint covariance,
and accumulated evidence. At least two rules are updated to check integration
resolution. An unresolved update rolls back; it never silently becomes Gaussian
or launches MC. Node locations are fixed for this cook; choose the rule budget
offline. This experimental backend still requires validation of the conditional
Gaussian approximation and lag on held-out cooks.
"""
function conditional_init(model::SystemModel;axes,orders=(5,9),max_nodes=729,
                          mean_tol=0.05,covariance_tol=0.1,kwargs...)
    length(orders)>=2 && issorted(collect(orders)) && allunique(orders) ||
        throw(ArgumentError("at least two strictly increasing quadrature rules are required"))
    mean_tol>0 && covariance_tol>0 || throw(ArgumentError("resolution tolerances must be positive"))
    any(k->haskey(kwargs,k),(:als,:drift,:graph,:hierarchy)) &&
        throw(ArgumentError("online conditional inference currently requires static single-cook latents; calibrate noise/hierarchies offline"))
    anchor=online_init(model;kwargs...)
    indices=[a isa Integer ? Int(a) : latent_index(model.blk,Symbol(a)) for a in axes]
    selected=[(anchor.key_z,i) for i in indices]
    off,_=_offsets(anchor.graph);ia=[off[k][i] for (k,i) in selected]
    C=joint_cov(anchor.graph)[1]
    center=anchor.graph.values[anchor.key_z][indices]
    scale=Matrix(cholesky(Symmetric(C[ia,ia])).L)
    banks=_ConditionalBank[];posteriors=ConditionalPosterior[]
    for order in orders
        p=_conditional_rule(anchor.graph,selected,center,scale,order;max_nodes,
                            iters=anchor.iters,tol=anchor.tol,ftol=anchor.ftol,
                            step_bounds=Dict(anchor.key_z=>anchor.latent_step),newton_tol=ONLINE_NEWTON_TOL)
        p.diagnostics.solves_ok || error("conditional prior solve did not converge")
        qs,lws=_tensor_quadrature(order,length(selected);max_nodes)
        logs=[lw+sum(abs2,q)/2+c.log_integral for (q,lw,c) in zip(qs,lws,p.nodes)]
        # Keep normalized LOG weights, including paths whose displayed mass may
        # underflow later. Future observations can revive those paths.
        logs .-= maximum(logs)+log(sum(exp.(logs.-maximum(logs))))
        sms=map(p.nodes) do c
            sm=deepcopy(anchor);sm.graph=c.graph;sm
        end
        push!(banks,_ConditionalBank(order,sms,p.nodes,logs));push!(posteriors,p)
    end
    r=_conditional_resolution(posteriors[end-1],posteriors[end])
    resolved=r.mean_error<=mean_tol && r.covariance_error<=covariance_tol
    p=posteriors[end]
    d=merge(p.diagnostics,r,(resolved=resolved,inference=:conditional_laplace,
            conditional_gaussian_validated=false,lag=anchor.lag))
    posterior=ConditionalPosterior(p.keys,p.offsets,p.axes,deepcopy(p.nodes),p.weights,p.mean,p.covariance,d)
    resolved || error("conditional prior integration is unresolved at the configured rules")
    return ConditionalSmoother(anchor,selected,banks,posterior,Float64(mean_tol),Float64(covariance_tol))
end

window_nodes(sm::ConditionalSmoother)=window_nodes(sm.anchor)
conditional_transport(sm::ConditionalSmoother)=conditional_transport(sm.posterior)

function _conditional_current(bank,axes;diagnostics=(;))
    _conditional_posterior(bank.nodes,bank.logweights,axes;diagnostics)
end

# A difficult conditional node may be far from the anchor's linear prediction.
# Retry from nearby converged conditional modes, in a fixed order. No failed node
# is accepted or dropped because its current quadrature weight happens to be small.
function _retry_conditional_nodes!(smoothers,nodes,axes)
    coordinates=reduce(hcat,[c.coordinates for c in nodes])
    scales=max.(vec(maximum(coordinates;dims=2)-minimum(coordinates;dims=2)),eps(Float64))
    for j in eachindex(nodes)
        inner=smoothers[j]
        (nodes[j].solve.converged || inner.iters==0 || inner.restarts==0) && continue
        neighbors=filter(i->i!=j && nodes[i].solve.converged,eachindex(nodes))
        sort!(neighbors;by=i->sum(abs2,(coordinates[:,i]-coordinates[:,j])./scales))
        off,n=_offsets(inner.graph)
        for i in neighbors[1:min(inner.restarts,length(neighbors))]
            for k in inner.graph.keys;inner.graph.values[k].=nodes[i].mean[off[k]];end
            nodes[j]=_conditional_node!(inner.graph,axes,coordinates[:,j];iters=inner.iters,
                tol=inner.tol,ftol=inner.ftol,step_bounds=Dict(inner.key_z=>inner.latent_step),
                newton_tol=ONLINE_NEWTON_TOL)
            nodes[j].solve.converged && break
        end
    end
    return nodes
end

function _solve_conditional_banks!(sm::ConditionalSmoother,recs;verbose=false)
    anchor=sm.anchor
    for rec in recs;add_measurement!(anchor,rec);end
    proposal=solve_window!(anchor;verbose)
    proposal.converged || return (ok=false,reason=:anchor_solve,diagnostics=proposal)
    for bank in sm.banks
        candidates=Vector{ConditionalNode}(undef,length(bank.nodes))
        # Components own their graphs and caches. Their solves are independent;
        # reduction of weights remains ordered and deterministic after the join.
        Threads.@threads for j in eachindex(bank.nodes)
          let inner=bank.smoothers[j],old=bank.nodes[j],c=nothing
            for rec in recs;add_measurement!(inner,rec);end
            # A single predictable process model for every component and rule.
            # Per-component re-anchoring would compare different likelihoods and
            # introduce component-dependent normalizing constants.
            for (edge,f) in inner.dyn
                a=anchor.dyn[edge].cache
                f.cache=merge(f.cache,(W=copy(a.W),wy=copy(a.wy)))
            end
            for _ in 1:inner.restarts+1
                c=_conditional_node!(inner.graph,sm.axes,old.coordinates;iters=inner.iters,
                    tol=inner.tol,ftol=inner.ftol,step_bounds=Dict(inner.key_z=>inner.latent_step),
                    newton_tol=ONLINE_NEWTON_TOL,
                    stationarity_scales=Dict(k=>[sensor(inner.model.log,id).noise.ar_sd]
                        for (id,chain) in inner.disc for (_,k) in chain))
                c.solve.converged && break
            end
            candidates[j]=c
          end
        end
        _retry_conditional_nodes!(bank.smoothers,candidates,sm.axes)
        for j in eachindex(bank.nodes)
            c=candidates[j];old=bank.nodes[j]
            c.solve.converged || return (ok=false,reason=:conditional_solve,
                diagnostics=(order=bank.order,node=j,coordinates=c.coordinates,solve=c.solve))
            # Ratio of conditional integrals, not a fresh fit to the surviving
            # factors. Constants common to every node cancel between weights.
            bank.logweights[j] += c.log_integral-old.log_integral
            bank.nodes[j]=c
        end
        shift=maximum(bank.logweights)
        bank.logweights .-= shift+log(sum(exp.(bank.logweights.-shift)))
    end
    return (ok=true,)
end

function _compact_conditional_banks!(sm::ConditionalSmoother)
    # Compression changes additive constants and conditional dimension. Store
    # the NEW local integral as next arrival's denominator, keeping all accumulated
    # evidence in logweights. Otherwise old measurements are double-counted/lost.
    for bank in sm.banks, j in eachindex(bank.nodes)
        inner=bank.smoothers[j]
        removed=marginalize_old!(inner)
        if !isempty(removed)
            bank.nodes[j]=_conditional_node!(inner.graph,sm.axes,bank.nodes[j].coordinates;
                iters=0,tol=inner.tol,ftol=inner.ftol)
        end
    end
    marginalize_old!(sm.anchor)
    return sm
end

function _conditional_advance!(sm::ConditionalSmoother,recs;verbose=false)
    solved=_solve_conditional_banks!(sm,recs;verbose)
    solved.ok || return solved
    anchor=sm.anchor
    fine=_conditional_current(sm.banks[end],sm.axes)
    coarse=_conditional_current(sm.banks[end-1],sm.axes)
    r=_conditional_resolution(coarse,fine;keys=[anchor.key_z,last(anchor.nodes)[2]])
    resolved=r.mean_error<=sm.mean_tol && r.covariance_error<=sm.covariance_tol
    d=merge(r,(resolved=resolved,inference=:conditional_laplace,
        conditional_gaussian_validated=false,lag=anchor.lag,
        orders=[b.order for b in sm.banks],nodes=[length(b.nodes) for b in sm.banks]))
    p=ConditionalPosterior(fine.keys,fine.offsets,fine.axes,deepcopy(fine.nodes),
                            copy(fine.weights),copy(fine.mean),copy(fine.covariance),d)
    resolved || return (ok=false,reason=:quadrature_resolution,diagnostics=d,candidate=p)
    sm.posterior=p
    _compact_conditional_banks!(sm)
    return (ok=true,diagnostics=d)
end

function assimilate!(sm::ConditionalSmoother,recs::AbstractVector{MeasurementRecord};verbose=false)
    isempty(recs) && throw(ArgumentError("nothing to assimilate"))
    for r in recs
        isfinite(r.t) && isfinite(r.value) || throw(ArgumentError("measurements must be finite"))
        sensor(sm.anchor.model.log,r.sensor)
        r.t>=last(sm.anchor.nodes)[1] || throw(ArgumentError("conditional online arrivals must be chronological"))
    end
    saved=deepcopy(sm);start=time()
    function restore!()
        for i in 1:fieldcount(ConditionalSmoother);setfield!(sm,i,getfield(saved,i));end
    end
    try
        sorted=sort(collect(recs);by=r->r.t,alg=MergeSort)
        first=1;result=nothing
        while first<=length(sorted)
            finish=first
            while finish<length(sorted) && sorted[finish+1].t==sorted[first].t;finish+=1;end
            result=_conditional_advance!(sm,sorted[first:finish];verbose)
            if !result.ok
                restore!()
                return merge(result,(committed=false,converged=false,wall=time()-start))
            end
            first=finish+1
        end
        belief=belief_at(sm)
        record=merge(belief,(committed=true,converged=true,inference=:conditional_laplace,
                            diagnostics=result.diagnostics,wall=time()-start))
        _push_record!(sm.anchor,record)
        return merge(record,(posterior=sm.posterior,))
    catch
        restore!();rethrow()
    end
end
assimilate!(sm::ConditionalSmoother,rec::MeasurementRecord;kwargs...)=assimilate!(sm,[rec];kwargs...)

function _conditional_belief(p,model,kz,ky,t)
    rz=p.offsets[kz];ry=p.offsets[ky]
    q=conditional_points(p;keys=[ky])
    X=reduce(hcat,[belief_decode(y,model.bl) for y in eachcol(q.points)])
    x=X*q.weights;dx=X.-x
    Cx=Matrix(Symmetric((dx.*q.weights')*dx'));pieces=Dict{Symbol,NamedTuple}()
    for id in model.bl.sys.pieces
        pc=piece(model.sys,id);rT=belief_index(model.bl,id,:T)
        if pc isa GridPiece
            ic=rT[pc.mesh.core];weights=zeros(length(rT))
            for (node,area) in pc.mesh.boundary;weights[node]+=area/pc.mesh.area;end
            surface=dot(weights,x[rT]);variance=dot(weights,Cx[rT,rT]*weights)
            pieces[id]=(T_centre=(x[ic],sqrt(max(Cx[ic,ic],0.0))),T_surface=(surface,sqrt(max(variance,0.0))))
        else
            ic,is=first(rT),last(rT)
            pieces[id]=(T_centre=(x[ic],sqrt(max(Cx[ic,ic],0.0))),T_surface=(x[is],sqrt(max(Cx[is,is],0.0))))
        end
    end
    ip=first(belief_index(model.bl,:pit,:T_pit));joint=vcat(collect(rz),collect(ry))
    return (t=Float64(t),z=p.mean[rz],y=p.mean[ry],x=x,z_cov=p.covariance[rz,rz],
            y_cov=p.covariance[ry,ry],zy_cov=p.covariance[joint,joint],x_cov=Cx,
            T_pit=(x[ip],sqrt(max(Cx[ip,ip],0.0))),pieces=pieces)
end
function belief_at(sm::ConditionalSmoother)
    p=sm.posterior
    p.diagnostics.resolved || throw(ArgumentError("conditional integration is unresolved"))
    t,ky=last(sm.anchor.nodes)
    return _conditional_belief(p,sm.anchor.model,sm.anchor.key_z,ky,t)
end

function update_execution!(sm::ConditionalSmoother;kwargs...)
    # Validate once before touching any component. Existing factors retain their
    # own historical model snapshots, as in the underlying graph backend.
    update_execution!(sm.anchor;kwargs...)
    for bank in sm.banks, inner in bank.smoothers
        update_execution!(inner;kwargs...)
    end
    return sm
end

"Offline full-history fit through the same declared factors; calibration remains outside live assimilation."
function conditional_fit(model::SystemModel,recs::AbstractVector{MeasurementRecord};axes,
                         iters=60,tol=1e-6,ftol=1e-8,kwargs...)
    inner=online_init(model;lag=typemax(Int),iters,tol,ftol)
    isempty(recs) || _append_sampled_measurements!(inner,sort(collect(recs);by=r->r.t,alg=MergeSort))
    indices=[a isa Integer ? Int(a) : latent_index(model.blk,Symbol(a)) for a in axes]
    selected=[(inner.key_z,i) for i in indices]
    return conditional_fit(inner.graph;axes=selected,iters,tol,ftol,
        check_keys=[inner.key_z,last(inner.nodes)[2]],step_bounds=Dict(inner.key_z=>inner.latent_step),
        newton_tol=ONLINE_NEWTON_TOL,kwargs...)
end
