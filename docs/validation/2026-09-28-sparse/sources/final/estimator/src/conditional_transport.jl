export TransportedConditionalPosterior, conditional_transport, conditional_shape_diagnostic,
       conditional_moments

struct _ConditionalTransport
    graph::FactorGraph
    offsets::Dict{VarKey,UnitRange{Int}}
    mean::Vector{Float64}
    root::Matrix{Float64}
    dynamics::Vector{DynamicsFactor}
end

function _conditional_transport(c::ConditionalNode)
    g=deepcopy(c.graph);off,_=_offsets(g);u=copy(c.mean);L=copy(c.root)
    dynamics=sort([f for f in g.factors if f isa DynamicsFactor];by=f->f.t1)
    for f in dynamics
        lin=factor_linearization(f,[g.values[k] for k in vars(f)])
        u[off[f.key_y1]]=lin.r
        L[off[f.key_y1],:].=0.0
        for (k,J) in zip(vars(f),lin.Js)
            L[off[f.key_y1],:].+=J*c.root[off[k],:]
        end
    end
    return _ConditionalTransport(g,off,u,L,dynamics)
end

function _conditional_decode(ref::_ConditionalTransport,u)
    x=copy(u)
    for f in ref.dynamics
        z=f.key_m===nothing ? x[ref.offsets[f.key_z]] :
            assemble_latents(f.split,x[ref.offsets[f.key_z]],x[ref.offsets[f.key_m]])
        y=propagate(f.model,x[ref.offsets[f.key_y0]],z,f.t0,f.t1;first_slice=f.first)[1]
        x[ref.offsets[f.key_y1]]=y+f.cache.W\u[ref.offsets[f.key_y1]]
    end
    return x
end

"""
    TransportedConditionalPosterior

A mixture of conditional Gaussians in parameter/process-innovation coordinates,
mapped NONLINEARLY into states by the executed physics. This keeps parameter/state
curvature that any state-coordinate Gaussian loses. It is an implicit distribution,
not a collection of Monte Carlo samples. Its moments require numerical integration.
"""
struct TransportedConditionalPosterior
    source::ConditionalPosterior
    components::Vector{_ConditionalTransport}
end

"""
    conditional_transport(p::ConditionalPosterior)

Change each local Gaussian's coordinates from states to process innovations, then
retain the nonlinear inverse map. Fixed process whiteners give a triangular map
with a CONSTANT Jacobian determinant; the normalized component evidence is unchanged.
This is a different approximation to the same posterior, not merely a different
plot of the state Gaussian. It does not restore nonlinear information already lost
through fixed-lag marginalization. State variables chosen as integration axes cannot
be outgoing states of a dynamics factor.
"""
function conditional_transport(p::ConditionalPosterior)
    components=[_conditional_transport(c) for c in p.nodes]
    generated=Set(f.key_y1 for f in first(components).dynamics)
    any(k in generated for (k,_) in p.axes) &&
        throw(ArgumentError("transport currently requires integration axes outside the generated states"))
    return TransportedConditionalPosterior(p,components)
end

"""
    conditional_points(p::TransportedConditionalPosterior; keys=p.source.keys, max_points=100_000)

Positive spherical-radial cubature in the conditional innovation coordinates,
followed by the nonlinear dynamics. This third-degree rule is a deterministic
integration approximation, NOT a certificate for higher moments or tail quantiles.
No Gaussian is substituted for the decoded joint points.
"""
function conditional_points(p::TransportedConditionalPosterior;keys=p.source.keys,max_points=100_000)
    count=sum(max(1,2size(c.root,2)) for c in p.components)
    count<=max_points || throw(ArgumentError("conditional moment integration needs $count points, above max_points=$max_points"))
    idx=reduce(vcat,[collect(p.source.offsets[k]) for k in keys];init=Int[])
    X=zeros(length(idx),count);weights=zeros(count);j=0
    for (wi,c) in zip(p.source.weights,p.components)
        d=size(c.root,2)
        if d==0
            j+=1;X[:,j]=_conditional_decode(c,c.mean)[idx];weights[j]=wi
        else
            for i in 1:d,sign in (-1,1)
                j+=1;X[:,j]=_conditional_decode(c,c.mean+sign*sqrt(d)*c.root[:,i])[idx]
                weights[j]=wi/(2d)
            end
        end
    end
    return (points=X,weights=weights)
end

function conditional_expectation(f,p::TransportedConditionalPosterior;kwargs...)
    q=conditional_points(p;kwargs...)
    return sum(w*f(x) for (w,x) in zip(q.weights,eachcol(q.points)))
end

function conditional_moments(p::Union{ConditionalPosterior,TransportedConditionalPosterior};kwargs...)
    q=conditional_points(p;kwargs...)
    mean=q.points*q.weights;delta=q.points.-mean
    return (mean=mean,covariance=Matrix(Symmetric((delta.*q.weights')*delta')),
            rule=:spherical_radial,points=length(q.weights),higher_moments_validated=false)
end

"""
    conditional_shape_diagnostic(p; key, max_components=3)

Check exact log-density departures from the local quadratic along +/- one-sd
conditional directions of the coordinates in `key`. The largest-weight components
are checked without any simulation truth, sensor-name assumptions, or random draws.
Works in state coordinates for `ConditionalPosterior`, or with nonlinear physics
decoding for `TransportedConditionalPosterior`. A large defect disproves the local
Gaussian assumption; a small defect in these finite probes does NOT prove accuracy.
`scores` are posterior-weighted absolute log-density defects on the checked subset.
"""
function conditional_shape_diagnostic(p::Union{ConditionalPosterior,TransportedConditionalPosterior};
                                      key::VarKey,max_components::Int=3)
    max_components>0 || throw(ArgumentError("check at least one component"))
    transported=p isa TransportedConditionalPosterior
    source=transported ? p.source : p
    ids=sortperm(source.weights;rev=true)[1:min(max_components,length(source.weights))]
    scores=zeros(length(source.offsets[key]));evaluations=0
    mass=sum(source.weights[ids])
    for j in ids
        c=source.nodes[j]
        ref=transported ? p.components[j] : nothing
        g=deepcopy(transported ? ref.graph : c.graph);off,_=_offsets(g)
        root=transported ? ref.root : c.root
        J,r,_,_=linearize(g);F=objective(g)
        for i in eachindex(scores)
            row=off[key][i];sd=norm(root[row,:]);sd>0 || continue
            direction=root[row,:]/sd
            delta=c.root*direction;linear=dot(r,J*delta);quadratic=sum(abs2,J*delta)/2
            xp=transported ? _conditional_decode(ref,ref.mean+root*direction) : c.mean+delta
            xm=transported ? _conditional_decode(ref,ref.mean-root*direction) : c.mean-delta
            fp=_objective_at(g,xp,off);fm=_objective_at(g,xm,off)
            defect=max(abs(fp-F-linear-quadratic),abs(fm-F+linear-quadratic))
            scores[i]+=source.weights[j]*defect/mass;evaluations+=2
        end
    end
    return (scores=scores,maximum_defect=maximum(scores),evaluations=evaluations,
            checked_mass=mass,components=ids,coordinates=transported ? :innovations : :states,
            validated=false)
end
