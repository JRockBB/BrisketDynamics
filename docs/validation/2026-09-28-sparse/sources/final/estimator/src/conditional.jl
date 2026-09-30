# Deterministic nested integration. The retained distribution is a collection of
# conditional Gaussians, not a Gaussian fitted to the collection's moments.
export ConditionalPosterior, conditional_fit, conditional_expectation, conditional_points

struct _ConditionedFactor <: Factor
    parent::Factor
    fixed::Dict{VarKey,Vector{Float64}}
    free::Dict{VarKey,Vector{Int}}
end
vars(f::_ConditionedFactor) = vars(f.parent)
ndim(f::_ConditionedFactor) = ndim(f.parent)
provenance(f::_ConditionedFactor) = "conditional " * provenance(f.parent)
function _expand_condition(f::_ConditionedFactor, vals)
    map(zip(vars(f), vals)) do (k, v)
        x = f.fixed[k] .+ zero(eltype(v))
        x[f.free[k]] = v
        x
    end
end
residual(f::_ConditionedFactor, vals) = residual(f.parent, _expand_condition(f, vals))
loss(f::_ConditionedFactor, vals) = loss(f.parent, _expand_condition(f, vals))
function factor_linearization(f::_ConditionedFactor, vals)
    lin = factor_linearization(f.parent, _expand_condition(f, vals))
    return (r=lin.r, Js=[J[:, f.free[k]] for (J,k) in zip(lin.Js,vars(f))])
end
jacobians(f::_ConditionedFactor, vals) = factor_linearization(f, vals).Js

# Exact coordinate elimination: no small-variance pseudo-priors, penalty scale,
# or fictitious information in a fixed integration coordinate.
function _condition_graph(g, axes, values)
    length(axes) == length(values) || throw(DimensionMismatch("one value per integration axis"))
    length(unique(axes)) == length(axes) || throw(ArgumentError("duplicate integration axis"))
    fixed = Dict(k=>zeros(g.dims[k]) for k in g.keys)
    free = Dict(k=>collect(1:g.dims[k]) for k in g.keys)
    for ((k,i),v) in zip(axes,values)
        has_variable(g,k) && 1 <= i <= g.dims[k] || throw(ArgumentError("invalid integration axis $((k,i))"))
        isfinite(v) || throw(ArgumentError("integration coordinates must be finite"))
        fixed[k][i] = v
        filter!(!=(i),free[k])
    end
    reduced = FactorGraph()
    for k in g.keys
        add_variable!(reduced,k,g.values[k][free[k]])
    end
    for f in g.factors
        # A conditioned prior row is constant. Keep it in the ORIGINAL graph for
        # the evidence, but omit it from the optimization. Otherwise its arbitrary
        # offset changes floating-point line searches and its zero Jacobian row
        # can contaminate a very ill-conditioned least-squares right-hand side.
        # An entirely unobserved integration coordinate must not move other modes.
        all(isempty(free[k]) for k in vars(f)) && continue
        parent=f
        if f isa SitePriorFactor
            indices=intersect(f.indices,free[f.key])
            isempty(indices) && continue
            parent=SitePriorFactor(f.key,f.sites;indices,source=f.source)
        elseif f isa GaussianPrior
            rows=findall(i->any(!iszero,f.R[i,free[f.key]]),Base.axes(f.R,1))
            isempty(rows) && continue
            parent=GaussianPrior(f.key,f.m,f.R[rows,:];source=f.source)
        elseif f isa LinearFactor
            rows=findall(i->any(any(!iszero,A[i,free[k]]) for (k,A) in zip(f.keys,f.A)),eachindex(f.b))
            isempty(rows) && continue
            parent=LinearFactor(f.keys,[A[rows,:] for A in f.A],f.b[rows];source=f.source)
        end
        # The wrapper must not hide a stale dynamics derivative from the solver's
        # final stationarity check. Conditional graphs always use fresh derivatives.
        if f isa DynamicsFactor
            f.relin_tol = 0.0
            f.relin_tol_state = 0.0
        end
        add_factor!(reduced,_ConditionedFactor(parent,fixed,free))
    end
    return reduced,fixed,free
end

# Column-scaled QR retains tiny spatial contrasts next to large common modes.
# root*root' is the covariance; logvolume is half logdet(precision).
function _conditional_root(g)
    J,_,off,n = linearize(g)
    n > 0 || return zeros(0,0),0.0,off
    scales = vec(sqrt.(sum(abs2,J;dims=1)))
    all(x->isfinite(x) && x>0,scales) || error("conditional graph has an unidentified coordinate")
    R = Matrix(qr(J ./ scales').R)
    d = abs.(diag(R))
    size(R,1) >= n && minimum(d) > eps(Float64)*max(size(J)...)*maximum(d) ||
        error("conditional graph is rank deficient")
    R = UpperTriangular(R[1:n,:])
    return (R \ Matrix{Float64}(I,n,n)) ./ scales, sum(log,scales)+sum(log,d),off
end

struct ConditionalNode
    graph::FactorGraph
    coordinates::Vector{Float64}
    mean::Vector{Float64}
    root::Matrix{Float64}
    log_integral::Float64
    solve::NamedTuple
end

_conditional_noise_scales(g)=Dict(f.key_d=>[f.spec.noise.ar_sd] for f in g.factors
    if f isa ObservationFactor && f.key_d!==nothing)

function _conditional_node!(g, axes, point; iters=60,tol=1e-6,ftol=1e-8,
                            step_bounds=Dict{VarKey,Float64}(), newton_tol=0.0,
                            stationarity_scales=_conditional_noise_scales(g))
    reduced,fixed,free = _condition_graph(g,axes,point)
    reduced_scales=Dict(k=>(v isa Real ? v : v[free[k]]) for (k,v) in stationarity_scales)
    result = if sum(values(reduced.dims))==0
        (F=objective(reduced),hist=NamedTuple[],converged=true,gradient=0.0,newton_step=0.0,newton_sd=0.0)
    else
        gauss_newton!(reduced;iters,tol,ftol,step_bounds,newton_tol,stationarity_scales=reduced_scales,
                      resolution_gtol=max(2tol,5e-4))
    end
    for k in g.keys
        g.values[k] .= fixed[k]
        g.values[k][free[k]] = reduced.values[k]
    end
    L,logvolume,off = _conditional_root(reduced)
    full_off,n = _offsets(g)
    rows = reduce(vcat,[collect(full_off[k])[free[k]] for k in g.keys];init=Int[])
    root = zeros(n,size(L,2));root[rows,:] = L
    mean = _stacked(g,g.keys,full_off,n)
    return ConditionalNode(g,collect(point),mean,root,-objective(g)-logvolume,result)
end

"""
    ConditionalPosterior

Deterministic quadrature over `axes`, with a separately optimized joint conditional
Gaussian at EVERY node. `mean` and `covariance` are summaries; use
`conditional_expectation`/`conditional_points` for nonlinear decisions. Diagnostics
separate integration resolution from the unverified conditional-Laplace assumption.
"""
struct ConditionalPosterior
    keys::Vector{VarKey}
    offsets::Dict{VarKey,UnitRange{Int}}
    axes::Vector{Tuple{VarKey,Int}}
    nodes::Vector{ConditionalNode}
    weights::Vector{Float64}
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    diagnostics::NamedTuple
end

# Golub-Welsch rule for EXPECTATIONS under N(0,1), with positive weights.
function _normal_quadrature(order::Int)
    order >= 2 || throw(ArgumentError("quadrature order must be at least two"))
    e = eigen(SymTridiagonal(zeros(order),sqrt.(Float64.(1:order-1))))
    return e.values,vec(e.vectors[1,:].^2)
end
function _tensor_quadrature(order,dim;max_nodes=729)
    dim > 0 || throw(ArgumentError("at least one integration axis is required"))
    order^big(dim) <= max_nodes || throw(ArgumentError("quadrature exceeds the node budget ($order^$dim > $max_nodes)"))
    x,w = _normal_quadrature(order)
    indices = collect(Iterators.product(ntuple(_->1:order,dim)...))[:]
    sort!(indices;by=is->sum(abs2,x[collect(is)]))
    points = [x[collect(is)] for is in indices]
    logs = [sum(log,w[collect(is)]) for is in indices]
    return points,logs
end
function _conditional_weights(logs)
    all(isfinite,logs) || error("a conditional component has non-finite evidence")
    w = exp.(logs .- maximum(logs))
    return w ./ sum(w)
end
function _conditional_posterior(nodes,logweights,axes;diagnostics=(;))
    g=first(nodes).graph;off,n=_offsets(g)
    all(c->c.graph.keys==g.keys && length(c.mean)==n,nodes) ||
        throw(ArgumentError("conditional components do not share a variable layout"))
    w=_conditional_weights(logweights)
    m=sum(wi .* c.mean for (wi,c) in zip(w,nodes))
    C=zeros(n,n)
    for (wi,c) in zip(w,nodes)
        d=c.mean-m; C .+= wi .* (c.root*c.root' + d*d')
    end
    return ConditionalPosterior(copy(g.keys),off,collect(axes),nodes,w,m,Matrix(Symmetric(C)),diagnostics)
end

# Compare the entire joint geometry in the fine rule's units, including small
# contrasts. Build whitening from component roots; don't invert a rounded Gram.
function _conditional_resolution(a::ConditionalPosterior,b::ConditionalPosterior;keys=b.keys)
    idx=reduce(vcat,[collect(b.offsets[k]) for k in keys];init=Int[])
    a.keys==b.keys || throw(ArgumentError("posterior layouts differ"))
    center=b.mean[idx]
    design=reduce(vcat,[sqrt(w).*vcat(c.root[idx,:]',(c.mean[idx]-center)') for (w,c) in zip(b.weights,b.nodes)])
    scales=vec(sqrt.(sum(abs2,design;dims=1)))
    all(>(0),scales) || error("zero variance in a resolution check")
    R=Matrix(qr(design./scales').R)
    white(v)=LowerTriangular(R')\(v./scales)
    function moments(p)
        m=zeros(length(idx)); C=zeros(length(idx),length(idx))
        for (w,c) in zip(p.weights,p.nodes)
            d=white(c.mean[idx]-center);L=white(c.root[idx,:])
            m .+= w.*d; C .+= w.*(L*L'+d*d')
        end
        return m,Matrix(Symmetric(C-m*m'))
    end
    ma,Ca=moments(a);mb,Cb=moments(b)
    return (mean_error=norm(ma-mb), covariance_error=opnorm(Symmetric(Ca-Cb)))
end

function _conditional_rule(g,axes,center,scale,order;max_nodes=729,kwargs...)
    points,logs=_tensor_quadrature(order,length(axes);max_nodes)
    off,n=_offsets(g);ia=[off[k][i] for (k,i) in axes]
    # A conditional Gaussian is a starting value only. Each graph is then solved
    # at the exact fixed coordinates and retains its OWN covariance.
    C=joint_cov(g)[1];m=_stacked(g,g.keys,off,n)
    slope=C[:,ia]/C[ia,ia]
    nodes=ConditionalNode[]; logweights=Float64[]
    for (q,lw) in zip(points,logs)
        point=center+scale*q
        gc=deepcopy(g);start=m+slope*(point-m[ia])
        for k in gc.keys;gc.values[k].=start[off[k]];end
        c=_conditional_node!(gc,axes,point;kwargs...)
        push!(nodes,c)
        # GH integrates against a normal density: divide that proposal density
        # back out. Its constant Jacobian cancels in normalized weights.
        push!(logweights,lw+sum(abs2,q)/2+c.log_integral)
    end
    return _conditional_posterior(nodes,logweights,axes;
        diagnostics=(order=order,nodes=length(nodes),solves_ok=all(c->c.solve.converged,nodes)))
end

"""
    conditional_fit(graph; axes, orders=(3,5,9), max_nodes=729, ...)

Offline nested Laplace integration of the SAME factor objective. `axes` is a list
of `(VarKey, coordinate_index)` pairs, chosen from calibration/physics or an
explicit approximation study, never from cook truth. All other variables are
optimized and their joint covariance retained separately at each node. Positive
Gaussian quadrature is centered/scaled at the fitted marginal for the axes.

Refinement checks joint means AND covariances, not optimizer convergence alone.
`resolved` only certifies agreement of consecutive integration rules; conditional
Gaussian error, missed remote modes and model calibration need independent checks.
The input graph is not mutated. MC is not used.
"""
function conditional_fit(graph::FactorGraph;axes,orders=(3,5,9),max_nodes=729,
                         mean_tol=0.05,covariance_tol=0.1,check_keys=graph.keys,
                         iters=60,tol=1e-6,ftol=1e-8,stationarity_scales=_conditional_noise_scales(graph),kwargs...)
    length(orders)>=2 && issorted(collect(orders)) && allunique(orders) ||
        throw(ArgumentError("at least two strictly increasing quadrature orders are required"))
    mean_tol>0 && covariance_tol>0 || throw(ArgumentError("resolution tolerances must be positive"))
    g=deepcopy(graph)
    _condition_graph(g,axes,[g.values[k][i] for (k,i) in axes]) # validate the coordinate declaration before solving
    mode=gauss_newton!(g;iters,tol,ftol,stationarity_scales,kwargs...)
    mode.converged || error("unconverged global mode: cannot construct conditional integration")
    off,_=_offsets(g);ia=[off[k][i] for (k,i) in axes]
    C=joint_cov(g)[1];center=[g.values[k][i] for (k,i) in axes]
    scale=Matrix(cholesky(Symmetric(C[ia,ia])).L)
    previous=nothing;history=NamedTuple[];p=nothing;resolved=false
    for order in orders
        p=_conditional_rule(g,axes,center,scale,order;max_nodes,iters,tol,ftol,stationarity_scales,kwargs...)
        resolution=previous===nothing ? (mean_error=Inf,covariance_error=Inf) :
            _conditional_resolution(previous,p;keys=check_keys)
        push!(history,merge((order=order,nodes=length(p.nodes),solves_ok=p.diagnostics.solves_ok),resolution))
        resolved=p.diagnostics.solves_ok && previous!==nothing && previous.diagnostics.solves_ok &&
                 resolution.mean_error<=mean_tol && resolution.covariance_error<=covariance_tol
        resolved && break
        previous=p
    end
    d=merge(p.diagnostics,(resolved=resolved,inference=:conditional_laplace,
            conditional_gaussian_validated=false,refinement=history))
    return ConditionalPosterior(p.keys,p.offsets,p.axes,p.nodes,p.weights,p.mean,p.covariance,d)
end

conditional_fit(problem::OfflineProblem;kwargs...)=conditional_fit(problem.graph;kwargs...)

"Positive spherical-radial cubature within each conditional Gaussian; no moment collapse or random draws."
function conditional_points(p::ConditionalPosterior;keys=p.keys)
    idx=reduce(vcat,[collect(p.offsets[k]) for k in keys];init=Int[])
    X=Vector{Vector{Float64}}();w=Float64[]
    for (wi,c) in zip(p.weights,p.nodes)
        # Only the requested marginal is needed; an SVD also handles fixed axes.
        s=svd(c.root[idx,:];full=false)
        L=s.U*Diagonal(s.S);d=size(L,2)
        if d==0
            push!(X,c.mean[idx]);push!(w,wi)
        else
            for j in 1:d,sign in (-1,1)
                push!(X,c.mean[idx]+sign*sqrt(d)*L[:,j]);push!(w,wi/(2d))
            end
        end
    end
    return (points=reduce(hcat,X),weights=w)
end
"Integrate a nonlinear function over the conditional representation using positive deterministic cubature."
function conditional_expectation(f,p::ConditionalPosterior;kwargs...)
    q=conditional_points(p;kwargs...)
    return sum(w*f(x) for (w,x) in zip(q.weights,eachcol(q.points)))
end
