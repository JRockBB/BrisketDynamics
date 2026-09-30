export ConditionalDensity, conditional_density, conditional_logpdf, conditional_density_distance,
       conditional_cdf

"A continuous parameter marginal with conditional Gaussian innovations and a nonlinear state decoder."
struct ConditionalDensity
    transported::TransportedConditionalPosterior
    axisrows::Vector{Int}
    free::Vector{Int}
    center::Vector{Float64}
    scale::Matrix{Float64}
    grid::Vector{Float64}
    slots::Dict{Tuple,Int}
    base_mean::Vector{Float64}
    base_root::Matrix{Float64}
    log_residual::Vector{Float64}
    mean_trend::Matrix{Float64}
    mean_residual::Vector{Vector{Float64}}
    roots::Vector{Matrix{Float64}}
    log_normalizer::Float64
    normalization_error::Float64
    log_jacobian::Float64
end

_conditional_logsum(a,b) = max(a,b)+log1p(exp(-abs(a-b)))
_conditional_phi(x) = .5erfc(-x/sqrt(2.0))
_conditional_phiinv(u) = -sqrt(2.0)*erfcinv(2u)

# Models are copied with each component. Object identity cannot establish that
# they implement the same decoder; compare every declaration, including controls,
# event history, parameter layout, tolerances and optional reduced-order model.
function _same_decoder_value(a,b)
    typeof(a)===typeof(b) || return false
    isequal(a,b) && return true
    if a isa AbstractArray
        return axes(a)==axes(b) && all(_same_decoder_value(x,y) for (x,y) in zip(a,b))
    elseif a isa AbstractDict
        return keys(a)==keys(b) && all(_same_decoder_value(a[k],b[k]) for k in keys(a))
    elseif a isa Number || a isa AbstractString || a isa Function || a isa Type || a isa Module
        return false
    end
    n=fieldcount(typeof(a))
    return n>0 && all(_same_decoder_value(getfield(a,i),getfield(b,i)) for i in 1:n)
end
function _same_conditional_decoder(a::_ConditionalTransport,b::_ConditionalTransport)
    a.offsets==b.offsets && length(a.dynamics)==length(b.dynamics) || return false
    return all(all(_same_decoder_value(getfield(x,k),getfield(y,k)) for k in
                   (:model,:key_z,:key_m,:split,:key_y0,:key_y1,:t0,:t1,:first)) &&
               x.cache.W==y.cache.W for (x,y) in zip(a.dynamics,b.dynamics))
end

function _canonical_conditional_root(root)
    n=size(root,1);n==0 && return zeros(0,0)
    scales=vec(sqrt.(sum(abs2,root;dims=2)))
    all(>(0),scales) || error("a conditional innovation has zero variance")
    R=Matrix(qr((root./scales)').R)
    L=scales.*Matrix(R')
    for j in 1:n
        L[j,j]==0 && error("conditional innovation covariance is singular")
        L[:,j].*=sign(L[j,j])
    end
    return L
end

function _density_corners(d::ConditionalDensity,q)
    k=length(q);indices=Vector{Vector{Int}}();weights=Vector{Vector{Float64}}()
    for i in 1:k
        n=min(4,length(d.grid));start=clamp(searchsortedlast(d.grid,q[i])-1,1,length(d.grid)-n+1)
        js=collect(start:start+n-1);x=clamp(q[i],first(d.grid),last(d.grid))
        ws=[prod((x-d.grid[b])/(d.grid[a]-d.grid[b]) for b in js if b!=a) for a in js]
        push!(indices,js);push!(weights,ws)
    end
    out=Tuple{Int,Float64}[]
    for slots in Iterators.product((eachindex(js) for js in indices)...)
        index=Tuple(indices[i][slots[i]] for i in 1:k)
        w=prod(weights[i][slots[i]] for i in 1:k)
        w!=0 && push!(out,(d.slots[index],w))
    end
    return out
end
function _density_residual(d,q)
    stencil=_density_corners(d,q);values=[d.log_residual[i] for (i,_) in stencil]
    # Interpolation weights apply to functions, never posterior masses. Bound
    # overshoot of the log-density remainder; the resulting density is positive.
    return clamp(sum(w*d.log_residual[i] for (i,w) in stencil),minimum(values),maximum(values))
end
function _density_geometry(d,q)
    m=d.mean_trend*vcat(1.0,q);L=zeros(length(d.free),length(d.free))
    stencil=_density_corners(d,q);logs=zeros(length(d.free))
    lower=fill(Inf,length(d.free));upper=fill(-Inf,length(d.free))
    for (i,w) in stencil
        m .+= w.*d.mean_residual[i];L .+= w.*d.roots[i]
        values=log.(diag(d.roots[i]));logs .+= w.*values
        lower=min.(lower,values);upper=max.(upper,values)
    end
    for i in eachindex(logs);L[i,i]=exp(clamp(logs[i],lower[i],upper[i]));end
    return m,L
end
function _density_base_logpdf(d,q)
    v=LowerTriangular(d.base_root)\(q-d.base_mean)
    return -.5sum(abs2,v)-sum(log,diag(d.base_root))-.5length(q)*log(2pi)
end
_density_marginal_logpdf(d,q)=_density_base_logpdf(d,q)+_density_residual(d,q)-d.log_normalizer

function _density_normalizer(d,order)
    qs,logs=_tensor_quadrature(order,length(d.center);max_nodes=100_000)
    values=[lw+_density_residual(d,d.base_mean+d.base_root*q) for (q,lw) in zip(qs,logs)]
    shift=maximum(values)
    return shift+log(sum(exp.(values.-shift)))
end

"""
    conditional_density(p; center, scale, order, logweights=nothing)

Reconstruct a CONTINUOUS density from a tensor conditional rule. A fitted quadratic
log marginal supplies a normalized Gaussian reference; only its nonquadratic
remainder is interpolated. Likewise an affine trend is removed from conditional
innovation means. Thus linear Gaussian updating remains exact even between nodes
and when the posterior concentrates more narrowly than the original grid spacing.
Local cubic interpolation acts on functions, not probability masses; logarithmic
Cholesky diagonals keep every reconstructed conditional covariance positive definite.
The nonlinear physics decoder is retained. No Monte Carlo or truth is used.

`logweights` must be the accumulated bank log weights online (not logs of displayed
weights, which can underflow). Offline they can be reconstructed from the full
graph integrals. The bounded remainders are clamped outside the grid; this is an
explicit approximation, not evidence that remote posterior modes are absent.
"""
function conditional_density(p::ConditionalPosterior;center,scale,order::Int,logweights=nothing,
                             normalization_orders=(16,32))
    all(c->c.solve.converged,p.nodes) || throw(ArgumentError("conditional density requires converged node solves"))
    order>=3 || throw(ArgumentError("continuous reconstruction needs at least three nodes per axis"))
    tp=conditional_transport(p);k=length(p.axes)
    1<=k<=3 || throw(ArgumentError("continuous conditional reconstruction currently supports one to three integration axes"))
    qs,lq=_tensor_quadrature(order,k;max_nodes=100_000)
    length(qs)==length(p.nodes) || throw(ArgumentError("posterior is not the declared tensor rule"))
    center=collect(Float64,center);scale=Matrix{Float64}(scale)
    size(scale)==(k,k) && length(center)==k || throw(DimensionMismatch("integration chart dimensions differ"))
    logabsdet(scale)[2]>0 || throw(ArgumentError("integration chart must have positive determinant"))
    grid=first(_normal_quadrature(order));slots=Dict{Tuple,Int}()
    ia=[p.offsets[key][i] for (key,i) in p.axes]
    free=setdiff(1:length(p.mean),ia)
    means=Vector{Float64}[];roots=Matrix{Float64}[]
    reference=first(tp.components)
    for (j,(c,q)) in enumerate(zip(tp.components,qs))
        isapprox(p.nodes[j].coordinates,center+scale*q;atol=1e-8,rtol=1e-10) ||
            throw(ArgumentError("node coordinates disagree with the integration chart"))
        _same_conditional_decoder(c,reference) ||
            throw(ArgumentError("conditional density requires common dynamics and covariance anchors"))
        slots[Tuple(argmin(abs.(grid.-v)) for v in q)]=j
        push!(means,c.mean[free]);push!(roots,_canonical_conditional_root(c.root[free,:]))
    end
    logs=logweights===nothing ? [lw+sum(abs2,q)/2+c.log_integral for (lw,q,c) in zip(lq,qs,p.nodes)] : collect(logweights)
    length(logs)==length(qs) && all(isfinite,logs) || throw(ArgumentError("finite accumulated log weights are required"))
    tilt=logs-lq
    # A quadratic Gaussian marginal is represented exactly, rather than being
    # approximated by a lattice or linear interpolation of its log density.
    features(q)=vcat(1.0,q,[i==j ? q[i]^2/2 : q[i]*q[j] for i in 1:k for j in i:k])
    D=reduce(vcat,[features(q)' for q in qs]);w=exp.(lq)
    beta=(D.*sqrt.(w))\(tilt.*sqrt.(w))
    H=zeros(k,k);j=k+1
    for a in 1:k,b in a:k;j+=1;H[a,b]=H[b,a]=beta[j];end
    P=Matrix{Float64}(I,k,k)-H
    if isposdef(Symmetric(P))
        bm=P\beta[2:k+1];br=Matrix(cholesky(Symmetric(inv(P))).L)
    else
        bm=zeros(k);br=Matrix{Float64}(I,k,k)
    end
    res=[-.5sum(abs2,q)+v+.5sum(abs2,LowerTriangular(br)\(q-bm)) for (q,v) in zip(qs,tilt)]
    res .-= maximum(res)
    trend=reduce(hcat,means)*(reduce(hcat,[vcat(1.0,q) for q in qs]).*w')'
    mr=[m-trend*vcat(1.0,q) for (m,q) in zip(means,qs)]
    jac=sum(logabsdet(f.cache.W)[1] for f in reference.dynamics;init=0.0)-logabsdet(scale)[1]
    d=ConditionalDensity(tp,ia,free,center,scale,grid,slots,bm,br,res,trend,mr,roots,0.0,Inf,jac)
    z0=_density_normalizer(d,first(normalization_orders));z1=_density_normalizer(d,last(normalization_orders))
    return ConditionalDensity(tp,ia,free,center,scale,grid,slots,bm,br,res,trend,mr,roots,z1,abs(z1-z0),jac)
end

function _conditional_encode(ref::_ConditionalTransport,x)
    u=copy(x)
    for f in ref.dynamics
        z=f.key_m===nothing ? x[ref.offsets[f.key_z]] :
            assemble_latents(f.split,x[ref.offsets[f.key_z]],x[ref.offsets[f.key_m]])
        y=propagate(f.model,x[ref.offsets[f.key_y0]],z,f.t0,f.t1;first_slice=f.first)[1]
        u[ref.offsets[f.key_y1]]=f.cache.W*(x[ref.offsets[f.key_y1]]-y)
    end
    return u
end

"Normalized joint log density in the original graph coordinates, including the change-of-variables determinant."
function conditional_logpdf(d::ConditionalDensity,x::AbstractVector)
    length(x)==length(d.transported.source.mean) || throw(DimensionMismatch("joint graph coordinates required"))
    q=d.scale\(x[d.axisrows]-d.center);m,L=_density_geometry(d,q)
    u=_conditional_encode(first(d.transported.components),x)
    delta=isempty(d.free) ? Float64[] : LowerTriangular(L)\(u[d.free]-m)
    return _density_marginal_logpdf(d,q)-.5sum(abs2,delta)-sum(log,diag(L))-
           .5length(delta)*log(2pi)+d.log_jacobian
end

# Stable KL between Gaussians: don't subtract two nearly equal dimension-sized
# traces. Shared row scaling also preserves very small initial-state contrasts.
function _conditional_gaussian_kl(ma,La,mb,Lb)
    isempty(ma) && return 0.0
    scales=vec(sqrt.(sum(abs2,Lb;dims=2)))
    B=LowerTriangular(Lb./scales)
    X=B\(La./scales);v=B\((ma-mb)./scales)
    value=sum(abs2,v)
    for j in axes(X,2)
        a=X[j,j]-1
        value+=max(0.0,a*a+2*(a-log1p(a)))
        for i in j+1:size(X,1);value+=X[i,j]^2;end
    end
    return .5value
end

"""
    conditional_density_distance(coarse, fine; orders=(5,9,17,33), atol=1e-3)

Estimate a total-variation bound from marginal density differences and conditional
Gaussian KL/Pinsker bounds. It is invariant under their COMMON invertible physics
decoder, so it compares the curved joint state distribution without fitting state
ellipses. Two deterministic integration rules check the bound's numerical stability.
This compares two approximations; it does not certify either against the true posterior.
"""
function conditional_density_distance(a::ConditionalDensity,b::ConditionalDensity;
                                      orders=(5,9,17,33),atol=1e-3,max_nodes=4096)
    a.axisrows==b.axisrows && a.free==b.free && a.center==b.center && a.scale==b.scale ||
        throw(ArgumentError("density comparison requires a common coordinate chart"))
    _same_conditional_decoder(first(a.transported.components),first(b.transported.components)) ||
        throw(ArgumentError("density comparison requires identical nonlinear decoders"))
    history=Float64[]
    for order in orders
        order^big(length(a.center))<=max_nodes || break
        qs,lw=_tensor_quadrature(order,length(a.center);max_nodes)
        total=0.0
        for base in (a,b),(eta,logw) in zip(qs,lw)
            q=base.base_mean+base.base_root*eta
            qa=_density_base_logpdf(a,q);qb=_density_base_logpdf(b,q)
            proposal=_conditional_logsum(qa,qb)-log(2.0)
            pa=exp(_density_marginal_logpdf(a,q)-proposal)
            pb=exp(_density_marginal_logpdf(b,q)-proposal)
            ma,La=_density_geometry(a,q);mb,Lb=_density_geometry(b,q)
            kl=min(_conditional_gaussian_kl(ma,La,mb,Lb),_conditional_gaussian_kl(mb,Lb,ma,La))
            total+=.5exp(logw)*(.5abs(pa-pb)+min(pa,pb)*min(1.0,sqrt(kl/2)))
        end
        push!(history,total)
        length(history)>=2 && abs(history[end]-history[end-1])<=atol && break
    end
    isempty(history) && throw(ArgumentError("density comparison exceeds the integration budget"))
    return (tv_bound_estimate=last(history),integration_change=length(history)<2 ? Inf : abs(history[end]-history[end-1]),
            normalization_error=max(a.normalization_error,b.normalization_error),history=history)
end

function _unit_legendre(order)
    j=Float64.(1:order-1);e=eigen(SymTridiagonal(zeros(order),j./sqrt.(4j.^2 .-1)))
    return (e.values.+1)./2,vec(e.vectors[1,:].^2)
end

"A deterministic marginal CDF for parameters/initial states; generated state CDFs require nonlinear integration."
function conditional_cdf(d::ConditionalDensity,key::VarKey,index::Int,value::Real;order=24)
    any(f.key_y1==key for f in first(d.transported.components).dynamics) &&
        throw(ArgumentError("generated state CDFs require nonlinear predictive integration"))
    row=d.transported.source.offsets[key][index];axis=findfirst(==(row),d.axisrows)
    if axis===nothing
        j=findfirst(==(row),d.free);qs,lw=_tensor_quadrature(order,length(d.center);max_nodes=100_000)
        total=0.0
        for (eta,l) in zip(qs,lw)
            q=d.base_mean+d.base_root*eta;m,L=_density_geometry(d,q)
            total+=exp(l+_density_residual(d,q)-d.log_normalizer)*_conditional_phi((value-m[j])/norm(L[j,:]))
        end
        return clamp(total,0.0,1.0)
    end
    direction=vec(d.scale[axis,:]'*d.base_root)
    sd=norm(direction);bound=(value-d.center[axis]-dot(d.scale[axis,:],d.base_mean))/sd
    mass=_conditional_phi(bound);mass==0 && return 0.0
    unit=direction/sd;e1=zeros(length(unit));e1[1]=1.0;v=e1-unit
    rotation=norm(v)<1e-12 ? Matrix{Float64}(I,length(unit),length(unit)) :
        Matrix{Float64}(I,length(unit),length(unit))-2*(v*v')/dot(v,v)
    ts,ws=_unit_legendre(order)
    rest,logs=length(unit)==1 ? ([Float64[]],[0.0]) : _tensor_quadrature(order,length(unit)-1;max_nodes=100_000)
    total=0.0
    for (t,w) in zip(ts,ws),(eta,l) in zip(rest,logs)
        q=d.base_mean+d.base_root*rotation*vcat(_conditional_phiinv(mass*t),eta)
        total+=mass*w*exp(l+_density_residual(d,q)-d.log_normalizer)
    end
    return clamp(total,0.0,1.0)
end
