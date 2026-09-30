# Experimental deterministic integration. Included explicitly by the validation
# driver until full cook / tail / conditional-shape qualification is complete.
module AdaptiveDensities

using LinearAlgebra

export AdaptiveDensity, adaptive_density, adaptive_logpdf, adaptive_expectation,
       adaptive_marginal_logpdf, adaptive_geometry, adaptive_cdf, adaptive_density_distance

"An oracle value: unnormalized marginal log density and Gaussian remainder."
struct Node
    logmass::Float64
    mean::Vector{Float64}
    root::Matrix{Float64}  # lower triangular, positive diagonal
end
function Node(value)
    m=collect(Float64,value.mean);L=Matrix{Float64}(value.root)
    isfinite(value.logmass) && all(isfinite,m) && all(isfinite,L) ||
        throw(ArgumentError("the density oracle returned a nonfinite value"))
    size(L)==(length(m),length(m)) && istril(L) && all(>(0),diag(L)) ||
        throw(ArgumentError("the remainder needs a nonsingular lower triangular root"))
    return Node(value.logmass,m,L)
end

struct Cell
    lo::Vector{Float64}
    hi::Vector{Float64}
    nodes::Vector{Node}
end

"""
A positive, normalized density on a finite box with a conditional Gaussian
remainder. The integrated coordinates have a piecewise polynomial LOG density;
they are not Gaussian and are not a discrete set of quadrature atoms. Conditional
means and log-Cholesky geometry vary over that box. Outside the box density is
zero: truncation and unobserved modes MUST be checked separately by the caller.
"""
struct AdaptiveDensity
    cells::Vector{Cell}
    log_normalizer::Float64
    integration_order::Int
    diagnostics::NamedTuple
end

struct NodeBudget <: Exception end

function _cell(oracle,cache,lo,hi,max_nodes)
    points=[lo+(hi-lo).*(collect(t).+1)./2 for t in Iterators.product(ntuple(_->(-1.,0.,1.),length(lo))...)]
    missing=count(q->!haskey(cache,Tuple(q)),points)
    length(cache)+missing<=max_nodes || throw(NodeBudget())
    # Visit the center first to help expensive optimizers warm-start near mass.
    for q in sort(vec(points);by=q->sum(abs2,(q-(lo+hi)/2)./(hi-lo)))
        get!(cache,Tuple(q)) do
            Node(oracle(q))
        end
    end
    ns=[cache[Tuple(q)] for q in vec(points)]
    return Cell(copy(lo),copy(hi),ns)
end

function _geometry(c::Cell,q)
    z=2 .* (q-c.lo)./(c.hi-c.lo).-1
    ws=[(v*(v-1)/2,1-v*v,v*(v+1)/2) for v in z]
    logmass=0.;n=length(first(c.nodes).mean);m=zeros(n);L=zeros(n,n);ld=zeros(n)
    for (node,t) in zip(c.nodes,Iterators.product(ntuple(_->1:3,length(q))...))
        w=prod(ws[j][t[j]] for j in eachindex(q))
        logmass+=w*node.logmass;m .+= w.*node.mean;L .+= w.*node.root
        ld .+= w.*log.(diag(node.root))
    end
    for i in 1:n;L[i,i]=exp(ld[i]);end
    return Node(logmass,m,L)
end

function _logvalue(c::Cell,q)
    z=2 .* (q-c.lo)./(c.hi-c.lo).-1
    ws=[(v*(v-1)/2,1-v*v,v*(v+1)/2) for v in z]
    return sum(prod(ws[j][t[j]] for j in eachindex(q))*node.logmass
        for (node,t) in zip(c.nodes,Iterators.product(ntuple(_->1:3,length(q))...)))
end

function _cell_at(cells,q)
    i=findfirst(c->length(q)==length(c.lo) && all(c.lo.<=q) && all(q.<=c.hi),cells)
    i===nothing && throw(DomainError(q,"outside the adaptive density's integration box"))
    return cells[i]
end
adaptive_geometry(d::AdaptiveDensity,q)=_geometry(_cell_at(d.cells,q),q)
function adaptive_marginal_logpdf(d::AdaptiveDensity,q)
    any(c->length(q)==length(c.lo) && all(c.lo.<=q) && all(q.<=c.hi),d.cells) || return -Inf
    return _logvalue(_cell_at(d.cells,q),q)-d.log_normalizer
end
function adaptive_logpdf(d::AdaptiveDensity,q,y)
    p=adaptive_marginal_logpdf(d,q);p==-Inf && return p
    n=adaptive_geometry(d,q);v=LowerTriangular(n.root)\(y-n.mean)
    return p-.5sum(abs2,v)-sum(log,diag(n.root))-.5length(v)*log(2pi)
end

function _legendre(order)
    j=Float64.(1:order-1);e=eigen(SymTridiagonal(zeros(order),j./sqrt.(4j.^2 .-1)))
    return (e.values.+1)./2,vec(e.vectors[1,:].^2)
end
function _points(c,order)
    x,w=_legendre(order);volume=prod(c.hi-c.lo)
    return ((c.lo+(c.hi-c.lo).*x[collect(t)],volume*prod(w[collect(t)]))
            for t in Iterators.product(ntuple(_->1:order,length(c.lo))...))
end
function _logsum(v)
    m=maximum(v)
    m == -Inf && return -Inf
    return m+log(sum(exp.(v.-m)))
end
_logabsdiff(a,b)=a==b ? -Inf : max(a,b)+log(-expm1(-abs(a-b)))
function _logmass(cells,order)
    return _logsum([log(w)+_logvalue(c,q) for c in cells for (q,w) in _points(c,order)])
end

# Tensor Bernstein coefficients bound the entire interpolated log density,
# including peaks between quadrature points. Two quadrature orders can both
# miss a narrow mode, so agreement alone must not accept an unsampled peak.
function _log_peak_bound(c)
    k=length(c.lo);coeff=reshape([n.logmass for n in c.nodes],ntuple(_->3,k))
    for axis in 1:k
        old=copy(coeff)
        for index in CartesianIndices(coeff)
            index[axis]==2 || continue
            left=CartesianIndex(ntuple(j->j==axis ? 1 : index[j],k))
            right=CartesianIndex(ntuple(j->j==axis ? 3 : index[j],k))
            coeff[index]=2old[index]-(old[left]+old[right])/2
        end
    end
    return maximum(coeff)
end
function _quadrature_guard(c,order)
    peak=_log_peak_bound(c)
    sampled=maximum(_logvalue(c,q) for (q,_) in _points(c,order))
    # If no point is within one log unit of a certified peak upper bound,
    # charge the whole cell's mass upper bound as unresolved quadrature.
    return peak-sampled>1. ? peak+sum(log,c.hi-c.lo) : -Inf
end

# Dimensionless Gaussian KL, preserving very small but informative contrasts.
function _kl(a,b)
    isempty(a.mean) && return 0.
    scales=vec(sqrt.(sum(abs2,b.root;dims=2)))
    B=LowerTriangular(b.root./scales);X=B\(a.root./scales);v=B\((a.mean-b.mean)./scales)
    total=sum(abs2,v)
    for j in axes(X,2)
        delta=X[j,j]-1
        total+=max(0.,delta*delta+2*(delta-log1p(delta)))
        for i in j+1:size(X,1);total+=X[i,j]^2;end
    end
    return total/2
end

# Log of the local, UNNORMALIZED joint L1 discrepancy. The
# positive integration weights and analytic Gaussian KL include covariance and
# correlation changes, even if marginal evidence is exactly unchanged.
function _logdifference(parent,children,order)
    terms=Float64[]
    for c in children,(q,w) in _points(c,order)
        a=_geometry(parent,q);b=_geometry(c,q)
        tv=min(1.,sqrt(min(_kl(a,b),_kl(b,a))/2))
        marginal=_logabsdiff(a.logmass,b.logmass)
        conditional=log(2.)+min(a.logmass,b.logmass)+log(tv)
        push!(terms,log(w)+_logsum((marginal,conditional)))
    end
    return _logsum(terms)
end

function _split(oracle,cache,c,axis,max_nodes)
    mid=(c.lo[axis]+c.hi[axis])/2
    h=copy(c.hi);h[axis]=mid;l=copy(c.lo);l[axis]=mid
    return [_cell(oracle,cache,c.lo,h,max_nodes),_cell(oracle,cache,l,c.hi,max_nodes)]
end

"""
    adaptive_density(oracle, lower, upper; tolerance=.01, max_nodes=729, ...)

Refine a tensor quadratic interpolant locally, one coordinate at a time. Oracle
values are reused exactly. Every direction of EVERY leaf is checked before a
resolution result can be reported. Refinement compares the joint density using
conditional Gaussian KL, not just the normalizer or the first two moments.

`resolved` means successive approximations agree at the stated numerical
resolution on this box. It is NOT a bound against the true target, a tail bound,
or a validation of the Gaussian remainder. A budget failure stays unresolved;
failed/nonfinite oracle evaluations are never silently dropped.
"""
function adaptive_density(oracle,lower,upper;tolerance=.01,max_nodes=729,max_cells=128,
                          orders=(5,9),integration_tolerance=tolerance/10,
                          cache=Dict{Tuple,Node}(),progress=nothing)
    lo=collect(Float64,lower);hi=collect(Float64,upper);k=length(lo)
    k>0 && length(hi)==k && all(isfinite,lo) && all(isfinite,hi) && all(hi.>lo) ||
        throw(ArgumentError("finite nonempty integration bounds are required"))
    isfinite(tolerance) && tolerance>0 && isfinite(integration_tolerance) && integration_tolerance>0 &&
        length(orders)==2 && all(o->o isa Integer,orders) && 2<=orders[1]<orders[2] ||
        throw(ArgumentError("positive tolerances and two increasing integration orders are required"))
    max_nodes>=3^big(k) && max_cells>=1 || throw(ArgumentError("insufficient budget for the initial cell"))
    cells=[_cell(oracle,cache,lo,hi,max_nodes)]
    candidates=IdDict{Cell,Vector{Vector{Cell}}}();history=NamedTuple[]
    checks=IdDict{Cell,NamedTuple}()
    reason=:node_budget;resolved=false;indicator=Inf;integration_error=Inf;logz=NaN
    while true
        coarse_mass=[_logmass([c],orders[1]) for c in cells]
        fine_mass=[_logmass([c],orders[2]) for c in cells]
        logz=_logsum(fine_mass)
        mass_errors=_logabsdiff.(coarse_mass,fine_mass).-logz
        mass_errors=max.(mass_errors,[_quadrature_guard(c,orders[2])-logz for c in cells])
        log_integration_error=_logsum(mass_errors)
        values=fill(-Inf,length(cells),k);changes=fill(-Inf,length(cells),k)
        complete=true
        for (i,c) in enumerate(cells)
            cs=get!(candidates,c,Vector{Cell}[])
            try
                while length(cs)<k;push!(cs,_split(oracle,cache,c,length(cs)+1,max_nodes));end
            catch err
                err isa NodeBudget || rethrow()
                complete=false;break
            end
            check=get!(checks,c) do
                # A leaf's geometry does not change when another leaf splits.
                # Cache unnormalized comparisons in log units. A child may
                # discover a mode far above the parent's mass; converting to
                # parent-relative linear units would overflow before rescaling.
                coarse=[_logdifference(c,cs[axis],orders[1]) for axis in 1:k]
                fine=[_logdifference(c,cs[axis],orders[2]) for axis in 1:k]
                (coarse=coarse,fine=fine)
            end
            values[i,:]=check.fine.-logz
            changes[i,:]=_logabsdiff.(check.fine,check.coarse).-logz
        end
        log_indicator=complete ? _logsum(values) : Inf
        log_integration_error=complete ? max(log_integration_error,_logsum(changes)) : log_integration_error
        indicator=exp(log_indicator);integration_error=exp(log_integration_error)
        push!(history,(nodes=length(cache),cells=length(cells),log_normalizer=logz,
                       indicator=indicator,integration_error=integration_error,
                       log_indicator=log_indicator,log_integration_error=log_integration_error,complete=complete))
        progress===nothing || progress(last(history))
        complete || break
        if log_indicator<=log(tolerance) && log_integration_error<=log(integration_tolerance)
            resolved=true;reason=:resolution;break
        end
        length(cells)<max_cells || (reason=:cell_budget;break)
        # Quadrature error belongs to individual cells. Refine the largest
        # error, not the largest mass, which can starve a poorly integrated
        # mode once another mode already has accurate quadrature.
        if maximum(values)<log_integration_error
            comparison_errors=vec(maximum(changes;dims=2))
            i=argmax(max.(mass_errors,comparison_errors))
            axis=mass_errors[i]>=comparison_errors[i] ? argmax(cells[i].hi-cells[i].lo) : argmax(changes[i,:])
        else
            i,axis=Tuple(argmax(values))
        end
        c=cells[i];splice!(cells,i:i,candidates[c][axis]);delete!(candidates,c);delete!(checks,c)
    end
    diagnostics=(resolved=resolved,reason=reason,refinement_indicator=indicator,
        integration_error=integration_error,nodes=length(cache),cells=length(cells),
        lower=lo,upper=hi,history=history,tail_validated=false,conditional_validated=false,
        target_validated=false)
    return AdaptiveDensity(cells,logz,orders[2],diagnostics)
end

"Integrate a function of the axis point and its Gaussian-remainder geometry."
function adaptive_expectation(f,d::AdaptiveDensity;order=d.integration_order)
    return sum(w*exp(n.logmass-d.log_normalizer)*f(q,n)
               for c in d.cells for (q,w) in _points(c,order) for n in (_geometry(c,q),))
end

"Axis marginal CDF; split cells at the threshold instead of integrating an indicator."
function adaptive_cdf(d::AdaptiveDensity,axis::Int,value::Real;order=d.integration_order)
    1<=axis<=length(first(d.cells).lo) || throw(BoundsError())
    mass=0.
    for c in d.cells
        value<=c.lo[axis] && continue
        hi=copy(c.hi);hi[axis]=min(hi[axis],value)
        domain=Cell(c.lo,hi,c.nodes)
        for (q,w) in _points(domain,order)
            mass+=w*exp(_logvalue(c,q)-d.log_normalizer)
        end
    end
    return clamp(mass,0.,1.)
end

# Disjoint rectangular strips comprising c minus an axis-aligned box. This
# permits domain-expansion comparisons without double counting overlapping cells.
function _outside(c,lo,hi)
    a=max.(c.lo,lo);b=min.(c.hi,hi)
    all(b.>a) || return [(c.lo,c.hi)]
    out=Tuple{Vector{Float64},Vector{Float64}}[];left=copy(c.lo);right=copy(c.hi)
    for j in eachindex(a)
        if left[j]<a[j]
            h=copy(right);h[j]=a[j];push!(out,(copy(left),h));left[j]=a[j]
        end
        if right[j]>b[j]
            l=copy(left);l[j]=b[j];push!(out,(l,copy(right)));right[j]=b[j]
        end
    end
    return out
end

"""
Compare two normalized approximations in the SAME coordinates, including their
Gaussian remainder. Cell intersections remove interpolation breaks from the
integration domains. Different boxes include the exclusive tail strips. This is
a numerically integrated TV upper-bound estimate BETWEEN APPROXIMATIONS, not a
bound against the target posterior or against modes outside both boxes.
"""
function adaptive_density_distance(a::AdaptiveDensity,b::AdaptiveDensity;orders=(7,11))
    length(first(a.cells).lo)==length(first(b.cells).lo) || throw(DimensionMismatch())
    length(first(first(a.cells).nodes).mean)==length(first(first(b.cells).nodes).mean) ||
        throw(DimensionMismatch())
    parts=Tuple{Vector{Float64},Vector{Float64},Union{Nothing,Cell},Union{Nothing,Cell}}[]
    for ca in a.cells,cb in b.cells
        lo=max.(ca.lo,cb.lo);hi=min.(ca.hi,cb.hi)
        all(hi.>lo) && push!(parts,(lo,hi,ca,cb))
    end
    alo=reduce((x,y)->min.(x,y),(c.lo for c in a.cells))
    ahi=reduce((x,y)->max.(x,y),(c.hi for c in a.cells))
    blo=reduce((x,y)->min.(x,y),(c.lo for c in b.cells))
    bhi=reduce((x,y)->max.(x,y),(c.hi for c in b.cells))
    for c in a.cells,(lo,hi) in _outside(c,blo,bhi);push!(parts,(lo,hi,c,nothing));end
    for c in b.cells,(lo,hi) in _outside(c,alo,ahi);push!(parts,(lo,hi,nothing,c));end
    history=Float64[]
    for order in orders
        total=0.
        for (lo,hi,ca,cb) in parts
            domain=Cell(lo,hi,Node[])
            for (q,w) in _points(domain,order)
                na=ca===nothing ? nothing : _geometry(ca,q)
                nb=cb===nothing ? nothing : _geometry(cb,q)
                pa=na===nothing ? 0. : exp(na.logmass-a.log_normalizer)
                pb=nb===nothing ? 0. : exp(nb.logmass-b.log_normalizer)
                tv=na===nothing || nb===nothing ? 0. : min(1.,sqrt(min(_kl(na,nb),_kl(nb,na))/2))
                total+=w*(.5abs(pa-pb)+min(pa,pb)*tv)
            end
        end
        push!(history,total)
    end
    return (tv_bound_estimate=last(history),integration_change=length(history)>1 ? abs(history[end]-history[end-1]) : Inf,
        history=history,target_validated=false)
end

end # module
