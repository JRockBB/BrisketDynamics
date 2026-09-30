# Experimental sparse interpolation of a joint log density and log-Cholesky
# conditional geometry. Loaded explicitly; no live estimator defaults change.
module SparseDensities
using LinearAlgebra
using ..AdaptiveDensities
import ..AdaptiveDensities: adaptive_geometry, adaptive_marginal_logpdf, adaptive_logpdf,
    adaptive_expectation
const AD=AdaptiveDensities
export SparseDensity, sparse_density

struct SparseDensity
    lower::Vector{Float64}
    upper::Vector{Float64}
    sequence::Vector{Float64}
    indices::Vector{Tuple}
    coefficients::Matrix{Float64}
    remainder::Int
    cells::Vector{NamedTuple{(:lo,:hi),Tuple{Vector{Float64},Vector{Float64}}}}
    log_normalizer::Float64
    integration_order::Int
    diagnostics::NamedTuple
end

# Deterministic discrete Leja rule, nested one point at a time. This is a
# polynomial interpolation rule, never signed probability/quadrature weights.
function leja(n)
    pool=collect(range(-1.,1.;length=4097));xs=[0.];scores=zeros(length(pool))
    for j in 2:n
        scores .+= log.(abs.(pool.-last(xs)))
        push!(xs,pool[argmax(scores)])
    end
    xs
end
function basis(sequence,index,z)
    prod(prod((z[j]-sequence[r])/(sequence[index[j]+1]-sequence[r])
        for r in 1:index[j];init=1.) for j in eachindex(z))
end
weights(sequence,indices,z)=[basis(sequence,i,z) for i in indices]
function pack(n::AD.Node)
    r=length(n.mean)
    v=vcat(n.logmass,n.mean,[i==j ? log(n.root[i,j]) : n.root[i,j] for j in 1:r for i in j:r])
    v
end
function unpack(v,r)
    L=zeros(r,r);p=1+r
    for j in 1:r,i in j:r
        p+=1;L[i,j]=i==j ? exp(v[p]) : v[p]
    end
    AD.Node((logmass=v[1],mean=v[2:1+r],root=L))
end
standardize(lo,hi,q)=2 .* (q-lo)./(hi-lo).-1
point(lo,hi,z)=(lo+hi)/2+(hi-lo).*z/2
function adaptive_geometry(d::SparseDensity,q)
    length(q)==length(d.lower) && all(d.lower.<=q) && all(q.<=d.upper) || throw(DomainError(q))
    unpack(d.coefficients*weights(d.sequence,d.indices,standardize(d.lower,d.upper,q)),d.remainder)
end
logvalue(d,q)=dot(view(d.coefficients,1,:),weights(d.sequence,d.indices,standardize(d.lower,d.upper,q)))
function adaptive_marginal_logpdf(d::SparseDensity,q)
    length(q)==length(d.lower) && all(d.lower.<=q) && all(q.<=d.upper) || return -Inf
    logvalue(d,q)-d.log_normalizer
end
function adaptive_logpdf(d::SparseDensity,q,y)
    p=adaptive_marginal_logpdf(d,q);p==-Inf && return p
    n=adaptive_geometry(d,q);v=LowerTriangular(n.root)\(y-n.mean)
    p-.5sum(abs2,v)-sum(log,diag(n.root);init=0.)-.5length(v)*log(2pi)
end
function adaptive_expectation(f,d::SparseDensity;order=d.integration_order)
    sum(w*exp(n.logmass-d.log_normalizer)*f(q,n) for c in d.cells
        for (q,w) in AD._points(c,order) for n in (adaptive_geometry(d,q),))
end

function frontier(indices,dimension,max_degree)
    present=Set(indices);out=Set{Tuple}()
    for i in indices,j in 1:dimension
        i[j]<max_degree || continue
        a=ntuple(k->i[k]+(j==k),dimension)
        a in present && continue
        all(a[k]==0 || ntuple(l->a[l]-(l==k),dimension) in present for k in 1:dimension) && push!(out,a)
    end
    sort!(collect(out);by=a->(sum(a),a))
end

# A finite-point discrepancy, NOT an integrated TV error or a target bound.
# Conditional KL catches mean/correlation errors invisible to marginal mass.
function discrepancy(a,b)
    tv=min(1.,sqrt(max(0.,min(AD._kl(a,b),AD._kl(b,a)))/2))
    tanh(abs(a.logmass-b.logmass)/2)+tv
end

# Bounds on each univariate Newton polynomial via Bernstein coefficients on a
# subinterval. Their tensor product bounds the full interpolant, including peaks
# missed by both positive quadrature rules. No bound on the true oracle is claimed.
function basis_bernstein(xs,degree,lo,hi,elevated_degree=degree)
    bern=[1.]
    for k in 1:degree
        a=(lo-xs[k])/(xs[degree+1]-xs[k]);b=(hi-xs[k])/(xs[degree+1]-xs[k])
        out=zeros(k+1)
        for j in 0:k-1
            out[j+1]+=(k-j)/k*a*bern[j+1]
            out[j+2]+=(j+1)/k*b*bern[j+1]
        end
        bern=out
    end
    # Degree elevation uses convex combinations, avoiding the cancellation of
    # converting high-order power coefficients into the Bernstein basis.
    for k in degree+1:elevated_degree
        out=zeros(k+1)
        for j in 0:k-1
            out[j+1]+=(k-j)/k*bern[j+1];out[j+2]+=(j+1)/k*bern[j+1]
        end
        bern=out
    end
    bern
end
function peak_bound(d,c)
    lo=standardize(d.lower,d.upper,c.lo);hi=standardize(d.lower,d.upper,c.hi)
    active=findall(!iszero,view(d.coefficients,1,:))
    isempty(active) && return 0.
    degrees=[maximum(d.indices[i][j] for i in active) for j in eachindex(lo)]
    # Sum Bernstein coefficients BEFORE bounding whenever the small scalar
    # tensor fits. This preserves cancellations between Newton terms. Bounding
    # each term separately is valid but excessively pessimistic even for a
    # separable Gaussian. This tensor uses no additional graph oracle calls.
    if prod(big.(degrees.+1))<=65536
        bases=[[basis_bernstein(d.sequence,k,lo[j],hi[j],degrees[j]) for k in 0:degrees[j]] for j in eachindex(lo)]
        coeff=zeros(Tuple(degrees.+1))
        for i in active
            index=d.indices[i];v=d.coefficients[1,i]
            for t in CartesianIndices(coeff)
                coeff[t]+=v*prod(bases[j][index[j]+1][t[j]] for j in eachindex(lo))
            end
        end
        return maximum(coeff)
    end
    ranges=[[extrema(basis_bernstein(d.sequence,k,lo[j],hi[j])) for k in 0:degrees[j]] for j in eachindex(lo)]
    upper=0.
    for i in active
        index=d.indices[i];v=d.coefficients[1,i]
        a=b=1.
        for j in eachindex(lo)
            l,h=ranges[j][index[j]+1];a,b=extrema((a*l,a*h,b*l,b*h))
        end
        upper+=max(v*a,v*b)
    end
    upper
end
function integrate(d;orders,tolerance,max_cells,max_evaluations,checkpoint)
    required=sum(big(o)^length(d.lower) for o in orders)
    required<=max_evaluations || throw(ArgumentError("quadrature budget cannot evaluate one cell at these orders and dimension"))
    count=0
    function assess(c)
        checkpoint===nothing || checkpoint()
        vals=Vector{Float64}[];sample=-Inf
        for order in orders
            terms=Float64[]
            for (q,w) in AD._points(c,order)
                count+=1
                checkpoint===nothing || (count%256==1 && checkpoint())
                value=logvalue(d,q);push!(terms,log(w)+value)
                order==last(orders) && (sample=max(sample,value))
            end
            push!(vals,terms)
        end
        mass=AD._logsum(last(vals));error=AD._logabsdiff(AD._logsum(first(vals)),mass)
        peak=peak_bound(d,c)
        peak-sample>1. && (error=max(error,peak+log(prod(c.hi-c.lo))))
        (mass=mass,error=error)
    end
    cells=[(lo=copy(d.lower),hi=copy(d.upper))];checks=[assess(first(cells))]
    while true
        z=AD._logsum([c.mass for c in checks]);err=exp(AD._logsum([c.error for c in checks])-z)
        reason=err<=tolerance ? :resolution : length(cells)>=max_cells ? :cell_budget :
            count+2required>max_evaluations ? :evaluation_budget : :refine
        reason!=:refine && return (cells=cells,logz=z,error=err,resolved=err<=tolerance,reason=reason,evaluations=count)
        i=argmax([c.error for c in checks]);c=cells[i]
        # Relative width prevents arbitrary units from controlling subdivision.
        j=argmax((c.hi-c.lo)./(d.upper-d.lower));m=(c.lo[j]+c.hi[j])/2
        l=copy(c.lo);h=copy(c.hi);l[j]=m;h[j]=m
        children=[(lo=c.lo,hi=h),(lo=l,hi=c.hi)]
        splice!(cells,i:i,children);splice!(checks,i:i,assess.(children))
    end
end

function local_quadrature_bound(c,orders)
    n=big(c)
    (big(orders[1])+2big(orders[2]))*n*(n+1)÷2+4sum(big,orders)*(2n-1)
end
function local_cell_limit(max_cells,orders,budget)
    lo=0;hi=max_cells
    while lo<hi
        delta=hi-lo;mid=lo+delta÷2+isodd(delta)
        if local_quadrature_bound(mid,orders)<=budget;lo=mid;else;hi=mid-1;end
    end
    lo
end

"""
Sparse nested Leja interpolation with positive cubature of exp(log density).
Two deterministic off-grid panels exercise interactions absent from axis lines.
All oracle calls, including validation and unused frontier nodes, consume budget.
`resolved` only means these finite interpolation checks and surrogate cubature
passed. It is neither the old cell-integrated refinement indicator nor a bound
against the posterior. Remote modes, conditional shape and tails remain separate.
If one-dimensional global interpolation is unresolved, `local_fallback=true`
can return an `AdaptiveDensity` using its local integrated refinement diagnostic.
`diagnostic_kind` identifies that outcome. Both paths share all oracle evaluations
and the total fit budget; a failed fallback keeps the global candidate unresolved.
"""
function sparse_density(oracle,lower,upper;tolerance=.01,max_nodes=257,max_cells=128,
        orders=(5,9),max_degree=64,max_quadrature_evaluations=2_000_000,
        progress=nothing,checkpoint=nothing,screen_only=false,local_fallback=true)
    lo=collect(Float64,lower);hi=collect(Float64,upper);d=length(lo)
    d>0 && length(hi)==d && all(isfinite,lo) && all(isfinite,hi) && all(hi.>lo) || throw(ArgumentError("invalid box"))
    max_nodes isa Integer && max_cells isa Integer && max_degree isa Integer &&
        max_quadrature_evaluations isa Integer && max_quadrature_evaluations>=1 &&
        max_nodes>=1 && max_cells>=1 && 2<=max_degree<=64 && isfinite(tolerance) && tolerance>0 &&
        length(orders)==2 && all(o->o isa Integer,orders) && 2<=orders[1]<orders[2] || throw(ArgumentError("invalid sparse settings"))
    # Reject impossible first cubature cells before spending any graph fits.
    screen_only || sum(big(o)^d for o in orders)<=max_quadrature_evaluations ||
        throw(ArgumentError("quadrature budget cannot evaluate one cell at these orders and dimension"))
    seq=leja(max_degree+1);cache=Dict{Tuple,AD.Node}();indices=Tuple[ntuple(_->0,d)]
    function evaluate(z)
        q=point(lo,hi,z);key=Tuple(q)
        get!(cache,key) do
            length(cache)<max_nodes || throw(AD.NodeBudget())
            checkpoint===nothing || checkpoint()
            AD.Node(oracle(q))
        end
    end
    zero_node=evaluate(zeros(d));r=length(zero_node.mean);coef=reshape(pack(zero_node),:,1)
    history=NamedTuple[];reason=:node_budget;passed=false;verror=Inf;ferror=Inf
    iteration=0
    while !screen_only
        iteration+=1;active=frontier(indices,d,max_degree)
        isempty(active) && (reason=:degree_budget;break)
        # First establish all quadratic terms; subsequent additions are ranked
        # by joint geometry, with breadth steps to avoid starving interactions.
        errors=Float64[];newcoef=Vector{Float64}[];complete=true
        try
            for i in active
                z=seq[collect(i).+1];n=evaluate(z);pred=coef*weights(seq,indices,z)
                push!(errors,discrepancy(n,unpack(pred,r)));push!(newcoef,pack(n)-pred)
            end
            if all(sum(i)>=3 for i in active)
                # Traverse lazily: a small fit budget must not first allocate
                # the exponential panel, and screening requires no panel at all.
                checked_error=0.
                for panel_scale in (1/sqrt(3.),1/(2sqrt(3.)))
                    for t in Iterators.product(ntuple(_->(-1.,1.),d)...)
                        z=collect(t).*panel_scale
                        checked_error=max(checked_error,discrepancy(evaluate(z),unpack(coef*weights(seq,indices,z),r)))
                    end
                end
                verror=checked_error
            end
        catch err
            err isa AD.NodeBudget || rethrow();complete=false
        end
        ferror=isempty(errors) ? Inf : maximum(errors)
        h=(nodes=length(cache),terms=length(indices),frontier_error=ferror,validation_error=verror,complete=complete)
        push!(history,h);progress===nothing || progress(h)
        complete || break
        if ferror<=tolerance && verror<=tolerance
            passed=true;reason=:finite_checks;break
        end
        minlevel=minimum(sum,active)
        chosen=minlevel<3 || iteration%4==0 ? findall(i->sum(i)==minlevel,active) :
            sortperm(errors;rev=true)[1:min(d,length(active))]
        append!(indices,active[chosen]);coef=hcat(coef,newcoef[chosen]...)
    end
    screen_only && (reason=:shape_screen)
    fallback_diagnostics=nothing;fallback_quadrature_bound=0
    # Conservative count for the one-dimensional local algorithm: repeated
    # mass/peak quadrature over 1:C leaves plus two geometry evaluations at
    # each point of both trial children of at most 2C-1 distinct checked leaves.
    local_cells=local_cell_limit(max_cells,orders,max_quadrature_evaluations÷2)
    # A global polynomial is a poor uniform approximation to a piecewise-smooth
    # target. In one dimension there is no sparse/tensor distinction: reuse the
    # proven local integrator and ALL exact oracle values, within the SAME total
    # fit budget. Multidimensional tensor fallback would restore the fit barrier
    # this method is intended to remove, so it is not enabled implicitly.
    if !screen_only && !passed && d==1 && local_fallback && local_cells>0 && max_nodes-length(cache)>=4
        sparse_nodes=length(cache)
        local_result=AD.adaptive_density(oracle,lo,hi;tolerance,max_nodes,max_cells=local_cells,orders,cache,progress,checkpoint)
        fallback_quadrature_bound=Int(local_quadrature_bound(local_cells,orders))
        fallback_diagnostics=merge(local_result.diagnostics,(sparse_nodes=sparse_nodes,))
        if local_result.diagnostics.resolved
            diagnostics=merge(local_result.diagnostics,(fallback=:local_tensor,
                fallback_reason=reason,sparse_nodes=sparse_nodes,sparse_terms=length(indices),
                sparse_history=history,diagnostic_kind=:local_integrated_joint_refinement,
                quadrature_evaluation_bound=fallback_quadrature_bound))
            return AD.AdaptiveDensity(local_result.cells,local_result.log_normalizer,local_result.integration_order,diagnostics)
        end
        # An unsuccessful fallback must not discard the more detailed global
        # approximation. Its extra oracle work is still charged and reported.
    end
    shell=SparseDensity(lo,hi,seq,indices,coef,r,[(lo=lo,hi=hi)],0.,last(orders),(;))
    integral=screen_only ? (cells=shell.cells,logz=zero_node.logmass+log(prod(hi-lo)),error=Inf,
        resolved=false,reason=:screen_only,evaluations=0) :
        integrate(shell;orders,tolerance=tolerance/10,max_cells,
            max_evaluations=max_quadrature_evaluations-fallback_quadrature_bound,checkpoint)
    diag=(resolved=passed && integral.resolved,reason=passed && !integral.resolved ? :quadrature_budget : reason,
        nodes=length(cache),terms=length(indices),interpolation_checks_pass=passed,
        frontier_error=ferror,validation_error=verror,integration_error=integral.error,
        quadrature_reason=integral.reason,quadrature_evaluations=integral.evaluations,
        fallback=fallback_diagnostics===nothing ? :none : :unresolved_local_attempt,
        fallback_diagnostics=fallback_diagnostics,
        quadrature_evaluation_bound=integral.evaluations+fallback_quadrature_bound,
        history=history,lower=lo,upper=hi,tail_validated=false,target_validated=false,conditional_validated=false,
        diagnostic_kind=:finite_point_joint_discrepancy,
        oracle_points=[collect(q) for q in sort(collect(keys(cache)))])
    SparseDensity(lo,hi,seq,indices,coef,r,integral.cells,integral.logz,last(orders),diag)
end
end
