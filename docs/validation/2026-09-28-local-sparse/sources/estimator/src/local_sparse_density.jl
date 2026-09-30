# Experimental spatial refinement of sparse polynomial conditional densities.
module LocalSparseDensities
using LinearAlgebra
using ..AdaptiveDensities, ..SparseDensities
import ..AdaptiveDensities: adaptive_geometry, adaptive_marginal_logpdf,
    adaptive_logpdf, adaptive_expectation
const AD=AdaptiveDensities
const SD=SparseDensities
export LocalSparseDensity, local_sparse_density

struct LocalSparseDensity
    patches::Vector{SD.SparseDensity}
    log_normalizer::Float64
    integration_order::Int
    diagnostics::NamedTuple
end
in_patch(p,q)=length(q)==length(p.lower) && all(p.lower.<=q) && all(q.<=p.upper)
function patch_at(d,q)
    i=findfirst(p->in_patch(p,q),d.patches)
    i===nothing && throw(DomainError(q,"outside the local sparse density's box"))
    d.patches[i]
end
adaptive_geometry(d::LocalSparseDensity,q)=adaptive_geometry(patch_at(d,q),q)
function adaptive_marginal_logpdf(d::LocalSparseDensity,q)
    i=findfirst(p->in_patch(p,q),d.patches)
    i===nothing ? -Inf : SD.logvalue(d.patches[i],q)-d.log_normalizer
end
function adaptive_logpdf(d::LocalSparseDensity,q,y)
    p=adaptive_marginal_logpdf(d,q);p==-Inf && return p
    n=adaptive_geometry(d,q);v=LowerTriangular(n.root)\(y-n.mean)
    p-.5sum(abs2,v)-sum(log,diag(n.root);init=0.)-.5length(v)*log(2pi)
end
function adaptive_expectation(f,d::LocalSparseDensity;order=d.integration_order)
    sum(w*exp(n.logmass-d.log_normalizer)*f(q,n) for p in d.patches for c in p.cells
        for (q,w) in AD._points(c,order) for n in (adaptive_geometry(p,q),))
end

shell(lo,hi,seq,indices,coef,r)=SD.SparseDensity(lo,hi,seq,indices,coef,r,
    [(lo=lo,hi=hi)],0.,0,(;))
function constant_patch(lo,hi,seq,evaluate)
    n=evaluate((lo+hi)/2)
    shell(lo,hi,seq,Tuple[ntuple(_->0,length(lo))],reshape(SD.pack(n),:,1),length(n.mean))
end

# Complete total-degree spaces keep all mixed terms at the selected degree.
# The next total-degree frontier is evaluated before any success is reported.
function grow(p,degree,evaluate)
    indices=copy(p.indices);coef=copy(p.coefficients)
    try
        while true
            active=filter(i->sum(i)<=degree,SD.frontier(indices,length(p.lower),degree))
            isempty(active) && break
            level=minimum(sum,active)
            for i in filter(i->sum(i)==level,active)
                z=p.sequence[collect(i).+1];n=evaluate(SD.point(p.lower,p.upper,z))
                delta=SD.pack(n)-coef*SD.weights(p.sequence,indices,z)
                push!(indices,i);coef=hcat(coef,delta)
            end
        end
    catch err
        err isa AD.NodeBudget || rethrow()
        return shell(p.lower,p.upper,p.sequence,indices,coef,p.remainder),false
    end
    shell(p.lower,p.upper,p.sequence,indices,coef,p.remainder),true
end

function assess(p,degree,evaluate;previous_error=Inf,previous_frontier=Inf)
    d=length(p.lower);front=filter(i->sum(i)==degree+1,SD.frontier(p.indices,d,degree+1))
    ferr=0.;verr=0.;complete=true;directions=zeros(d)
    try
        for i in front
            z=p.sequence[collect(i).+1];n=evaluate(SD.point(p.lower,p.upper,z))
            err=SD.discrepancy(n,SD.unpack(p.coefficients*SD.weights(p.sequence,p.indices,z),p.remainder))
            ferr=max(ferr,err)
            # Joint, dimensionless geometry determines directional importance;
            # raw mean or Cholesky coefficients have incompatible units.
            directions=max.(directions,err.*collect(i)./sum(i))
        end
        # Mixed interactions invisible to the current polynomial and its
        # frontier must still be exercised. These points are never fitted.
        for scale in (1/sqrt(3.),1/(2sqrt(3.)))
            for signs in Iterators.product(ntuple(_->(-1.,1.),d)...)
                z=collect(signs).*scale;n=evaluate(SD.point(p.lower,p.upper,z))
                verr=max(verr,SD.discrepancy(n,SD.unpack(p.coefficients*SD.weights(p.sequence,p.indices,z),p.remainder)))
            end
        end
    catch err
        err isa AD.NodeBudget || rethrow();complete=false
    end
    (patch=p,degree=degree,error=complete ? max(ferr,verr) : Inf,
        frontier_error=ferr,validation_error=verr,complete=complete,
        previous_error=previous_error,previous_frontier=previous_frontier,directions=directions)
end

"""
Piecewise sparse polynomials of log density and conditional log-Cholesky geometry.
Order increases while the finite joint discrepancy decreases by at least a
factor of two; otherwise the failing region is bisected. Children start at total
degree two. All fitted, frontier and off-grid nodes share one exact oracle cache.

Every leaf must pass the SAME uniform finite-point criterion as the global
sparse method. No mass weighting relaxes errors in the tails. Positive cubature
and interpolant peak guards are separate checks. Neither check proves target,
tail, conditional-shape or cook coverage accuracy. Adjacent approximations may
have small boundary jumps; boundary agreement is not asserted.
"""
function local_sparse_density(oracle,lower,upper;tolerance=.01,max_nodes=257,max_cells=128,
        orders=(5,9),max_degree=8,max_quadrature_evaluations=2_000_000,
        progress=nothing,checkpoint=nothing)
    lo=collect(Float64,lower);hi=collect(Float64,upper);d=length(lo)
    d>0 && length(hi)==d && all(isfinite,lo) && all(isfinite,hi) && all(hi.>lo) || throw(ArgumentError("invalid box"))
    max_nodes isa Integer && max_cells isa Integer && max_degree isa Integer &&
        max_quadrature_evaluations isa Integer && max_nodes>=1 && max_cells>=1 &&
        3<=max_degree<=12 && isfinite(tolerance) && tolerance>0 &&
        length(orders)==2 && all(o->o isa Integer,orders) && 2<=orders[1]<orders[2] || throw(ArgumentError("invalid local sparse settings"))
    required=sum(big(o)^d for o in orders)
    required<=max_quadrature_evaluations || throw(ArgumentError("quadrature budget cannot evaluate one cell"))
    seq=SD.leja(max_degree+2);cache=Dict{Tuple,AD.Node}();history=NamedTuple[]
    function evaluate(q)
        get!(cache,Tuple(q)) do
            length(cache)<max_nodes || throw(AD.NodeBudget())
            checkpoint===nothing || checkpoint()
            AD.Node(oracle(q))
        end
    end
    p,built=grow(constant_patch(lo,hi,seq,evaluate),2,evaluate)
    leaves=[assess(p,2,evaluate)];reason=:node_budget
    while true
        checkpoint===nothing || checkpoint()
        errors=[c.error for c in leaves];complete=built && all(c.complete for c in leaves)
        h=(nodes=length(cache),cells=length(leaves),degrees=[c.degree for c in leaves],
            frontier_error=maximum(c.frontier_error for c in leaves),
            validation_error=maximum(c.validation_error for c in leaves),
            indicator=maximum(errors),complete=complete)
        push!(history,h);progress===nothing || progress(h)
        complete || break
        maximum(errors)<=tolerance && (reason=:finite_checks;break)
        i=argmax(errors);c=leaves[i]
        # An off-grid interaction may be invisible to the preceding frontier.
        # First incorporate its newly discovered polynomial terms before using
        # failure to improve as evidence for spatial refinement.
        split=c.degree>=max_degree || (c.degree>=3 && c.previous_frontier>tolerance && c.error>=.5c.previous_error)
        if !split
            next,built=grow(c.patch,c.degree+1,evaluate)
            leaves[i]=assess(next,c.degree+1,evaluate;previous_error=c.error,previous_frontier=c.frontier_error)
            continue
        end
        length(leaves)<max_cells || (reason=:cell_budget;break)
        (length(leaves)+1)*required<=max_quadrature_evaluations || (reason=:quadrature_budget;break)
        widths=(c.patch.upper-c.patch.lower)./(hi-lo)
        # Prefer the largest frontier discrepancy, using relative width only
        # for ties or an interaction invisible to the entire frontier.
        scores=c.directions
        candidates=findall(x->x>=maximum(scores)*(1-1e-12),scores)
        axis=candidates[argmax(widths[candidates])]
        mid=(c.patch.lower[axis]+c.patch.upper[axis])/2
        c.patch.lower[axis]<mid<c.patch.upper[axis] || (reason=:spatial_limit;break)
        left_hi=copy(c.patch.upper);left_hi[axis]=mid
        right_lo=copy(c.patch.lower);right_lo[axis]=mid
        children=typeof(c)[];ready=true
        # Keep the parent until both child polynomials exist. A fit-budget exit
        # cannot leave a hole or a partially changed domain in the candidate.
        try
            for (a,b) in ((c.patch.lower,left_hi),(right_lo,c.patch.upper))
                child,ok=grow(constant_patch(a,b,seq,evaluate),2,evaluate)
                ok || (ready=false;break)
                push!(children,assess(child,2,evaluate))
            end
        catch err
            err isa AD.NodeBudget || rethrow();ready=false
        end
        ready || break
        splice!(leaves,i:i,children)
    end
    patches=SD.SparseDensity[];integrals=NamedTuple[];evaluations=0
    for (i,c) in enumerate(leaves)
        available=max_quadrature_evaluations-evaluations-Int(required)*(length(leaves)-i)
        result=SD.integrate(c.patch;orders,tolerance=tolerance/10,max_cells,
            max_evaluations=available,checkpoint)
        evaluations+=result.evaluations;push!(integrals,result)
        p=c.patch
        push!(patches,SD.SparseDensity(p.lower,p.upper,p.sequence,p.indices,p.coefficients,
            p.remainder,result.cells,result.logz,last(orders),(;)))
    end
    logz=AD._logsum([r.logz for r in integrals])
    integration_error=all(isfinite(r.error) for r in integrals) ?
        sum(exp(r.logz-logz)*r.error for r in integrals) : Inf
    interpolation_pass=reason==:finite_checks
    quadrature_pass=all(r.resolved for r in integrals) && integration_error<=tolerance/10
    diag=(resolved=interpolation_pass && quadrature_pass,
        reason=interpolation_pass && !quadrature_pass ? :quadrature_budget : reason,
        nodes=length(cache),cells=length(leaves),degrees=[c.degree for c in leaves],
        interpolation_checks_pass=interpolation_pass,frontier_error=maximum(c.frontier_error for c in leaves),
        validation_error=maximum(c.validation_error for c in leaves),
        integration_error=integration_error,quadrature_evaluations=evaluations,
        quadrature_reasons=[r.reason for r in integrals],history=history,lower=lo,upper=hi,
        tail_validated=false,target_validated=false,conditional_validated=false,
        diagnostic_kind=:local_finite_point_joint_discrepancy,
        oracle_points=[collect(q) for q in sort(collect(keys(cache)))])
    LocalSparseDensity(patches,logz,last(orders),diag)
end
end
