using LinearAlgebra
candidate_chart(p::AdaptiveConditionalCandidate)=AffineChart(p.center,p.scale)
candidate_chart(p::MappedConditionalCandidate)=p.chart

"""
Compare candidates in common original graph axes and innovation coordinates,
not in their different reference coordinates q. Parameter axes normally use
unconstrained site coordinates; these are not automatically physical units.
Integrate a bounded discrepancy under the mixture of their own normalized
densities, so distinct curved supports are included. Two quadrature orders check
this estimate; it is not a rigorous target-error or remote-mode certificate.
"""
function mapped_density_distance(a,b;orders=(7,11),checkpoint=nothing)
    a.axes==b.axes && a.free==b.free && E._same_conditional_decoder(a.decoder,b.decoder) ||
        throw(ArgumentError("comparison requires identical target coordinates and decoder"))
    results=NamedTuple[]
    for order in orders
        total=0.;outside=0.;count=0
        for (source,other) in ((a,b),(b,a))
            sc=candidate_chart(source);oc=candidate_chart(other)
            for cell in source.density.cells,(q,w) in AdaptiveDensities._points(cell,order)
                count+=1;checkpoint===nothing || (count%32==1 && checkpoint())
                ns=AdaptiveDensities._geometry(cell,q)
                weight=w*exp(ns.logmass-source.density.log_normalizer)
                qo=chart_inverse(oc,chart_forward(sc,q))
                lb=adaptive_marginal_logpdf(other.density,qo)
                if lb==-Inf
                    total+=weight/2;outside+=weight/2;continue
                end
                na=ns;nb=adaptive_geometry(other.density,qo)
                la=ns.logmass-source.density.log_normalizer-chart_logjac(sc,q)
                lb-=chart_logjac(oc,qo)
                delta=abs(la-lb)
                tv=min(1.,sqrt(min(AdaptiveDensities._kl(na,nb),AdaptiveDensities._kl(nb,na))/2))
                total+=weight/2*(tanh(delta/2)+2exp(-delta)/(1+exp(-delta))*tv)
            end
        end
        push!(results,(order=order,tv_bound_estimate=total,exclusive_support_mass=outside))
    end
    return (tv_bound_estimate=last(results).tv_bound_estimate,
        integration_change=length(results)>1 ? abs(results[end].tv_bound_estimate-results[end-1].tv_bound_estimate) : Inf,
        history=results,target_validated=false)
end

"A selected graph-coordinate CDF; physical thresholds need the site's coordinate transform."
function mapped_axis_cdf(p,axis,value;order=p.density.integration_order,checkpoint=nothing)
    chart=candidate_chart(p);k=chart_dimension(chart)
    axis isa Integer && 1<=axis<=k || throw(BoundsError())
    order isa Integer && order>=2 && value isa Real && !isnan(value) || throw(ArgumentError("invalid CDF query"))
    istril(chart_base(chart).scale) && all(>(0),diag(chart_base(chart).scale)) ||
        throw(ArgumentError("CDF requires a monotone triangular chart"))
    xs,ws=AdaptiveDensities._legendre(order);total=0.;count=0
    for cell in p.density.cells,index in Iterators.product(ntuple(_->1:order,k)...)
        count+=1;checkpoint===nothing || (count%32==1 && checkpoint())
        q=zeros(k);weight=1.
        for j in 1:k
            width=cell.hi[j]-cell.lo[j]
            if j==axis
                a,s=ConditionalCharts.axis_intercept_scale(chart,q,j)
                upper=min(cell.hi[j],(value-a)/s)
                width=max(0.,upper-cell.lo[j])
            end
            q[j]=cell.lo[j]+width*xs[index[j]];weight*=width*ws[index[j]]
        end
        weight==0 && continue
        total+=weight*exp(AdaptiveDensities._logvalue(cell,q)-p.density.log_normalizer)
    end
    return total
end

"Separate numerical resolution of a nonlinear physical CDF from density resolution."
function mapped_axis_cdf_report(p,axis,value;orders=(17,33,65),tolerance=1e-5,
        max_evaluations=1_000_000,checkpoint=nothing)
    length(orders)>=2 && all(o->o isa Integer && o>=2,orders) && all(diff(collect(orders)).>0) &&
        isfinite(tolerance) && tolerance>0 && max_evaluations isa Integer && max_evaluations>0 ||
        throw(ArgumentError("invalid CDF resolution settings"))
    estimates=Float64[];completed=Int[];evaluations=big(0);budget=false
    for order in orders
        cost=big(length(p.density.cells))*big(order)^chart_dimension(candidate_chart(p))
        if evaluations+cost>max_evaluations;budget=true;break;end
        push!(estimates,mapped_axis_cdf(p,axis,value;order,checkpoint));push!(completed,order)
        evaluations+=cost
    end
    change=length(estimates)>=2 ? abs(estimates[end]-estimates[end-1]) : Inf
    resolved=!budget && change<=tolerance
    return (value=isempty(estimates) ? nothing : last(estimates),integration_change=change,resolved=resolved,
        reason=budget ? :evaluation_budget : resolved ? :integration_resolution : :integration_unresolved,
        estimates=estimates,orders=completed,requested_orders=orders,evaluations=evaluations,target_validated=false)
end
