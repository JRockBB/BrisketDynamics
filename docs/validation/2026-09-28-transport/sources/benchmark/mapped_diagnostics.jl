using LinearAlgebra
candidate_chart(p::AdaptiveConditionalCandidate)=AffineChart(p.center,p.scale)
candidate_chart(p::MappedConditionalCandidate)=p.chart

"""
Compare candidates in common PHYSICAL axes and common innovation coordinates.
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

"A physical selected-axis CDF, splitting the mapped integration interval at its threshold."
function mapped_axis_cdf(p,axis,value;order=p.density.integration_order)
    chart=candidate_chart(p);k=chart_dimension(chart)
    1<=axis<=k || throw(BoundsError())
    istril(chart_base(chart).scale) && all(>(0),diag(chart_base(chart).scale)) ||
        throw(ArgumentError("CDF requires a monotone triangular chart"))
    xs,ws=AdaptiveDensities._legendre(order);total=0.
    for cell in p.density.cells,index in Iterators.product(ntuple(_->1:order,k)...)
        q=zeros(k);weight=1.
        for j in 1:k
            width=cell.hi[j]-cell.lo[j]
            if j==axis
                q[j]=0.;a=chart_forward(chart,q)[j]
                q[j]=1.;s=chart_forward(chart,q)[j]-a
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
function mapped_axis_cdf_report(p,axis,value;orders=(17,33,65),tolerance=1e-5)
    length(orders)>=2 && all(diff(collect(orders)).>0) && tolerance>0 || throw(ArgumentError("invalid CDF resolution settings"))
    estimates=Float64[]
    for order in orders
        push!(estimates,mapped_axis_cdf(p,axis,value;order))
    end
    change=abs(estimates[end]-estimates[end-1])
    return (value=last(estimates),integration_change=change,resolved=change<=tolerance,
        estimates=estimates,orders=orders,target_validated=false)
end
