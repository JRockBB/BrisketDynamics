# Direct deterministic posterior integrals. Include after BrisketEstimator.
# No Gaussian conditional evidence and no Monte Carlo inference are used here.
isdefined(@__MODULE__,:DirectQuadrature) || include(joinpath(@__DIR__,"../src/direct_quadrature.jl"))
using .DirectQuadrature, LinearAlgebra
isdefined(@__MODULE__,:AbstractDirectChart) || include(joinpath(@__DIR__,"direct_transport.jl"))

struct DirectPosterior
    reference::NamedTuple
    center::Vector{Float64}
    root::Matrix{Float64}
    transport::AbstractDirectChart
    origin_potential::Float64
    origin_logjac::Float64
    normalizer::Float64
    log_normalizer::Float64
    mean::Vector{Float64}
    covariance::Matrix{Float64}
    covariance_root::Matrix{Float64}
    integration_rotation::Matrix{Float64}
    physical_mean::Vector{Float64}
    physical_variance::Vector{Float64}
    diagnostics::NamedTuple
    settings::NamedTuple
end

function direct_chart(sm)
    ref=E._posterior_transport(sm)
    A=Matrix{Float64}(I,ref.n,ref.n)+ref.V*Diagonal(ref.a.-1)*ref.V'
    # QR of a covariance root avoids forming its ill-conditioned Gram matrix.
    R=Matrix(qr(A').R);L=Matrix(LowerTriangular(R'))
    L=L*Diagonal(sign.(diag(L)))
    return ref,copy(ref.center),L
end

function direct_logpdf(p::DirectPosterior,u;allow_unresolved=false)
    allow_unresolved || p.diagnostics.resolved || throw(ArgumentError("posterior normalizer is unresolved; diagnostic access requires allow_unresolved=true"))
    length(u)==length(p.center) || throw(DimensionMismatch())
    -E._transport_draw(p.reference,u).potential-p.log_normalizer
end

function _direct_batch(f)
    function batch(points)
        out=Vector{Vector{Float64}}(undef,length(points))
        Threads.@threads for j in eachindex(points)
            out[j]=collect(Float64,f(collect(points[j])))
        end
        out
    end
end

"""
Integrate the original full-history posterior in ALL independent prior/process
coordinates. The local Gaussian is an importance coordinate system only. Every
quadrature point receives the exact nonlinear likelihood correction. Gaussian OU
sensor histories are marginalized analytically from their original graph factors.

Means/covariances are integrated summaries of the full target; they do not define
the posterior. A signed cubature is never exposed as a discrete probability law.
`resolved` describes estimated quadrature precision, not statistical calibration.
"""
function direct_posterior_fit(sm;atol=.002,rtol=.002,max_evaluations=100_000,
        max_level=5,max_seconds=600.,progress=nothing,chart=nothing,rule=:hermite,transport_axes=0,
        curvature_rotation=false,moments=true,physical_moments=false)
    start=time();transport_diagnostics=nothing
    if chart===nothing && transport_axes>0
        fitted=fit_direct_transport(sm;max_axes=transport_axes,max_seconds)
        ref=fitted.reference;center=fitted.chart.center;L=fitted.base_root;transport=fitted.chart
        transport_diagnostics=fitted.diagnostics
    else
        ref,center,L=chart===nothing ? direct_chart(sm) : chart
        transport=DirectAffineChart(center,L)
    end
    origin_map=direct_map(transport,zeros(ref.n))
    origin=E._transport_draw(ref,origin_map.u);g=length(origin.draw);ny=sm.model.bl.n
    root,_,offsets=E._conditional_root(deepcopy(sm.graph))
    rows=vcat(collect(offsets[sm.key_z]),collect(offsets[last(sm.nodes)[2]]))
    R=Matrix(qr(root[rows,:]').R);S=Matrix(LowerTriangular(R'))
    S=S*Diagonal(sign.(diag(S)));Sinv=inv(LowerTriangular(S))
    base=origin.draw;base_x=E.belief_decode(base[end-ny+1:end],sm.model.bl)
    J=E.ForwardDiff.jacobian(y->E.belief_decode(y,sm.model.bl),base[end-ny+1:end])
    sx=sqrt.(vec(sum(abs2,J*S[end-ny+1:end,:];dims=2)))
    all(>(0),sx) || error("degenerate physical output scale")
    pairs=[(i,j) for j in 1:g for i in j:g]
    function base_integrand(v)
        mapped=direct_map(transport,v);r=E._transport_draw(ref,mapped.u)
        logratio=origin.potential-r.potential+sum(abs2,v)/2+mapped.logjac-origin_map.logjac
        weight=exp(logratio)
        isfinite(weight) || error("nonfinite direct likelihood correction")
        moments || return [weight]
        x=physical_moments ? E.belief_decode(r.draw[end-ny+1:end],sm.model.bl) : Float64[]
        # Explicit locals: this callback is evaluated concurrently. In Julia,
        # reusing a name assigned later in the enclosing function creates a
        # shared closure box, even when that assignment has not executed yet.
        local standardized=Sinv*(r.draw-base)
        local physical_standardized=physical_moments ? (x-base_x)./sx : Float64[]
        weight.*vcat(1.,standardized,[standardized[i]*standardized[j] for (i,j) in pairs],
            physical_standardized,physical_standardized.^2)
    end
    rotation=Matrix{Float64}(I,ref.n,ref.n);rotation_diagnostics=nothing
    if curvature_rotation
        function design(v)
            mapped=direct_map(transport,v);r=E._transport_draw(ref,mapped.u)
            logratio=origin.potential-r.potential+sum(abs2,v)/2+mapped.logjac-origin_map.logjac
            moments ? vcat(logratio,Sinv*(r.draw-base)) : [logratio]
        end
        fitted_rotation=direct_curvature_rotation(design,ref.n,length(ref.zm);max_seconds=max_seconds-(time()-start))
        rotation=fitted_rotation.rotation;rotation_diagnostics=fitted_rotation.diagnostics
    end
    integrand(v)=base_integrand(rotation*v)
    function moment_scale(v)
        mass=max(abs(v[1]),eps(Float64));shift=v[2:1+g]/mass
        diag_=zeros(g)
        for ((i,j),a) in zip(pairs,v[2+g:1+g+length(pairs)]/mass)
            i==j && (diag_[i]=a)
        end
        spread=sqrt.(max.(abs.(diag_-shift.^2),eps(Float64)))
        at=1+g+length(pairs);xmean=physical_moments ? v[at+1:at+ny]/mass : Float64[]
        xspread=physical_moments ? sqrt.(max.(abs.(v[at+ny+1:at+2ny]/mass-xmean.^2),eps(Float64))) : Float64[]
        # Precision is expressed in the estimated posterior's standard
        # deviations, not the old MAP covariance. Broad nonlinear directions
        # must not inherit arbitrarily small numerical units from that ellipse.
        # First-moment scales also account for their effect on centered moments.
        mass.*vcat(1.,spread./(1 .+2abs.(shift)./spread),
            [spread[i]*spread[j] for (i,j) in pairs],
            xspread./(1 .+2abs.(xmean)./xspread),xspread.^2)
    end
    settings=(atol=Float64(atol),rtol=Float64(rtol),max_evaluations=max_evaluations,
        max_level=max_level,max_seconds=Float64(max_seconds),rule=rule)
    remaining=max_seconds-(time()-start)
    remaining>0 || error("direct posterior chart exhausted the integration time budget")
    result=integrate_normal(integrand,ref.n;merge(settings,(max_seconds=remaining,))...,evaluate_batch=_direct_batch(integrand),progress,
        error_scale=moments ? moment_scale : nothing)
    Z=result.value[1];valid=isfinite(Z)&&Z>0
    normalizer_valid=valid
    mu=fill(NaN,g);C=fill(NaN,g,g);Cr=fill(NaN,g,g);mx=fill(NaN,ny);vx=fill(NaN,ny)
    if valid && moments
        z=result.value[2:1+g]/Z;M=zeros(g,g)
        for ((i,j),v) in zip(pairs,result.value[2+g:1+g+length(pairs)]/Z);M[i,j]=M[j,i]=v;end
        white=Matrix(Symmetric(M-z*z'));mu=base+S*z;C=Matrix(Symmetric(S*white*S'))
        physical_valid=true
        if physical_moments
            at=1+g+length(pairs);a=result.value[at+1:at+ny]/Z
            av=result.value[at+ny+1:at+2ny]/Z-a.^2
            mx=base_x+sx.*a;vx=sx.^2 .* av
            physical_valid=all(>=(0),vx)
        end
        valid=isposdef(Symmetric(white)) && physical_valid
        if valid;Cr=S*cholesky(Symmetric(white)).L;end
    end
    moments_valid=moments && valid
    logZ=Z>0 ? log(Z)-origin.potential+ref.n/2*log(2pi)+origin_map.logjac : NaN
    diagnostics=merge(result.diagnostics,(resolved=result.diagnostics.resolved&&valid,
        moments_requested=moments,moments_valid=moments_valid,
        physical_moments_requested=physical_moments,
        physical_moments_resolved=physical_moments&&moments_valid&&result.diagnostics.resolved,
        moments_resolved=moments_valid&&result.diagnostics.resolved,
        normalizer_valid=normalizer_valid,normalizer=Z,normalizer_error=result.error[1],
        coordinates=ref.n,gaussian_is_proposal_only=true,full_target=true,
        statistically_consistent=false,query_precision_required=true,transport=transport_diagnostics,
        integration_rotation=rotation_diagnostics,total_wall=time()-start))
    fitsettings=merge(settings,(transport_axes=transport_axes,curvature_rotation=curvature_rotation,
        moments=moments,physical_moments=physical_moments,supplied_chart=chart!==nothing))
    DirectPosterior(ref,center,L,transport,origin.potential,origin_map.logjac,Z,logZ,mu,C,Cr,rotation,mx,vx,diagnostics,fitsettings)
end

"Stable squared posterior radius, including weak contrasts in a marginal block."
function direct_radius(p::DirectPosterior,target;indices=eachindex(p.mean),allow_unresolved=false)
    allow_unresolved || p.diagnostics.moments_resolved || throw(ArgumentError("posterior moments are unresolved; diagnostic access requires allow_unresolved=true"))
    p.diagnostics.moments_valid || throw(ArgumentError("posterior moments are not valid"))
    ix=collect(indices);G=p.covariance_root[ix,:]
    scales=sqrt.(vec(sum(abs2,G;dims=2)))
    R=Matrix(qr((G./scales)').R)
    sum(abs2,LowerTriangular(R')\((target[ix]-p.mean[ix])./scales))
end

const _SQRT2=sqrt(2.)
_normal_cdf(x)=E.erfc(-x/_SQRT2)/2
_normal_quantile(p)=-_SQRT2*E.erfcinv(2p)

"""
Direct CDF of a latent or latest-state coordinate. The event is integrated by
truncating a standard normal coordinate, rather than summing discontinuous signed
quadrature weights or imposing a Gaussian interval. For latest states the final
process innovation enters linearly conditional on the previous history, so this
truncation is exact even though the entire preceding dynamics are nonlinear.
"""
function direct_cdf(p::DirectPosterior,index::Int,value::Real;atol=.002,rtol=.002,
        max_evaluations=p.settings.max_evaluations,max_level=p.settings.max_level,
        max_seconds=p.settings.max_seconds,progress=nothing,rule=p.settings.rule)
    ref=p.reference;nz=length(ref.zm);ny=ref.model.bl.n
    1<=index<=nz+ny && isfinite(value) || throw(ArgumentError("invalid posterior CDF query"))
    function base_integrand(q)
        condition=direct_conditioning(p.transport,ref,index,q)
        B=condition.rotation;target=condition.target;width=condition.width
        isfinite(width)&&width>0 || error("CDF conditioning direction is degenerate")
        w=copy(q);w[target]=0.;v=B*w;mapped=direct_map(p.transport,v)
        threshold=if index<=nz
            constant=ref.zm[index]+ref.zs[index]*mapped.u[index]
            (value-constant)/width
        else
            r0=E._transport_draw(ref,mapped.u)
            (value-r0.draw[index])/width
        end
        mass=_normal_cdf(threshold)
        mass==0 && return [0.]
        probability=mass*_normal_cdf(q[target])
        # Quadrature nodes are finite, but a far positive tail can round Phi to
        # one. The nearest interior float avoids an infinite, zero-weight solve.
        w[target]=_normal_quantile(clamp(probability,nextfloat(0.),prevfloat(1.)))
        v=B*w;mapped=direct_map(p.transport,v);r=E._transport_draw(ref,mapped.u)
        ratio=exp(p.origin_potential-r.potential+sum(abs2,v)/2+mapped.logjac-p.origin_logjac)
        [mass*ratio]
    end
    integrand(q)=base_integrand(p.integration_rotation*q)
    result=integrate_normal(integrand,ref.n;atol=atol*max(p.normalizer,eps()),rtol,
        max_evaluations,max_level,max_seconds,evaluate_batch=_direct_batch(integrand),progress,rule)
    valid=isfinite(p.normalizer)&&p.normalizer>0
    cdf=valid ? result.value[1]/p.normalizer : NaN
    denominator=p.normalizer-p.diagnostics.normalizer_error
    error=valid&&denominator>0 ? (result.error[1]+abs(cdf)*p.diagnostics.normalizer_error)/denominator : Inf
    resolved=p.diagnostics.resolved && result.diagnostics.resolved && 0<=cdf<=1 && error<=atol+rtol*abs(cdf)
    (value=cdf,error=error,resolved=resolved,diagnostics=result.diagnostics)
end

"""
CDF of a new, uncensored reading at the posterior's terminal time. Integrates
the conditional sensor discrepancy analytically, including correlation with all
past readings. This is a new white-noise replicate, not the CDF of an observation
already included in the likelihood. Use a forecast posterior for future times.
"""
function direct_predictive_cdf(p::DirectPosterior,sensor_id::Symbol,value::Real;
        atol=.002,rtol=.002,max_evaluations=p.settings.max_evaluations,
        max_level=p.settings.max_level,max_seconds=p.settings.max_seconds,progress=nothing,rule=p.settings.rule)
    isfinite(value) || throw(ArgumentError("predictive threshold must be finite"))
    ref=p.reference;model=ref.model;spec=E.sensor(model.log,sensor_id);noise=spec.noise
    h=E.compile_observable(spec.observable,model.sys,model.bl,model.blk,model.log)
    observed=findall(f->f.spec.id==sensor_id&&f.key_d!==nothing,ref.fs)
    k=zeros(length(ref.fs));discmean=0.;variance=noise.ar_sd^2+noise.white_sd^2
    if !isempty(observed)
        lastobs=observed[argmax([ref.fs[i].t for i in observed])];f=ref.fs[lastobs]
        row=findfirst(==(f.key_d),ref.dkeys)
        rho,sd=ref.t==f.t ? (1.,0.) : E.ou_step(noise,ref.t-f.t)
        k=rho.*vec(ref.Dobs*ref.D[row,:]);discmean=rho*ref.dm[row]
        variance=rho^2*sum(abs2,ref.D[row,:])+sd^2+noise.white_sd^2
    end
    slope=ref.W'*(ref.W*k);variance-=dot(k,slope)
    isfinite(variance)&&variance>0 || error("nonpositive conditional predictive noise variance")
    sd=sqrt(variance);observations=[f.value for f in ref.fs];nz=length(ref.zm)
    function base_integrand(v)
        mapped=direct_map(p.transport,v);r=E._transport_draw(ref,mapped.u)
        z=r.draw[1:nz];state=E.system_from_belief(E.belief_decode(r.draw[nz+1:end],model.bl),model.bl)
        params,_,tk=E._point_params(model,z,ref.t)
        mean=h(state,z,params,tk,ref.t)+discmean+dot(slope,observations-r.predictions-ref.obsbias)
        weight=exp(p.origin_potential-r.potential+sum(abs2,v)/2+mapped.logjac-p.origin_logjac)
        [weight*_normal_cdf((value-mean)/sd)]
    end
    integrand(v)=base_integrand(p.integration_rotation*v)
    result=integrate_normal(integrand,ref.n;atol=atol*max(p.normalizer,eps()),rtol,
        max_evaluations,max_level,max_seconds,evaluate_batch=_direct_batch(integrand),progress,rule)
    cdf=result.value[1]/p.normalizer
    denominator=p.normalizer-p.diagnostics.normalizer_error
    error=denominator>0 ? (result.error[1]+abs(cdf)*p.diagnostics.normalizer_error)/denominator : Inf
    resolved=p.diagnostics.resolved&&result.diagnostics.resolved&&0<=cdf<=1&&error<=atol+rtol*abs(cdf)
    (value=cdf,error=error,resolved=resolved,diagnostics=result.diagnostics)
end
