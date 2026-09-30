module ConditionalCharts

using LinearAlgebra
export AffineChart, TriangularChart, chart_forward, chart_inverse, chart_logjac,
       fit_triangular_chart, chart_dimension, chart_base

struct AffineChart
    center::Vector{Float64}
    scale::Matrix{Float64}
    function AffineChart(center,scale)
        c=collect(Float64,center);L=Matrix{Float64}(scale)
        !isempty(c) && size(L)==(length(c),length(c)) && all(isfinite,c) &&
            all(isfinite,L) && isfinite(logabsdet(L)[1]) || throw(ArgumentError("invalid affine chart"))
        new(c,L)
    end
end
chart_dimension(c::AffineChart)=length(c.center)
chart_base(c::AffineChart)=c
chart_forward(c::AffineChart,q)=c.center+c.scale*q
chart_inverse(c::AffineChart,z)=c.scale\(z-c.center)
chart_logjac(c::AffineChart,q)=logabsdet(c.scale)[1]

function _features(t,radius)
    x=t./radius
    return vcat(1.,x,[x[i]*x[j] for j in eachindex(x) for i in 1:j])
end

"""
An invertible triangular change of integration coordinates, not an assumed
posterior distribution. In affine coordinates t, each t[j] = m(t[1:j-1]) +
exp(s(t[1:j-1])) q[j]. Quadratic functions m and s describe conditional mode
and log-scale pilots. A poor fit can waste work but never replaces the target
density: every downstream oracle evaluation includes the exact map Jacobian.

A finite q box has a curved image in the physical axes. Domain coverage must be
checked independently; resolution on that box does not validate tails or modes.
"""
struct TriangularChart
    base::AffineChart
    means::Vector{Vector{Float64}}
    logscales::Vector{Vector{Float64}}
    radius::Float64
    function TriangularChart(base,means,logscales,radius)
        k=chart_dimension(base)
        istril(base.scale) && all(>(0),diag(base.scale)) || throw(ArgumentError("triangular chart needs a lower triangular positive base"))
        length(means)==length(logscales)==k-1 && isfinite(radius) && radius>0 || throw(ArgumentError("invalid triangular coefficients"))
        ms=[collect(Float64,v) for v in means];ss=[collect(Float64,v) for v in logscales]
        for p in 1:k-1
            n=1+p+p*(p+1)÷2
            length(ms[p])==length(ss[p])==n && all(isfinite,ms[p]) && all(isfinite,ss[p]) ||
                throw(ArgumentError("invalid triangular polynomial"))
        end
        new(base,ms,ss,Float64(radius))
    end
end
chart_dimension(c::TriangularChart)=chart_dimension(c.base)
chart_base(c::TriangularChart)=c.base
function _stage(c,j,t)
    f=_features(t[1:j-1],c.radius)
    return dot(c.means[j-1],f),dot(c.logscales[j-1],f)
end
function _forward(c::TriangularChart,q)
    length(q)==chart_dimension(c) || throw(DimensionMismatch())
    t=copy(q);logjac=logabsdet(c.base.scale)[1]
    for j in 2:length(t)
        m,s=_stage(c,j,t);t[j]=m+exp(s)*q[j];logjac+=s
    end
    all(isfinite,t) && isfinite(logjac) || throw(ArgumentError("nonfinite triangular map"))
    return chart_forward(c.base,t),logjac
end
chart_forward(c::TriangularChart,q)=first(_forward(c,q))
chart_logjac(c::TriangularChart,q)=last(_forward(c,q))
function chart_inverse(c::TriangularChart,z)
    t=chart_inverse(c.base,z);q=copy(t)
    for j in 2:length(t)
        m,s=_stage(c,j,t);q[j]=(t[j]-m)*exp(-s)
    end
    return q
end

# Fixed, symmetric, overdetermined quadratic design. No truth, sensor labels,
# or case-specific point placement enters the chart construction.
function _pilots(p,radius)
    points=[zeros(p)]
    for i in 1:p,scale in (-1.,-.5,.5,1.)
        x=zeros(p);x[i]=scale*radius;push!(points,x)
    end
    for j in 2:p,i in 1:j-1,a in (-1.,1.),b in (-1.,1.)
        x=zeros(p);x[i]=a*radius;x[j]=b*radius;push!(points,x)
    end
    return points
end

"Fit a chart from conditional pilot means/roots in physical selected-axis coordinates."
function fit_triangular_chart(base::AffineChart,oracle;radius=2.,max_fits=100)
    k=chart_dimension(base)
    istril(base.scale) && all(>(0),diag(base.scale)) || throw(ArgumentError("pilot construction requires a triangular base"))
    isfinite(radius) && radius>0 || throw(ArgumentError("invalid pilot radius"))
    required=sum(1+2p+2p*p for p in 1:k-1;init=0)
    required<=max_fits || throw(ArgumentError("map construction exceeds the fit budget"))
    ms=Vector{Float64}[];ss=Vector{Float64}[];records=NamedTuple[];stages=NamedTuple[]
    for j in 2:k
        p=j-1;points=_pilots(p,radius);values=Float64[];logs=Float64[]
        for t in points
            prefix=base.center[1:p]+base.scale[1:p,1:p]*t
            result=oracle(prefix)
            length(result.mean)==k && size(result.root,1)==k && all(isfinite,result.mean) &&
                all(isfinite,result.root) || throw(ArgumentError("invalid conditional pilot"))
            sd=norm(result.root[j,:])/base.scale[j,j]
            isfinite(sd) && sd>0 || throw(ArgumentError("singular conditional pilot scale"))
            m=(result.mean[j]-base.center[j]-dot(base.scale[j,1:p],t))/base.scale[j,j]
            push!(values,m);push!(logs,log(sd))
            push!(records,(stage=j,prefix=prefix,standardized=t,mean=m,logscale=log(sd)))
        end
        A=reduce(vcat,[permutedims(_features(t,radius)) for t in points])
        cm=A\values;cs=A\logs;push!(ms,cm);push!(ss,cs)
        push!(stages,(stage=j,fits=length(points),scaled_mean_error=maximum(abs.((A*cm-values)./exp.(logs))),
            logscale_error=maximum(abs.(A*cs-logs))))
    end
    return (chart=TriangularChart(base,ms,ss,radius),diagnostics=(fits=length(records),stages=stages,
        pilots=records,posterior_assumption=false,tail_validated=false))
end

end
