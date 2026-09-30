using Test,LinearAlgebra,SpecialFunctions
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    const E=BrisketEstimator
end
isdefined(@__MODULE__,:MappedConditionalCandidate) || include(joinpath(@__DIR__,"../scripts/adaptive_conditional.jl"))
include(joinpath(@__DIR__,"../scripts/fit_conditional_chart.jl"))
include(joinpath(@__DIR__,"../scripts/mapped_diagnostics.jl"))

struct ChartBananaResidual <: E.Factor
    key::E.VarKey
end
E.vars(f::ChartBananaResidual)=[f.key]
E.ndim(::ChartBananaResidual)=2
E.residual(f::ChartBananaResidual,v)=[v[1][1],(v[1][2]-v[1][1]^2)/(.3exp(.2v[1][1]))]
E.jacobians(f::ChartBananaResidual,v)=[E.ForwardDiff.jacobian(z->E.residual(f,[z]),v[1])]
E.loss(f::ChartBananaResidual,v)=sum(abs2,E.residual(f,v))/2+log(.3)+.2v[1][1]

struct ChartProbeResidual <: E.Factor
    key::E.VarKey
    value::Float64
    sigma::Float64
end
E.vars(f::ChartProbeResidual)=[f.key]
E.ndim(::ChartProbeResidual)=1
E.residual(f::ChartProbeResidual,v)=[(v[1][1]-f.value)/f.sigma]
E.jacobians(f::ChartProbeResidual,v)=[reshape([1/f.sigma,0.],1,2)]

@testset "mapped graph target and budget" begin
    key=E.VarKey(:params,:chart_test,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(2));E.add_factor!(g,ChartBananaResidual(key));saved=deepcopy(g)
    axes=[(key,1),(key,2)];base=AffineChart(zeros(2),[1. 0.;0. .3])
    fit=fit_conditional_chart(g;axes,center=base.center,scale=base.scale,max_fits=5)
    @test fit.diagnostics.fits==5
    @test fit.diagnostics.optimizer_attempts==5
    q=[.371,-.829];point=chart_forward(fit.chart,q)
    @test point≈[q[1],q[1]^2+.3exp(.2q[1])*q[2]] atol=1e-8
    candidate=adaptive_conditional_fit(g;axes,chart=fit.chart,lower=[-5.,-5.],upper=[5.,5.],
        tolerance=.001,max_nodes=729-fit.diagnostics.fits,orders=(7,11),
        curvature=polished_gn_conditional_curvature,shape_key=key)
    @test candidate isa MappedConditionalCandidate
    @test candidate.density.diagnostics.resolved
    @test candidate.diagnostics.shape_pass
    @test candidate.diagnostics.graph_solves+fit.diagnostics.fits<=729
    @test adaptive_conditional_logpdf(candidate,point)≈-E.loss(first(g.factors),[point])-log(2pi) atol=3e-6
    @test adaptive_conditional_logpdf(candidate,chart_forward(fit.chart,[6.,0.]))==-Inf
    @test !candidate.diagnostics.committed && !candidate.diagnostics.consistent
    @test E._same_decoder_value(g,saved)
    @test_throws ArgumentError adaptive_conditional_fit(g;axes,chart=fit.chart,center=base.center,
        scale=base.scale,lower=[-5.,-5.],upper=[5.,5.])
    phi(x)=.5erfc(-x/sqrt(2));mass5=erf(5/sqrt(2))
    @test mapped_axis_cdf(candidate,1,.37;order=17)≈(phi(.37)-phi(-5.))/mass5 atol=1e-6
    # A small scale around a large center must not be computed by subtraction
    # of two nearly identical mapped values when forming the CDF threshold.
    narrowchart=AffineChart([1e4,0.],[1e-9 0.;0. 1.])
    narrow=MappedConditionalCandidate(candidate.density,candidate.decoder,candidate.axes,candidate.axisrows,
        candidate.free,narrowchart,candidate.diagnostics)
    threshold=1e4+.371e-9;standardized=(threshold-1e4)/1e-9
    @test mapped_axis_cdf(narrow,1,threshold;order=33)≈(phi(standardized)-phi(-5.))/mass5 atol=1e-6
    xs,ws=AdaptiveDensities._legendre(201)
    reference=sum(10w*exp(-x^2/2)/sqrt(2pi)/mass5*
        clamp((phi(-x^2/(.3exp(.2x)))-phi(-5.))/mass5,0.,1.) for (x,w) in zip(10xs.-5,ws))
    query=mapped_axis_cdf_report(candidate,2,0.)
    @test query.resolved
    @test query.value≈reference atol=2e-5
    @test_throws ArgumentError mapped_axis_cdf(candidate,1,NaN)
    @test_throws ArgumentError mapped_axis_cdf(candidate,1,0.;order=0)
    @test_throws ArgumentError mapped_axis_cdf_report(candidate,1,0.;tolerance=Inf)
    limited=mapped_axis_cdf_report(candidate,2,0.;max_evaluations=1)
    @test !limited.resolved && limited.reason==:evaluation_budget && limited.value===nothing && limited.evaluations==0
    @test_throws ErrorException mapped_axis_cdf(candidate,1,0.;checkpoint=()->error("query deadline"))
    @test mapped_density_distance(candidate,candidate).tv_bound_estimate<1e-10
    # Same physical density and curved support in a shifted reference chart.
    # Comparing q coordinates directly would incorrectly report a difference.
    a=.7;shifted=TriangularChart(AffineChart([a,0.],Matrix{Float64}(I,2,2)),
        [[a^2,4a,4.]],[[log(.3)+.2a,.4,0.]],2.)
    sd=adaptive_density(q->(logmass=-((q[1]+a)^2+q[2]^2)/2,mean=Float64[],root=zeros(0,0)),
        [-5.0-a,-5.],[5.0-a,5.];tolerance=.001,max_nodes=724,orders=(7,11))
    equivalent=MappedConditionalCandidate(sd,candidate.decoder,candidate.axes,candidate.axisrows,candidate.free,shifted,candidate.diagnostics)
    @test mapped_density_distance(candidate,equivalent).tv_bound_estimate<1e-8

    # Reuse the same nonlinear coordinates through two target updates. Every
    # update includes original factors once, never the previous posterior as data.
    updated=deepcopy(g);precision=1.;information=0.;previous=candidate
    for (value,sigma) in ((.7,.5),(-.2,.8))
        before=deepcopy(previous.decoder.graph)
        E.add_factor!(updated,ChartProbeResidual(key,value,sigma))
        precision+=inv(sigma^2);information+=value/sigma^2
        current=adaptive_conditional_fit(updated;axes,chart=fit.chart,lower=[-5.,-5.],upper=[5.,5.],
            tolerance=.001,max_nodes=724,orders=(7,11),
            curvature=polished_gn_conditional_curvature,shape_key=key)
        mu=information/precision;sd=inv(sqrt(precision))
        massx=phi((5-mu)/sd)-phi((-5-mu)/sd)
        logx=-(point[1]-mu)^2/(2sd^2)-log(sd)-log(2pi)/2-log(massx)
        logy=-(point[2]-point[1]^2)^2/(2(.3exp(.2point[1]))^2)-log(.3)-.2point[1]-log(2pi)/2-log(mass5)
        @test current.density.diagnostics.resolved
        @test adaptive_conditional_logpdf(current,point)≈logx+logy atol=2e-6
        @test mapped_axis_cdf(current,1,mu;order=33)≈.5 atol=2e-6
        @test E._same_decoder_value(before,previous.decoder.graph)
        previous=current
    end
end
