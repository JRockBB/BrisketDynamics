using Test,LinearAlgebra,SpecialFunctions
isdefined(@__MODULE__,:ConditionalCharts) || include(joinpath(@__DIR__,"../src/conditional_charts.jl"))
isdefined(@__MODULE__,:AdaptiveDensities) || include(joinpath(@__DIR__,"../src/adaptive_density.jl"))
using .ConditionalCharts,.AdaptiveDensities

@testset "triangular coordinate map" begin
    base=AffineChart([.3,-.2],[1.2 0.;.4 .7])
    manual=TriangularChart(base,[[.1,.3,.7]],[[.2,-.4,.1]],2.)
    q=[.371,-.829];z=chart_forward(manual,q)
    @test chart_inverse(manual,z)≈q atol=1e-12
    h=1e-5;eye2=Matrix{Float64}(I,2,2)
    J=hcat([(chart_forward(manual,q+h*eye2[:,i])-chart_forward(manual,q-h*eye2[:,i]))/(2h) for i in 1:2]...)
    @test logabsdet(J)[1]≈chart_logjac(manual,q) atol=1e-8
    @test chart_forward(manual,[0,1])≈chart_forward(manual,[0.,1.])
    for badwidth in (-1000.,-745.,1000.)
        bad=TriangularChart(AffineChart(zeros(2),Matrix{Float64}(I,2,2)),
            [[0.,0.,0.]],[[badwidth,0.,0.]],2.)
        @test_throws ArgumentError chart_forward(bad,[0.,1.])
        @test_throws ArgumentError chart_inverse(bad,[0.,1.])
    end
    @test_throws ArgumentError AffineChart(zeros(2),zeros(2,2))
    @test_throws ArgumentError TriangularChart(AffineChart(zeros(2),[1. 1.;0. 1.]),[[0.,0.,0.]],[[0.,0.,0.]],2.)
    @test_throws ArgumentError fit_triangular_chart(base,p->error("budget must be checked first");max_fits=4)

    # Learn a curved mean and a nonconstant conditional width. The production
    # learner receives no density samples or truth, just conditional pilots.
    eye=AffineChart(zeros(2),Matrix{Float64}(I,2,2))
    pilot(p)=(mean=[p[1],p[1]^2],root=reshape([0.,.3exp(.2p[1])],2,1))
    fit=fit_triangular_chart(eye,pilot)
    @test fit.diagnostics.fits==5
    @test maximum(s.scaled_mean_error for s in fit.diagnostics.stages)<1e-12
    @test chart_forward(fit.chart,q)≈[q[1],q[1]^2+.3exp(.2q[1])*q[2]] atol=1e-12
    @test chart_logjac(fit.chart,q)≈log(.3)+.2q[1] atol=1e-12
    @test chart_inverse(fit.chart,chart_forward(fit.chart,q))≈q atol=1e-12
    # In 3D, later stages condition on already transformed coordinates.
    c3=TriangularChart(AffineChart(zeros(3),Matrix{Float64}(I,3,3)),
        [[0.,0.,1.],[.2,.3,-.4,.1,.5,.2]],[[0.,.1,0.],[0.,-.1,.2,0.,0.,0.]],1.)
    q3=[.4,-.8,.9]
    @test chart_inverse(c3,chart_forward(c3,q3))≈q3 atol=1e-12
    J3=hcat([(chart_forward(c3,q3+h*Matrix{Float64}(I,3,3)[:,i])-chart_forward(c3,q3-h*Matrix{Float64}(I,3,3)[:,i]))/(2h) for i in 1:3]...)
    @test logabsdet(J3)[1]≈chart_logjac(c3,q3) atol=1e-8

    logtarget(z)=-z[1]^2/2-((z[2]-z[1]^2)/(.3exp(.2z[1])))^2/2-log(.3)-.2z[1]
    oracle(q)=(logmass=logtarget(chart_forward(fit.chart,q))+chart_logjac(fit.chart,q),mean=Float64[],root=zeros(0,0))
    density=adaptive_density(oracle,[-5.,-5.],[5.,5.];tolerance=.001,max_nodes=724,orders=(7,11))
    @test density.diagnostics.resolved
    @test density.log_normalizer≈log(2pi) atol=3e-6
    @test adaptive_expectation((q,n)->chart_forward(fit.chart,q)[1],density)≈0. atol=1e-8
    @test adaptive_expectation((q,n)->chart_forward(fit.chart,q)[2],density)≈1. atol=2e-5
    # Exact truncated-normal fourth minus second moment. At radius five the
    # missing fourth-moment tail is 4e-4 despite only 6e-7 missing probability.
    mass5=erf(5/sqrt(2))
    truncated_cross=2-2*5*(5^2+2)*exp(-5^2/2)/sqrt(2pi)/mass5
    @test adaptive_expectation((q,n)->q[1]^2*(chart_forward(fit.chart,q)[2]-1),density)≈truncated_cross atol=1e-6
    @test adaptive_marginal_logpdf(density,q)-chart_logjac(fit.chart,q)≈logtarget(chart_forward(fit.chart,q))-density.log_normalizer atol=1e-10

    # The chart is only a coordinate change. Two remaining modes must survive;
    # their weights cannot be collapsed into the Gaussian pilot geometry.
    logadd(a,b)=max(a,b)+log1p(exp(-abs(a-b)))
    bimodal(z)=-z[1]^2/2+logadd(-((z[2]-z[1]^2-1)/.4)^2/2,-((z[2]-z[1]^2+1)/.4)^2/2)
    bimodalfit=fit_triangular_chart(eye,p->(mean=[p[1],p[1]^2],root=reshape([0.,1.],2,1)))
    bm(q)=(logmass=bimodal(chart_forward(bimodalfit.chart,q))+chart_logjac(bimodalfit.chart,q),mean=Float64[],root=zeros(0,0))
    b=adaptive_density(bm,[-5.,-5.],[5.,5.];tolerance=.005,max_nodes=724,orders=(7,11))
    @test b.diagnostics.resolved
    @test b.log_normalizer≈log(2*.4*2pi) atol=.002
    @test adaptive_cdf(b,2,0.)≈.5 atol=.002
    @test adaptive_expectation((q,n)->q[2]^2,b)≈1.16 atol=.005

    # A local mode/curvature pilot is legitimate as a map, but its finite box
    # can exclude a second mode. Exact Jacobians do not fix missing support.
    # This is why the cook driver reports domain_validated=false separately.
    localmode=fit_triangular_chart(eye,p->(mean=[p[1],p[1]^2+1],root=reshape([0.,.4],2,1))).chart
    localdensity(q)=(logmass=bimodal(chart_forward(localmode,[0.,q[1]]))+chart_logjac(localmode,[0.,q[1]]),mean=Float64[],root=zeros(0,0))
    narrow=adaptive_density(localdensity,[-4.],[4.];tolerance=.005,max_nodes=257,orders=(7,11))
    wide=adaptive_density(localdensity,[-10.],[10.];tolerance=.005,max_nodes=257,orders=(7,11))
    @test narrow.diagnostics.resolved
    @test wide.diagnostics.resolved
    @test exp(narrow.log_normalizer-log(2*.4*sqrt(2pi)))<.6
    @test exp(wide.log_normalizer-log(2*.4*sqrt(2pi)))>.999
    @test adaptive_cdf(wide,1,-2.5)≈.5 atol=.002
end
