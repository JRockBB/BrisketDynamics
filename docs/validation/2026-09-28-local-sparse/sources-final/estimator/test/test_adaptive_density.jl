using Test, LinearAlgebra
include(joinpath(@__DIR__,"../src/adaptive_density.jl"))
using .AdaptiveDensities

@testset "adaptive continuous density" begin
    # A rotated Gaussian with a strongly correlated conditional remainder is
    # represented exactly, including off-grid values and its small contrast.
    H=[2. .6;.6 1.];mu=[.3,-.4];L=[2. 0.;1.999 1e-4]
    oracle(q)=(logmass=-dot(q-mu,H*(q-mu))/2,mean=[q[1]+2q[2],q[1]-q[2]],root=L)
    p=adaptive_density(oracle,[-8.,-8.],[8.,8.];tolerance=1e-5,max_nodes=729,orders=(7,11))
    @test p.diagnostics.resolved
    @test !p.diagnostics.target_validated
    @test !p.diagnostics.tail_validated
    @test p.log_normalizer≈log(2pi)-logdet(H)/2 atol=2e-6
    @test adaptive_expectation((q,n)->q,p)≈mu atol=1e-6
    q=[.179,-.631];v=adaptive_geometry(p,q)
    @test v.mean≈oracle(q).mean atol=1e-12
    @test v.root≈L atol=1e-12
    @test adaptive_marginal_logpdf(p,q)≈oracle(q).logmass-p.log_normalizer atol=1e-12
    @test adaptive_logpdf(p,q,v.mean)≈oracle(q).logmass-p.log_normalizer-sum(log,diag(L))-log(2pi) atol=1e-12
    @test adaptive_marginal_logpdf(p,[20.,0.]) == -Inf
    @test adaptive_cdf(p,1,-20.)==0.
    @test adaptive_cdf(p,1,20.)≈1. atol=1e-12
    @test adaptive_cdf(p,1,mu[1])≈.5 atol=1e-6

    # A symmetric pair of modes defeats a single Gaussian, but is integrated
    # without random draws, atoms, or fitting a Gaussian over the two modes.
    logadd(a,b)=max(a,b)+log1p(exp(-abs(a-b)))
    bimodal(q)=(logmass=logadd(-.5((q[1]-2)/.4)^2,-.5((q[1]+2)/.4)^2),mean=Float64[],root=zeros(0,0))
    b=adaptive_density(bimodal,[-5.],[5.];tolerance=.0001,max_nodes=257,orders=(7,11))
    @test b.diagnostics.resolved
    @test b.log_normalizer≈log(2*.4*sqrt(2pi)) atol=5e-4
    @test adaptive_expectation((q,n)->q[1]^2,b)≈4.16 atol=.002
    @test adaptive_cdf(b,1,0.)≈.5 atol=5e-4
    @test adaptive_marginal_logpdf(b,[2.])>adaptive_marginal_logpdf(b,[0.])+10
    @test adaptive_density_distance(b,b).tv_bound_estimate<1e-12

    # Two quadrature orders can both miss one narrow mode. Refinement must
    # inspect a bound on the entire interpolant, and log-space errors must
    # rank remote mass without Inf*0 or Inf-Inf producing NaN.
    narrow(q)=(logmass=logadd(-.5((q[1]-2)/.001)^2,-.5((q[1]+2)/.001)^2),
               mean=Float64[],root=zeros(0,0))
    sharp=adaptive_density(narrow,[-5.],[5.];tolerance=.0001,max_nodes=257,orders=(7,11))
    @test sharp.diagnostics.resolved
    @test sharp.log_normalizer≈log(.002sqrt(2pi)) atol=1e-7
    @test adaptive_cdf(sharp,1,0.)≈.5 atol=1e-7
    @test all(h->!isnan(h.indicator) && !isnan(h.integration_error),sharp.diagnostics.history)
    @test all(h->isfinite(h.log_indicator) && isfinite(h.log_integration_error),sharp.diagnostics.history)
    shifted_narrow(q)=merge(narrow(q),(logmass=narrow(q).logmass-1e6,))
    sharp_shift=adaptive_density(shifted_narrow,[-5.],[5.];tolerance=.0001,max_nodes=257,orders=(7,11))
    @test sharp_shift.diagnostics.resolved
    @test sharp_shift.log_normalizer+1e6≈sharp.log_normalizer atol=1e-7
    @test adaptive_cdf(sharp_shift,1,0.)≈.5 atol=1e-7

    # Marginal evidence and means do not change at all. Only nonlinear
    # covariance/correlation in the remainder can trigger this refinement.
    remainder(q)=(logmass=-q[1]^2/2,mean=[0.,0.],
        root=[1. 0.;.7sin(2q[1]) exp(.3cos(q[1]))])
    r=adaptive_density(remainder,[-5.],[5.];tolerance=.005,max_nodes=257,orders=(5,9))
    @test r.diagnostics.resolved
    @test r.diagnostics.cells>1
    @test first(r.diagnostics.history).indicator>.1
    @test adaptive_geometry(r,[.37]).root≈remainder([.37]).root atol=.004

    # A curved conditional mean retains a nonzero third cross moment which a
    # joint Gaussian would lose, even when it matches the mean and covariance.
    banana(q)=(logmass=-q[1]^2/2,mean=[q[1]^2],root=reshape([.2],1,1))
    c=adaptive_density(banana,[-7.],[7.];tolerance=1e-5,max_nodes=257,orders=(7,11))
    @test c.diagnostics.resolved
    @test adaptive_expectation((q,n)->q[1]^2*(n.mean[1]-1),c)≈2. atol=2e-6

    calls=Ref(0)
    counted(q)=(calls[]+=1;bimodal(q))
    limited=adaptive_density(counted,[-5.],[5.];tolerance=1e-12,max_nodes=7)
    @test !limited.diagnostics.resolved
    @test limited.diagnostics.reason==:node_budget
    @test calls[]==limited.diagnostics.nodes<=7
    @test_throws ErrorException adaptive_density(q->error("failed conditional solve"),[-1.],[1.])
    @test_throws ArgumentError adaptive_density(q->(logmass=NaN,mean=[0.],root=ones(1,1)),[-1.],[1.])
    @test_throws ArgumentError adaptive_density(bimodal,[-Inf],[1.])
    @test_throws ArgumentError adaptive_density(bimodal,[-1.],[1.];orders=(9,5))
    @test_throws ArgumentError adaptive_density(bimodal,[-1.],[1.];tolerance=Inf)

    uniform(q)=(logmass=0.,mean=Float64[],root=zeros(0,0))
    small=adaptive_density(uniform,[-1.,-1.],[1.,1.])
    large=adaptive_density(uniform,[-2.,-2.],[2.,2.])
    far=adaptive_density(uniform,[4.,4.],[6.,6.])
    shifted=adaptive_density(uniform,[0.,-1.],[2.,1.])
    @test adaptive_density_distance(small,large).tv_bound_estimate≈.75 atol=1e-12
    @test adaptive_density_distance(small,far).tv_bound_estimate≈1. atol=1e-12
    @test adaptive_density_distance(small,shifted).tv_bound_estimate≈.5 atol=1e-12
    @test adaptive_density_distance(large,small).tv_bound_estimate≈.75 atol=1e-12
    @test adaptive_density_distance(small,large).integration_change<1e-12
end
