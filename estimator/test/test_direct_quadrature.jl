using Test, LinearAlgebra
include(joinpath(@__DIR__,"../src/direct_quadrature.jl"))
using .DirectQuadrature
using SpecialFunctions: erfc

@testset "direct Gaussian integration" begin
    for level in 0:4
        x,w=normal_rule(level)
        @test sum(w)≈1 atol=2e-15
        @test all(>(0),w)
        @test sum(x.*w)≈0 atol=2e-15
        level==0 || @test sum(x.^2 .* w)≈1 atol=1e-13
    end
    # Exact full joint moments including a nonlinear generated state. No local
    # covariance approximation can reproduce this banana's extra variance.
    f(x)=begin
        a=x[1];b=x[2]+2*(a*a-1)
        [1.,a,b,a*a,a*b,b*b]
    end
    r=integrate_normal(f,4;atol=1e-10,rtol=1e-10,max_evaluations=10000)
    @test r.diagnostics.resolved
    @test r.diagnostics.pair_shell_checked
    @test r.value≈[1.,0.,0.,1.,0.,9.] atol=1e-10
    @test !r.diagnostics.gaussian_remainder
    @test !r.diagnostics.monte_carlo
end

@testset "piecewise integration of nonsmooth likelihood structure" begin
    # A C1 hinge has a known Gaussian integral and exercises the same regularity
    # class as piecewise probe interpolation without using any cooking case.
    c=.137;exact=(1+c*c)*erfc(c/sqrt(2))/2-c*exp(-c*c/2)/sqrt(2pi)
    r=integrate_normal(x->[max(x[1]-c,0.)^2],1;atol=1e-7,rtol=1e-7,
        max_level=8,max_evaluations=10000,rule=:composite)
    @test r.diagnostics.resolved
    @test only(r.value)≈exact atol=2e-7
    for level in 2:4
        x,w=DirectQuadrature.composite_normal_rule(level)
        @test sum(w)≈1 atol=1e-14
        @test all(>(0),w)
        @test sum(w.*x)≈0 atol=1e-13
        @test sum(w.*x.^2)≈1 atol=1e-6
        @test minimum(x)<-8 && maximum(x)>8
    end
end

@testset "direct non-Gaussian density correction" begin
    # A normalized, asymmetric two-mode target relative to standard normal.
    # Its first two moments are known independently of the quadrature.
    function mixture(x)
        w=.3exp(-2x[1]-2)+.7exp(2x[1]-2)
        w.*[1.,x[1],x[1]^2,x[2],x[1]*x[2],x[2]^2]
    end
    r=integrate_normal(mixture,3;atol=2e-7,rtol=2e-7,max_evaluations=20000,max_level=6)
    @test r.diagnostics.resolved
    @test r.value≈[1.,.8,5.,0.,0.,1.] atol=2e-6
    # All one-dimensional evaluations of this correction are exactly constant.
    # The complete pair shell must find the missing interaction and normalizer.
    r=integrate_normal(x->[1+x[1]^2*x[2]^2],5;atol=1e-10,rtol=1e-10)
    @test r.diagnostics.resolved
    @test only(r.value)≈2 atol=1e-10
    # Completely hidden from every axial AND pair panel. A separate rotation
    # of the integration measure must prevent a false convergence declaration.
    hidden=integrate_normal(x->[1+prod(x.^2)],3;atol=1e-9,rtol=1e-9,max_evaluations=10000)
    @test hidden.diagnostics.resolved
    @test hidden.diagnostics.rotation_checked
    @test only(hidden.value)≈2 atol=1e-8
end

@testset "direct integration budgets and failures" begin
    calls=Ref(0)
    f(x)=(calls[]+=1;[exp(.2sum(x))])
    r=integrate_normal(f,5;max_evaluations=12)
    @test !r.diagnostics.resolved
    @test !r.diagnostics.complete_frontier
    @test r.diagnostics.reason==:evaluation_budget
    @test calls[]==r.diagnostics.evaluations<=12
    @test all(isinf,r.error)
    @test_throws ErrorException integrate_normal(x->[x[1]==0 ? 1. : NaN],2)
    @test_throws ArgumentError integrate_normal(identity,0)
    @test_throws ArgumentError integrate_normal(identity,2;max_level=10)
    # A parallel batch callback must have identical numerical semantics.
    f2(x)=[1.,sum(x.^2),prod(1 .+ x.^2)]
    a=integrate_normal(f2,3;atol=1e-10,rtol=1e-10)
    b=integrate_normal(f2,3;atol=1e-10,rtol=1e-10,evaluate_batch=qs->[f2(collect(q)) for q in qs])
    @test a.value==b.value
    @test a.indices==b.indices
    @test a.diagnostics.evaluations==b.diagnostics.evaluations
end
