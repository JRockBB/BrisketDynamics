using Test, LinearAlgebra
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    const E=BrisketEstimator
end
include(joinpath(@__DIR__,"../scripts/adaptive_conditional.jl"))
include(joinpath(@__DIR__,"../scripts/curvature_audit.jl"))
include(joinpath(@__DIR__,"../scripts/adaptive_curvature.jl"))

@testset "adaptive full-target conditional adapter" begin
    key=E.VarKey(:params,:adaptive_test,0);g=E.FactorGraph()
    E.add_variable!(g,key,[.4,-.3])
    covariance=[1. .7;.7 1.];mean=[.4,-.3]
    E.add_factor!(g,E.GaussianPrior(key,mean,Symmetric(covariance)))
    original=deepcopy(g)
    p=adaptive_conditional_fit(g;axes=[(key,1)],center=[.4],scale=ones(1,1),
        lower=[-7.],upper=[7.],tolerance=1e-6,max_nodes=65,orders=(9,15),shape_key=key)
    @test p.density.diagnostics.resolved
    @test p.diagnostics.shape_pass
    @test p.diagnostics.shape_defect<1e-8
    @test !p.diagnostics.committed && !p.diagnostics.consistent
    @test p.diagnostics.curvature==:gauss_newton
    x=[.17,.39];delta=x-mean
    exact=-dot(delta,covariance\delta)/2-logdet(covariance)/2-log(2pi)
    @test adaptive_conditional_logpdf(p,x)≈exact atol=1e-8
    @test adaptive_geometry(p.density,[.25]).mean≈[-.3+.7*.25]
    @test adaptive_geometry(p.density,[.25]).root[1,1]≈sqrt(.51)
    @test g.values==original.values
    @test E.objective(g)==E.objective(original)
    @test_throws ArgumentError adaptive_conditional_fit(g;axes=[(key,1),(key,1)],center=[0.,0.],
        scale=Matrix{Float64}(I,2,2),lower=[-1.,-1.],upper=[1.,1.])

    # Test the correction contract: covariance and evidence change together.
    correction(node,ref)=(root=ref.root/2,log_integral=node.log_integral-log(2.),diagnostics=(method=:test,))
    pc=adaptive_conditional_fit(g;axes=[(key,1)],center=[.4],scale=ones(1,1),
        lower=[-7.],upper=[7.],tolerance=1e-6,max_nodes=65,orders=(9,15),curvature=correction)
    @test pc.diagnostics.curvature==:corrected
    @test !pc.diagnostics.shape_complete && !pc.diagnostics.shape_pass
    @test adaptive_geometry(pc.density,[.25]).root[1,1]≈sqrt(.51)/2
    @test pc.density.log_normalizer≈p.density.log_normalizer-log(2.) atol=1e-8

    # A curvature hook must not silently change the meaning of coordinates.
    function invalid_decoder(node,ref)
        bad=deepcopy(ref);bad.offsets[key]=2:3
        return (root=ref.root,log_integral=node.log_integral,decoder=bad,diagnostics=(method=:test,))
    end
    @test_throws ErrorException adaptive_conditional_fit(g;axes=[(key,1)],center=[.4],scale=ones(1,1),
        lower=[-2.],upper=[2.],curvature=invalid_decoder)

    # A deliberately unfinished first optimizer must not be silently accepted.
    # The independent score solve can finish it and recover the exact posterior,
    # including the mean (not merely a new covariance at the unfinished point).
    unfinished=deepcopy(g);unfinished.values[key][2]=2.
    pp=adaptive_conditional_fit(unfinished;axes=[(key,1)],center=[.4],scale=ones(1,1),
        lower=[-7.],upper=[7.],tolerance=1e-6,max_nodes=65,orders=(9,15),iters=0,
        curvature=polished_gn_conditional_curvature,shape_key=key)
    @test any(!n.initial_solve_converged for n in pp.diagnostics.nodes)
    @test pp.diagnostics.shape_pass
    @test adaptive_conditional_logpdf(pp,x)≈exact atol=1e-8
    @test maximum(n.curvature.score for n in pp.diagnostics.nodes)<=1e-5
end
