using Test, LinearAlgebra
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    const E=BrisketEstimator
end
include(joinpath(@__DIR__,"../scripts/curvature_audit.jl"))
include(joinpath(@__DIR__,"../scripts/adaptive_curvature.jl"))

struct AdaptiveQuadraticResidual <: E.Factor
    key::E.VarKey
end
E.vars(f::AdaptiveQuadraticResidual)=[f.key]
E.ndim(::AdaptiveQuadraticResidual)=1
E.residual(f::AdaptiveQuadraticResidual,v)=[sqrt(1+v[1][2]^2)]
E.jacobians(f::AdaptiveQuadraticResidual,v)=[reshape([0.,v[1][2]/sqrt(1+v[1][2]^2)],1,:)]

struct AdaptiveDoubleWellResidual <: E.Factor
    key::E.VarKey
end
E.vars(f::AdaptiveDoubleWellResidual)=[f.key]
E.ndim(::AdaptiveDoubleWellResidual)=1
E.residual(f::AdaptiveDoubleWellResidual,v)=[1-v[1][2]^2]
E.jacobians(f::AdaptiveDoubleWellResidual,v)=[reshape([0.,-2v[1][2]],1,:)]

@testset "observed adaptive covariance and evidence" begin
    key=E.VarKey(:params,:curvature_test,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(2));E.add_factor!(g,E.GaussianPrior(key,zeros(2),Matrix{Float64}(I,2,2)))
    E.add_factor!(g,AdaptiveQuadraticResidual(key))
    node=E._conditional_node!(g,[(key,1)],[.2]);ref=E._conditional_transport(node)
    corrected=observed_conditional_curvature(node,ref)
    @test sum(abs2,corrected.root[2,:])≈.5 atol=1e-10
    @test corrected.log_integral-node.log_integral≈-.5log(2.) atol=1e-10
    @test corrected.diagnostics.step_change<1e-10
    @test !corrected.diagnostics.process_noise_changed
    @test !corrected.diagnostics.eigenvalues_clipped
    @test corrected.diagnostics.stationarity<1e-10

    displaced=deepcopy(ref);displaced.mean[2]=.1
    fit=polish_conditional_innovations(displaced)
    @test fit.ref.mean[2]≈0. atol=1e-5
    @test fit.score<=1e-5
    @test all(diff([h.objective for h in fit.history]).<=0)
    polished=observed_conditional_curvature(node,displaced;polish=true)
    @test polished.mean[2]≈0. atol=1e-5
    @test polished.diagnostics.polished
    @test polished.log_integral-node.log_integral≈-.5log(2.) atol=1e-9

    bad=E.FactorGraph();E.add_variable!(bad,key,zeros(2))
    E.add_factor!(bad,E.GaussianPrior(key,zeros(2),Matrix{Float64}(I,2,2)))
    E.add_factor!(bad,AdaptiveDoubleWellResidual(key))
    saddle=E._conditional_node!(bad,[(key,1)],[0.])
    @test_throws ErrorException observed_conditional_curvature(saddle,E._conditional_transport(saddle))

    fully_fixed=E._conditional_node!(g,[(key,1),(key,2)],[.2,.3])
    exact=observed_conditional_curvature(fully_fixed,E._conditional_transport(fully_fixed))
    @test exact.log_integral==fully_fixed.log_integral
    @test size(exact.root)==(2,0)
end
