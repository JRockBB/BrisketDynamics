# Pure statistical oracles: these do not load the cooking model or reuse its inference calculations.
using LinearAlgebra, Random, Statistics, Test
include(joinpath(@__DIR__,"..","scripts","posterior_comparison.jl"))
oracle_quantiles(p)=permutedims(hcat([quantile(vec(p.draws[i,:]),[.025,.5,.975]) for i in axes(p.draws,1)]...))
function comparison_oracle(X)
    n=size(X,2);C=cov(X;dims=2)
    return (draws=X,weights=fill(1/n,n),mean=vec(mean(X;dims=2)),covariance=C,
        diagnostics=(replicates=8,mean_mcse=sqrt.(diag(C)./n)))
end
@testset "online comparison detects mean and dependence errors" begin
    X=randn(MersenneTwister(70102),2,8192);a=comparison_oracle(X)
    compare(a,b)=compare_sampled_posteriors(a,b;quantiles=oracle_quantiles)
    @test compare(a,a).passed
    shifted=copy(X);shifted[1,:].+=.3
    @test !compare(a,comparison_oracle(shifted)).passed
    dependent=copy(X);dependent[2,:]=.8X[1,:]+.6X[2,:]
    result=compare(a,comparison_oracle(dependent))
    @test result.max_mean_mc_z<6
    @test result.max_cdf_mc_z<6
    @test result.max_joint_moment_mc_z>6
    @test !result.passed
end
