using Test,LinearAlgebra
if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    const E=BrisketEstimator
end
isdefined(@__MODULE__,:adaptive_conditional_fit) || include(joinpath(@__DIR__,"../scripts/adaptive_conditional.jl"))
isdefined(@__MODULE__,:fit_conditional_chart) || include(joinpath(@__DIR__,"../scripts/fit_conditional_chart.jl"))

struct TestTransportStop <: Exception end
@testset "transport budget abort and ownership" begin
    calls=Ref(0)
    oracle(q)=(calls[]+=1;(logmass=-sum(abs2,q)/2,mean=Float64[],root=zeros(0,0)))
    @test_throws TestTransportStop adaptive_density(oracle,[-4.],[4.];checkpoint=()->throw(TestTransportStop()))
    @test calls[]==0
    key=E.VarKey(:params,:abort_test,0);g=E.FactorGraph()
    E.add_variable!(g,key,zeros(2))
    E.add_factor!(g,E.GaussianPrior(key,zeros(2),Matrix{Float64}(I,2,2)))
    saved=deepcopy(g);axes=[(key,1),(key,2)]
    failure=try
        fit_conditional_chart(g;axes,center=zeros(2),scale=Matrix{Float64}(I,2,2),
            checkpoint=()->throw(TestTransportStop()))
        nothing
    catch err
        err
    end
    @test failure isa ChartFitFailure
    @test failure.cause isa TestTransportStop
    @test failure.completed==0
    @test failure.optimizer_attempts==0
    @test E._same_decoder_value(saved,g)
    @test_throws TestTransportStop adaptive_conditional_fit(g;axes,
        chart=AffineChart(zeros(2),Matrix{Float64}(I,2,2)),lower=[-4.,-4.],upper=[4.,4.],
        checkpoint=()->throw(TestTransportStop()))
    @test E._same_decoder_value(saved,g)
end
