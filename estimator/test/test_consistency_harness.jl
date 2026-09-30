using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_posterior.jl"))

@testset "independent posterior validation bookkeeping" begin
    model=consistency_model(nx=4,suite=:core,stop=900.0)
    @test model.log.t_grid[end]>=900.0
    @test all(diff(collect(model.log.t_grid)).==120.0)
    @test_throws ArgumentError consistency_model(stop=0.0)
    # A stratified uniform rank grid and its actual central interval membership.
    records=[(completed=true,mc_precision_ok=true,ranks=[(i-.5)/64],
              joint_ranks=fill((i-.5)/64,3),covered=[.025<=(i-.5)/64<=.975]) for i in 1:64]
    @test posterior_control_summary(records,64).passed
    concentrated=[merge(r,(ranks=[.5],)) for r in records]
    @test !posterior_control_summary(concentrated,64).passed
    imprecise=copy(records);imprecise[1]=merge(imprecise[1],(mc_precision_ok=false,))
    @test !posterior_control_summary(imprecise,64).passed
    @test !posterior_control_summary(records[1:8],8).statistically_eligible
    @test !posterior_control_summary(records[1:63],64).passed
    with_failure=Any[records[1:63]...,(completed=false,mc_precision_ok=false,error="failed cook")]
    @test !posterior_control_summary(with_failure,64).passed
    @test !posterior_control_summary(NamedTuple[],64).passed
end
