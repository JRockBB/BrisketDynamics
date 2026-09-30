if !isdefined(@__MODULE__,:E)
    include(joinpath(@__DIR__,"../src/BrisketEstimator.jl"))
    using .BrisketEstimator
    const E=BrisketEstimator
end
using Test
include(joinpath(@__DIR__,"../scripts/consistency_direct.jl"))

@testset "marginal checks do not certify an estimator" begin
    records=[(completed=true,ranks=[(i-.5)/100],rank_errors=[0.],
        nees_z=nothing,nees_y=nothing,gaussian_nees_y=1.) for i in 1:100]
    r=direct_consistency_summary(records,100)
    @test r.eligible
    @test r.marginal_checks_passed
    @test !r.passed
    @test r.mean_nees_z===nothing
    @test r.mean_nees_y===nothing
    @test !isempty(r.qualification_missing)
    failed=vcat(records,[(completed=false,)])
    r=direct_consistency_summary(failed,101)
    @test !r.eligible
    @test !r.passed
    @test r.completed==100
    @test !direct_consistency_summary(NamedTuple[],1).passed
    wide=[merge(r,(rank_errors=[.2],)) for r in records]
    @test !direct_consistency_summary(wide,100).marginal_checks_passed
    data=_direct_json_value((a=NaN,b=[Inf,1.],c=(d=-Inf,),e=Dict("x"=>NaN)))
    @test data.a===nothing
    @test data.b==[nothing,1.]
    @test data.c.d===nothing
    @test data.e["x"]===nothing
    @test !occursin("NaN",E.JSON.json(data))
end