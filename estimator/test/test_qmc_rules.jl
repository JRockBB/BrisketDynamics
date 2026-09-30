push!(LOAD_PATH,joinpath(@__DIR__,"../qmc"))
using Test, Random, Sobol, Statistics
include(joinpath(@__DIR__,"../src/qmc_rules.jl"))
using .QMCNormalRules
@testset "Sobol net and scramble contracts" begin
    for d in (1,3,17)
        a=uniform_net(256,d;scramble=false)
        s=Sobol.SobolSeq(d)
        @test a[:,1]==zeros(d)
        @test a[:,2:end]==hcat([Sobol.next!(s) for _ in 2:256]...)
        b=uniform_net(256,d;rng=MersenneTwister(11))
        @test all(x->0<x<1,b)
        @test b==uniform_net(256,d;rng=MersenneTwister(11))
        @test b!=uniform_net(256,d;rng=MersenneTwister(12))
        @test b[:,1:128]==uniform_net(128,d;rng=MersenneTwister(11))
        for i in 1:d;@test sort(floor.(Int,256b[i,:]))==collect(0:255);end
    end
    @test_throws ArgumentError uniform_net(123,2)
    @test_throws ArgumentError normal_net(16,2;scramble=false)
    @test_throws ArgumentError uniform_net(16,0)
end
@testset "independent scrambled normal quadrature" begin
    estimates=Vector{Float64}[]
    for seed in 1:16
        x=normal_net(16384,4;rng=MersenneTwister(seed))
        # Includes a higher-order interaction hidden from axial cubature.
        push!(estimates,[mean(x[1,:]),mean(x[1,:].^2),mean(exp.(x[2,:])),mean(prod(x.^2;dims=1))])
    end
    @test vec(mean(reduce(hcat,estimates);dims=2))≈[0.,1.,exp(.5),1.] atol=.02
    @test all(isfinite,reduce(hcat,estimates))
end