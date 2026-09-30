using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "sampled estimator transactions" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.0)
    @test_throws ArgumentError sampled_init(model;lag=0)
    @test_throws ArgumentError sampled_init(model;als=nothing)
    @test_throws ArgumentError sampled_init(model;particles=32,max_particles=16)
    sm=sampled_init(model;mode=:offline,particles=16,max_particles=16,rng=MersenneTwister(818))
    @test sm.replicates==8
    inner=sm.graph_backend;before=deepcopy(inner.graph.values);factors=copy(inner.graph.factors)
    posterior=sm.posterior;nodes=copy(window_nodes(sm));records=inner.record_count
    reading=belief_at(sm).T_pit[1]+2.0
    result=assimilate!(sm,MeasurementRecord(:pit,0.0,reading))
    @test !result.committed
    @test result.reason===:monte_carlo_precision
    @test sm.posterior===posterior
    @test inner.graph.values==before
    @test inner.graph.factors==factors
    @test window_nodes(sm)==nodes
    @test inner.record_count==records
    @test isempty(inner.obs)
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord[])
    @test_throws ArgumentError assimilate!(sm,MeasurementRecord(:pit,NaN,reading))
    @test inner.graph.values==before

    capped=SensorSpec(:pit,PitTemperature(),NoiseModel(white_sd=.1,ar_sd=1.5,tau=3000.0,cap=300.0))
    capped_model=SystemModel(model.sys,CookLog(collect(model.log.t_grid),collect(model.log.u);sensors=[capped]),model.blk)
    capped_sm=sampled_init(capped_model;mode=:offline,particles=16,max_particles=16,rng=MersenneTwister(820))
    capped_prior=capped_sm.posterior
    @test_throws ArgumentError assimilate!(capped_sm,MeasurementRecord(:pit,0.0,300.0))
    @test capped_sm.posterior===capped_prior
    @test isempty(capped_sm.graph_backend.obs)

    # An analytically Gaussian case must succeed through the same public backend.
    sm.particles=2048;sm.max_particles=2048
    result=assimilate!(sm,MeasurementRecord(:pit,0.0,reading))
    @test result.committed
    @test result.inference===:tempered_smc
    @test result.posterior===sm.posterior
    @test result.diagnostics.mc_precision_ok
    @test length(inner.obs)==1
    @test inner.record_count==records+1
    @test belief_at(sm).x==result.x
    ky=inner.nodes[1][2];ip=first(E.belief_index(model.bl,:pit,:T_pit))
    prior=only(f for f in factors_of(inner.graph,ky) if f isa GaussianPrior)
    C0=inv(prior.R'*prior.R);v0=C0[ip,ip]
    spec=model.log.sensors[1];R=spec.noise.white_sd^2+spec.noise.ar_sd^2
    expected=prior.m[ip]+v0/(v0+R)*(reading-prior.m[ip])
    @test abs(result.T_pit[1]-expected)<6sqrt(v0*R/(v0+R)/result.diagnostics.n)
end
