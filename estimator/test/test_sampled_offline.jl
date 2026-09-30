using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","BrisketEstimator.jl"))
using .BrisketEstimator,BrisketDynamics,LinearAlgebra,Random,Test
const E=BrisketEstimator
include(joinpath(@__DIR__,"..","scripts","consistency_control.jl"))

@testset "offline sampled entry point and retained histories" begin
    model=consistency_model(nx=4,suite=:pit,stop=600.)
    sm=online_init(model);ky=sm.nodes[1][2];ip=first(E.belief_index(model.bl,:pit,:T_pit));nz=length(model.blk)
    prior=only(f for f in factors_of(sm.graph,ky) if f isa GaussianPrior)
    v=inv(prior.R'*prior.R)[ip,ip];spec=only(model.log.sensors);R=spec.noise.ar_sd^2+spec.noise.white_sd^2/3
    readings=fill(MeasurementRecord(:pit,0.,prior.m[ip]+1.),3);saved=copy(readings)
    p=sampled_fit(model,readings;particles=1024,max_particles=2048,rng=MersenneTwister(471))
    @test readings==saved
    @test p.diagnostics.mc_precision_ok
    @test p.history!==nothing
    @test abs(p.mean[nz+ip]-(prior.m[ip]+v/(v+R)))<6p.diagnostics.mean_mcse[nz+ip]
    @test abs(p.covariance[nz+ip,nz+ip]/(v*R/(v+R))-1)<.08
    for j in (1,length(p.weights))
        @test E._transport_draw(p.history.reference,p.history.coordinates[:,j]).draw≈p.draws[:,j]
    end
    @test_throws ArgumentError sampled_fit(model,readings;particles=1)
    @test_throws ArgumentError sampled_fit(model,readings;lag=3)
    @test_throws ArgumentError sampled_fit(model,[MeasurementRecord(:pit,NaN,300.)])
end
